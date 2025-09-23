# recommender_v2.py — final optimized version
import numpy as np
import pandas as pd
from numpy.linalg import norm
from huggingface_hub import snapshot_download
from pathlib import Path


class RecommenderV2:
    """
    - 两阶段召回：argpartition 快速取 Top-K 候选
    - 融合：RRF（稳健）/ 幂平均（可调 p）
    - 重排：MMR 提升多样性
    - 预处理：向量 L2 归一化 + float32，点积即余弦
    """

    def __init__(self, card_df: pd.DataFrame,
                 art_embs: np.ndarray,
                 lore_embs: np.ndarray,
                 meta_embs: np.ndarray):
        self.db = card_df.reset_index(drop=True)
        # 预 L2 归一化，后续使用点积即可
        self.art = self._l2(art_embs)
        self.lore = self._l2(lore_embs)
        self.meta = self._l2(meta_embs)

        # name/idx 快速映射
        self.name2idx = pd.Series(self.db.index.values,
                                  index=self.db["name"].astype(str)).to_dict()

    @staticmethod
    def _l2(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        n = norm(X, axis=1, keepdims=True) + 1e-9
        return X / n

    # --------- 召回工具 ----------
    @staticmethod
    def _topk_idx(sims: np.ndarray, k: int, exclude: int) -> np.ndarray:
        k = min(k, len(sims) - 1)
        idx = np.argpartition(-sims, k)[:k + 1]
        # 排除查询自身
        idx = idx[idx != exclude]
        return idx[np.argsort(-sims[idx])]

    # --------- 融合 ----------
    @staticmethod
    def _rrf(ranks_dict: dict, k: int = 60) -> dict[int, float]:
        """Reciprocal Rank Fusion; 对分数尺度不敏感，更稳健。"""
        fused = {}
        for ranks in ranks_dict.values():
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r)
        return fused

    @staticmethod
    def _power_mean(sim_dict: dict[int, dict[int, float]], p: float = 1.5) -> dict[int, float]:
        """幂平均融合；p>1 时奖励“多模态同时高分”的候选。"""
        fused = {}
        # 汇总所有候选 id
        all_ids = set()
        for d in sim_dict.values():
            all_ids.update(d.keys())
        for cid in all_ids:
            s = 0.0
            cnt = 0
            for m in ("art", "lore", "meta"):
                v = sim_dict[m].get(cid, 0.0)
                s += v ** p
                cnt += 1
            fused[cid] = (s / cnt) ** (1.0 / p)
        return fused

    @staticmethod
    def _minmax_on_pool(scores: np.ndarray, pool: list[int]) -> dict[int, float]:
        v = scores[pool]
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            return {i: 0.0 for i in pool}
        return {i: float((scores[i] - lo) / (hi - lo)) for i in pool}

    # --------- MMR 重排 ----------
    def _mmr_rerank(self, cand_idx: list[int], rel_scores: dict[int, float],
                    top_n: int, lam: float) -> list[int]:
        if not cand_idx or len(cand_idx) <= 1:
            return cand_idx[:top_n]

        sub = self.art[cand_idx]
        # 候选之间的“艺术相似度”
        sim_mat = np.dot(sub, sub.T).astype(np.float32)
        np.fill_diagonal(sim_mat, 0.0)

        selected = []
        remaining = list(range(len(cand_idx)))

        rel_vec = np.array([rel_scores.get(cand_idx[i], 0.0) for i in remaining], dtype=np.float32)
        # 先选相关性最高的
        first = int(np.argmax(rel_vec))
        selected.append(remaining.pop(first))

        while len(selected) < min(top_n, len(cand_idx)) and remaining:
            # 与已选集合的最大相似度 = 多样性惩罚
            penal = sim_mat[remaining][:, selected].max(axis=1)
            mmr = lam * rel_vec[remaining] - (1.0 - lam) * penal
            nxt_local = int(np.argmax(mmr))
            nxt_global = remaining.pop(nxt_local)
            selected.append(nxt_global)

        return [cand_idx[i] for i in selected]

    # --------- 主流程 ----------
    def recommend(self, query_name: str,
                  top_n: int = 12,
                  k_each: int = 150,
                  fusion: str = "rrf",  # "rrf" | "power_mean"
                  p_power: float = 1.5,
                  use_mmr: bool = True,
                  mmr_lambda: float = 0.7) -> pd.DataFrame:
        if query_name not in self.name2idx:
            raise ValueError(f"Card '{query_name}' not found.")
        q = self.name2idx[query_name]

        # 1) 三模态相似度（已归一化 → 点积即余弦）
        art_s = np.dot(self.art, self.art[q])
        lore_s = np.dot(self.lore, self.lore[q])
        meta_s = np.dot(self.meta, self.meta[q])

        # 2) 各取 Top-K 候选 + 合并去重
        art_c = self._topk_idx(art_s, k_each, q)
        lore_c = self._topk_idx(lore_s, k_each, q)
        meta_c = self._topk_idx(meta_s, k_each, q)
        pool = list(set(art_c) | set(lore_c) | set(meta_c))

        # 3) 融合打分
        if fusion == "rrf":
            # 用“排名”而非“分数”融合，更稳健
            def ranks_from(scores: np.ndarray) -> dict[int, int]:
                # 仅在候选池内排序
                return {cid: r + 1 for r, cid in enumerate(sorted(pool, key=lambda i: -scores[i]))}

            fused = self._rrf({
                "art": ranks_from(art_s),
                "lore": ranks_from(lore_s),
                "meta": ranks_from(meta_s)
            })
        else:
            # 先做池内 min-max 归一化，再走幂平均
            fused = self._power_mean({
                "art": self._minmax_on_pool(art_s, pool),
                "lore": self._minmax_on_pool(lore_s, pool),
                "meta": self._minmax_on_pool(meta_s, pool),
            }, p=p_power)

        # 4) （可选）MMR 多样性重排
        pre_ranked = sorted(pool, key=lambda i: -fused.get(i, 0.0))
        if use_mmr:
            final_idx = self._mmr_rerank(pre_ranked[:3 * top_n], fused, top_n, mmr_lambda)
        else:
            final_idx = pre_ranked[:top_n]

        # 5) 组织输出（附带各模态相似度与最终分）
        out = self.db.iloc[final_idx].copy()
        out["art_sim"] = [float(art_s[i]) for i in final_idx]
        out["lore_sim"] = [float(lore_s[i]) for i in final_idx]
        out["meta_sim"] = [float(meta_s[i]) for i in final_idx]
        out["final_score"] = [float(fused.get(i, 0.0)) for i in final_idx]
        return out

    # --------- 便捷构造（从 Hugging Face 数据集加载） ----------
    @classmethod
    def from_hf(cls, repo_id: str):
        local_dir = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.parquet", "*.npz"]
        ))
        df = pd.read_parquet(local_dir / "card_database.parquet")
        art = np.load(local_dir / "art_embs.npz")["data"]
        lore = np.load(local_dir / "lore_embs.npz")["data"]
        meta = np.load(local_dir / "meta_embs.npz")["data"]
        return cls(df, art, lore, meta)
