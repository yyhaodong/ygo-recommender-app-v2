# recommender_v2.py — final optimized (MetaEngine + debug_meta_components) [UNK-safe]
import numpy as np
import pandas as pd
from numpy.linalg import norm
from huggingface_hub import snapshot_download
from pathlib import Path
from dataclasses import dataclass

# ---------------------- Helper: one-hot / multi-hot ----------------------
def _build_onehot(series: pd.Series) -> tuple[np.ndarray, dict]:
    """单值类别 → one-hot（dense float32）；返回矩阵与{值→列}映射"""
    vals = series.astype(str).fillna("UNK").values
    uniq = sorted(pd.unique(vals).tolist())
    if "UNK" not in uniq:
        uniq = ["UNK"] + [u for u in uniq if u != "UNK"]
    idx = {v: i for i, v in enumerate(uniq)}
    X = np.zeros((len(vals), len(uniq)), dtype=np.float32)
    unk_i = idx["UNK"]
    for r, v in enumerate(vals):
        j = idx.get(v, unk_i)
        X[r, j] = 1.0
    return X, idx

def _split_multi(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    for sep in ["|", ",", "/", ";"]:
        if sep in s:
            toks = [t.strip() for t in s.split(sep)]
            return [t for t in toks if t]
    return [s]

def _build_multihot(series: pd.Series) -> tuple[np.ndarray, dict, bool]:
    """
    多值类别 → multi-hot（优先布尔矩阵，便于 Jaccard）
    - 永远保证 vocab 里有 'UNK'，避免 idx['UNK'] KeyError
    - 对任何异常 token 兜底到 UNK
    """
    raw_list = series.fillna("").astype(str).tolist()
    tokens_list = [_split_multi(s) for s in raw_list]

    # 建 vocab，并**强制加入 'UNK'**
    vocab = {tok for toks in tokens_list for tok in toks if tok}
    vocab.add("UNK")
    uniq = sorted(vocab)
    idx = {v: i for i, v in enumerate(uniq)}
    unk_i = idx["UNK"]

    # 构造布尔 multi-hot
    N, D = len(tokens_list), len(uniq)
    X = np.zeros((N, D), dtype=bool)
    for r, toks in enumerate(tokens_list):
        if not toks:
            X[r, unk_i] = True
        else:
            for t in toks:
                j = idx.get(t, unk_i)
                X[r, j] = True

    return X, idx, True

def _l2_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

# ---------------------- MetaEngine（游戏逻辑友好） ----------------------
@dataclass
class MetaWeights:
    w_cat: float = 0.40  # 类别大块
    w_level: float = 0.30
    w_atk: float = 0.15
    w_def: float = 0.15
    # 类别内部按组（type/attribute/race）
    w_type: float = 0.50
    w_attr: float = 0.25
    w_race: float = 0.25

class MetaEngine:
    """
    从 DataFrame 拆出 Meta 的数值 + 类别特征，并提供“以某张卡为查询”的相似度向量。
    - 数值（level/atk/def）：稳健高斯核（带宽=IQR/1.349），可按需改窄
    - 类别：type/attribute 用 cosine；race 多热优先 Jaccard
    """
    def __init__(self,
                 df: pd.DataFrame,
                 level_col: str = "level",
                 atk_col: str = "atk",
                 def_col: str = "def",
                 type_col: str = "type",
                 attribute_col: str = "attribute",
                 race_col: str = "race",
                 meta_w: MetaWeights = MetaWeights()):

        self.df = df.reset_index(drop=True)
        self.w = meta_w

        # -------- 数值矩阵：N x 3 --------
        for col in [level_col, atk_col, def_col]:
            if col not in self.df.columns:
                raise ValueError(f"MetaEngine: 缺少列 '{col}'")
        num = self.df[[level_col, atk_col, def_col]].to_numpy(dtype=np.float32)
        self.num = num
        # 稳健带宽
        q25 = np.percentile(num, 25, axis=0)
        q75 = np.percentile(num, 75, axis=0)
        iqr = np.maximum(q75 - q25, 1e-8)
        self.sigma = iqr / 1.349  # 如需更“锋利”，可改为 iqr / 2.0 或手动设置

        # -------- 类别矩阵 --------
        self.has_type = type_col in self.df.columns
        self.has_attr = attribute_col in self.df.columns
        self.has_race = race_col in self.df.columns

        if self.has_type:
            Xtype, _ = _build_onehot(self.df[type_col].fillna("UNK").astype(str))
            self.type_norm = _l2_rows(Xtype)
        else:
            self.type_norm = None

        if self.has_attr:
            Xattr, _ = _build_onehot(self.df[attribute_col].fillna("UNK").astype(str))
            self.attr_norm = _l2_rows(Xattr)
        else:
            self.attr_norm = None

        if self.has_race:
            # 关键：传入 fillna("").astype(str) 保证稳定
            Xrace, _, is_bool = _build_multihot(self.df[race_col].fillna("").astype(str))
            self.race_bool = Xrace if is_bool else None
            self.race_norm = None if is_bool else _l2_rows(Xrace.astype(np.float32))
        else:
            self.race_bool = None
            self.race_norm = None

        # 预先归一化好的数值权重（level/atk/def）
        num_w = np.array([self.w.w_level, self.w.w_atk, self.w.w_def], dtype=np.float32)
        self.num_w = num_w / (num_w.sum() + 1e-9)

    def _num_sim_vec(self, q_idx: int) -> np.ndarray:
        z = (self.num - self.num[q_idx]) / (self.sigma + 1e-8)  # N x 3
        quad = np.sum(self.num_w * (z ** 2), axis=1)            # N
        return np.exp(-0.5 * quad).astype(np.float32)

    def _cat_sim_vec(self, q_idx: int) -> np.ndarray:
        parts, ws = [], []
        if self.type_norm is not None:
            parts.append(np.dot(self.type_norm, self.type_norm[q_idx]))
            ws.append(self.w.w_type)
        if self.attr_norm is not None:
            parts.append(np.dot(self.attr_norm, self.attr_norm[q_idx]))
            ws.append(self.w.w_attr)
        if self.race_bool is not None:
            a = self.race_bool
            q = a[q_idx]
            inter = (a & q).sum(axis=1).astype(np.float32)
            union = (a | q).sum(axis=1).astype(np.float32)
            parts.append(np.divide(inter, np.maximum(union, 1e-8)))
            ws.append(self.w.w_race)
        elif self.race_norm is not None:
            parts.append(np.dot(self.race_norm, self.race_norm[q_idx]))
            ws.append(self.w.w_race)

        if not parts:
            return np.zeros(len(self.df), dtype=np.float32)
        W = np.array(ws, dtype=np.float32)
        W = W / (W.sum() + 1e-9)
        S = np.vstack(parts).T   # N x G
        return (S * W).sum(axis=1).astype(np.float32)

    def similarities(self, q_idx: int) -> np.ndarray:
        s_num = self._num_sim_vec(q_idx)
        s_cat = self._cat_sim_vec(q_idx)
        w_cat = self.w.w_cat
        return ((1.0 - w_cat) * s_num + w_cat * s_cat).astype(np.float32)

# ---------------------- 主推荐器 ----------------------
class RecommenderV2:
    """
    一个功能完整的、经过性能优化的多模态推荐器。
    - 两阶段召回：argpartition 快速取 Top-K 候选
    - 融合：RRF（可加模态权重）/ 幂平均（可加模态权重）
    - 重排：MMR 提升多样性
    - 预处理：向量 L2 归一化 + float32，点积即余弦
    - 新：可选 MetaEngine（游戏逻辑友好数值+类别混合度量）
    """

    def __init__(self, card_df: pd.DataFrame,
                 art_embs: np.ndarray,
                 lore_embs: np.ndarray,
                 meta_embs: np.ndarray,
                 *,
                 use_meta_engine: bool = False,
                 meta_engine_kwargs: dict = None):

        self.db = card_df.reset_index(drop=True)
        # 预 L2 归一化，后续使用点积即可
        self.art = self._l2(art_embs)
        self.lore = self._l2(lore_embs)

        # 旧：向量化 meta（兼容保留）
        self.meta = self._l2(meta_embs) if meta_embs is not None else None

        # 可选：启用 MetaEngine
        self.meta_engine = None
        if use_meta_engine:
            meta_engine_kwargs = meta_engine_kwargs or {}
            self.meta_engine = MetaEngine(self.db, **meta_engine_kwargs)

        # name/idx 快速映射
        self.name2idx = pd.Series(
            self.db.index.values,
            index=self.db["name"].astype(str)
        ).to_dict()

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
        idx = idx[idx != exclude]  # 排除查询自身
        return idx[np.argsort(-sims[idx])]

    # --------- 融合（支持模态权重） ----------
    @staticmethod
    def _rrf(ranks_dict: dict, k: int = 60, modality_weights: dict | None = None) -> dict[int, float]:
        mw = modality_weights or {"art": 1.0, "lore": 1.0, "meta": 1.0}
        fused = {}
        for m, ranks in ranks_dict.items():
            w = float(mw.get(m, 1.0))
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + w * (1.0 / (k + r))
        return fused

    @staticmethod
    def _power_mean(sim_dict: dict[str, dict[int, float]],
                    p: float = 1.5,
                    modality_weights: dict | None = None) -> dict[int, float]:
        mw = modality_weights or {"art": 1.0, "lore": 1.0, "meta": 1.0}
        fused = {}
        all_ids = set()
        for d in sim_dict.values():
            all_ids.update(d.keys())
        mw_vec = np.array([mw.get("art",1.0), mw.get("lore",1.0), mw.get("meta",1.0)], dtype=np.float32)
        mw_vec = mw_vec / (mw_vec.sum() + 1e-9)
        for cid in all_ids:
            v_art = sim_dict.get("art", {}).get(cid, 0.0)
            v_lor = sim_dict.get("lore", {}).get(cid, 0.0)
            v_met = sim_dict.get("meta", {}).get(cid, 0.0)
            v = np.array([v_art, v_lor, v_met], dtype=np.float32)
            s = np.sum(mw_vec * (v ** p))
            fused[cid] = float((s) ** (1.0 / p))
        return fused

    @staticmethod
    def _minmax_on_pool(scores: np.ndarray, pool: list[int]) -> dict[int, float]:
        if not pool:
            return {}
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
        sim_mat = np.dot(sub, sub.T).astype(np.float32)
        np.fill_diagonal(sim_mat, 0.0)

        selected_indices_in_pool = []
        remaining_indices_in_pool = list(range(len(cand_idx)))
        rel_vec_all = np.array([rel_scores.get(cand_idx[i], 0.0) for i in range(len(cand_idx))], dtype=np.float32)

        first_local_idx = int(np.argmax(rel_vec_all[remaining_indices_in_pool]))
        selected_indices_in_pool.append(remaining_indices_in_pool.pop(first_local_idx))

        while len(selected_indices_in_pool) < min(top_n, len(cand_idx)) and remaining_indices_in_pool:
            penal = sim_mat[remaining_indices_in_pool][:, selected_indices_in_pool].max(axis=1)
            rel_now = rel_vec_all[remaining_indices_in_pool]
            mmr = lam * rel_now - (1.0 - lam) * penal
            nxt_local = int(np.argmax(mmr))
            nxt_global = remaining_indices_in_pool.pop(nxt_local)
            selected_indices_in_pool.append(nxt_global)

        return [cand_idx[i] for i in selected_indices_in_pool]

    # --------- 主流程（按名称） ----------
    def recommend(self, query_name: str,
                  top_n: int = 12,
                  k_each: int = 150,
                  fusion: str = "rrf",
                  p_power: float = 1.5,
                  use_mmr: bool = True,
                  mmr_lambda: float = 0.7,
                  # 高层模态权重（来自主滑块）
                  w_art: float = 1.0,
                  w_lore: float = 1.0,
                  w_meta: float = 1.0) -> pd.DataFrame:

        if query_name not in self.name2idx:
            raise ValueError(f"Card '{query_name}' not found.")
        q = self.name2idx[query_name]

        # --- 三通道相似度向量 ---
        art_s = np.dot(self.art, self.art[q]).astype(np.float32)
        lore_s = np.dot(self.lore, self.lore[q]).astype(np.float32)

        if self.meta_engine is not None:
            meta_s = self.meta_engine.similarities(q)  # 新版：数值+类别混合
        else:
            if self.meta is None:
                meta_s = np.zeros(len(self.db), dtype=np.float32)
            else:
                meta_s = np.dot(self.meta, self.meta[q]).astype(np.float32)

        # --- 召回池 ---
        art_c  = self._topk_idx(art_s,  k_each, q)
        lore_c = self._topk_idx(lore_s, k_each, q)
        meta_c = self._topk_idx(meta_s, k_each, q)
        pool = list(set(art_c) | set(lore_c) | set(meta_c))

        # --- 融合（支持模态权重） ---
        mw = {"art": w_art, "lore": w_lore, "meta": w_meta}
        if fusion == "rrf":
            def ranks_from(scores: np.ndarray) -> dict[int, int]:
                return {cid: r + 1 for r, cid in enumerate(sorted(pool, key=lambda i: -scores[i]))}
            fused = self._rrf({
                "art":  ranks_from(art_s),
                "lore": ranks_from(lore_s),
                "meta": ranks_from(meta_s)
            }, modality_weights=mw)
        else:
            fused = self._power_mean({
                "art":  self._minmax_on_pool(art_s,  pool),
                "lore": self._minmax_on_pool(lore_s, pool),
                "meta": self._minmax_on_pool(meta_s, pool),
            }, p=p_power, modality_weights=mw)

        # --- MMR 重排 ---
        pre_ranked = sorted(pool, key=lambda i: -fused.get(i, 0.0))
        if use_mmr:
            final_idx = self._mmr_rerank(pre_ranked[:3 * top_n], fused, top_n, mmr_lambda)
        else:
            final_idx = pre_ranked[:top_n]

        out = self.db.iloc[final_idx].copy()
        out["art_sim"]   = [float(art_s[i]) for i in final_idx]
        out["lore_sim"]  = [float(lore_s[i]) for i in final_idx]
        out["meta_sim"]  = [float(meta_s[i]) for i in final_idx]
        out["final_score"] = [float(fused.get(i, 0.0)) for i in final_idx]
        return out

    # --------- 调试方法A：打印数值/类别/融合后的 meta 组件 ----------
    def debug_meta_components(self, query_name: str, rank: int = 0):
        """
        打印并返回：数值相似 s_num、类别相似 s_cat、融合后的 meta 相似 s_meta。
        rank=0 表示top1候选，rank=1 表示第二名……
        仅在 use_meta_engine=True 时有效。
        """
        if query_name not in self.name2idx:
            raise ValueError(f"Card '{query_name}' not found.")
        if self.meta_engine is None:
            print("[DEBUG] MetaEngine disabled → 正在使用旧的 meta_emb 余弦相似。")
            return None

        q = self.name2idx[query_name]
        # 三通道
        art_s  = np.dot(self.art,  self.art[q]).astype(np.float32)
        lore_s = np.dot(self.lore, self.lore[q]).astype(np.float32)
        meta_s = self.meta_engine.similarities(q)

        # 构造候选池（与 recommend 保持一致）
        k_each = 150
        art_c  = self._topk_idx(art_s,  k_each, q)
        lore_c = self._topk_idx(lore_s, k_each, q)
        meta_c = self._topk_idx(meta_s, k_each, q)
        pool   = list(set(art_c) | set(lore_c) | set(meta_c))

        # 仅按 meta_s 排一下，查看第 rank 名
        pre = sorted(pool, key=lambda i: -meta_s[i])
        if rank >= len(pre):
            raise ValueError(f"rank {rank} >= pool size {len(pre)}")
        i = pre[rank]

        # 组件值
        s_num  = float(self.meta_engine._num_sim_vec(q)[i])
        s_cat  = float(self.meta_engine._cat_sim_vec(q)[i])
        s_meta = float(meta_s[i])

        print(f"[DEBUG] query='{query_name}' q_idx={q}  cand_idx={i}  name={self.db.iloc[i]['name']}")
        print(f"[DEBUG] s_num={s_num:.6f}  s_cat={s_cat:.6f}  s_meta={s_meta:.6f}")
        print(f"[DEBUG] sigma(level, atk, def) = {self.meta_engine.sigma}")
        return {"q": int(q), "i": int(i), "name": str(self.db.iloc[i]["name"]),
                "s_num": s_num, "s_cat": s_cat, "s_meta": s_meta,
                "sigma": self.meta_engine.sigma.copy()}

    # --------- 辅助：以图搜图找到最近邻的英文卡名 ----------
    def nearest_card_by_art(self, art_vec: np.ndarray) -> tuple[int, str, float]:
        v = art_vec.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = np.dot(self.art, v).astype(np.float32)
        idx = int(np.argmax(sims))
        name = str(self.db.iloc[idx]["name"])
        return idx, name, float(sims[idx])

    # 可选：直接用图片向量做推荐（不映射卡名）
    def recommend_by_art_vector(self, art_vec: np.ndarray,
                                top_n: int = 12,
                                k_each: int = 150,
                                fusion: str = "rrf",
                                p_power: float = 1.5,
                                use_mmr: bool = True,
                                mmr_lambda: float = 0.7,
                                w_art: float = 1.0,
                                w_lore: float = 1.0,
                                w_meta: float = 1.0) -> pd.DataFrame:
        v = art_vec.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        art_s = np.dot(self.art, v).astype(np.float32)
        lore_s = np.zeros_like(art_s, dtype=np.float32)
        meta_s = np.zeros_like(art_s, dtype=np.float32)

        def _topk(scores):
            k = min(k_each, len(scores))
            idx = np.argpartition(-scores, k-1)[:k]
            return idx[np.argsort(-scores[idx])]
        art_c = _topk(art_s)
        pool = list(set(art_c))

        mw = {"art": w_art, "lore": w_lore, "meta": w_meta}
        if fusion == "rrf":
            ranks = {cid: r+1 for r, cid in enumerate(sorted(pool, key=lambda i: -art_s[i]))}
            fused = self._rrf({"art": ranks}, modality_weights=mw)
        else:
            fused = self._power_mean({"art": self._minmax_on_pool(art_s, pool),
                                      "lore": {i:0.0 for i in pool},
                                      "meta": {i:0.0 for i in pool}}, p=p_power, modality_weights=mw)

        pre = sorted(pool, key=lambda i: -fused.get(i, 0.0))
        final_idx = self._mmr_rerank(pre[:3*top_n], fused, top_n, mmr_lambda) if use_mmr else pre[:top_n]

        out = self.db.iloc[final_idx].copy()
        out["art_sim"] = [float(art_s[i]) for i in final_idx]
        out["lore_sim"] = [0.0 for _ in final_idx]
        out["meta_sim"] = [0.0 for _ in final_idx]
        out["final_score"] = [float(fused.get(i, 0.0)) for i in final_idx]
        return out

    # --------- 从 Hugging Face 数据集加载 ----------
    @classmethod
    def from_hf(cls, repo_id: str,
                *,
                use_meta_engine: bool = False,
                meta_engine_kwargs: dict = None):
        local_dir = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.parquet", "*.npz"]
        ))
        df   = pd.read_parquet(local_dir / "card_database.parquet")
        art  = np.load(local_dir / "art_embs.npz")["data"]
        lore = np.load(local_dir / "lore_embs.npz")["data"]
        meta = np.load(local_dir / "meta_embs.npz")["data"]  # 兼容旧

        return cls(df, art, lore, meta,
                   use_meta_engine=use_meta_engine,
                   meta_engine_kwargs=meta_engine_kwargs)
