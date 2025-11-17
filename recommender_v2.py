# recommender_v2.py — A/B 切替対応（System A: Linear / System B: Gaussian）
import numpy as np
import pandas as pd
from numpy.linalg import norm
from huggingface_hub import snapshot_download
from pathlib import Path
from dataclasses import dataclass

# =========================================================
# ヘルパ関数
# =========================================================
def _l2_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

def _build_onehot(series: pd.Series) -> tuple[np.ndarray, dict]:
    """単一カテゴリ列 → one-hot（float32）。UNK を必ず含める。"""
    vals = series.fillna("UNK").astype(str).values
    uniq = sorted(pd.unique(vals).tolist())
    if "UNK" not in uniq:
        uniq = ["UNK"] + [u for u in uniq if u != "UNK"]
    idx = {v: i for i, v in enumerate(uniq)}
    X = np.zeros((len(vals), len(uniq)), dtype=np.float32)
    unk_i = idx["UNK"]
    for r, v in enumerate(vals):
        X[r, idx.get(v, unk_i)] = 1.0
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
    多値カテゴリ列 → multi-hot（bool）。UNK を必ず含め、未知語は UNK にフォールバック。
    """
    raw_list = series.fillna("").astype(str).tolist()
    tokens_list = [_split_multi(s) for s in raw_list]
    vocab = {tok for toks in tokens_list for tok in toks if tok}
    vocab.add("UNK")
    uniq = sorted(vocab)
    idx = {v: i for i, v in enumerate(uniq)}
    unk_i = idx["UNK"]
    N, D = len(tokens_list), len(uniq)
    X = np.zeros((N, D), dtype=bool)
    for r, toks in enumerate(tokens_list):
        if not toks:
            X[r, unk_i] = True
        else:
            for t in toks:
                X[r, idx.get(t, unk_i)] = True
    return X, idx, True

# =========================================================
# System A: BaselineEngine（線形正規化 + 余弦）
# =========================================================
@dataclass
class LinearWeights:
    # 大枠：カテゴリ vs 数値
    w_cat: float = 0.40
    w_level: float = 0.30
    w_atk: float = 0.15
    w_def: float = 0.15
    # カテゴリ内の重み
    w_type: float = 0.50
    w_attr: float = 0.25
    w_race: float = 0.25

class BaselineEngine:
    """
    System A（Baseline）: Linear Normalization（Min–Max）+ Cosine Similarity
    学術注釈:
      - Linear Normalization は「知覚差 ~ 物理差」に線形比例を仮定する。
      - ゲーム数値（ATK/DEF/Level）では中～高域で知覚歪み（圧縮/膨張）が生じやすい。
      - 本研究では System B（非線形・RBF）との対照群として用いる。
    """
    def __init__(self,
                 df: pd.DataFrame,
                 level_col: str = "level",
                 atk_col: str = "atk",
                 def_col: str = "def",
                 type_norm: np.ndarray | None = None,
                 attr_norm: np.ndarray | None = None,
                 race_bool: np.ndarray | None = None,
                 race_norm: np.ndarray | None = None,
                 lin_w: LinearWeights = LinearWeights()):
        self.df = df.reset_index(drop=True)
        self.w = lin_w

        # 数値列チェック
        for col in [level_col, atk_col, def_col]:
            if col not in self.df.columns:
                raise ValueError(f"BaselineEngine: 列 '{col}' が存在しません。")

        # 中央値補完
        num = self.df[[level_col, atk_col, def_col]].to_numpy(dtype=np.float32)
        num[~np.isfinite(num)] = np.nan
        med = np.nanmedian(num, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        inds = np.where(~np.isfinite(num))
        if inds[0].size:
            num[inds] = med[inds[1]]

        # 列ごとの Min–Max（[0,1]）
        lo = np.nanmin(num, axis=0)
        hi = np.nanmax(num, axis=0)
        span = np.maximum(hi - lo, 1e-9).astype(np.float32)
        self.num_linear = (num - lo) / span

        # 数値サブ重みの正規化
        num_w = np.array([self.w.w_level, self.w.w_atk, self.w.w_def], dtype=np.float32)
        self.num_w = num_w / (num_w.sum() + 1e-9)

        # カテゴリ表現（外部から受け取り・再利用）
        self.type_norm = type_norm
        self.attr_norm = attr_norm
        self.race_bool = race_bool
        self.race_norm = race_norm

    def _num_sim_vec(self, q_idx: int) -> np.ndarray:
        # 線形正規化後の 3次元を重み付けし、余弦類似度で評価
        X = self.num_linear.astype(np.float32, copy=False)
        Xw = X * self.num_w
        qw = Xw[q_idx]
        nX = np.linalg.norm(Xw, axis=1) + 1e-9
        nq = float(np.linalg.norm(qw) + 1e-9)
        s = (Xw @ qw) / (nX * nq)
        return np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0).astype(np.float32)

    def _cat_sim_vec(self, q_idx: int) -> np.ndarray:
        parts, ws = [], []
        if self.type_norm is not None:
            parts.append(np.dot(self.type_norm, self.type_norm[q_idx])); ws.append(self.w.w_type)
        if self.attr_norm is not None:
            parts.append(np.dot(self.attr_norm, self.attr_norm[q_idx])); ws.append(self.w.w_attr)
        if self.race_bool is not None:
            a = self.race_bool; q = a[q_idx]
            inter = (a & q).sum(axis=1).astype(np.float32)
            union = (a | q).sum(axis=1).astype(np.float32)
            eps = 1e-6
            parts.append((inter + eps) / (np.maximum(union, eps))); ws.append(self.w.w_race)
        elif self.race_norm is not None:
            parts.append(np.dot(self.race_norm, self.race_norm[q_idx])); ws.append(self.w.w_race)

        if not parts:
            return np.zeros(len(self.df), dtype=np.float32)
        W = np.array(ws, dtype=np.float32); W = W / (W.sum() + 1e-9)
        S = np.vstack(parts).T
        return np.clip((S * W).sum(axis=1).astype(np.float32), 0.0, 1.0)

    def similarities(self, q_idx: int) -> np.ndarray:
        s_num = self._num_sim_vec(q_idx)
        s_cat = self._cat_sim_vec(q_idx)
        w_cat = self.w.w_cat
        s = (1.0 - w_cat) * s_num + w_cat * s_cat
        return np.clip(s.astype(np.float32), 0.0, 1.0)

# =========================================================
# System B: MetaEngine（ゲーム単位スケーリング + RBF）
# =========================================================
@dataclass
class MetaWeights:
    # 大枠：カテゴリ vs 数値
    w_cat: float = 0.40
    w_level: float = 0.30
    w_atk: float = 0.15
    w_def: float = 0.15
    # カテゴリ内の重み
    w_type: float = 0.50
    w_attr: float = 0.25
    w_race: float = 0.25

class MetaEngine:
    """
    Meta = 数値(Level/ATK/DEF) + カテゴリ(Type/Attribute/Race)
    - 数値：ゲーム単位空間（Level=1、ATK/DEF=100）で RBF（ガウス核）相似
      学術注釈: Non-linear Perception Mapping（心理物理の近似）として RBF を採用。
    - カテゴリ：type/attribute は余弦，race は Jaccard（multi-hot）
    """
    def __init__(self,
                 df: pd.DataFrame,
                 level_col: str = "level",
                 atk_col: str = "atk",
                 def_col: str = "def",
                 type_col: str = "type",
                 attribute_col: str = "attribute",
                 race_col: str = "race",
                 meta_w: MetaWeights = MetaWeights(),
                 units: tuple[float, float, float] = (1.0, 100.0, 100.0),
                 min_sigma: tuple[float, float, float] = (1.0, 3.0, 3.0),
                 sigma_scale: float = 1.0):

        self.df = df.reset_index(drop=True)
        self.w = meta_w

        # -------- 数値特性の取得とクリーニング --------
        for col in [level_col, atk_col, def_col]:
            if col not in self.df.columns:
                raise ValueError(f"MetaEngine: 列 '{col}' が存在しません。")

        num_raw = self.df[[level_col, atk_col, def_col]].to_numpy(dtype=np.float32)

        num = num_raw.copy()
        num[~np.isfinite(num)] = np.nan
        med = np.nanmedian(num, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        inds = np.where(~np.isfinite(num))
        if inds[0].size:
            num[inds] = med[inds[1]]

        # ゲーム単位スケーリング（Level=1, ATK/DEF=100）
        self.units = np.array(units, dtype=np.float32)
        num_scaled = num / (self.units + 1e-9)
        self.num = num_scaled  # 以降はこの空間で処理

        # RBF の帯域（σ）推定：IQR 基準 + 下限 + スケール
        q25 = np.nanpercentile(num_scaled, 25, axis=0)
        q75 = np.nanpercentile(num_scaled, 75, axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
        sigma = iqr / 1.349
        sigma = np.maximum(sigma, np.array(min_sigma, dtype=np.float32))
        sigma = sigma * float(sigma_scale)
        self.sigma = sigma.astype(np.float32)

        # -------- カテゴリ特性 --------
        self.has_type = type_col in self.df.columns
        self.has_attr = attribute_col in self.df.columns
        self.has_race = race_col in self.df.columns

        if self.has_type:
            Xtype, _ = _build_onehot(self.df[type_col])
            self.type_norm = _l2_rows(Xtype)
        else:
            self.type_norm = None

        if self.has_attr:
            Xattr, _ = _build_onehot(self.df[attribute_col])
            self.attr_norm = _l2_rows(Xattr)
        else:
            self.attr_norm = None

        if self.has_race:
            Xrace, _, is_bool = _build_multihot(self.df[race_col])
            self.race_bool = Xrace if is_bool else None
            self.race_norm = None if is_bool else _l2_rows(Xrace.astype(np.float32))
        else:
            self.race_bool = None
            self.race_norm = None

        # 数値サブ重みの正規化
        num_w = np.array([self.w.w_level, self.w.w_atk, self.w.w_def], dtype=np.float32)
        self.num_w = num_w / (num_w.sum() + 1e-9)

    # -------- 数値相似（RBF） --------
    def _num_sim_vec(self, q_idx: int) -> np.ndarray:
        # z = 差分 / σ，二乗和を重み付きで集約し，exp(-0.5 * quad)
        z = (self.num - self.num[q_idx]) / (self.sigma + 1e-9)
        quad = np.sum(self.num_w * (z ** 2), axis=1)
        s = np.exp(-0.5 * quad).astype(np.float32)
        return np.clip(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    # -------- カテゴリ相似 --------
    def _cat_sim_vec(self, q_idx: int) -> np.ndarray:
        parts, ws = [], []
        if self.type_norm is not None:
            parts.append(np.dot(self.type_norm, self.type_norm[q_idx])); ws.append(self.w.w_type)
        if self.attr_norm is not None:
            parts.append(np.dot(self.attr_norm, self.attr_norm[q_idx])); ws.append(self.w.w_attr)
        if self.race_bool is not None:
            a = self.race_bool; q = a[q_idx]
            inter = (a & q).sum(axis=1).astype(np.float32)
            union = (a | q).sum(axis=1).astype(np.float32)
            eps = 1e-6
            parts.append((inter + eps) / (np.maximum(union, eps))); ws.append(self.w.w_race)
        elif self.race_norm is not None:
            parts.append(np.dot(self.race_norm, self.race_norm[q_idx])); ws.append(self.w.w_race)

        if not parts:
            return np.zeros(len(self.df), dtype=np.float32)
        W = np.array(ws, dtype=np.float32); W = W / (W.sum() + 1e-9)
        S = np.vstack(parts).T
        out = (S * W).sum(axis=1).astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    def similarities(self, q_idx: int) -> np.ndarray:
        s_num = self._num_sim_vec(q_idx)
        s_cat = self._cat_sim_vec(q_idx)
        w_cat = self.w.w_cat
        s = (1.0 - w_cat) * s_num + w_cat * s_cat
        return np.clip(s.astype(np.float32), 0.0, 1.0)

# =========================================================
# メイン推薦器（A/B 分流対応）
# =========================================================
class RecommenderV2:
    """
    二段階召喚（候補収集）+ 融合（RRF/冪平均）+ MMR 再ランキング。
    MetaEngine（RBF）/ BaselineEngine（Linear）を併存し，A/B 実験で切替可能。
    """
    def __init__(self, card_df: pd.DataFrame,
                 art_embs: np.ndarray,
                 lore_embs: np.ndarray,
                 meta_embs: np.ndarray,
                 *,
                 use_meta_engine: bool = False,
                 meta_engine_kwargs: dict = None):

        self.db = card_df.reset_index(drop=True)
        self.art = self._l2(art_embs)
        self.lore = self._l2(lore_embs)
        self.meta = self._l2(meta_embs) if meta_embs is not None else None

        # A/B 用エンジン
        self.meta_engine = None    # System B
        self.linear_engine = None  # System A

        if use_meta_engine:
            meta_engine_kwargs = meta_engine_kwargs or {}
            self.meta_engine = MetaEngine(self.db, **meta_engine_kwargs)
            # 既に構築したカテゴリ表現を BaselineEngine に再利用
            self.linear_engine = BaselineEngine(
                self.db,
                level_col=meta_engine_kwargs.get("level_col", "level"),
                atk_col=meta_engine_kwargs.get("atk_col", "atk"),
                def_col=meta_engine_kwargs.get("def_col", "def"),
                type_norm=getattr(self.meta_engine, "type_norm", None),
                attr_norm=getattr(self.meta_engine, "attr_norm", None),
                race_bool=getattr(self.meta_engine, "race_bool", None),
                race_norm=getattr(self.meta_engine, "race_norm", None),
                lin_w=LinearWeights(
                    w_cat=self.meta_engine.w.w_cat,
                    w_level=self.meta_engine.w.w_level,
                    w_atk=self.meta_engine.w.w_atk,
                    w_def=self.meta_engine.w.w_def,
                    w_type=self.meta_engine.w.w_type,
                    w_attr=self.meta_engine.w.w_attr,
                    w_race=self.meta_engine.w.w_race,
                ),
            )

        self.name2idx = pd.Series(
            self.db.index.values,
            index=self.db["name"].astype(str)
        ).to_dict()

    @staticmethod
    def _l2(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        n = norm(X, axis=1, keepdims=True) + 1e-9
        return X / n

    @staticmethod
    def _topk_idx(sims: np.ndarray, k: int, exclude: int) -> np.ndarray:
        k = min(k, len(sims) - 1)
        idx = np.argpartition(-sims, k)[:k + 1]
        idx = idx[idx != exclude]
        return idx[np.argsort(-sims[idx])]

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

    def _mmr_rerank(self, cand_idx: list[int], rel_scores: dict[int, float],
                    top_n: int, lam: float) -> list[int]:
        if not cand_idx or len(cand_idx) <= 1:
            return cand_idx[:top_n]
        sub = self.art[cand_idx]
        sim_mat = np.dot(sub, sub.T).astype(np.float32)
        np.fill_diagonal(sim_mat, 0.0)
        selected = []
        remaining = list(range(len(cand_idx)))
        rel_vec_all = np.array([rel_scores.get(cand_idx[i], 0.0) for i in range(len(cand_idx))], dtype=np.float32)
        first = int(np.argmax(rel_vec_all[remaining]))
        selected.append(remaining.pop(first))
        while len(selected) < min(top_n, len(cand_idx)) and remaining:
            penal = sim_mat[remaining][:, selected].max(axis=1)
            rel_now = rel_vec_all[remaining]
            mmr = lam * rel_now - (1.0 - lam) * penal
            nxt_local = int(np.argmax(mmr))
            selected.append(remaining.pop(nxt_local))
        return [cand_idx[i] for i in selected]

    def recommend(self, query_name: str,
                  top_n: int = 12,
                  k_each: int = 150,
                  fusion: str = "rrf",
                  p_power: float = 1.5,
                  use_mmr: bool = True,
                  mmr_lambda: float = 0.7,
                  w_art: float = 1.0,
                  w_lore: float = 1.0,
                  w_meta: float = 1.0,
                  ab_system: str = "B"  # "A"=Baseline（Linear）, "B"=Proposed（RBF）
                  ) -> pd.DataFrame:

        if query_name not in self.name2idx:
            raise ValueError(f"Card '{query_name}' not found.")
        q = self.name2idx[query_name]

        # 画像・テキスト類似（共通）
        art_s  = np.dot(self.art,  self.art[q]).astype(np.float32)
        lore_s = np.dot(self.lore, self.lore[q]).astype(np.float32)

        # ---------- A/B 切替（数値知覚モデリングの実験変数） ----------
        # System A: Linear Normalization + Cosine（対照）
        # System B: Non-linear Perception Mapping with Gaussian Kernel（提案）
        if (ab_system or "B").upper().startswith("A"):
            if self.linear_engine is not None:
                meta_s = self.linear_engine.similarities(q)
            else:
                meta_s = np.dot(self.meta, self.meta[q]).astype(np.float32) if self.meta is not None else \
                         np.zeros(len(self.db), dtype=np.float32)
        else:
            if self.meta_engine is not None:
                meta_s = self.meta_engine.similarities(q)
            else:
                meta_s = np.dot(self.meta, self.meta[q]).astype(np.float32) if self.meta is not None else \
                         np.zeros(len(self.db), dtype=np.float32)

        # 候補集合
        art_c  = self._topk_idx(art_s,  k_each, q)
        lore_c = self._topk_idx(lore_s, k_each, q)
        meta_c = self._topk_idx(meta_s, k_each, q)
        pool = list(set(art_c) | set(lore_c) | set(meta_c))

        # 融合（RRF / 冪平均）
        mw = {"art": w_art, "lore": w_lore, "meta": w_meta}
        if fusion == "rrf":
            def ranks_from(scores: np.ndarray) -> dict[int, int]:
                return {cid: r + 1 for r, cid in enumerate(sorted(pool, key=lambda i: -scores[i]))}
            fused = self._rrf({"art": ranks_from(art_s),
                               "lore": ranks_from(lore_s),
                               "meta": ranks_from(meta_s)}, modality_weights=mw)
        else:
            fused = self._power_mean({
                "art":  self._minmax_on_pool(art_s,  pool),
                "lore": self._minmax_on_pool(lore_s, pool),
                "meta": self._minmax_on_pool(meta_s, pool),
            }, p=p_power, modality_weights=mw)

        pre = sorted(pool, key=lambda i: -fused.get(i, 0.0))
        final_idx = self._mmr_rerank(pre[:3 * top_n], fused, top_n, mmr_lambda) if use_mmr else pre[:top_n]

        out = self.db.iloc[final_idx].copy()
        out["art_sim"]     = [float(art_s[i]) for i in final_idx]
        out["lore_sim"]    = [float(lore_s[i]) for i in final_idx]
        out["meta_sim"]    = [float(meta_s[i]) for i in final_idx]
        out["final_score"] = [float(fused.get(i, 0.0)) for i in final_idx]
        return out

    # -------- デバッグ：数値/カテゴリ/統合の分解 --------
    def debug_meta_components(self, query_name: str, rank: int = 0):
        if query_name not in self.name2idx:
            raise ValueError(f"Card '{query_name}' not found.")
        if self.meta_engine is None:
            print("[DEBUG] MetaEngine disabled.")
            return None
        q = self.name2idx[query_name]
        art_s  = np.dot(self.art,  self.art[q]).astype(np.float32)
        lore_s = np.dot(self.lore, self.lore[q]).astype(np.float32)
        meta_s = self.meta_engine.similarities(q)
        k_each = 150
        art_c  = self._topk_idx(art_s,  k_each, q)
        lore_c = self._topk_idx(lore_s, k_each, q)
        meta_c = self._topk_idx(meta_s, k_each, q)
        pool   = list(set(art_c) | set(lore_c) | set(meta_c))
        pre = sorted(pool, key=lambda i: -meta_s[i])
        if rank >= len(pre):
            raise ValueError(f"rank {rank} >= pool size {len(pre)}")
        i = pre[rank]
        s_num  = float(self.meta_engine._num_sim_vec(q)[i])
        s_cat  = float(self.meta_engine._cat_sim_vec(q)[i])
        s_meta = float(meta_s[i])
        return {"q": int(q), "i": int(i), "name": str(self.db.iloc[i]["name"]),
                "s_num": s_num, "s_cat": s_cat, "s_meta": s_meta,
                "sigma": self.meta_engine.sigma.copy()}

    # -------- Hugging Face データセットからロード --------
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
        meta = np.load(local_dir / "meta_embs.npz")["data"]
        return cls(df, art, lore, meta,
                   use_meta_engine=use_meta_engine,
                   meta_engine_kwargs=meta_engine_kwargs)