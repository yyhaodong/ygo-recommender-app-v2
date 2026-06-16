# recommender_v2.py — A/B 切替対応（System A: Linear / System B: Gaussian Route-B）
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
    w_cat: float = 0.40
    w_level: float = 0.30
    w_atk: float = 0.15
    w_def: float = 0.15
    w_type: float = 0.50
    w_attr: float = 0.25
    w_race: float = 0.25


class BaselineEngine:
    def __init__(self, df, level_col="level", atk_col="atk", def_col="def",
                 type_norm=None, attr_norm=None, race_bool=None, race_norm=None,
                 lin_w=LinearWeights()):
        self.df = df.reset_index(drop=True)
        self.w = lin_w
        for col in [level_col, atk_col, def_col]:
            if col not in self.df.columns:
                raise ValueError(f"BaselineEngine: 列 '{col}' が存在しません。")
        num = self.df[[level_col, atk_col, def_col]].to_numpy(dtype=np.float32)
        num[~np.isfinite(num)] = np.nan
        med = np.nanmedian(num, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        inds = np.where(~np.isfinite(num))
        if inds[0].size:
            num[inds] = med[inds[1]]
        lo = np.nanmin(num, axis=0)
        hi = np.nanmax(num, axis=0)
        span = np.maximum(hi - lo, 1e-9).astype(np.float32)
        self.num_linear = (num - lo) / span
        num_w = np.array([self.w.w_level, self.w.w_atk, self.w.w_def], dtype=np.float32)
        self.num_w = num_w / (num_w.sum() + 1e-9)
        self.type_norm = type_norm
        self.attr_norm = attr_norm
        self.race_bool = race_bool
        self.race_norm = race_norm

    def _num_sim_vec(self, q_idx):
        X = self.num_linear.astype(np.float32, copy=False)
        Xw = X * self.num_w
        qw = Xw[q_idx]
        nX = np.linalg.norm(Xw, axis=1) + 1e-9
        nq = float(np.linalg.norm(qw) + 1e-9)
        s = (Xw @ qw) / (nX * nq + 1e-9)
        return np.clip(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0).astype(np.float32)

    def _cat_sim_vec(self, q_idx):
        parts, ws = [], []
        if self.type_norm is not None:
            parts.append(np.dot(self.type_norm, self.type_norm[q_idx])); ws.append(self.w.w_type)
        if self.attr_norm is not None:
            parts.append(np.dot(self.attr_norm, self.attr_norm[q_idx])); ws.append(self.w.w_attr)
        if self.race_bool is not None:
            a = self.race_bool; q = a[q_idx]
            inter = (a & q).sum(axis=1).astype(np.float32)
            union = (a | q).sum(axis=1).astype(np.float32)
            parts.append((inter + 1e-6) / np.maximum(union, 1e-6)); ws.append(self.w.w_race)
        elif self.race_norm is not None:
            parts.append(np.dot(self.race_norm, self.race_norm[q_idx])); ws.append(self.w.w_race)
        if not parts:
            return np.zeros(len(self.df), dtype=np.float32)
        W = np.array(ws, dtype=np.float32); W = W / (W.sum() + 1e-9)
        return np.clip((np.vstack(parts).T * W).sum(axis=1).astype(np.float32), 0.0, 1.0)

    def similarities(self, q_idx):
        s = (1.0 - self.w.w_cat) * self._num_sim_vec(q_idx) + self.w.w_cat * self._cat_sim_vec(q_idx)
        return np.clip(s.astype(np.float32), 0.0, 1.0)


# =========================================================
# System B: MetaEngine（ゲーム単位スケーリング + RBF）
# =========================================================
@dataclass
class MetaWeights:
    w_cat: float = 0.50
    w_level: float = 0.25
    w_atk: float = 0.15
    w_def: float = 0.10
    w_type: float = 0.50
    w_attr: float = 0.25
    w_race: float = 0.25


class MetaEngine:
    def __init__(self, df, level_col="level", atk_col="atk", def_col="def",
                 type_col="type", attribute_col="attribute", race_col="race",
                 meta_w=MetaWeights(), units=(1.0, 100.0, 100.0),
                 min_sigma=(1.0, 3.0, 3.0), sigma_scale=2.0):
        self.df = df.reset_index(drop=True)
        self.w = meta_w
        for col in [level_col, atk_col, def_col]:
            if col not in self.df.columns:
                raise ValueError(f"MetaEngine: 列 '{col}' が存在しません。")
        num = self.df[[level_col, atk_col, def_col]].to_numpy(dtype=np.float32)
        num[~np.isfinite(num)] = np.nan
        med = np.nanmedian(num, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        inds = np.where(~np.isfinite(num))
        if inds[0].size:
            num[inds] = med[inds[1]]
        self.units = np.array(units, dtype=np.float32)
        self.num = num / (self.units + 1e-9)
        q25 = np.nanpercentile(self.num, 25, axis=0)
        q75 = np.nanpercentile(self.num, 75, axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
        sigma = np.maximum(iqr / 1.349, np.array(min_sigma, dtype=np.float32)) * float(sigma_scale)
        self.sigma = sigma.astype(np.float32)
        self.has_type = type_col in self.df.columns
        self.has_attr = attribute_col in self.df.columns
        self.has_race = race_col in self.df.columns
        self.type_norm = _l2_rows(_build_onehot(self.df[type_col])[0]) if self.has_type else None
        self.attr_norm = _l2_rows(_build_onehot(self.df[attribute_col])[0]) if self.has_attr else None
        if self.has_race:
            Xrace, _, is_bool = _build_multihot(self.df[race_col])
            self.race_bool = Xrace if is_bool else None
            self.race_norm = None if is_bool else _l2_rows(Xrace.astype(np.float32))
        else:
            self.race_bool = self.race_norm = None
        num_w = np.array([self.w.w_level, self.w.w_atk, self.w.w_def], dtype=np.float32)
        self.num_w = num_w / (num_w.sum() + 1e-9)

    def _num_sim_vec(self, q_idx):
        z = (self.num - self.num[q_idx]) / (self.sigma + 1e-9)
        s = np.exp(-0.5 * np.sum(self.num_w * (z ** 2), axis=1)).astype(np.float32)
        return np.clip(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    def _cat_sim_vec(self, q_idx):
        parts, ws = [], []
        if self.type_norm is not None:
            parts.append(np.dot(self.type_norm, self.type_norm[q_idx])); ws.append(self.w.w_type)
        if self.attr_norm is not None:
            parts.append(np.dot(self.attr_norm, self.attr_norm[q_idx])); ws.append(self.w.w_attr)
        if self.race_bool is not None:
            a = self.race_bool; q = a[q_idx]
            inter = (a & q).sum(axis=1).astype(np.float32)
            union = (a | q).sum(axis=1).astype(np.float32)
            parts.append((inter + 1e-6) / np.maximum(union, 1e-6)); ws.append(self.w.w_race)
        elif self.race_norm is not None:
            parts.append(np.dot(self.race_norm, self.race_norm[q_idx])); ws.append(self.w.w_race)
        if not parts:
            return np.zeros(len(self.df), dtype=np.float32)
        W = np.array(ws, dtype=np.float32); W = W / (W.sum() + 1e-9)
        return np.clip((np.vstack(parts).T * W).sum(axis=1).astype(np.float32), 0.0, 1.0)

    def similarities(self, q_idx):
        s = (1.0 - self.w.w_cat) * self._num_sim_vec(q_idx) + self.w.w_cat * self._cat_sim_vec(q_idx)
        return np.clip(s.astype(np.float32), 0.0, 1.0)


# =========================================================
# メイン推薦器
# =========================================================
class RecommenderV2:
    """
    二段階召喚 + 融合（RRF / power mean）+ MMR 再ランキング。
    + search_by_text: 自由テキストからカードを検索する新機能。
    """

    # CLIP model name (detected by detect_clip_model.py)
    _CLIP_MODEL_NAME = "ViT-B-32"
    _CLIP_PRETRAINED = "openai"

    def __init__(self, card_df, art_embs, lore_embs, meta_embs,
                 *, use_meta_engine=False, meta_engine_kwargs=None):
        self.db = card_df.reset_index(drop=True)
        self.art  = self._l2(art_embs)
        self.lore = self._l2(lore_embs)
        self.meta = self._l2(meta_embs) if meta_embs is not None else None
        self.meta_engine   = None
        self.linear_engine = None

        # CLIP model (lazy-loaded on first call to search_by_text)
        self._clip_model = None
        self._clip_tokenizer = None

        if use_meta_engine:
            meta_engine_kwargs = meta_engine_kwargs or {}
            self.meta_engine = MetaEngine(self.db, **meta_engine_kwargs)
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

        if "name" not in self.db.columns:
            raise ValueError("card_df must contain a 'name' column.")

        self.name2idx = pd.Series(
            self.db.index.values, index=self.db["name"].astype(str)
        ).to_dict()

    # ---------------------------------------------------------
    # ★ 新機能: 自由テキスト → カード検索
    # ---------------------------------------------------------
    def _load_clip(self):
        """CLIPモデルを遅延ロード（初回呼び出し時のみ）"""
        if self._clip_model is not None:
            return
        try:
            import torch
            import open_clip
        except ImportError:
            raise ImportError("open_clip が必要です: pip install open-clip-torch")

        print(f"[CLIP] Loading {self._CLIP_MODEL_NAME} / {self._CLIP_PRETRAINED} ...")
        model, _, _ = open_clip.create_model_and_transforms(
            self._CLIP_MODEL_NAME, pretrained=self._CLIP_PRETRAINED
        )
        model.eval()
        self._clip_model = model
        self._clip_tokenizer = open_clip.get_tokenizer(self._CLIP_MODEL_NAME)
        self._torch = torch
        print("[CLIP] Model loaded.")

    def search_by_text(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """
        自由テキストで最も近いカードを返す。
        結果の先頭カード名を recommend() に渡せばそのまま推薦に使える。

        Parameters
        ----------
        query  : 検索クエリ（英語推奨）例: "cute girl magician"
        top_n  : 返すカード数

        Returns
        -------
        DataFrame with columns [...card columns..., 'text_sim']
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        self._load_clip()

        with self._torch.no_grad():
            tokens = self._clip_tokenizer([query])
            text_feat = self._clip_model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        text_vec = text_feat.cpu().numpy().astype(np.float32)[0]

        # art_embs はすでに L2 正規化済み → 内積 = コサイン類似度
        sims = np.dot(self.art, text_vec)

        top_idx = np.argsort(-sims)[:top_n].astype(np.int64)
        result = self.db.iloc[top_idx].copy()
        result["text_sim"] = sims[top_idx].astype(float)
        return result

    def search_and_recommend(self, query: str, top_n: int = 12,
                             n_candidates: int = 5, **recommend_kwargs) -> dict:
        """
        テキスト検索 → 上位1件を基準カードとして推薦まで一気に実行。

        Parameters
        ----------
        query          : 自由テキスト（英語）
        top_n          : 最終推薦件数
        n_candidates   : テキスト検索で候補として表示するカード数
        recommend_kwargs: recommend() に渡す追加引数

        Returns
        -------
        {
          "query":      入力テキスト,
          "candidates": テキスト検索の候補 DataFrame,
          "pivot":      推薦の基準カード名,
          "results":    推薦結果 DataFrame
        }
        """
        candidates = self.search_by_text(query, top_n=n_candidates)
        pivot_name = str(candidates.iloc[0]["name"])
        results = self.recommend(pivot_name, top_n=top_n, **recommend_kwargs)
        return {
            "query":      query,
            "candidates": candidates,
            "pivot":      pivot_name,
            "results":    results,
        }

    # ---------------------------------------------------------
    # 既存メソッド（変更なし）
    # ---------------------------------------------------------
    @staticmethod
    def _l2(X):
        X = X.astype(np.float32, copy=False)
        return X / (norm(X, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def _validate_weights(modality_weights):
        mw = modality_weights or {"art": 1.0, "lore": 1.0, "meta": 1.0}
        w = np.array([mw.get("art", 1.0), mw.get("lore", 1.0), mw.get("meta", 1.0)], dtype=np.float32)
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("modality weights must be finite and non-negative.")
        s = float(w.sum())
        if s <= 1e-12:
            raise ValueError("at least one modality weight must be positive.")
        return w / s

    @staticmethod
    def _topk_idx(sims, k, exclude):
        sims = np.asarray(sims, dtype=np.float32)
        n = sims.shape[0]
        if n <= 1 or k <= 0:
            return np.empty(0, dtype=np.int64)
        if not 0 <= exclude < n:
            raise IndexError(f"exclude index out of range: {exclude}")
        k_eff = min(int(k) + 1, n)
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[idx != exclude]
        if idx.size == 0:
            return idx.astype(np.int64)
        return idx[np.argsort(-sims[idx], kind="mergesort")][:k].astype(np.int64)

    @staticmethod
    def _rrf(ranks_dict, k=60, modality_weights=None):
        mw = modality_weights or {"art": 1.0, "lore": 1.0, "meta": 1.0}
        fused: dict[int, float] = {}
        for m, ranks in ranks_dict.items():
            w = float(mw.get(m, 1.0))
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + w * (1.0 / (k + int(r)))
        return fused

    @staticmethod
    def _rrf_on_pool(score_mat, pool, modality_weights=None, rrf_k=60):
        if pool.size == 0:
            return np.empty(0, dtype=np.float32), {}
        w = RecommenderV2._validate_weights(modality_weights)
        n, m = score_mat.shape
        ranks = np.empty((n, m), dtype=np.float32)
        for j in range(m):
            order = np.argsort(-score_mat[:, j], kind="mergesort")
            ranks[order, j] = np.arange(1, n + 1, dtype=np.float32)
        fused_arr = (w[None, :] / (float(rrf_k) + ranks)).sum(axis=1).astype(np.float32)
        return fused_arr, {int(cid): float(s) for cid, s in zip(pool, fused_arr)}

    @staticmethod
    def _power_mean(sim_dict, p=1.5, modality_weights=None):
        if not np.isfinite(p) or p <= 0:
            raise ValueError("p_power must be finite and > 0.")
        all_ids = set()
        for d in sim_dict.values(): all_ids.update(d.keys())
        mw_vec = RecommenderV2._validate_weights(modality_weights)
        fused: dict[int, float] = {}
        for cid in all_ids:
            v = np.clip(np.nan_to_num(np.array([
                sim_dict.get("art",  {}).get(cid, 0.0),
                sim_dict.get("lore", {}).get(cid, 0.0),
                sim_dict.get("meta", {}).get(cid, 0.0),
            ], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            fused[cid] = float(np.power(np.sum(mw_vec * np.power(v, p)), 1.0 / p))
        return fused

    @staticmethod
    def _power_mean_on_pool(score_mat, pool, p=1.5, modality_weights=None):
        if pool.size == 0:
            return np.empty(0, dtype=np.float32), {}
        if not np.isfinite(p) or p <= 0:
            raise ValueError("p_power must be finite and > 0.")
        w = RecommenderV2._validate_weights(modality_weights)
        v = np.clip(np.nan_to_num(score_mat, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        fused_arr = np.power(np.sum(w[None, :] * np.power(v, p), axis=1), 1.0 / p).astype(np.float32)
        return fused_arr, {int(cid): float(s) for cid, s in zip(pool, fused_arr)}

    @staticmethod
    def _minmax_array_on_pool(scores, pool, neutral_value=0.5):
        if pool.size == 0:
            return np.empty(0, dtype=np.float32)
        v = np.nan_to_num(np.asarray(scores, dtype=np.float32)[pool], nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            return np.full(pool.shape[0], neutral_value, dtype=np.float32)
        return ((v - lo) / (hi - lo)).astype(np.float32)

    @staticmethod
    def _minmax_on_pool(scores, pool):
        pool_arr = np.asarray(pool, dtype=np.int64)
        arr = RecommenderV2._minmax_array_on_pool(scores, pool_arr)
        return {int(i): float(v) for i, v in zip(pool_arr, arr)}

    def _mmr_rerank(self, cand_idx, rel_scores, top_n, lam):
        if not 0.0 <= float(lam) <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1].")
        if top_n <= 0 or not cand_idx:
            return []
        if len(cand_idx) <= 1:
            return cand_idx[:top_n]
        cand_arr = np.asarray(cand_idx, dtype=np.int64)
        m = cand_arr.size
        sub = self.art[cand_arr]
        sim_mat = np.dot(sub, sub.T).astype(np.float32)
        np.fill_diagonal(sim_mat, 0.0)
        rel_vec = np.nan_to_num(
            np.array([rel_scores.get(int(c), 0.0) for c in cand_arr], dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        selected: list[int] = []
        available = np.ones(m, dtype=bool)
        first = int(np.argmax(rel_vec))
        selected.append(first); available[first] = False
        max_penalty = sim_mat[:, first].copy()
        limit = min(int(top_n), m)
        while len(selected) < limit and available.any():
            mmr = float(lam) * rel_vec - (1.0 - float(lam)) * max_penalty
            mmr[~available] = -np.inf
            nxt = int(np.argmax(mmr))
            selected.append(nxt); available[nxt] = False
            max_penalty = np.maximum(max_penalty, sim_mat[:, nxt])
        return cand_arr[selected].astype(int).tolist()

    def recommend(self, query_name, top_n=12, k_each=150, fusion="rrf",
                  p_power=1.5, use_mmr=True, mmr_lambda=0.7,
                  w_art=1.0, w_lore=1.0, w_meta=1.0, ab_system="B"):
        if query_name not in self.name2idx:
            raise ValueError(f"Card '{query_name}' not found.")
        if top_n <= 0:
            return self.db.iloc[[]].copy()
        if k_each <= 0:
            raise ValueError("k_each must be positive.")

        fusion_key = (fusion or "rrf").lower()
        if fusion_key not in {"rrf", "power_mean", "power"}:
            raise ValueError("fusion must be 'rrf' or 'power_mean'.")

        ab_key = (ab_system or "B").upper()[0]
        if ab_key not in {"A", "B"}:
            raise ValueError("ab_system must be 'A' or 'B'.")

        q = self.name2idx[query_name]
        art_s  = np.dot(self.art,  self.art[q]).astype(np.float32)
        lore_s = np.dot(self.lore, self.lore[q]).astype(np.float32)

        if ab_key == "A":
            if self.linear_engine is None:
                raise RuntimeError("System A requires use_meta_engine=True.")
            meta_s = self.linear_engine.similarities(q)
        else:
            if self.meta_engine is None:
                raise RuntimeError("System B requires use_meta_engine=True.")
            meta_s = self.meta_engine.similarities(q)

        art_c  = self._topk_idx(art_s,  k_each, q)
        lore_c = self._topk_idx(lore_s, k_each, q)
        meta_c = self._topk_idx(meta_s, k_each, q)
        pool = np.union1d(np.union1d(art_c, lore_c), meta_c).astype(np.int64)

        if pool.size == 0:
            return self.db.iloc[[]].copy()

        mw = {"art": w_art, "lore": w_lore, "meta": w_meta}

        if fusion_key == "rrf":
            score_mat = np.column_stack([art_s[pool], lore_s[pool], meta_s[pool]]).astype(np.float32)
            fused_arr, fused = self._rrf_on_pool(score_mat, pool, modality_weights=mw)
        else:
            score_mat = np.column_stack([
                self._minmax_array_on_pool(art_s,  pool),
                self._minmax_array_on_pool(lore_s, pool),
                self._minmax_array_on_pool(meta_s, pool),
            ]).astype(np.float32)
            fused_arr, fused = self._power_mean_on_pool(score_mat, pool, p=p_power, modality_weights=mw)

        order = np.argsort(-fused_arr, kind="mergesort")
        pre = pool[order].astype(int).tolist()
        mmr_pool_size = min(len(pre), max(top_n, 3 * top_n))

        if use_mmr:
            final_idx = self._mmr_rerank(pre[:mmr_pool_size], fused, top_n, mmr_lambda)
        else:
            final_idx = pre[:top_n]

        out = self.db.iloc[final_idx].copy()
        final_arr = np.asarray(final_idx, dtype=np.int64)
        out["art_sim"]     = art_s[final_arr].astype(float)
        out["lore_sim"]    = lore_s[final_arr].astype(float)
        out["meta_sim"]    = meta_s[final_arr].astype(float)
        out["final_score"] = [float(fused.get(int(i), 0.0)) for i in final_idx]

        out.attrs["recommend_config"] = {
            "ab_system": ab_key,
            "fusion": "power_mean" if fusion_key == "power" else fusion_key,
            "p_power": float(p_power), "use_mmr": bool(use_mmr),
            "mmr_lambda": float(mmr_lambda), "k_each": int(k_each), "top_n": int(top_n),
            "weights": {"art": float(w_art), "lore": float(w_lore), "meta": float(w_meta)},
        }
        return out

    def debug_meta_components(self, query_name, rank=0):
        if query_name not in self.name2idx:
            raise ValueError(f"Card '{query_name}' not found.")
        if self.meta_engine is None:
            print("[DEBUG] MetaEngine disabled."); return None
        q = self.name2idx[query_name]
        art_s  = np.dot(self.art,  self.art[q]).astype(np.float32)
        lore_s = np.dot(self.lore, self.lore[q]).astype(np.float32)
        meta_s = self.meta_engine.similarities(q)
        pool = np.union1d(np.union1d(
            self._topk_idx(art_s, 150, q),
            self._topk_idx(lore_s, 150, q)),
            self._topk_idx(meta_s, 150, q))
        pre = pool[np.argsort(-meta_s[pool], kind="mergesort")]
        if rank >= len(pre):
            raise ValueError(f"rank {rank} >= pool size {len(pre)}")
        i = int(pre[rank])
        return {"q": int(q), "i": i, "name": str(self.db.iloc[i]["name"]),
                "s_num": float(self.meta_engine._num_sim_vec(q)[i]),
                "s_cat": float(self.meta_engine._cat_sim_vec(q)[i]),
                "s_meta": float(meta_s[i]),
                "sigma": self.meta_engine.sigma.copy()}

    @classmethod
    def from_hf(cls, repo_id, *, use_meta_engine=False, meta_engine_kwargs=None):
        local_dir = Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            allow_patterns=["*.parquet", "*.npz"]
        ))
        df   = pd.read_parquet(local_dir / "card_database.parquet")
        art  = np.load(local_dir / "art_embs.npz")["data"]
        lore = np.load(local_dir / "lore_embs.npz")["data"]
        meta = np.load(local_dir / "meta_embs.npz")["data"]
        return cls(df, art, lore, meta,
                   use_meta_engine=use_meta_engine,
                   meta_engine_kwargs=meta_engine_kwargs)
