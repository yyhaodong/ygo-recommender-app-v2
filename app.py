# app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List

from recommender_v2 import RecommenderV2

# ====== 基本設定 ======
PAGE_TITLE = "🔮 遊戯王カード 多モーダル推薦エンジン"
HF_DATASET_REPO = "oneonehaodong/ygo-recommender-data"  # 変更する場合はここだけ

st.set_page_config(page_title="YGO Recommender", layout="wide")


# ====== ユーティリティ ======
def _first_exist(colnames: List[str], df: pd.DataFrame) -> Optional[str]:
    """候補リストのうち最初に存在する列名を返す。見つからなければ None。"""
    for c in colnames:
        if c in df.columns:
            return c
    return None


def _col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    return _first_exist(list(cands), df)


def _image_url_from_row(row: pd.Series, df: pd.DataFrame) -> Optional[str]:
    img_col = _col(df, "image_url", "img_url", "image", "img", "picture_url")
    return None if img_col is None else row.get(img_col, None)


def _text_from_row(row: pd.Series, df: pd.DataFrame, cands: List[str], default: str = "—") -> str:
    c = _first_exist(cands, df)
    if c is None:
        return default
    v = row.get(c, default)
    if pd.isna(v):
        return default
    return str(v)


# ====== モデル・データのロード ======
@st.cache_resource(show_spinner=True)
def load_recommender() -> RecommenderV2:
    return RecommenderV2.from_hf(HF_DATASET_REPO)


rec = load_recommender()
DB = rec.db  # 便宜用


# ====== サイドバー（操作） ======
st.sidebar.title("🛠 検索パラメータ")

# カード選択
all_names = DB["name"].astype(str).sort_values().tolist()
selected_name = st.sidebar.selectbox(
    "カード名を選択",
    options=["（選択してください）"] + all_names,
    index=0,
)

# 表示枚数
top_n = st.sidebar.slider(
    "表示する枚数",
    min_value=5, max_value=30, value=12, step=1,
    help="推薦結果として表示するカードの枚数（5〜30）。"
)

# 融合戦略
fusion = st.sidebar.selectbox(
    "融合戦略",
    options=["rrf", "power_mean"],
    index=0,
    help=(
        "複数モダリティ（イラスト・フレーバー・メタ）のスコアをどう統合するか。\n"
        "・rrf：順位ベースの統合。スコアの尺度差に強く、安定。\n"
        "・power_mean：スコアをプール平均。全体的に高い候補が有利。"
    )
)

# 多様性（MMR）
use_mmr = st.sidebar.checkbox("多様性 (MMR) を有効化", value=True)
mmr_lambda = st.sidebar.slider(
    "関連性 vs 多様性 (λ)",
    min_value=0.0, max_value=1.0, value=0.70, step=0.01,
    help="1.0 に近いほど関連性を重視、0.0 に近いほど多様性を重視。"
)

# ====== メイン（UI） ======
st.title(PAGE_TITLE)

# ヒント
if selected_name == "（選択してください）":
    st.info("左サイドバーからカードを選んでください。")
    st.stop()

# 選択カードの行
try:
    row = DB.loc[DB["name"].astype(str) == selected_name].iloc[0]
except IndexError:
    st.error("選択したカードが見つかりませんでした。")
    st.stop()

# ---------- 選択カードの情報（画面最上段） ----------
st.subheader("🎴 選択中のカード")
col_img, col_info = st.columns([1, 2])  # ← vertical_alignment は使えない（存在しない引数）

with col_img:
    url = _image_url_from_row(row, DB)
    if url:
        st.image(url, use_column_width=True)
    else:
        st.markdown("（画像なし）")

with col_info:
    name = str(row.get("name", "—"))
    tpe = _text_from_row(row, DB, ["type", "card_type", "category"])
    race = _text_from_row(row, DB, ["race", "tribe", "attribute"])
    arche = _text_from_row(row, DB, ["archetype", "series"])
    atk = _text_from_row(row, DB, ["atk"], "—")
    dfn = _text_from_row(row, DB, ["def", "defense"], "—")
    desc = _text_from_row(row, DB, ["desc", "description", "lore"], "—")

    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:6px;">
            <h3 style="margin:0;">{name}</h3>
            <div>タイプ：<b>{tpe}</b>　/　分類：<b>{race}</b>　/　シリーズ：<b>{arche}</b></div>
            <div>ATK：<b>{atk}</b>　/　DEF：<b>{dfn}</b></div>
            <div style="margin-top:6px; line-height:1.5;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------- 推薦の実行 ----------
@st.cache_data(show_spinner=True, ttl=600)
def _recommend_cached(qname: str, n: int, fusion_label: str, use_mmr_flag: bool, lam: float) -> pd.DataFrame:
    return rec.recommend(
        query_name=qname,
        top_n=int(n),
        k_each=150,
        fusion=fusion_label,
        use_mmr=use_mmr_flag,
        mmr_lambda=float(lam),
    )


st.subheader("✨ 推薦結果")
with st.spinner("計算中..."):
    results = _recommend_cached(selected_name, top_n, fusion, use_mmr, mmr_lambda)

if results.empty:
    st.warning("推薦結果が空でした。別のカードでお試しください。")
    st.stop()

# ---------- グリッド表示 ----------
# 画像列名を特定
img_colname = _col(results, "image_url", "img_url", "image", "img", "picture_url")

# 4列グリッドに表示
N_COLS = 4
rows = int(np.ceil(len(results) / N_COLS))

for r in range(rows):
    cols = st.columns(N_COLS)
    for c in range(N_COLS):
        idx = r * N_COLS + c
        if idx >= len(results):
            continue
        item = results.iloc[idx]
        with cols[c]:
            # 画像
            if img_colname and pd.notna(item.get(img_colname, None)):
                st.image(item[img_colname], use_column_width=True)
            else:
                st.markdown("（画像なし）")

            # テキスト：名前＋スコア
            nm = str(item.get("name", "—"))
            score = float(item.get("final_score", 0.0))
            st.markdown(f"**{nm}**  \nScore: `{score:.4f}`")

            # 追加情報（必要に応じて）
            if "art_sim" in item and "lore_sim" in item and "meta_sim" in item:
                st.caption(
                    f"art: {item['art_sim']:.3f} / lore: {item['lore_sim']:.3f} / meta: {item['meta_sim']:.3f}"
                )

st.caption("提示されたスコアは候補集合内での相対値です。MMR を有効化すると多様性が高まります。")

