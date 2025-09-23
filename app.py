# app.py — final optimized version
import os
import streamlit as st
import pandas as pd
from recommender_v2 import RecommenderV2

st.set_page_config(page_title="遊戯王カード 多モーダル推薦", page_icon="🔮", layout="wide")

REPO_ID = "oneonehaodong/ygo-recommender-data"  # 你的 HF 数据集

@st.cache_resource(show_spinner="Initializing recommender (first run only)...")
def load_recommender():
    return RecommenderV2.from_hf(REPO_ID)

rec = load_recommender()

st.title("🔮 遊戯王カード 多モーダル推薦エンジン")

# ---------- 控件 ----------
with st.sidebar:
    st.header("⚙️ 推薦パラメータ")
    names = [""] + sorted(rec.db["name"].astype(str).unique().tolist())
    query_name = st.selectbox("検索したいカードを選択してください:", options=names, index=0)

    top_n = st.slider("表示する枚数 (Top-N)", 6, 30, 12)
    k_each = st.slider("各モダリティの候補数 (K)", 50, 400, 150, step=10)

    fusion = st.selectbox("融合戦略", ["rrf", "power_mean"])
    p_power = 1.5
    if fusion == "power_mean":
        p_power = st.slider("Power Mean の p 値", 1.0, 5.0, 1.5, 0.1)

    use_mmr = st.checkbox("多様性を高める (MMR)", value=True)
    mmr_lambda = 0.7
    if use_mmr:
        mmr_lambda = st.slider("関連性 vs 多様性 (λ)", 0.0, 1.0, 0.7, 0.05)

if not query_name:
    st.info("左のサイドバーからカードを選択して推薦を開始してください。")
    st.stop()

st.subheader(f"「{query_name}」への推薦結果")
with st.spinner("計算中..."):
    df = rec.recommend(
        query_name=query_name,
        top_n=top_n,
        k_each=k_each,
        fusion=fusion,
        p_power=p_power,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda
    )

# ---------- 展示 ----------
if df.empty:
    st.warning("推薦結果が見つかりませんでした。")
else:
    num_cols = 6
    cols = st.columns(num_cols)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % num_cols]:
            card_id = int(row["id"]) if "id" in row and pd.notna(row["id"]) else None
            if card_id:
                img = f"https://images.ygoprodeck.com/images/cards/{card_id}.jpg"
                st.image(img, use_column_width=True, caption=f"{row['name']}")
            st.caption(f"art:{row['art_sim']:.3f} / lore:{row['lore_sim']:.3f} / meta:{row['meta_sim']:.3f}\nscore:{row['final_score']:.4f}")

st.markdown(
    "<small>データ提供: YGOPRODeck API ／ 画像: images.ygoprodeck.com</small>",
    unsafe_allow_html=True
)
