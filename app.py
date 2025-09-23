# app.py — minimal, production-safe
import streamlit as st
import pandas as pd
from recommender_v2 import RecommenderV2

st.set_page_config(page_title="遊戯王カード 多モーダル推薦", page_icon="🔮", layout="wide")

REPO_ID = "oneonehaodong/ygo-recommender-data"

@st.cache_resource(show_spinner="Loading recommender & data...")
def get_recommender():
    # 重要：recommender_v2.from_hf 内部使用 *.parquet / *.npz
    return RecommenderV2.from_hf(REPO_ID)

rec = get_recommender()

st.title("🔮 遊戯王カード 多モーダル推薦エンジン")

with st.sidebar:
    st.header("⚙️ 検索パラメータ")
    names = [""] + rec.db["name"].astype(str).tolist()
    query = st.selectbox("カード名を選択", options=names)
    top_n = st.slider("表示する枚数", 5, 30, 12)
    st.markdown("---")
    fusion = st.selectbox("融合戦略", ["rrf", "power_mean"])
    p_power = 1.5
    if fusion == "power_mean":
        p_power = st.slider("Power mean の p", 1.0, 5.0, 1.5, 0.1)
    use_mmr = st.checkbox("多様性 (MMR) を有効化", value=True)
    mmr_lambda = 0.7
    if use_mmr:
        mmr_lambda = st.slider("関連性 vs 多様性 (λ)", 0.0, 1.0, 0.7, 0.05)

if query:
    with st.spinner("計算中..."):
        try:
            df = rec.recommend(
                query_name=query,
                top_n=top_n,
                fusion=fusion,
                p_power=p_power,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
            )
        except Exception as e:
            st.error(f"エラー: {e}")
            st.stop()

    cols = st.columns(6)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % 6]:
            cid = int(row["id"]) if "id" in row else None
            if cid:
                st.image(f"https://images.ygoprodeck.com/images/cards/{cid}.jpg",
                         caption=f"{row['name']}\nScore: {row['final_score']:.4f}")
            else:
                st.write(row["name"])
else:
    st.info("左のサイドバーからカードを選んでください。")
