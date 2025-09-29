# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from recommender_v2 import RecommenderV2

# -----------------------------
# 基本設定
# -----------------------------
st.set_page_config(
    page_title="遊戯王カード 多モーダル推薦エンジン",
    page_icon="🔮",
    layout="wide",
)

HF_REPO = "oneonehaodong/ygo-recommender-data"

@st.cache_resource(show_spinner=True)
def load_recommender():
    return RecommenderV2.from_hf(HF_REPO)

rec = load_recommender()
DB: pd.DataFrame = rec.db

# 画像URLの取り出し（データに合わせて安全に）
def get_image_url(row: pd.Series) -> str | None:
    # 1) もし image_url 列があればそれを使う
    for key in ["image_url", "img_url", "card_image", "picture"]:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    # 2) もし id 系の数値があれば一般的な YGOPRODECK URL を試す（なければ None）
    for key in ["id", "card_id", "ygoprodeck_id"]:
        if key in row and pd.notna(row[key]):
            try:
                cid = int(row[key])
                return f"https://images.ygoprodeck.com/images/cards_cropped/{cid}.jpg"
            except Exception:
                pass
    return None

# テキスト（日本語）整形
def safe_text(v, default="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    s = str(v).strip()
    return s if s else default

# -----------------------------
# サイドバー：パラメータ
# -----------------------------
with st.sidebar:
    st.markdown("### 🔍 検索パラメータ")

    # カード名
    names = DB["name"].astype(str).dropna().sort_values().unique().tolist()
    selected_name = st.selectbox(
        "カード名を選択",
        options=names,
        index=0 if names else None,
        help="推薦の基準となるカードを選びます。右側に推薦結果が表示されます。",
    )

    # 表示枚数
    top_n = st.slider(
        "表示する枚数",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
        help="推薦結果として表示するカードの枚数（5〜30）。",
    )

    # 融合戦略
    fusion_label = st.selectbox(
        "融合戦略",
        options=["rrf", "power_mean"],
        index=0,
        help=(
            "複数モダリティのスコアをどう統合するかを選びます。\n"
            "・rrf：順位を重視し、スコアの尺度差に強い（安定）。\n"
            "・power_mean：スコアのバランスを重視（全体的に高い候補が有利）。"
        ),
    )

    # 多様性 (MMR)
    use_mmr = st.checkbox(
        "多様性 (MMR) を有効化",
        value=True,
        help="似すぎたカードが並び過ぎないようにする再ランキングを有効にします。",
    )

    lam = st.slider(
        "関連性 vs 多様性 (λ)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help="0に近いほど関連性重視、1に近いほど多様性重視になります。",
    )

# -----------------------------
# メインヘッダ
# -----------------------------
st.markdown(
    "<h1 style='margin-top:0'>🔮 遊戯王カード 多モーダル推薦エンジン</h1>",
    unsafe_allow_html=True,
)

# -----------------------------
# 選択中カードの情報ボックス
# -----------------------------
if selected_name:
    base_row = DB.loc[DB["name"].astype(str) == selected_name]
    if not base_row.empty:
        base_row = base_row.iloc[0]
        img_url = get_image_url(base_row)

        with st.container(border=True):
            st.markdown("#### 🃏 選択中のカード")
            col_img, col_info = st.columns([1, 2], vertical_alignment="top")
            with col_img:
                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.info("画像が見つかりませんでした。")

            with col_info:
                st.markdown(f"**名前**：{safe_text(base_row.get('name'))}")
                # 代表的な補助情報（存在すれば表示）
                line = []
                for k in ["type", "race", "attribute", "frameType", "category"]:
                    if k in DB.columns:
                        v = safe_text(base_row.get(k))
                        if v != "—":
                            line.append(f"{k}: {v}")
                if line:
                    st.caption(" / ".join(line))

                # 効果テキスト等（列名に応じて拾う）
                effect_key = None
                for k in ["desc", "effect", "lore", "text", "jp_text", "ja_text"]:
                    if k in DB.columns:
                        effect_key = k
                        break
                if effect_key:
                    txt = safe_text(base_row.get(effect_key))
                    with st.expander("効果テキストを表示", expanded=False):
                        st.write(txt)
    else:
        st.warning("選択したカードがデータベースで見つかりませんでした。")

# -----------------------------
# 推薦の実行
# -----------------------------
if selected_name:
    try:
        df_rec = rec.recommend(
            query_name=selected_name,
            top_n=int(top_n),
            k_each=150,
            fusion=fusion_label,
            p_power=1.5,
            use_mmr=use_mmr,
            mmr_lambda=float(lam),
        )

        if df_rec.empty:
            st.info("推薦候補が見つかりませんでした。パラメータを調整してください。")
        else:
            st.markdown("#### 🎯 推薦結果")
            # グリッド表示
            ncols = 5 if top_n >= 15 else 4
            rows = (len(df_rec) + ncols - 1) // ncols
            for r in range(rows):
                cols = st.columns(ncols, vertical_alignment="top")
                for c in range(ncols):
                    i = r * ncols + c
                    if i >= len(df_rec):
                        break
                    row = df_rec.iloc[i]
                    with cols[c]:
                        img = get_image_url(row)
                        if img:
                            st.image(img, use_container_width=True)
                        st.markdown(f"**{safe_text(row.get('name'))}**")
                        # スコアの小さな説明
                        st.caption(
                            "art: {:.4f} / lore: {:.4f} / meta: {:.4f} / final: {:.4f}".format(
                                float(row.get("art_sim", 0.0)),
                                float(row.get("lore_sim", 0.0)),
                                float(row.get("meta_sim", 0.0)),
                                float(row.get("final_score", 0.0)),
                            )
                        )
    except Exception as e:
        st.error(f"推薦中にエラーが発生しました：{e}")
else:
    st.info("左のサイドバーからカードを選んでください。")
