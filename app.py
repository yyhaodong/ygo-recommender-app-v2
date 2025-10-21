# app.py — 遊戯王カード 多モーダル推薦（日本語UI・名称/画像/カメラ・左右対比）
from __future__ import annotations
import os, re, json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent

# ======== CLIP 設定（512 次元で確認） ========
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# 任意: clip_config.json があれば上書き
if os.path.exists("clip_config.json"):
    try:
        _cfg = json.load(open("clip_config.json", "r", encoding="utf-8"))
        MODEL_NAME = _cfg.get("MODEL_NAME", MODEL_NAME)
        PRETRAINED = _cfg.get("PRETRAINED", PRETRAINED)
    except Exception:
        pass

# =========================
# 画像表示（URL優先）
# =========================
def show_image_url(value: str | None, *, caption=None):
    if not value:
        st.write("—"); return
    url = str(value)
    try:
        if url.startswith(("http://", "https://")):
            r = requests.get(url, timeout=8); r.raise_for_status()
            st.image(BytesIO(r.content), caption=caption, use_column_width=True)
        else:
            st.warning("画像URLが無効です（ローカルパス検出）。")
            st.caption(url)
    except Exception as e:
        st.warning("画像の読み込みに失敗しました。")
        st.caption(url); st.caption(f"→ {e}")

def safe_columns(n: int):
    try: n = int(n or 1)
    except Exception: n = 1
    return st.columns(max(1, min(n, 6)), gap="small")

def pill(text: str):
    st.markdown(
        f"""<span style="display:inline-block;padding:2px 8px;border-radius:999px;
        background:#eef2ff;border:1px solid #c7d2fe;font-size:12px;">{text}</span>""",
        unsafe_allow_html=True,
    )

def fmt(v): return "-" if pd.isna(v) else str(v)

# --- 展示の丸め修正：100%に“誤って”ならないように（真=1.0の時だけ100） ---
def similarity_bar(label: str, value: float, note: str=""):
    try:
        v = float(value); v = 0.0 if np.isnan(v) else max(0.0, min(1.0, v))
    except Exception:
        v = 0.0
    if v >= 1 - 1e-6:
        pct = 100
    else:
        pct = int(np.floor(v * 100))  # 向下取整
    st.markdown(f"**{label}：{pct}%**  {note}")
    st.markdown(
        f"""
        <div style="background:#eee;border-radius:10px;height:12px;overflow:hidden;">
          <div style="width:{pct}%;height:100%;background:#16a34a;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# ページ設定 & データ読込
# =========================
st.set_page_config(page_title="遊戯王カード 多モーダル推薦", page_icon="🔮", layout="wide")
st.title("🔮 遊戯王カード 多モーダル推薦エンジン")

@st.cache_resource(show_spinner="推薦エンジンとデータを読み込み中…")
def get_recommender():
    # --- ここが重要：MetaEngine を有効化してロード ---
    from recommender_v2 import RecommenderV2, MetaWeights
    return RecommenderV2.from_hf(
        "oneonehaodong/ygo-recommender-data",
        use_meta_engine=True,                                   # ← 新規
        meta_engine_kwargs=dict(                                # ← 新規（列名はHFデータに合わせて必要なら調整）
            level_col="level", atk_col="atk", def_col="def",
            type_col="type", attribute_col="attribute", race_col="race",
            meta_w=MetaWeights(  # 既定: Cat 0.40, Level 0.30, ATK 0.15, DEF 0.15
                w_cat=0.40, w_level=0.30, w_atk=0.15, w_def=0.15,
                w_type=0.50, w_attr=0.25, w_race=0.25
            )
        )
    )

rec = get_recommender()
DF: pd.DataFrame = rec.db.copy()

# 実行時に画像URL作成（既存URL列 → 数字IDで YGOPRO）
def make_runtime_image_url(df: pd.DataFrame) -> pd.Series:
    for col in ["image_url", "img_url", "thumbnail_url", "card_image_url", "url"]:
        if col in df.columns:
            s = df[col].astype(str)
            http_mask = s.str.startswith(("http://", "https://"), na=False)
            if http_mask.any():
                return s.where(http_mask, other=None)
    cand = None
    for col in ["image_path", "img", "thumbnail", "id", "passcode", "konami_id", "code"]:
        if col in df.columns:
            cand = df[col].astype(str); break
    if cand is None: return pd.Series([None]*len(df), index=df.index)
    ids = cand.str.extract(r"(\d{5,})", expand=False)
    base = "https://images.ygoprodeck.com/images/cards/"
    url = base + ids + ".jpg"
    return url.where(ids.notna(), other=None)

DF["image_url_runtime"] = make_runtime_image_url(DF)

# 列名フォールバック
COL_NAME  = "name" if "name" in DF.columns else DF.columns[0]
COL_TYPE  = next((c for c in ["type", "card_type", "race", "frameType"] if c in DF.columns), None)
COL_ATK   = next((c for c in ["atk", "ATK"] if c in DF.columns), None)
COL_DEF   = next((c for c in ["def", "DEF", "defe"] if c in DF.columns), None)
COL_RAR   = next((c for c in ["rarity", "Rarity"] if c in DF.columns), None)
COL_DESC  = next((c for c in ["desc", "effect", "text"] if c in DF.columns), None)
COL_ID    = next((c for c in ["id", "passcode", "konami_id", "code"] if c in DF.columns), None)

def image_url_for_row(row: pd.Series) -> str | None:
    if "image_url_runtime" in row and pd.notna(row["image_url_runtime"]):
        return str(row["image_url_runtime"])
    if COL_ID and pd.notna(row.get(COL_ID)):
        try:
            cid = int(row[COL_ID])
            return f"https://images.ygoprodeck.com/images/cards/{cid}.jpg"
        except Exception:
            return None
    return None

# =========================
# CLIP エンコーダ（未導入でも名称検索は動く）
# =========================
ENCODER_OK = False
try:
    import open_clip, torch
    @st.cache_resource(show_spinner="画像エンコーダを読み込み中…")
    def get_img_encoder():
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        model.eval()
        return model, preprocess, torch
    model_clip, preprocess_clip, torch = get_img_encoder()
    ENCODER_OK = True
    st.caption(f"🔧 CLIP: {MODEL_NAME} / {PRETRAINED}")
except Exception:
    st.info("画像検索は未有効（torch/open-clip 未インストールまたは読み込み失敗）。")
    ENCODER_OK = False

def encode_pil_to_vec(pil_img: Image.Image) -> np.ndarray:
    if not ENCODER_OK:
        raise RuntimeError("Image encoder not available.")
    with torch.no_grad():
        x = preprocess_clip(pil_img.convert("RGB")).unsqueeze(0)
        feat = model_clip.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)

# =========================
# サイドバー
# =========================
with st.sidebar:
    st.header("🛠 検索パラメータ")
    mode = st.radio("モード", ["かんたん", "上級"], horizontal=True)

    tab_name, tab_image, tab_camera = st.tabs(["カード名", "画像から", "カメラ"])
    effective_query_name = None
    query = None

    with tab_name:
        names = [""] + DF[COL_NAME].astype(str).tolist()
        query = st.selectbox("カード名を選択", options=names)

    with tab_image:
        if ENCODER_OK:
            up = st.file_uploader("画像をアップロード", type=["jpg","jpeg","png"])
            url = st.text_input("または画像URLを貼り付け")
            pil = None
            if up:
                pil = Image.open(up)
            elif url:
                try:
                    b = requests.get(url, timeout=6).content
                    pil = Image.open(BytesIO(b))
                except Exception:
                    st.error("画像URLの取得に失敗しました。")
            if pil is not None:
                st.image(pil, caption="クエリ画像プレビュー", use_column_width=True)
                with st.spinner("画像特徴を抽出中…"):
                    try:
                        v = encode_pil_to_vec(pil)
                        idx, nn_name, sim = rec.nearest_card_by_art(v)
                        effective_query_name = nn_name
                        st.success(f"最も近いカード：**{nn_name}**（sim={sim:.3f}）")
                    except Exception as e:
                        st.error(f"画像検索エラー：{e}")
        else:
            st.info("torch/open-clip が未インストールのため、画像検索は無効です。")

    with tab_camera:
        if ENCODER_OK:
            cam = st.camera_input("カメラで撮影して検索", label_visibility="collapsed")
            if cam is not None:
                pil = Image.open(cam)
                st.image(pil, caption="カメラ画像プレビュー", use_column_width=True)
                with st.spinner("画像特徴を抽出中…"):
                    try:
                        v = encode_pil_to_vec(pil)
                        idx, nn_name, sim = rec.nearest_card_by_art(v)
                        effective_query_name = nn_name
                        st.success(f"最も近いカード：**{nn_name}**（カメラ, sim={sim:.3f}）")
                    except Exception as e:
                        st.error(f"画像検索エラー：{e}")
        else:
            st.info("torch/open-clip が未インストールのため、カメラ検索は無効です。")

    if not effective_query_name:
        effective_query_name = query or None

    topk    = st.slider("Top-K（表示件数）", 6, 36, 18, 2)
    fusion  = st.selectbox("融合方式", ["rrf", "power_mean"], index=0,
                           help="RRF：スコア尺度に頑健。power_mean：複数モダリティ同時高得点を優遇。")
    p_power = st.slider("冪平均 p（>1 ほど“同時に高得点”を優遇）", 1.0, 3.0, 1.5, 0.1,
                        disabled=(fusion != "power_mean"))

    if mode == "かんたん":
        k_each, use_mmr, mmr_lambda = 150, True, 0.7
    else:
        k_each     = st.slider("各モダリティの候補数 k_each", 50, 400, 150, 10)
        use_mmr    = st.checkbox("MMR による多様性再ランキングを使用", True)
        mmr_lambda = st.slider("MMR λ（関連性 vs 反冗長）", 0.1, 0.95, 0.7, 0.05)

    st.divider()
    debug = st.toggle("🔧 デバッグ情報を表示", value=False)
    fire  = st.button("🔮 検索", use_container_width=True)

# =========================
# レンダリング
# =========================
def render_card_full(row: pd.Series | Dict[str, Any]):
    d = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    left, right = st.columns([1, 2], gap="small")
    with left:
        show_image_url(image_url_for_row(row), caption=d.get(COL_NAME))
    with right:
        st.subheader(str(d.get(COL_NAME, "Unknown")))
        mc = safe_columns(4)
        mc[0].metric("種別",   fmt(d.get(COL_TYPE)))
        mc[1].metric("ATK",    fmt(d.get(COL_ATK)))
        mc[2].metric("DEF",    fmt(d.get(COL_DEF)))
        mc[3].metric("レア度", fmt(d.get(COL_RAR)))
        st.write(""); pill("テキスト特徴"); pill("画像特徴"); pill("メタデータ/OCR")
        with st.expander("効果テキスト / Notes", expanded=True):
            st.write(d.get(COL_DESC) or "—")

def render_card_compact(row: pd.Series | Dict[str, Any]):
    d = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    show_image_url(image_url_for_row(row), caption=None)
    st.markdown(f"**{d.get(COL_NAME, 'Unknown')}**")
    with st.expander("詳細を見る"):
        similarity_bar("🖼️ 画像類似度",  d.get("art_sim", 0.0),  "絵柄・色味などの近さ")
        similarity_bar("📖 テキスト類似度", d.get("lore_sim", 0.0), "効果テキストの意味の近さ")
        similarity_bar("🔢 メタデータ類似度", d.get("meta_sim", 0.0), "種別・ATK/DEF 等の一致度")
        similarity_bar("⭐ 総合スコア",   d.get("final_score", 0.0), "上記を融合した最終評価")
        st.write(f"種別: {fmt(d.get(COL_TYPE))}")
        st.write(f"ATK : {fmt(d.get(COL_ATK))}")
        st.write(f"DEF : {fmt(d.get(COL_DEF))}")
        if COL_DESC:
            st.markdown("**効果テキスト / Notes**")
            st.write(d.get(COL_DESC) or "—")

# デバッグ
if debug:
    http_ok = DF["image_url_runtime"].astype(str).str.startswith(("http://","https://"), na=False).sum()
    st.info("🔧 デバッグ情報")
    st.code(
        "CWD = {}\nApp root = {}\nRows = {}\nHTTPな画像URL行数 = {}\n".format(
            os.getcwd(), APP_ROOT, len(DF), int(http_ok)
        )
    )
    st.dataframe(DF[[COL_NAME, "image_url_runtime"]].head(10))

# =========================
# メイン処理
# =========================
if fire:
    if not effective_query_name:
        st.warning("基準となるカード（または画像）を選んでください。")
    else:
        base_df = DF[DF[COL_NAME] == effective_query_name]
        if len(base_df):
            st.subheader("🔎 基準カード")
            render_card_full(base_df.iloc[0])
            st.divider()

        with st.spinner("計算中…"):
            try:
                results: pd.DataFrame = rec.recommend(
                    query_name=effective_query_name,
                    top_n=int(topk), k_each=int(k_each),
                    fusion=fusion, p_power=float(p_power),
                    use_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda)
                    # ※ UIは変えない前提なので w_art/w_lore/w_meta は既定値のまま
                )
            except Exception as e:
                st.error("推薦の計算に失敗しました。"); st.exception(e); results = None

        if results is not None and len(results):
            results = results.join(DF["image_url_runtime"], how="left")
            st.subheader(f"Top-{topk} の結果")
            cols = safe_columns(3)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    render_card_compact(row)

            # -------- ここから追加：開発者用デバッグ（UI外観は保持、折りたたみで表示） --------
            with st.expander("🔧 開発者デバッグ（Meta 相似分解）", expanded=False):
                st.write("MetaEngine 状態：", "✅ 有効" if rec.meta_engine is not None else "❌ 無効")
                if rec.meta_engine is not None:
                    st.caption(f"σ (Level, ATK, DEF) = {list(map(float, rec.meta_engine.sigma))}")
                    rank_to_debug = st.number_input("対象：メタで並べた上位から何番目を確認（0=1位）", 0, max(0, len(results)-1), 0, 1)
                    if st.button("この候補の s_num / s_cat / s_meta を表示"):
                        dbg = rec.debug_meta_components(effective_query_name, rank=int(rank_to_debug))
                        if dbg is None:
                            st.warning("MetaEngine が無効です。")
                        else:
                            st.json(dbg)
                            st.caption("s_num=数値核(ATK/DEF/Level), s_cat=カテゴリ(Type/Attribute/Race), s_meta=Meta内部融合")
                else:
                    st.info("MetaEngine が無効のため、分解は利用できません。from_hf の引数を確認してください。")
            # -----------------------------------------------------------------------

        else:
            st.info("該当する結果がありません。")
else:
    st.info("左側でカード名を選ぶか、画像/カメラで検索して「🔮 検索」を押してください。")
