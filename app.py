# app.py — 遊戯王カード 多モーダル推薦（A/B実験・4列グリッド・画像互換対応）
from __future__ import annotations
import os, json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# --- HEIC/HEIF 画像サポート（iPhone対策：未導入でも可）---
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

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
# 画像表示ユーティリティ（後方互換）
# =========================
def _img_fit(image_obj, *, caption=None):
    """Streamlitのバージョン差を吸収しつつ、列幅にフィットして画像を表示する。"""
    try:
        # 新しめのStreamlit
        st.image(image_obj, caption=caption, use_container_width=True)
    except TypeError as e:
        # 古いStreamlitは use_container_width を受け付けない
        if "use_container_width" in str(e):
            st.image(image_obj, caption=caption, use_column_width=True)
        else:
            raise

# =========================
# 画像表示（URL優先）
# =========================
def show_image_url(value: str | None, *, caption=None):
    """列幅いっぱいに画像を自動フィットさせる（レスポンシブ）。"""
    if not value:
        st.write("—")
        return
    url = str(value)
    try:
        if url.startswith(("http://", "https://")):
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            _img_fit(BytesIO(r.content), caption=caption)  # 後方互換ハンドラ経由
        else:
            st.warning("画像URLが無効です（ローカルパス検出）。")
            st.caption(url)
    except Exception as e:
        st.warning("画像の読み込みに失敗しました。")
        st.caption(url)
        st.caption(f"→ {e}")

def safe_columns(n: int):
    """古いコード互換（不要なら未使用でOK）。"""
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

# 表示の丸め修正：1.0 のみ 100%、それ以外は切り捨て整数％
def similarity_bar(label: str, value: float, note: str=""):
    try:
        v = float(value); v = 0.0 if np.isnan(v) else max(0.0, min(1.0, v))
    except Exception:
        v = 0.0
    pct = 100 if v >= 1 - 1e-6 else int(np.floor(v * 100))
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
    """
    RecommenderV2 を Hugging Face データセットから構築。
    - MetaEngine を有効化（System B の数値カーネル用）
    - Baseline（System A）は meta 埋め込みのコサインにフォールバック
    """
    from recommender_v2 import RecommenderV2, MetaWeights
    return RecommenderV2.from_hf(
        "oneonehaodong/ygo-recommender-data",
        use_meta_engine=True,  # B で使用
        meta_engine_kwargs=dict(
            level_col="level", atk_col="atk", def_col="def",
            type_col="type", attribute_col="attribute", race_col="race",
            meta_w=MetaWeights(
                w_cat=0.40, w_level=0.30, w_atk=0.15, w_def=0.15,
                w_type=0.50, w_attr=0.25, w_race=0.25
            ),
            # ゲーム単位スケーリング + σ 下限（縮尺空間）
            units=(1.0, 100.0, 100.0),   # Level=1, ATK/DEF=100
            min_sigma=(1.0, 3.0, 3.0),  # 1級 / 300 ATK / 300 DEF
            sigma_scale=1.0
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
    """画像URLを安全に取得（YGOPROのID補完を含む）。"""
    url_rt = row.get("image_url_runtime", None)
    if pd.notna(url_rt) and isinstance(url_rt, (str, bytes)) and str(url_rt):
        return str(url_rt)
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
    """PIL 画像を CLIP ベクトルへ変換（L2 正規化）。"""
    if not ENCODER_OK:
        raise RuntimeError("Image encoder not available.")
    with torch.no_grad():
        x = preprocess_clip(pil_img.convert("RGB")).unsqueeze(0)
        feat = model_clip.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)

def nearest_card_by_art(rec_obj, v: np.ndarray) -> Tuple[int, str, float]:
    """
    画像ベクトル v と rec.art のコサインで最も近いカードを返す。
    - RecommenderV2 に専用APIが無い場合のフォールバック。
    """
    sims = np.dot(rec_obj.art, v.astype(np.float32))
    i = int(np.argmax(sims))
    name = str(rec_obj.db.iloc[i]["name"])
    return i, name, float(sims[i])

# =========================
# サイドバー（A/B 切替・検索入力・高度設定）
# =========================
with st.sidebar:
    st.header("アルゴリズム切替")
    ab_label = st.radio(
        "A/B テスト用：数値類似の計算法を切り替え",
        options=[
            "System A: Baseline (Min-Max + Cosine)",
            "System B: Proposed (Gaussian Kernel)"
        ],
        index=1
    )
    ab_system = "B" if ab_label.startswith("System B") else "A"

    st.header("🛠 検索パラメータ")
    tab_name, tab_image, tab_camera = st.tabs(["カード名", "画像から", "カメラ"])
    effective_query_name = None
    query = None

    with tab_name:
        names = [""] + DF[COL_NAME].astype(str).tolist()
        query = st.selectbox("カード名を選択", options=names)

    with tab_image:
        # いつでもアップロード表示（エンコーダ有無に依存しない）
        up = st.file_uploader("画像をアップロード", type=["jpg","jpeg","png","webp","gif","heic","heif"])
        url = st.text_input("または画像URLを貼り付け")
        pil = None
        if up:
            try:
                pil = Image.open(up)
            except Exception as e:
                st.error(f"画像の読み込みに失敗しました：{e}")
        elif url:
            try:
                b = requests.get(url, timeout=6).content
                pil = Image.open(BytesIO(b))
            except Exception as e:
                st.error(f"画像URLの取得に失敗しました：{e}")

        if pil is not None:
            _img_fit(pil, caption="クエリ画像プレビュー")  # 後方互換
            if ENCODER_OK:
                with st.spinner("画像特徴を抽出中…"):
                    try:
                        v = encode_pil_to_vec(pil)
                        idx, nn_name, sim = nearest_card_by_art(rec, v)  # フォールバック近傍検索
                        effective_query_name = nn_name
                        st.success(f"最も近いカード：**{nn_name}**（sim={sim:.3f}）")
                    except Exception as e:
                        st.error(f"画像検索エラー：{e}")
            else:
                st.info("画像のプレビューのみ（エンコーダ未有効のため検索は省略）。")

    with tab_camera:
        if ENCODER_OK:
            cam = st.camera_input("カメラで撮影して検索", label_visibility="collapsed")
            if cam is not None:
                try:
                    pil = Image.open(cam)
                    _img_fit(pil, caption="カメラ画像プレビュー")  # 後方互換
                    with st.spinner("画像特徴を抽出中…"):
                        v = encode_pil_to_vec(pil)
                        idx, nn_name, sim = nearest_card_by_art(rec, v)
                        effective_query_name = nn_name
                        st.success(f"最も近いカード：**{nn_name}**（カメラ, sim={sim:.3f}）")
                except Exception as e:
                    st.error(f"画像検索エラー：{e}")
        else:
            st.info("torch/open-clip が未インストールのため、カメラ検索は無効です。")

    if not effective_query_name:
        effective_query_name = query or None

    with st.expander("Advanced（研究者向け）", expanded=True):
        topk    = st.slider("Top-K（表示件数）", 6, 36, 18, 2)
        fusion  = st.selectbox("融合方式", ["rrf", "power_mean"], index=0,
                               help="RRF：スコア尺度に頑健。power_mean：複数モダリティ同時高得点を優遇。")
        p_power = st.slider("冪平均 p（>1 ほど“同時に高得点”を優遇）", 1.0, 3.0, 1.5, 0.1,
                            disabled=(fusion != "power_mean"))
        k_each     = st.slider("各モダリティの候補数 k_each", 50, 400, 150, 10)
        use_mmr    = st.checkbox("MMR による多様性再ランキングを使用", True)
        mmr_lambda = st.slider("MMR λ（関連性 vs 反冗長）", 0.1, 0.95, 0.7, 0.05)

    st.divider()
    debug = st.toggle("🔧 デバッグ情報を表示", value=False)
    fire  = st.button("🔮 検索", use_container_width=True)

# =========================
# 結果カードの描画
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

def render_results_grid(results_df: pd.DataFrame, n_cols: int = 4):
    """
    レスポンシブ CSS grid で結果を表示する。
    スマホ: 2列, タブレット: 3列, PC: 4列以上に自動調整。
    """
    st.markdown("""
    <style>
    .ygo-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 12px;
        width: 100%;
    }
    .ygo-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: #1a1a2e;
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #333;
    }
    .ygo-card img {
        width: 100%;
        height: auto;
        border-radius: 4px;
    }
    .ygo-card-name {
        font-size: 12px;
        text-align: center;
        margin-top: 6px;
        color: #eee;
        word-break: break-word;
    }
    @media (max-width: 480px) {
        .ygo-grid { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """, unsafe_allow_html=True)

    cards_html = '<div class="ygo-grid">'
    for _, row in results_df.iterrows():
        d = row.to_dict()
        name = str(d.get(COL_NAME, "Unknown"))
        img_url = image_url_for_row(row) or ""
        img_tag = f'<img src="{img_url}" alt="{name}">' if img_url else ""
        cards_html += f'<div class="ygo-card">{img_tag}<div class="ygo-card-name">{name}</div></div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    with st.expander("📋 全結果の詳細スコアを見る", expanded=False):
        for _, row in results_df.iterrows():
            d = row.to_dict()
            st.markdown(f"**{d.get(COL_NAME, 'Unknown')}**")
            similarity_bar("🖼️ 画像類似度",      d.get("art_sim", 0.0),    "絵柄・色味などの近さ")
            similarity_bar("📖 テキスト類似度",   d.get("lore_sim", 0.0),   "効果テキストの意味の近さ")
            similarity_bar("🔢 メタデータ類似度", d.get("meta_sim", 0.0),   "種別・ATK/DEF 等の一致度")
            similarity_bar("⭐ 総合スコア",       d.get("final_score", 0.0), "上記を融合した最終評価")
            st.divider()

# =========================
# A/B 横断の通知バナー
# =========================
if ab_system == "A":
    st.info("🧪 現在テスト中：System A（Baseline: Min-Max + Cosine）", icon="🧪")
else:
    st.success("🧪 現在テスト中：System B（Proposed: Gaussian Kernel）", icon="🧪")

# デバッグ（任意）
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
# メイン処理（A/B 切替はメタ経路の切換で実装）
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
                # --- A/B の切替：
                # RecommenderV2.recommend の実装に ab_system 引数が無くても動くよう
                # meta_engine の有効/無効を一時的に切り替えて制御する。
                _saved_engine = rec.meta_engine
                if ab_system == "A":
                    rec.meta_engine = None  # Baseline: メタは埋め込みのコサイン
                else:
                    rec.meta_engine = _saved_engine  # Proposed: MetaEngine（RBF/Gaussian）

                results: pd.DataFrame = rec.recommend(
                    query_name=effective_query_name,
                    top_n=int(topk), k_each=int(k_each),
                    fusion=fusion, p_power=float(p_power),
                    use_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda)
                )
            except Exception as e:
                st.error("推薦の計算に失敗しました。")
                st.exception(e)
                results = None
            finally:
                # インスタンスを元の状態に戻す（再実行の整合性を保つ）
                rec.meta_engine = _saved_engine

        if results is not None and len(results):
            results = results.join(DF["image_url_runtime"], how="left")

            st.subheader(f"Top-{topk} の結果")
            screenshot_mode = st.toggle("📸 スクリーンショット用（先頭4枚を4列グリッドで表示）", value=False)
            if screenshot_mode:
                render_results_grid(results.head(4), n_cols=4)
            else:
                render_results_grid(results, n_cols=4)

            # ---- 開発者向け：Meta の分解（B の理解補助）----
            with st.expander("🔧 開発者デバッグ（Meta 相似分解）", expanded=False):
                st.write("MetaEngine 状態：", "✅ 有効" if rec.meta_engine is not None else "❌ 無効")
                if rec.meta_engine is not None:
                    st.caption(f"σ (Level, ATK, DEF) = {list(map(float, rec.meta_engine.sigma))}")
                else:
                    st.caption("Baseline（System A）ではメタは埋め込みコサインで計算。")
            # --------------------------------------------------
        else:
            st.info("該当する結果がありません。")
else:
    st.info("左側でカード名を選ぶか、画像/カメラで検索して「🔮 検索」を押してください。")
