# app.py — 遊戯王カード 多モーダル推薦（A/B実験・4列グリッド・画像互換対応）
from __future__ import annotations
import os, json, hashlib
from html import escape as _esc
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps

# --- HEIC/HEIF 画像サポート（iPhone対策：未導入でも可）---
HEIF_OK = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_OK = True
except Exception:
    pass

APP_ROOT = Path(__file__).resolve().parent

# ======== CLIP 設定（512 次元で確認） ========
# OpenAI の重みは QuickGELU で学習されている。"ViT-B-32" だと活性化関数が合わず警告が出て、
# カードDBの画像ベクトル（HF transformers 版で生成）と少しずれる（Colab 診断：余弦の中央値 0.965）。
MODEL_NAME = "ViT-B-32-quickgelu"
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
    try:
        st.image(image_obj, caption=caption, use_container_width=True)
    except TypeError as e:
        if "use_container_width" in str(e):
            st.image(image_obj, caption=caption, use_column_width=True)
        else:
            raise

# サーバー側で画像を取りに行くのは、カード画像の配信元だけに限定する（任意 URL の取得はしない）
ALLOWED_IMAGE_HOSTS = ("images.ygoprodeck.com",)
REMOTE_IMAGE_MAX_BYTES = 5 * 1024 * 1024


def _is_allowed_image_url(url: str) -> bool:
    try:
        u = urlparse(url)
    except Exception:
        return False
    return u.scheme == "https" and (u.hostname or "") in ALLOWED_IMAGE_HOSTS


def show_image_url(value: str | None, *, caption=None):
    if not value:
        st.write("—")
        return
    url = str(value)
    try:
        if _is_allowed_image_url(url):
            r = requests.get(url, timeout=8, allow_redirects=False)
            r.raise_for_status()
            if len(r.content) > REMOTE_IMAGE_MAX_BYTES:
                raise ValueError("画像が大きすぎます")
            _img_fit(BytesIO(r.content), caption=caption)
        else:
            st.warning("この画像URLは表示できません（カード画像の配信元以外は取得しません）。")
            st.caption(url)
    except Exception as e:
        st.warning("画像の読み込みに失敗しました。")
        st.caption(url)
        st.caption(f"→ {e}")

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

def _num_str(v) -> str:
    """ATK/DEF などを '1800.0' ではなく '1800' で表示する（欠損は '-'）"""
    if pd.isna(v):
        return "-"
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except Exception:
        return _esc(str(v))

def _bar_html(label: str, value: float, note: str = "") -> str:
    """カード内に埋め込む HTML 版の類似度バー（改行を含めないこと：Markdown に解釈されるため）"""
    try:
        v = float(value); v = 0.0 if np.isnan(v) else max(0.0, min(1.0, v))
    except Exception:
        v = 0.0
    pct = 100 if v >= 1 - 1e-6 else int(np.floor(v * 100))
    return (f'<div class="ygo-sim"><div class="ygo-sim-label">{label}：{pct}%</div>'
            f'<div class="ygo-bar"><div class="ygo-bar-fill" style="width:{pct}%"></div></div>'
            f'<div class="ygo-sim-note">{note}</div></div>')

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

def synced_slider(label: str, vmin, vmax, default, step, key: str, disabled: bool = False):
    """スライダーと数値入力を連動させる（スライダー操作が苦手でも、数値を直接入力できる）。"""
    is_int = all(isinstance(x, int) for x in (vmin, vmax, default, step))
    ks, kn = f"{key}_slider", f"{key}_num"
    if ks not in st.session_state:
        st.session_state[ks] = default
    if kn not in st.session_state:
        st.session_state[kn] = default

    def _snap(v):
        v = min(max(v, vmin), vmax)
        v = round(round((v - vmin) / step) * step + vmin, 6)
        return int(round(v)) if is_int else float(v)

    def _slider_changed():
        st.session_state[kn] = _snap(st.session_state[ks])

    def _number_changed():
        v = _snap(st.session_state[kn])
        st.session_state[kn] = v
        st.session_state[ks] = v

    st.markdown(f"<div style='font-size:14px;margin-bottom:-6px'>{label}</div>", unsafe_allow_html=True)
    c_slider, c_num = st.columns([3, 2])
    with c_slider:
        st.slider(label, vmin, vmax, step=step, key=ks, on_change=_slider_changed,
                  disabled=disabled, label_visibility="collapsed")
    with c_num:
        st.number_input(label, vmin, vmax, step=step, key=kn, on_change=_number_changed,
                        disabled=disabled, label_visibility="collapsed",
                        format="%d" if is_int else "%.2f")
    return st.session_state[ks]

# =========================
# ページ設定 & データ読込
# =========================
st.set_page_config(page_title="遊戯王カード 多モーダル推薦", page_icon="🔮", layout="wide")
st.title("🔮 遊戯王カード 多モーダル推薦エンジン")

@st.cache_resource(show_spinner="推薦エンジンとデータを読み込み中…")
def get_recommender():
    from recommender_v2 import RecommenderV2, MetaWeights
    return RecommenderV2.from_hf(
        "oneonehaodong/ygo-recommender-data",
        use_meta_engine=True,
        meta_engine_kwargs=dict(
            level_col="level", atk_col="atk", def_col="def",
            type_col="type", attribute_col="attribute", race_col="race",
            meta_w=MetaWeights(
                w_cat=0.40, w_level=0.30, w_atk=0.15, w_def=0.15,
                w_type=0.50, w_attr=0.25, w_race=0.25
            ),
            units=(2.0, 50.0, 50.0),
            min_sigma=(1.0, 3.0, 3.0),
            sigma_scale=2.5
        )
    )

rec = get_recommender()
DF: pd.DataFrame = rec.db.copy()

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

COL_NAME  = "name" if "name" in DF.columns else DF.columns[0]
COL_TYPE  = next((c for c in ["type", "card_type", "race", "frameType"] if c in DF.columns), None)
COL_ATK   = next((c for c in ["atk", "ATK"] if c in DF.columns), None)
COL_DEF   = next((c for c in ["def", "DEF", "defe"] if c in DF.columns), None)
COL_RAR   = next((c for c in ["rarity", "Rarity"] if c in DF.columns), None)
COL_DESC  = next((c for c in ["desc", "effect", "text"] if c in DF.columns), None)
COL_ID    = next((c for c in ["id", "passcode", "konami_id", "code"] if c in DF.columns), None)

def image_url_for_row(row: pd.Series) -> str | None:
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
# CLIP エンコーダ
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
    ENCODER_LABEL = f"CLIP: {MODEL_NAME} / {PRETRAINED}"
except Exception:
    ENCODER_OK = False
    ENCODER_LABEL = "画像検索は無効（torch/open-clip を読み込めません）"

def encode_pil_to_vec(pil_img: Image.Image) -> np.ndarray:
    if not ENCODER_OK:
        raise RuntimeError("Image encoder not available.")
    with torch.no_grad():
        x = preprocess_clip(pil_img.convert("RGB")).unsqueeze(0)
        feat = model_clip.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)

# =========================
# 画像からカード候補を探す（推薦アルゴリズム・推薦用ベクトルは変更しない）
# =========================
IMG_MAX_BYTES = 10 * 1024 * 1024          # アップロード上限 10MB
IMG_MAX_PIXELS = 40_000_000               # デコード後の画素数上限（例: 8000×5000）
Image.MAX_IMAGE_PIXELS = IMG_MAX_PIXELS   # これを大きく超える画像は PIL 側でも拒否（解凍爆弾対策）
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "gif"] + (["heic", "heif"] if HEIF_OK else [])
CARD_HW = 614 / 421                       # 遊戯王カード画像の 縦/横
# カードDBの画像ベクトルを作ったときの「絵柄部分の切り出し」（source_data/main_config.yaml の adaptive_crop）
# 2011 年を境にカードの枠が変わるため 2 種類ある。アップロード画像の年代は分からないので両方試す。
ART_BOXES = {"post": (0.11, 0.25, 0.89, 0.59), "pre": (0.125, 0.26, 0.875, 0.58)}
IMAGE_TOPK = 5
# 複数の切り出し方の結果をどうまとめるか: "max"=余弦の最大 / "z"=切り出し方ごとに標準化してから最大 / "rrf"=順位で統合
IMAGE_MERGE = "z"


def load_query_image(data: bytes) -> Image.Image:
    """アップロードされたバイト列を安全に画像として開く。問題があれば ValueError（日本語メッセージ）"""
    if len(data) > IMG_MAX_BYTES:
        raise ValueError(f"ファイルが大きすぎます（上限 {IMG_MAX_BYTES // (1024 * 1024)}MB）。")
    try:
        im = Image.open(BytesIO(data))
        w, h = im.size
        if w * h > IMG_MAX_PIXELS:
            raise ValueError(f"画像の解像度が大きすぎます（{w}×{h}）。")
        im.load()
    except ValueError:
        raise
    except Exception:
        raise ValueError("画像として読み込めませんでした（対応形式: " + ", ".join(UPLOAD_TYPES) + "）。")
    im = ImageOps.exif_transpose(im)   # スマホ写真の向き（EXIF）を直す
    return im.convert("RGB")


def classify_query_image(w: int, h: int) -> str:
    """縦横比だけで入力の種類を推定する: card=カード全体 / art=絵柄だけ(ほぼ正方形) / other=それ以外（写真・スクショ等）"""
    r = h / max(w, 1)
    if abs(r / CARD_HW - 1) <= 0.10:
        return "card"
    if 0.85 <= r <= 1.18:
        return "art"
    return "other"


def query_views(img: Image.Image) -> Tuple[str, Dict[str, Image.Image]]:
    """入力の種類に応じて、CLIP に渡す画像（切り出し方）を用意する"""
    kind = classify_query_image(*img.size)
    w, h = img.size
    crops = {k: img.crop((int(l * w), int(t * h), int(r * w), int(b * h))) for k, (l, t, r, b) in ART_BOXES.items()}
    if kind == "card":
        return kind, crops                          # カード全体 → 絵柄部分だけ（カードDBと同じ切り出し）
    if kind == "art":
        return kind, {"direct": img}                # すでに絵柄だけ → そのまま
    return kind, {"direct": img, **crops}           # 判断できない → 全部試す


def merge_view_scores(S: np.ndarray, method: str = IMAGE_MERGE) -> np.ndarray:
    """S: (切り出し方の数, カード数) の余弦。1 本のスコアにまとめる（大きいほど候補として上）"""
    if S.shape[0] == 1 or method == "max":
        return S.max(axis=0)
    if method == "z":   # 切り出し方ごとに余弦の分布（平均・ばらつき）が違うので、標準化してから比べる
        Z = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)
        return Z.max(axis=0)
    if method == "rrf":
        ranks = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
        return (1.0 / (60.0 + ranks)).sum(axis=0)
    raise ValueError(f"unknown merge method: {method}")


def image_candidates(art: np.ndarray, img: Image.Image, encode, k: int = IMAGE_TOPK,
                     method: str = IMAGE_MERGE) -> Dict[str, Any]:
    """画像 → 上位 k 枚の候補（行番号・参考スコア・どの切り出しで一番近かったか）"""
    kind, views = query_views(img)
    names = list(views)
    S = np.stack([art @ encode(views[n]).astype(np.float32) for n in names])
    score = merge_view_scores(S, method)
    top = np.argsort(-score)[:k]
    return {"kind": kind, "views": names, "idx": top.tolist(),
            "cos": S[:, top].max(axis=0).tolist(),                 # 表示用の参考値（最大の余弦）
            "best_view": [names[j] for j in S[:, top].argmax(axis=0)]}

def encode_text_to_vec(text: str) -> np.ndarray:
    if not ENCODER_OK:
        raise RuntimeError("CLIP not available.")
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    with torch.no_grad():
        tokens = tokenizer([text])
        feat = model_clip.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)[0]


# =========================
# 候補（テキスト・画像・カメラ）と基準カードの状態
#   ・基準カードは st.session_state["name_select"]（ドロップダウン）だけで決める
#   ・テキスト/画像/カメラは「候補」を作るだけ。候補をクリックしたときにドロップダウンへ書き込む
#   ・新しい候補が来たら、古い選択は外す（ユーザーが選び直す）
#   ・画像は内容のハッシュで覚えておき、同じ画像を再計算しない
# =========================
_SOURCE_LABEL = {"text": "テキスト検索", "image": "アップロード画像", "camera": "カメラ画像"}


def _clear_candidates(source: str | None = None):
    if source is None or st.session_state.get("cand_source") == source:
        for k in ("cand_df", "cand_source", "cand_label", "cand_info", "cand_picked"):
            st.session_state.pop(k, None)


def _set_candidates(df: pd.DataFrame, source: str, label: str, info: Dict[str, Any] | None = None):
    st.session_state.update(cand_df=df, cand_source=source, cand_label=label, cand_info=info or {}, cand_picked=None)
    st.session_state["name_select"] = ""      # 古い基準カードの選択を外す
    st.session_state.pop("pick_note", None)
    st.session_state.pop("img_error", None)


def _on_name_change():
    """ドロップダウンで直接カードを選んだ → それが最優先。候補一覧は閉じる（画像の計算結果はキャッシュに残る）"""
    st.session_state.pop("pick_note", None)
    _clear_candidates()


def _pick_candidate(name: str):
    src, label = st.session_state.get("cand_source", ""), st.session_state.get("cand_label", "")
    st.session_state["name_select"] = name
    st.session_state["cand_picked"] = name
    st.session_state["pick_note"] = (name, f"{'💬' if src == 'text' else '🖼'} {_SOURCE_LABEL.get(src, '')}"
                                           f"「**{label}**」→ 候補から選んだ **{name}** で推薦します")


def _repick():
    st.session_state["cand_picked"] = None


def _on_text_search():
    q = (st.session_state.get("text_q") or "").strip()
    if not q:
        return
    if not ENCODER_OK:
        st.session_state["img_error"] = ("text", "CLIP が未ロードです。")
        return
    try:
        sims = np.dot(rec.art, encode_text_to_vec(q))
        top_idx = np.argsort(-sims)[: int(st.session_state.get("text_n", 5))]
        df = rec.db.iloc[top_idx].copy()
        df["cand_score"] = sims[top_idx]
        _set_candidates(df.join(DF["image_url_runtime"], how="left"), "text", q)
    except Exception as e:
        st.session_state["img_error"] = ("text", f"テキスト検索エラー：{e}")


def _image_to_candidates(data: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    img = load_query_image(data)
    res = image_candidates(rec.art, img, encode_pil_to_vec)
    df = rec.db.iloc[res["idx"]].copy()
    df["cand_score"] = res["cos"]
    df["cand_view"] = res["best_view"]
    thumb = img.copy(); thumb.thumbnail((360, 360))
    return df.join(DF["image_url_runtime"], how="left"), {"kind": res["kind"], "views": res["views"], "thumb": thumb}


def _on_image_input(widget_key: str, source: str):
    """画像アップロード／カメラが変わったとき（Streamlit のコールバック：同じ画像なら再計算しない）"""
    f = st.session_state.get(widget_key)
    if f is None:                                  # ファイルを消した → その入力の候補だけ消す（選んだ基準カードは残す）
        _clear_candidates(source)
        st.session_state.pop("img_error", None)
        return
    if not ENCODER_OK:
        st.session_state["img_error"] = (source, "画像エンコーダが無効のため、画像検索はできません。")
        return
    data = f.getvalue()
    h = hashlib.sha1(data).hexdigest()
    cache = st.session_state.setdefault("img_cache", {})
    try:
        if h not in cache:
            cache.clear()                          # 直近 1 枚分だけ覚える（メモリ節約）
            cache[h] = _image_to_candidates(data)
        df, info = cache[h]
        _set_candidates(df, source, getattr(f, "name", "") or _SOURCE_LABEL[source], info)
    except ValueError as e:                        # サイズ超過・読めない画像など
        _clear_candidates(source)
        st.session_state["img_error"] = (source, str(e))
    except Exception as e:
        _clear_candidates(source)
        st.session_state["img_error"] = (source, f"画像検索エラー：{e}")


def _source_status(source: str):
    """サイドバーの各タブに、その入力の状態（エラー・選択中・選び直し）を表示する"""
    err = st.session_state.get("img_error")
    if err and err[0] == source:
        st.error(err[1])
    if st.session_state.get("cand_source") != source:
        return
    picked = st.session_state.get("cand_picked")
    if picked:
        st.success(f"🎯 選択中：**{picked}**")
        c1, c2 = st.columns(2)
        c1.button("↩ 候補を選び直す", key=f"repick_{source}", on_click=_repick, use_container_width=True)
        c2.button("❌ 解除", key=f"clear_{source}", on_click=_clear_and_unselect, use_container_width=True)
    else:
        st.info("右側の候補から基準カードを選んでください。")


def _clear_and_unselect():
    _clear_candidates()
    st.session_state["name_select"] = ""
    st.session_state.pop("pick_note", None)

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
    tab_name, tab_text, tab_image, tab_camera = st.tabs(["カード名", "テキスト検索", "画像から", "カメラ"])

    with tab_name:
        names = [""] + DF[COL_NAME].astype(str).tolist()
        # 基準カードはこのドロップダウンの値だけで決まる（候補から選んだときもここに書き込む）
        st.selectbox("カード名を選択", options=names, key="name_select", on_change=_on_name_change)

    with tab_text:
        st.caption("英語でキーワードを入力")
        st.text_input("🔍 フリーテキスト検索", placeholder="e.g. cute girl magician", key="text_q")
        st.slider("候補カード数", 3, 10, 5, 1, key="text_n")
        st.button("🔍 候補を探す", use_container_width=True, on_click=_on_text_search)
        _source_status("text")

    with tab_image:
        st.file_uploader(f"画像をアップロード（{IMG_MAX_BYTES // (1024 * 1024)}MB まで）", type=UPLOAD_TYPES,
                         key="img_upload", on_change=_on_image_input, args=("img_upload", "image"))
        st.caption("カード全体が写った画像なら、絵柄部分を切り出して比べます。候補 5 枚から選んでください。")
        _source_status("image")

    with tab_camera:
        if ENCODER_OK:
            st.camera_input("カメラで撮影して検索", label_visibility="collapsed",
                            key="cam_input", on_change=_on_image_input, args=("cam_input", "camera"))
            _source_status("camera")
        else:
            st.info("torch/open-clip が未インストールのため、カメラ検索は無効です。")

    # 基準カード = ドロップダウンの値（最後にユーザーが明示的に選んだカード）
    effective_query_name = st.session_state.get("name_select") or None

    with st.expander("Advanced（研究者向け）", expanded=True):
        topk    = synced_slider("Top-K（表示件数）", 6, 36, 18, 2, key="topk")
        fusion  = st.selectbox("融合方式", ["rrf", "power_mean"], index=0,
                               help="RRF：順位ベースで、スコアの尺度差に頑健。power_mean：各モダリティのスコアの冪平均。p が大きいほど『特に高い1つ』を重視し、p=1 は単純な加重平均。")
        p_power = synced_slider("冪平均 p（大きいほど 1つでも高得点のカードを優遇）", 1.0, 3.0, 1.5, 0.1,
                                key="p_power", disabled=(fusion != "power_mean"))
        k_each     = synced_slider("各モダリティの候補数 k_each", 50, 400, 150, 10, key="k_each")
        use_mmr    = st.checkbox("MMR による多様性再ランキングを使用", True)
        mmr_lambda = synced_slider("MMR λ（関連性 vs 反冗長）", 0.1, 0.95, 0.7, 0.05, key="mmr_lambda")

    with st.expander("ℹ️ ATK/DEF などの数値がないカードは、どう比べている？", expanded=False):
        st.markdown(
            """
**メタデータ類似度の中身**

- 種別・属性・種族が一致するか（40%）
- レベル・ATK・DEF が近いか（60%）

**魔法・罠（数値がないカード）の扱い**

1. レベル・ATK・DEF がないので、**全カードの中央値で補って**計算しています。
2. そのため、魔法・罠どうしは数値の部分が常に一致（100%）になり、差は種別と種族（通常・永続・速攻など）だけで決まります。
3. 基準が魔法・罠のとき、モンスターは種別が違うので0点ですが、数値が「中央値のモンスター」に近いと、メタデータ類似度が実態より高く出ます。

**この方法の限界と判断**

- 数値がないカードのメタデータ類似度は**目安**です。魔法・罠では画像・テキストの類似度のほうが判断材料になります。
- 「数値がない＝似ていない」と扱う案も試しましたが、同じ系列のカードを拾える割合が少し下がったため、現状は据え置いています。
- 今後は、魔法・罠の種族や効果の種類を使った別の比べ方を検討したいと考えています。
            """
        )

    st.divider()
    debug = st.toggle("🔧 デバッグ情報を表示", value=False)
    fire  = st.button("🔮 検索", use_container_width=True)

# =========================
# 結果カードの描画
# =========================
# --- 基準カードのサイズ設定 ---------------------------------------------
# 下の推薦グリッド（render_results_grid の .ygo-grid）と同じ値。グリッド側を変えたらここも合わせる。
GRID_MIN_PX = 160      # grid-template-columns: repeat(auto-fill, minmax(160px, 1fr))
GRID_GAP_PX = 12       # gap: 12px
CARD_INSET_PX = 18     # .ygo-card の padding 8px×2 + border 1px×2（画像はこの分だけ列幅より狭い）
BASE_AREA_RATIO = 2.0  # 基準カードの面積 ÷ 推薦カード1枚の面積（1.5〜2 の範囲で調整可）
CARD_ASPECT = "421 / 614"  # YGOPRODeck のカード画像の縦横比（変形・切り抜きなし）


def _base_card_css() -> str:
    """基準カードの幅を「推薦カード1枚の幅 × √面積比」にする CSS を作る。

    推薦グリッドは auto-fill なので、列数 n はコンテナ幅 W で決まる：
        n = floor((W + gap) / (min + gap))、列幅 w = (W − gap·(n−1)) / n
    コンテナクエリで n ごとの区間を書き、その区間の w を 100cqw（= W）から計算する。
    これで画面幅・ズームが変わっても、推薦カードとの面積比が常に BASE_AREA_RATIO になる。
    見た目（背景・枠・角丸・文字色）は下の .ygo-card と揃えている。
    """
    r = BASE_AREA_RATIO ** 0.5
    step = GRID_MIN_PX + GRID_GAP_PX

    def width_for(n: int) -> str:
        return (f"calc({r:.4f} * ((100cqw - {GRID_GAP_PX * (n - 1)}px) / {n} - {CARD_INSET_PX}px))")

    rules = [
        ".ygo-base-wrap{container-type:inline-size;container-name:ygo-base;width:100%;margin:0 0 .5rem 0;}",
        # パネル：推薦カード（.ygo-card）と同じ背景・枠・角丸
        ".ygo-base{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;box-sizing:border-box;width:100%;"
        "background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;}",
        # 1行目はコンテナクエリ非対応ブラウザ用の控えめな固定上限、2行目が本来の計算式
        f".ygo-base-img{{flex:0 0 auto;max-width:100%;width:min(100%,220px);width:{width_for(1)};}}",
        f".ygo-base-img img{{display:block;width:100%;height:auto;aspect-ratio:{CARD_ASPECT};"
        "object-fit:contain;border-radius:4px;}",
        ".ygo-base-info{flex:1 1 280px;min-width:0;max-width:72ch;}",
        ".ygo-base-label{font-size:.8rem;color:#9db4ff;letter-spacing:.04em;margin-bottom:.25rem;}",
        ".ygo-base-name{font-size:1.6rem;font-weight:700;line-height:1.25;color:#eee;margin:0 0 .6rem 0;}",
        ".ygo-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:.8rem;}",
        ".ygo-chip{font-size:.8rem;line-height:1.5;padding:1px 10px;border-radius:999px;"
        "background:#23233d;border:1px solid #3a3a5c;color:#ccd;white-space:nowrap;}",
        ".ygo-chip b{color:#eee;font-weight:600;margin-left:4px;}",
        ".ygo-base-desc{font-size:1rem;line-height:1.7;color:#ddd;word-break:break-word;}",
    ]
    for n in range(2, 25):  # 列数 2〜24（W ≈ 4,100px まで）
        rules.append(f"@container ygo-base (min-width:{step * n - GRID_GAP_PX}px)"
                     f"{{.ygo-base-img{{width:{width_for(n)};}}}}")
    # 文字が画像の下に回る幅（3列以下・約545px未満）では、画像を中央に置いて右側の空白をなくす
    rules.append("@container ygo-base (max-width:545px){.ygo-base{justify-content:center;}}")
    # 推薦グリッドは画面幅 480px 以下で 2 列固定になるので、そこに合わせる（最後に置いて優先させる）
    rules.append(f"@media (max-width:480px){{.ygo-base-img{{width:{width_for(2)};}}}}")
    return "<style>" + "".join(rules) + "</style>"


def _chip(label: str, value) -> str:
    """値がある項目だけチップにする（欠損・空文字は出さない）"""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() in ("", "nan", "None"):
        return ""
    v = _num_str(value) if isinstance(value, (int, float, np.integer, np.floating)) else _esc(str(value))
    return f'<span class="ygo-chip">{_esc(label)}<b>{v}</b></span>'


def render_card_full(row: pd.Series | Dict[str, Any]):
    """基準カード：画像は推薦カード1枚の約 2 倍の面積、右に名前・属性チップ・効果テキスト。

    st.columns の比率ではなく HTML + CSS（flex + コンテナクエリ）で描く。
    Streamlit 内部のクラス名には依存しない（自前の .ygo-base-* だけを使う）。
    """
    d = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    img_url = image_url_for_row(row) or ""
    name = _esc(str(d.get(COL_NAME, "Unknown")))
    type_raw = str(d.get(COL_TYPE) or "")
    is_spell_trap = ("Spell" in type_raw) or ("Trap" in type_raw)
    chips = "".join([
        _chip("種別", d.get(COL_TYPE)),
        _chip("属性", d.get("attribute")),
        _chip("種類" if is_spell_trap else "種族", d.get("race")),
        _chip("レベル", d.get("level")),
        _chip("ATK", d.get(COL_ATK)),
        _chip("DEF", d.get(COL_DEF)),
        _chip("系列", d.get("archetype")),
    ])
    desc_html = ""
    if COL_DESC:
        # 空行があると Markdown の HTML ブロックが途切れるので、改行は <br> にする
        desc = str(d.get(COL_DESC) or "—")
        desc_html = '<div class="ygo-base-desc">' + _esc(desc).replace("\r", "").replace("\n", "<br>") + "</div>"
    img_html = (f'<div class="ygo-base-img"><img src="{_esc(img_url)}" alt="{name}"></div>' if img_url else "")
    html = (
        _base_card_css()
        + '<div class="ygo-base-wrap"><div class="ygo-base">'
        + img_html
        + '<div class="ygo-base-info">'
        + '<div class="ygo-base-label">🔎 基準カード</div>'
        + f'<div class="ygo-base-name">{name}</div>'
        + f'<div class="ygo-chips">{chips}</div>'
        + desc_html
        + "</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

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

# ※ 下の 160px / 12px / padding 8px は GRID_MIN_PX / GRID_GAP_PX / CARD_INSET_PX（基準カードの大きさ計算）と連動
def render_results_grid(results_df: pd.DataFrame, n_cols: int = 4):
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
    .ygo-grid { align-items: start; }
    .ygo-detail { width: 100%; margin-top: 6px; font-size: 11px; color: #ddd; }
    .ygo-detail summary { cursor: pointer; text-align: center; color: #9db4ff; font-size: 11px; }
    .ygo-sim { margin-top: 6px; }
    .ygo-sim-label { font-size: 11px; color: #eee; }
    .ygo-sim-note { font-size: 10px; color: #aaa; margin-top: 2px; }
    .ygo-bar { background: #333; border-radius: 6px; height: 8px; overflow: hidden; margin-top: 2px; }
    .ygo-bar-fill { height: 100%; background: #16a34a; }
    @media (max-width: 480px) {
        .ygo-grid { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """, unsafe_allow_html=True)

    # 総合スコアは融合方式によって尺度が違う（RRF は 0.01 前後、冪平均は 0〜1）ため、
    # 表示では「表示中の1位を100%とした相対値」にして、どちらの方式でも読めるようにする
    try:
        max_final = float(pd.to_numeric(results_df["final_score"], errors="coerce").max())
    except Exception:
        max_final = 0.0

    cards_html = '<div class="ygo-grid">'
    for _, row in results_df.iterrows():
        d = row.to_dict()
        name = _esc(str(d.get(COL_NAME, "Unknown")))
        img_url = image_url_for_row(row) or ""
        img_tag = f'<img src="{img_url}" alt="{name}">' if img_url else ""
        total_rel = (float(d.get("final_score", 0.0)) / max_final) if max_final > 0 else 0.0
        details = (
            '<details class="ygo-detail"><summary>推薦の根拠を見る</summary>'
            + _bar_html("🖼️ 画像類似度", d.get("art_sim", 0.0), "絵柄・色味などの近さ")
            + _bar_html("📖 テキスト類似度", d.get("lore_sim", 0.0), "効果テキストの意味の近さ")
            + _bar_html("🔢 メタデータ類似度", d.get("meta_sim", 0.0), "種別・ATK/DEF 等の一致度")
            + _bar_html("⭐ 総合スコア", total_rel, "上記を融合した評価（表示中の1位=100%）")
            + f'<div class="ygo-sim-note">種別: {_esc(fmt(d.get(COL_TYPE)))} / ATK: {_num_str(d.get(COL_ATK))} / DEF: {_num_str(d.get(COL_DEF))}</div>'
            + '</details>'
        )
        cards_html += f'<div class="ygo-card">{img_tag}<div class="ygo-card-name">{name}</div>{details}</div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

# =========================
# A/B 通知バナー
# =========================
_sys_label = ("System A（Baseline: Min-Max + Cosine）" if ab_system == "A"
              else "System B（Proposed: Gaussian Kernel）")
_sys_cls = "ygo-pill-a" if ab_system == "A" else "ygo-pill-b"
# 上部は1行のステータスだけにする（履歴のクリアはページ下部へ移動）
st.markdown(
    "<style>"
    ".ygo-status{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:-.25rem 0 .25rem 0;}"
    ".ygo-pill{font-size:.85rem;line-height:1.6;padding:2px 12px;border-radius:999px;border:1px solid #3a3a5c;"
    "background:#23233d;color:#ccd;}"
    ".ygo-pill-b{background:rgba(22,163,74,.15);border-color:rgba(22,163,74,.6);color:#bfe8cc;}"
    ".ygo-pill-a{background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.6);color:#c7dbff;}"
    "</style>"
    f'<div class="ygo-status"><span class="ygo-pill {_sys_cls}">🎛 選択中：{_esc(_sys_label)}</span>'
    f'<span class="ygo-pill">🔧 {_esc(ENCODER_LABEL)}</span></div>',
    unsafe_allow_html=True,
)


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
# ★ 候補カード（テキスト検索・画像・カメラ共通）を主画面に表示。クリックで基準カードに決める
# =========================
_cands = st.session_state.get("cand_df")
if _cands is not None and not st.session_state.get("cand_picked"):
    _src = st.session_state.get("cand_source", "")
    _label = st.session_state.get("cand_label", "")
    _info = st.session_state.get("cand_info", {}) or {}
    if _src == "text":
        st.subheader(f"🔍「{_label}」の候補カード")
    else:
        st.subheader(f"🖼 {_SOURCE_LABEL.get(_src, '画像')}の候補カード（上位 {len(_cands)} 枚）")
        _kind_text = {"card": "カード全体の画像 → 絵柄部分を切り出して比較しました",
                      "art": "絵柄だけの画像 → そのまま比較しました",
                      "other": "カード全体か判断できない画像 → 画像全体と、絵柄の切り出しの両方で比較しました"}
        c_img, c_txt = st.columns([1, 5])
        if _info.get("thumb") is not None:
            c_img.image(_info["thumb"], width=120)
        c_txt.caption("判定：" + _kind_text.get(_info.get("kind", ""), "—"))
        c_txt.caption("※ 下の数値は CLIP の画像ベクトルの近さ（参考値）で、正しく識別できた確率ではありません。"
                      "正しいカードが 1 位とは限らないので、目で確認して選んでください。")
    st.caption("カードを選ぶと、そのカードを基準に推薦します（その後「🔮 検索」を押してください）")

    cols = st.columns(len(_cands))
    for i, (col, (_, row)) in enumerate(zip(cols, _cands.iterrows())):
        with col:
            card_name = str(row[COL_NAME])
            img_url = image_url_for_row(row) or ""
            if img_url and _is_allowed_image_url(img_url):
                st.image(img_url, use_column_width=True)   # URL のまま渡す＝ブラウザが直接読み込む（サーバー側では取得しない）
            st.caption(f"**{card_name}**")
            st.caption(f"近さ（参考）: {float(row['cand_score']):.3f}")
            st.button("✅ 選択", key=f"main_sel_{i}", use_container_width=True,
                      on_click=_pick_candidate, args=(card_name,))

    st.divider()

# =========================
# 結果の描画（検索履歴つき）
# =========================
MAX_HISTORY = 5   # 1ページに残す検索結果の数（最新 + 過去4件）

def _entry_cfg_text(cfg: Dict[str, Any]) -> str:
    """検索時に保存した設定（entry）を1行の文字列にする"""
    t = f"System {cfg['ab_system']} ／ 融合: {cfg['fusion']}"
    if cfg["fusion"] == "power_mean":
        t += f"（p={cfg['p_power']:.1f}）"
    t += " ／ MMR: " + (f"λ={cfg['mmr_lambda']:.2f}" if cfg["use_mmr"] else "OFF")
    t += f" ／ k_each={cfg['k_each']} ／ Top-{cfg['topk']}"
    return t


def _cfg_changes(entry: Dict[str, Any], current: Dict[str, Any]) -> list[str]:
    """表示中の結果（entry に保存した設定）と、いまサイドバーで選ばれている設定の違いを列挙する。
    結果に効かない値（rrf のときの p、MMR OFF のときの λ）は比べない。"""
    ch = []
    if current.get("query_name") != entry["query_name"]:
        ch.append(f"基準カード: {entry['query_name']} → {current.get('query_name') or '未選択'}")
    if current["ab_system"] != entry["ab_system"]:
        ch.append(f"System {entry['ab_system']} → {current['ab_system']}")
    if current["fusion"] != entry["fusion"]:
        ch.append(f"融合 {entry['fusion']} → {current['fusion']}")
    elif current["fusion"] == "power_mean" and abs(current["p_power"] - entry["p_power"]) > 1e-9:
        ch.append(f"p {entry['p_power']:.1f} → {current['p_power']:.1f}")
    if current["use_mmr"] != entry["use_mmr"]:
        ch.append("MMR " + ("ON → OFF" if entry["use_mmr"] else "OFF → ON"))
    elif current["use_mmr"] and abs(current["mmr_lambda"] - entry["mmr_lambda"]) > 1e-9:
        ch.append(f"λ {entry['mmr_lambda']:.2f} → {current['mmr_lambda']:.2f}")
    if current["k_each"] != entry["k_each"]:
        ch.append(f"k_each {entry['k_each']} → {current['k_each']}")
    if current["topk"] != entry["topk"]:
        ch.append(f"Top-K {entry['topk']} → {current['topk']}")
    return ch


def render_history_entry(entry: Dict[str, Any], nested: bool = False, current: Dict[str, Any] | None = None):
    """1回分の検索結果（基準カード・設定・Top-N）を描画する。nested=True は expander の中（expander を入れ子にできない）。
    表示する設定は、必ず検索したときに entry に保存した値を使う（サイドバーの現在値は使わない）。
    current（いまの設定）を渡すと、違いがあるときに「再検索で反映」の警告を出す。"""
    if current is not None:
        changes = _cfg_changes(entry, current)
        if changes:
            st.warning("設定が変更されています。再検索すると反映されます。\n\n"
                       "下の結果は前回の設定のままです（変更：" + "、".join(changes) + "）", icon="⚠️")
    if entry.get("text_note"):
        st.info(entry["text_note"])
    base_row = entry.get("base_row")
    if base_row is not None:
        render_card_full(base_row)

    results = entry["results"]
    # 見出しの右にスクリーンショット切替（key を付けて状態を保持：検索履歴が session_state にあるので消えない）
    head_col, shot_col = st.columns([3, 1])
    with head_col:
        st.subheader(f"Top-{entry['topk']} の結果")
    with shot_col:
        screenshot_mode = st.toggle("📸 先頭4枚のみ", value=False, key=f"shot_{entry['id']}",
                                    help="スクリーンショット用：先頭4枚だけ表示します")
    _cls = "ygo-pill-a" if entry["ab_system"] == "A" else "ygo-pill-b"
    st.markdown(f'<div class="ygo-status"><span class="ygo-pill {_cls}">🧾 この結果の設定：'
                f'{_esc(_entry_cfg_text(entry))}</span></div>', unsafe_allow_html=True)
    render_results_grid(results.head(4) if screenshot_mode else results, n_cols=4)

    # 注意書きは結果の下にまとめる
    st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)

    if entry["use_mmr"]:
        st.caption("※ MMR が有効なので、並び順は総合スコア順ではありません。"
                   "「関連性（総合スコア）」と「すでに選んだカードと絵柄が似すぎていないか」のバランスで決まるため、"
                   "総合スコアが高くても、先に並んだカードと絵柄が近いと順位が下がります。")
    qtype = entry.get("query_type", "")
    if ("Spell" in qtype) or ("Trap" in qtype):
        st.caption("※ 基準カードが魔法・罠のため、レベル/ATK/DEF の数値がありません。"
                   "数値は全カードの中央値で補って計算しているので、モンスターの「メタデータ類似度」は"
                   "実態より高く出ることがあります（目安として見てください）。")

    if not nested:
        with st.expander("🔧 開発者デバッグ（Meta 相似分解）", expanded=False):
            st.write("MetaEngine 状態：", "✅ 有効" if rec.meta_engine is not None else "❌ 無効")
            if rec.meta_engine is not None:
                st.caption(f"σ (Level, ATK, DEF) = {list(map(float, rec.meta_engine.sigma))}")
            else:
                st.caption("Baseline（System A）ではメタは埋め込みコサインで計算。")

# =========================
# メイン処理
# =========================
if fire:
    if not effective_query_name:
        st.warning("基準となるカード（または画像）を選んでください。")
    else:
        text_note = None
        _pn = st.session_state.get("pick_note")
        if _pn and _pn[0] == effective_query_name:   # 候補から選んだカードで検索したときだけ、経緯を表示
            text_note = _pn[1]

        base_df = DF[DF[COL_NAME] == effective_query_name]

        with st.spinner("計算中…"):
            try:
                # A/B の切替は recommend() の ab_system 引数で行う
                # （以前は rec.meta_engine を一時的に None にしていたが、System A でエラーになるバグがあった）
                results: pd.DataFrame = rec.recommend(
                    query_name=effective_query_name,
                    top_n=int(topk), k_each=int(k_each),
                    fusion=fusion, p_power=float(p_power),
                    use_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda),
                    ab_system=ab_system,
                )
            except Exception as e:
                st.error("推薦の計算に失敗しました。")
                st.exception(e)
                results = None

        if results is not None and len(results):
            results = results.join(DF["image_url_runtime"], how="left")
            seq = st.session_state.get("hist_seq", 0) + 1
            st.session_state["hist_seq"] = seq
            entry = {
                "id": seq,
                "query_name": effective_query_name,
                "query_type": str(base_df.iloc[0].get(COL_TYPE, "")) if len(base_df) else "",
                "base_row": base_df.iloc[0] if len(base_df) else None,
                "results": results,
                "topk": int(topk), "k_each": int(k_each),
                "fusion": fusion, "p_power": float(p_power),
                "use_mmr": bool(use_mmr), "mmr_lambda": float(mmr_lambda),
                "ab_system": ab_system,
                "text_note": text_note,
            }
            hist = [entry] + st.session_state.get("history", [])
            st.session_state["history"] = hist[:MAX_HISTORY]
        elif results is not None:
            st.info("該当する結果がありません。")

# 履歴は session_state にあるので、ボタンやスライダーを操作して画面が再実行されても消えない
history = st.session_state.get("history", [])
if history:
    current_cfg = {
        "query_name": effective_query_name, "ab_system": ab_system, "fusion": fusion,
        "p_power": float(p_power), "use_mmr": bool(use_mmr), "mmr_lambda": float(mmr_lambda),
        "k_each": int(k_each), "topk": int(topk),
    }
    render_history_entry(history[0], current=current_cfg)
    if len(history) > 1:
        st.divider()
        st.markdown(f"#### 🕘 過去の検索（{len(history) - 1}件・最大{MAX_HISTORY}件まで保持）")
        for e in history[1:]:
            label = f"#{e['id']}  {e['query_name']} ／ System {e['ab_system']}・{e['fusion']}・Top-{e['topk']}"
            with st.expander(label, expanded=False):
                render_history_entry(e, nested=True)
    # 履歴のクリアは一番下に置く（上部をすっきりさせるため）
    if st.button("🗑 検索履歴をクリア"):
        st.session_state["history"] = []
        st.rerun()
elif st.session_state.get("cand_df") is None:
    st.info("左側でカード名を選ぶか、テキスト/画像/カメラで検索して「🔮 検索」を押してください。")
