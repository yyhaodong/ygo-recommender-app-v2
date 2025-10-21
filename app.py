# app.py — 遊戯王カード 多モーダル推薦（日本語UI・名称/画像/カメラ・左右対比）
from __future__ import annotations
import os, json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent

# ======== CLIP 設定 ========
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
if os.path.exists("clip_config.json"):
    try:
        _cfg = json.load(open("clip_config.json", "r", encoding="utf-8"))
        MODEL_NAME = _cfg.get("MODEL_NAME", MODEL_NAME)
        PRETRAINED = _cfg.get("PRETRAINED", PRETRAINED)
    except Exception:
        pass


def show_image_url(value: str | None, *, caption=None):
    if not value:
        st.write("—")
        return
    url = str(value)
    try:
        if url.startswith(("http://", "https://")):
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            st.image(BytesIO(r.content), caption=caption, use_column_width=True)
        else:
            st.warning("画像URLが無効です（ローカルパス検出）。")
            st.caption(url)
    except Exception as e:
        st.warning("画像の読み込みに失敗しました。")
        st.caption(url)
        st.caption(f"→ {e}")


def safe_columns(n: int):
    try:
        n = int(n or 1)
    except Exception:
        n = 1
    return st.columns(max(1, min(n, 6)), gap="small")


def pill(text: str):
    st.markdown(
        f"""<span style="display:inline-block;padding:2px 8px;border-radius:999px;
        background:#eef2ff;border:1px solid #c7d2fe;font-size:12px;">{text}</span>""",
        unsafe_allow_html=True,
    )


def fmt(v):
    return "-" if pd.isna(v) else str(v)


def similarity_bar(label: str, value: float, note: str = ""):
    try:
        v = float(value)
        v = 0.0 if np.isnan(v) else max(0.0, min(1.0, v))
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
    from recommender_v2 import RecommenderV2, MetaWeights
    return RecommenderV2.from_hf(
        "oneonehaodong/ygo-recommender-data",
        use_meta_engine=True,
        meta_engine_kwargs=dict(
            level_col="level",
            atk_col="atk",
            def_col="def",
            type_col="type",
            attribute_col="attribute",
            race_col="race",
            meta_w=MetaWeights(
                w_cat=0.40, w_level=0.30, w_atk=0.15, w_def=0.15,
                w_type=0.50, w_attr=0.25, w_race=0.25
            ),
            units=(1.0, 100.0, 100.0),
            min_sigma=(1.0, 3.0, 3.0),
            sigma_scale=1.0
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
            cand = df[col].astype(str)
            break
    if cand is None:
        return pd.Series([None] * len(df), index=df.index)
    ids = cand.str.extract(r"(\d{5,})", expand=False)
    base = "https://images.ygoprodeck.com/images/cards/"
    url = base + ids + ".jpg"
    return url.where(ids.notna(), other=None)


DF["image_url_runtime"] = make_runtime_image_url(DF)

COL_NAME = "name" if "name" in DF.columns else DF.columns[0]
COL_TYPE = next((c for c in ["type", "card_type", "race", "frameType"] if c in DF.columns), None)
COL_ATK = next((c for c in ["atk", "ATK"] if c in DF.columns), None)
COL_DEF = next((c for c in ["def", "DEF", "defe"] if c in DF.columns), None)
COL_RAR = next((c for c in ["rarity", "Rarity"] if c in DF.columns), None)
COL_DESC = next((c for c in ["desc", "effect", "text"] if c in DF.columns), None)
COL_ID = next((c for c in ["id", "passcode", "konami_id", "code"] if c in DF.columns), None)


# ✅ 修正后的括号正确版本
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
