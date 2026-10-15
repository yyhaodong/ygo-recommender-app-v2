# image_query_validate.py —— 新しい画像検索ロジック（app.py の関数そのもの）を Colab で検証する（読むだけ・何も書き換えない）
# ------------------------------------------------------------------
# ・app.py から切り出し/統合の関数と定数を「そのまま」読み込んで使う（コピーではないので、検証対象 = 実装）
# ・前回の診断で使った 148〜150 枚（seed=0 の抽出）とは重ならない、独立したカードで評価する
# ・比較: 改修前（ViT-B-32・画像全体をそのまま）／ 改修後の各設定（モデル 2 種 × 統合方法 3 種 × 振り分けあり/全切り出し）
# ・入力: 官方完整卡图 / 官方插画裁剪 / 模拟拍照 / 商品截图风(白边) / 低清晰度
# ・命中率 = 正しいカードが上位 K 枚に入った割合（余弦スコアは識別の確率ではない）
# 用法(Colab):
#   !pip -q install open_clip_torch==3.2.0
#   (app.py を /content にアップロードしてから)  単元格に貼り付けて実行 or  !python image_query_validate.py --app app.py
# ------------------------------------------------------------------
import argparse, ast, collections, io, time
import numpy as np, pandas as pd
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter, ImageDraw

WANT_FUNCS = {"load_query_image", "classify_query_image", "query_views", "merge_view_scores", "image_candidates"}
WANT_VARS = {"IMG_MAX_BYTES", "IMG_MAX_PIXELS", "UPLOAD_TYPES", "CARD_HW", "ART_BOXES", "IMAGE_TOPK", "IMAGE_MERGE",
             "MODEL_NAME", "PRETRAINED"}


def find_app(path):
    import os
    for c in (path, "/content/app.py", "/content/ygo-recommender-app-v2/app.py", "/app.py"):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("新版 app.py が見つかりません。/content にアップロードしてください（%cd /content → files.upload()）。")


def load_app_functions(path):
    path = find_app(path)
    print("読み込んだ app.py:", path)
    src = open(path, encoding="utf-8").read()
    ns = {"np": np, "Image": Image, "ImageOps": ImageOps, "BytesIO": BytesIO, "HEIF_OK": False,
          "Dict": dict, "Any": object, "Tuple": tuple}
    exec("from __future__ import annotations", ns)
    for node in ast.parse(src).body:
        if (isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS) or \
           (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANT_VARS for t in node.targets)):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    missing = (WANT_FUNCS | {"ART_BOXES", "IMAGE_MERGE", "MODEL_NAME"}) - set(ns)
    assert not missing, f"app.py に見つからない: {missing}"
    return ns


def load_db(revision):
    from huggingface_hub import snapshot_download
    d = snapshot_download("oneonehaodong/ygo-recommender-data", repo_type="dataset", revision=revision,
                          allow_patterns=["*.parquet", "art_embs.npz"])
    db = pd.read_parquet(f"{d}/card_database.parquet").reset_index(drop=True)
    art = np.load(f"{d}/art_embs.npz")["data"].astype(np.float32)
    return db, art / (np.linalg.norm(art, axis=1, keepdims=True) + 1e-9)   # recommender_v2 と同じ正規化


def make_encoder(name, pretrained="openai"):
    import torch, open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model.eval()

    def enc(img):
        with torch.no_grad():
            f = model.encode_image(preprocess(img.convert("RGB")).unsqueeze(0))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()[0].astype(np.float32)
    return enc


def get(url):
    import requests
    r = requests.get(url, timeout=15, headers={"User-Agent": "ygo-validate/1.0"}); r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def fake_photo(img):
    bg = Image.new("RGB", (int(img.width * 1.6), int(img.height * 1.4)), (90, 85, 80))
    bg.paste(img.rotate(6, expand=True, fillcolor=(90, 85, 80)), (int(img.width * .25), int(img.height * .15)))
    bg = bg.filter(ImageFilter.GaussianBlur(1.5))
    buf = io.BytesIO(); bg.save(buf, "JPEG", quality=30); buf.seek(0)
    return Image.open(buf).convert("RGB")


def fake_screenshot(img):
    """商品ページのスクショ風: 白い余白 + 上に価格っぽい帯（カードは中央、横長の画面）"""
    W, H = int(img.height * 1.2), int(img.height * 1.15)
    bg = Image.new("RGB", (W, H), (250, 250, 250))
    bg.paste(img, ((W - img.width) // 2, int(H * 0.08)))
    ImageDraw.Draw(bg).rectangle([0, 0, W, int(H * 0.05)], fill=(30, 60, 120))
    return bg


def low_res(img, w=120):
    s = img.resize((w, int(img.height * w / img.width)), Image.BILINEAR)
    return s.resize(img.size, Image.BILINEAR)


INPUTS = {
    "官方完整卡图": lambda f, c: f,
    "官方插画裁剪": lambda f, c: c,
    "模拟拍照": lambda f, c: fake_photo(f),
    "商品截图风(白边)": lambda f, c: fake_screenshot(f),
    "低清晰度完整卡图": lambda f, c: low_res(f),
}
METHODS = ["max", "z", "rrf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="app.py")
    ap.add_argument("--revision", default="3d53e6c3484f8388ee5f44b6d8cda7ad6aff59c1")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    args, _ = ap.parse_known_args()

    A = load_app_functions(args.app)
    print("app.py:", "MODEL_NAME =", A["MODEL_NAME"], "| IMAGE_MERGE =", A["IMAGE_MERGE"], "| ART_BOXES =", A["ART_BOXES"])
    db, art = load_db(args.revision)
    names = db["name"].astype(str).to_numpy()

    used_before = set(np.random.default_rng(0).choice(len(db), size=150, replace=False).tolist())  # 前回の診断の抽出
    pool = np.array([i for i in range(len(db)) if i not in used_before])
    sample = np.random.default_rng(args.seed).choice(pool, size=min(args.n, len(pool)), replace=False)
    assert not (set(sample.tolist()) & used_before)
    print(f"独立サンプル {len(sample)} 枚（前回の 150 枚とは重複なし）")

    imgs = {}
    for k, i in enumerate(sample):
        cid = int(float(str(db.at[i, "id"])))
        try:
            imgs[int(i)] = (get(f"https://images.ygoprodeck.com/images/cards/{cid}.jpg"),
                            get(f"https://images.ygoprodeck.com/images/cards_cropped/{cid}.jpg"))
        except Exception as e:
            print("skip", cid, e)
        time.sleep(0.05)
    ok = list(imgs)
    print("画像取得", len(ok), "枚")

    encoders = {"ViT-B-32(改修前)": make_encoder("ViT-B-32"), "ViT-B-32-quickgelu": make_encoder("ViT-B-32-quickgelu")}
    rows = []
    for inp, fn in INPUTS.items():
        for i in ok:
            img = fn(*imgs[i])
            kind, views = A["query_views"](img)          # app.py の振り分け（gate）
            w, h = img.size                               # 比較用: 振り分けずに 3 通り全部（all）
            all_views = {"direct": img, **{k: img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
                                           for k, (l, t, r, b) in A["ART_BOXES"].items()}}
            for ename, enc in encoders.items():
                vecs = {v: enc(im) for v, im in all_views.items()}
                S_all = np.stack([art @ vecs[v] for v in all_views])
                S_gate = np.stack([art @ vecs[v] for v in views])
                # 改修前の挙動: 画像全体をそのまま 1 回だけ（direct）
                if ename.endswith("(改修前)"):
                    s0 = art @ enc(img)
                    rows.append(dict(input=inp, config="改修前: ViT-B-32・全体そのまま", card=i, kind=kind,
                                     rank=int((s0 > s0[i]).sum()) + 1, top1=names[int(np.argmax(s0))],
                                     name5=names[i] in set(names[np.argsort(-s0)[:5]])))
                for policy, S in (("gate", S_gate), ("all", S_all)):
                    for m in METHODS:
                        sc = A["merge_view_scores"](S, m)
                        top5 = np.argsort(-sc)[:5]
                        rows.append(dict(input=inp, config=f"{ename} + {m}" + ("" if policy == "gate" else " [全切り出し]"),
                                         card=i, kind=kind, rank=int((sc > sc[i]).sum()) + 1, top1=names[top5[0]],
                                         name5=names[i] in set(names[top5])))
                if ename == "ViT-B-32-quickgelu" and inp == "官方完整卡图":
                    for v in ("post", "pre", "direct"):
                        rows.append(dict(input=inp, config=f"self_cos[{v}]", card=i, kind=kind,
                                         rank=-1, top1="", name5=False, self_cos=float(art[i] @ vecs[v])))
        print("done:", inp)

    df = pd.DataFrame(rows)
    ev = df[df["rank"] > 0].copy()
    ev["hit1"] = ev["rank"] == 1; ev["hit5"] = ev["rank"] <= 5
    pd.set_option("display.width", 250)
    print("\n=== top-1 / top-5（ID 単位）: 入力 × 設定 ===")
    t = ev.groupby(["input", "config"], sort=False)[["hit1", "hit5"]].mean().unstack("input")
    print(t.round(3).to_string())
    print("\n=== top-5（卡名単位: 同名の別版も正解）===")
    print(ev.groupby(["input", "config"], sort=False)["name5"].mean().unstack("input").round(3).to_string())
    print("\n=== 入力の判定（kind）の内訳 ===")
    print(ev.drop_duplicates(["input", "card"]).groupby("input")["kind"].value_counts().unstack(fill_value=0).to_string())
    sc = df[df["config"].str.startswith("self_cos")]
    if len(sc):
        print("\n=== quickgelu + 卡库と同じ切り出しの self_cos（≈1.00 なら生成時と一致）===")
        print(sc.groupby("config")["self_cos"].describe()[["50%", "min", "max"]].round(4).to_string())
    app_cfg = f"{A['MODEL_NAME']} + {A['IMAGE_MERGE']}"
    print(f"\n=== app.py の現在設定（{app_cfg}）: top-1 に最も多く出たカード（Three of a Kind 集中の確認）===")
    for inp, g in ev[ev["config"] == app_cfg].groupby("input", sort=False):
        print(f"  {inp}: " + ", ".join(f"{n}×{k}" for n, k in collections.Counter(g["top1"]).most_common(3)))
    print("  （改修前）官方完整卡图: " + ", ".join(f"{n}×{k}" for n, k in collections.Counter(
        ev[(ev.config.str.startswith("改修前")) & (ev.input == "官方完整卡图")]["top1"]).most_common(3)))

    enc = encoders["ViT-B-32-quickgelu"]; rng = np.random.default_rng(1)
    noise = {"ランダム雑音": Image.fromarray(rng.integers(0, 255, (400, 300, 3), dtype=np.uint8)),
             "単色": Image.new("RGB", (400, 300), (200, 30, 30))}
    try:
        noise["人物写真(hopper)"] = get("https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.jpg")
    except Exception:
        pass
    print("\n=== 非カード画像: 最大の余弦（参考。カード画像と比べて低いかを見るだけ）===")
    for k, im in noise.items():
        res = A["image_candidates"](art, im, enc)
        print(f"  {k}: kind={res['kind']} 最大余弦={max(res['cos']):.3f} 1位={names[res['idx'][0]]}")
    df.to_csv("image_query_validate.csv", index=False)
    print("\n明細: image_query_validate.csv")


if __name__ == "__main__":
    main()
