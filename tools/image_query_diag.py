# image_query_diag.py —— 图片检索链路诊断 v2(只读:不改 App、不改推荐数据、不训练模型)
# ------------------------------------------------------------------
# 检查项(按优先级):
#   1. 自身向量一致性: 同一张卡的图 → 编码 → 和库里"这张卡自己的向量"的余弦(self_cos)。
#      若模型+预处理+图片都和生成 art_embs.npz 时相同, 应 ≈ 1.00。维度相同不代表一致。
#   2. 两套 CLIP 实现是否等价: open_clip ViT-B-32/openai(App 用) vs Hugging Face openai/clip-vit-base-patch32
#      (main_config.yaml 写的)。同一张图 → 两个向量的余弦。
#   3. 找出生成向量时的裁剪方式: 在 20 张卡上搜索裁剪框, 看哪个让 self_cos 最接近 1。
#   4. 各种输入下 top-1 / top-5 是否包含正确卡(同时按"卡 ID 所在行"和"卡名"两种口径计)。
#   5. 未命中案例: 列出实际 top-5 和分数; 统计哪些卡反复出现(例如 Three of a Kind)。
#   6. Three of a Kind 自身检查 + "被多少张卡当作最近邻"(hub 程度)。
# 注意: 余弦相似度不是识别正确率; 命中率只表示"正确卡是否出现在前 K 名"。
# 用法(Colab, 普通 CPU 即可): 方法1 把整个文件粘贴进一个单元格直接运行(默认就是正式数据版本、150 张卡)
# 方法2 上传文件后:
#   !pip -q install open_clip_torch==3.2.0
#   !python image_query_diag.py --revision 3d53e6c3484f8388ee5f44b6d8cda7ad6aff59c1 --n 150
# 输出: 屏幕上的汇总 + image_query_diag.csv(明细) + image_query_misses.csv(未命中案例)
# ------------------------------------------------------------------
import argparse, io, time, collections
import numpy as np, pandas as pd
from PIL import Image, ImageFilter

VARIANT_BOXES = {  # (left, top, right, bottom) 相对比例
    "配置裁剪 pre [0.125,0.26,0.875,0.58]": (0.125, 0.26, 0.875, 0.58),
    "配置裁剪 post [0.11,0.25,0.89,0.59]": (0.11, 0.25, 0.89, 0.59),
}


def load_db(revision):
    from huggingface_hub import snapshot_download
    d = snapshot_download("oneonehaodong/ygo-recommender-data", repo_type="dataset", revision=revision,
                          allow_patterns=["*.parquet", "art_embs.npz"])
    db = pd.read_parquet(f"{d}/card_database.parquet").reset_index(drop=True)
    art = np.load(f"{d}/art_embs.npz")["data"].astype(np.float32)
    return db, art / np.linalg.norm(art, axis=1, keepdims=True)


def make_openclip():
    import torch, open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")  # = app.py
    model.eval()
    print("open_clip 预处理 (= App):", preprocess)

    def enc(img):
        with torch.no_grad():
            f = model.encode_image(preprocess(img.convert("RGB")).unsqueeze(0))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()[0].astype(np.float32)
    return enc


def make_hf():
    """main_config.yaml 里写的 Hugging Face 版本; 没装 transformers 时跳过"""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        print("(跳过 HF 对比: 没有 transformers)", e); return None
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def enc(img):
        with torch.no_grad():
            px = p(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
            # 新版 transformers 的 get_image_features 返回对象而不是张量, 所以直接走 视觉塔 → 投影层(和 get_image_features 原本的计算相同)
            f = m.visual_projection(m.vision_model(pixel_values=px).pooler_output)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()[0].astype(np.float32)
    return enc


def get(url):
    import requests
    r = requests.get(url, timeout=15, headers={"User-Agent": "ygo-diag/1.0"}); r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def crop_rel(img, box):
    w, h = img.size; l, t, r, b = box
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def low_res(img, w=100):
    s = img.resize((w, int(img.height * w / img.width)), Image.BILINEAR)
    return s.resize(img.size, Image.BILINEAR)


def fake_photo(img):
    """模拟手机拍照: 灰色背景上(卡约占画面 60%)、旋转 6°、模糊、强 JPEG 压缩"""
    bg = Image.new("RGB", (int(img.width * 1.6), int(img.height * 1.4)), (90, 85, 80))
    bg.paste(img.rotate(6, expand=True, fillcolor=(90, 85, 80)), (int(img.width * .25), int(img.height * .15)))
    bg = bg.filter(ImageFilter.GaussianBlur(1.5))
    buf = io.BytesIO(); bg.save(buf, "JPEG", quality=30); buf.seek(0)
    return Image.open(buf).convert("RGB")


def card_id(db, i):
    return int(float(str(db.at[i, "id"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="3d53e6c3484f8388ee5f44b6d8cda7ad6aff59c1")
    ap.add_argument("--n", type=int, default=150, help="随机抽多少张卡测试")
    ap.add_argument("--n-crop-search", type=int, default=20, help="用多少张卡搜索生成时的裁剪框")
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()  # 直接粘贴到 Colab 单元格运行时, 会忽略 Jupyter 自带的 -f 参数

    db, art = load_db(args.revision)
    names = db["name"].astype(str).to_numpy()
    print("DB", db.shape, "art", art.shape, "| 同名多行的卡名数:", int((pd.Series(names).value_counts() > 1).sum()))
    enc = make_openclip()
    enc_hf = make_hf()
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(db), size=min(args.n, len(db)), replace=False)

    # ---- 下载图片(整卡 + 官方插画裁剪) ----
    imgs = {}
    for k, i in enumerate(sample):
        cid = card_id(db, i)
        try:
            imgs[i] = (get(f"https://images.ygoprodeck.com/images/cards/{cid}.jpg"),
                       get(f"https://images.ygoprodeck.com/images/cards_cropped/{cid}.jpg"))
        except Exception as e:
            print("skip", cid, e)
        time.sleep(0.05)
        if (k + 1) % 50 == 0:
            print(f"下载 {k+1}/{len(sample)}")
    ok = list(imgs)

    # ---- 2. 两套 CLIP 实现是否等价 ----
    if enc_hf is not None:
        try:  # 这一项失败也不影响后面的检查
            same = [float(enc(imgs[i][1]) @ enc_hf(imgs[i][1])) for i in ok[:10]]
            hf_self = [float(art[i] @ enc_hf(imgs[i][1])) for i in ok[:10]]
            print(f"\n=== 2. open_clip vs HF(同一张插画裁剪图)向量余弦: 中位数 {np.median(same):.4f}"
                  f"(≈1.00 表示两套实现等价) | HF 编码的 self_cos 中位数 {np.median(hf_self):.4f}")
        except Exception as e:
            print("\n=== 2. HF 对比失败, 跳过:", repr(e))

    # ---- 3. 搜索生成向量时的裁剪框 ----
    boxes = [(l, t, 1 - l, b) for l in (0.06, 0.09, 0.12, 0.15) for t in (0.15, 0.18, 0.21, 0.24, 0.27)
             for b in (0.55, 0.60, 0.65, 0.70, 0.74)]
    sub = ok[:args.n_crop_search]
    scores = []
    for box in boxes:
        scores.append((float(np.median([art[i] @ enc(crop_rel(imgs[i][0], box)) for i in sub])), box))
    scores.sort(reverse=True)
    print(f"\n=== 3. 裁剪框搜索({len(sub)} 张卡, 整卡图上按比例裁): self_cos 中位数前 5 ===")
    for s, box in scores[:5]:
        print(f"  {s:.4f}  box(l,t,r,b)={box}")
    best_box = scores[0][1]

    variants = {
        "整卡(官方图)": lambda f, c: f,
        "官方插画裁剪(cards_cropped)": lambda f, c: c,
        **{k: (lambda box: (lambda f, c: crop_rel(f, box)))(v) for k, v in VARIANT_BOXES.items()},
        f"搜索到的最佳裁剪 {best_box}": lambda f, c: crop_rel(f, best_box),
        "整卡·低清晰度(宽100px)": lambda f, c: low_res(f),
        "整卡·模拟拍照": lambda f, c: fake_photo(f),
        "最佳裁剪·模拟拍照(先拍照再按比例裁)": lambda f, c: crop_rel(fake_photo(f), best_box),
    }

    # ---- 1 & 4 & 5. 各输入下的排名、自身一致性、未命中案例 ----
    rows, misses, t_enc = [], [], []
    for vname, fn in variants.items():
        for i in ok:
            t0 = time.perf_counter(); v = enc(fn(*imgs[i])); t_enc.append(time.perf_counter() - t0)
            sims = art @ v
            order = np.argsort(-sims)[:5]
            rank = int((sims > sims[i]).sum()) + 1
            top5_names = [names[j] for j in order]
            rows.append(dict(card=int(i), name=names[i], variant=vname, self_cos=float(sims[i]), rank=rank,
                             top1_name=top5_names[0], top1_sim=float(sims[order[0]]),
                             hit1_id=rank == 1, hit5_id=rank <= 5,
                             hit1_name=top5_names[0] == names[i], hit5_name=names[i] in top5_names))
            if rank > 1:
                misses.append(dict(variant=vname, name=names[i], rank=rank, self_cos=round(float(sims[i]), 4),
                                   top5=" / ".join(f"{names[j]}({sims[j]:.3f})" for j in order)))
    df = pd.DataFrame(rows)
    summ = df.groupby("variant", sort=False).agg(
        n=("card", "count"), self_cos中位数=("self_cos", "median"),
        top1_ID=("hit1_id", "mean"), top5_ID=("hit5_id", "mean"),
        top1_卡名=("hit1_name", "mean"), top5_卡名=("hit5_name", "mean"), 排名中位数=("rank", "median"))
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print("\n=== 1 & 4. 各输入: 自身一致性 + top-1 / top-5 命中率 ===")
    print(summ.round(3).to_string())
    print("读法: self_cos≈1.00 → 和生成向量时的处理一致; 'ID' 口径=正确行, '卡名' 口径=同名卡也算对")

    print("\n=== 5a. 各输入下 top-1 最常出现的卡(检查 Three of a Kind 之类的集中现象) ===")
    for vname, g in df.groupby("variant", sort=False):
        c = collections.Counter(g["top1_name"]).most_common(3)
        print(f"  {vname}: " + ", ".join(f"{n}×{k}" for n, k in c))
    mdf = pd.DataFrame(misses)
    print("\n=== 5b. 未命中案例(每种输入前 3 个) ===")
    if len(mdf):
        print(mdf.groupby("variant", sort=False).head(3).to_string(index=False))

    # ---- 6. Three of a Kind 自身检查 + hub 程度 ----
    hub_name = "Three of a Kind"
    if hub_name in set(names):
        h = int(np.where(names == hub_name)[0][0]); cid = card_id(db, h)
        try:
            f, c = get(f"https://images.ygoprodeck.com/images/cards/{cid}.jpg"), get(f"https://images.ygoprodeck.com/images/cards_cropped/{cid}.jpg")
            print(f"\n=== 6. {hub_name}: 自己的整卡图 self_cos {art[h] @ enc(f):.4f} | 插画裁剪 self_cos {art[h] @ enc(c):.4f}"
                  f" | 最佳裁剪 self_cos {art[h] @ enc(crop_rel(f, best_box)):.4f}")
        except Exception as e:
            print("Three of a Kind 图片下载失败", e)
        nn_count = collections.Counter()
        for s in range(0, len(art), 2000):  # 库内每张卡的最近邻(排除自己)
            S = art[s:s + 2000] @ art.T
            S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -9
            nn_count.update(np.argmax(S, axis=1).tolist())
        top = nn_count.most_common(5)
        print("   库内'被当作最近邻'次数最多的卡: " + ", ".join(f"{names[j]}×{k}" for j, k in top)
              + f" | {hub_name}: {nn_count.get(h, 0)} 次")

    noise = [Image.fromarray(rng.integers(0, 255, (400, 300, 3), dtype=np.uint8)), Image.new("RGB", (400, 300), (200, 30, 30))]
    try:
        noise.append(get("https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.jpg"))
    except Exception:
        pass
    print("\n=== 非卡牌图 top-1 余弦(看能否设'不像卡牌'提示阈值; 不是识别率) ===",
          [round(float((art @ enc(im)).max()), 3) for im in noise])
    print(f"单张编码耗时 中位数 {np.median(t_enc)*1000:.0f} ms")
    df.to_csv("image_query_diag.csv", index=False); mdf.to_csv("image_query_misses.csv", index=False)
    print("明细: image_query_diag.csv / image_query_misses.csv")


if __name__ == "__main__":
    main()
