# detect_clip_model.py
import json, sys
import numpy as np
import torch, open_clip
from PIL import Image
import requests
from io import BytesIO

# ---------- 1) 载入你的向量 ----------
EMB_PATH = "art_embs.npz"     # 如在别处请改路径
KEY = "data"                  # 你的 npz 键
embs = np.load(EMB_PATH)[KEY].astype(np.float32)
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
print(f"✅ emb shape = {embs.shape}  (dim={embs.shape[1]})")

# ---------- 2) 准备一张测试图 ----------
IMG_URL = "https://images.ygoprodeck.com/images/cards/89631139.jpg"
img = Image.open(BytesIO(requests.get(IMG_URL).content)).convert("RGB")

# ---------- 3) 备选模型列表（可按需增删） ----------
candidates = [
    ("ViT-B-32", "openai"),
    ("ViT-B-16", "openai"),
    ("ViT-L-14", "openai"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-H-14", "laion2b_s32b_b79k"),
]

def test_one(name, pre):
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre)
    model.eval()
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    v = feat.cpu().numpy().astype(np.float32)[0]
    score = float(np.dot(embs, v).max())
    return score

scores = []
for name, pre in candidates:
    print(f"🔍 测试 {name:10s} ({pre}) ...", end="")
    try:
        s = test_one(name, pre)
        print(f" max cosine = {s:.4f}")
        scores.append((s, name, pre))
    except Exception as e:
        print(f" 失败：{e}")

if not scores:
    print("❌ 没有任何模型测试成功。请检查网络或 open_clip 安装。")
    sys.exit(1)

best = max(scores, key=lambda x: x[0])
best_score, best_name, best_pre = best

print("\n=== 判定结果 ===")
for s, n, p in sorted(scores, key=lambda x: -x[0]):
    print(f"{n:10s} | {p:22s} | {s:.4f}")
print("\n👉 请把下面两行粘贴进 app.py 顶部：")
print(f'MODEL_NAME = "{best_name}"')
print(f'PRETRAINED = "{best_pre}"')

# 同时写入到 clip_config.json，便于应用自动读取
cfg = {"MODEL_NAME": best_name, "PRETRAINED": best_pre, "score": best_score}
with open("clip_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print("\n已写入 clip_config.json：", cfg)
