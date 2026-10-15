# text_search_check.py —— テキスト検索（CLIP の文字→画像）が QuickGELU 変更でどう変わるかの確認（読むだけ）
# app.py の _on_text_search と同じ計算: sims = art @ encode_text(q) → 上位 N 枚
# 用法(Colab): !pip -q install open_clip_torch==3.2.0 → このファイルを単元格に貼って実行
import numpy as np, pandas as pd

REVISION = "3d53e6c3484f8388ee5f44b6d8cda7ad6aff59c1"
QUERIES = ["beautiful female magician", "cute girl magician"]
TARGET = "Ghostrick Witch"
TOPN = 10


def load_db():
    from huggingface_hub import snapshot_download
    d = snapshot_download("oneonehaodong/ygo-recommender-data", repo_type="dataset", revision=REVISION,
                          allow_patterns=["*.parquet", "art_embs.npz"])
    db = pd.read_parquet(f"{d}/card_database.parquet").reset_index(drop=True)
    art = np.load(f"{d}/art_embs.npz")["data"].astype(np.float32)
    return db, art / (np.linalg.norm(art, axis=1, keepdims=True) + 1e-9)


def text_encoder(name):
    import torch, open_clip
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained="openai"); model.eval()
    tok = open_clip.get_tokenizer(name)

    def enc(text):
        with torch.no_grad():
            f = model.encode_text(tok([text])); f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()[0].astype(np.float32)
    return enc


db, art = load_db()
names = db["name"].astype(str).to_numpy()
for model in ["ViT-B-32", "ViT-B-32-quickgelu"]:
    enc = text_encoder(model)
    tag = "（現在の公開版）" if model == "ViT-B-32" else "（新しい app.py）"
    print(f"\n==================== {model} {tag} ====================")
    for q in QUERIES:
        s = art @ enc(q)
        order = np.argsort(-s)
        rank = int(np.where(names[order] == TARGET)[0][0]) + 1 if TARGET in set(names) else None
        print(f"\n「{q}」→ {TARGET} の順位: {rank}（候補数スライダーは 3〜10、既定 5）")
        for r, j in enumerate(order[:TOPN], 1):
            print(f"  {r:2d}. {names[j]}  ({s[j]:.3f})")
