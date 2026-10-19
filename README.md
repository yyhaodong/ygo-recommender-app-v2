# YGO Recommender v2

画像・テキスト・数値の3要素を組み合わせて、
遊戯王カードを「感覚的な近さ」で推薦するAIシステムです。

## 設計の背景・詳細

👉 [note記事：システムの設計思想と使い方](https://note.com/haodong0409/n/n282d983b308b)

## 仕組み（公開中のアプリで実際に使っているもの）

| 観点 | 使っているもの | 補足 |
|---|---|---|
| 画像 | CLIP ViT-B/32（OpenAI の学習済み重み） | カードの絵柄部分を切り出してベクトル化（512 次元）。新旧のカード枠で切り出し位置を変えている。カード DB 側は Hugging Face transformers の `openai/clip-vit-base-patch32`、アプリの検索時は open_clip の `ViT-B-32-quickgelu`（同じ重み） |
| テキスト | `sentence-transformers/all-MiniLM-L6-v2`（英語） | カードの説明文（desc）をそのままベクトル化（384 次元）。初期版は日本語モデル `pkshatech/simcse-ja-bert-base-clcmlp` だったが、説明文が英語のため置き換えた |
| 数値・分類 | 学習済みモデルは使っていない | 種類・属性・種族の一致と、レベル・ATK・DEF の近さを計算。System B はガウス型、System A は線形正規化 + コサイン（比較用） |
| 統合 | RRF（既定）／ Power Mean | 3 つの観点の結果をまとめる |
| 多様性 | MMR | 絵柄が似すぎたカードが並ばないように並べ替える |

- どのモデルも学習済みのものをそのまま使っており、追加の学習（ファインチューニング）はしていない。
- `meta_embs.npz` は読み込まれるが、推薦には使っていない（数値・分類の類似度はその場で計算）。

## データ

- カード情報：YGOPRODeck API（13,342 枚。テキストは英語、1 カード名につき画像 1 枚）
- アプリは Hugging Face のデータセット [`oneonehaodong/ygo-recommender-data`](https://huggingface.co/datasets/oneonehaodong/ygo-recommender-data) を読み込む。
- `optimized_data_for_upload/` は、そのデータセット（revision `3d53e6c`）と同じベクトルにそろえてある。`source_data/lore_embeddings.npy` も同じ 384 次元版。

## 検証スクリプト

画像検索の確認に使ったスクリプトは [`tools/`](tools/README.md) にある（読むだけで、データやモデルは書き換えない）。

## 起動方法

```python
from recommender_v2 import RecommenderV2

rec = RecommenderV2.from_hf("oneonehaodong/ygo-recommender-data", use_meta_engine=True)

# カード名で推薦
results = rec.recommend("Dark Magician", top_n=12, ab_system="B")

# テキストで検索して推薦
results = rec.search_and_recommend("silver dragon with wings", top_n=12)
```
