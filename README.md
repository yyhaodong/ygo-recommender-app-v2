# YGO Recommender v2

画像・テキスト・数値の3要素を組み合わせて、
遊戯王カードを「感覚的な近さ」で推薦するAIシステムです。

## 設計の背景・詳細

👉 [note記事：システムの設計思想と使い方](https://note.com/haodong0409/n/n282d983b308b)

## 使用技術

- CLIP（画像特徴量）
- テキスト埋め込み（効果文の意味的類似度）
- RBFカーネル / 線形正規化（数値類似度の比較実験）
- RRF・Power Mean（スコア統合）
- MMR（推薦結果の多様性制御）

## 起動方法

```python
from recommender_v2 import RecommenderV2

rec = RecommenderV2.from_hf("oneonehaodong/ygo-recommender-data", use_meta_engine=True)

# カード名で推薦
results = rec.recommend("Dark Magician", top_n=12, ab_system="B")

# テキストで検索して推薦
results = rec.search_and_recommend("silver dragon with wings", top_n=12)
```
