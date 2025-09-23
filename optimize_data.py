# optimize_data.py (最终版，使用相对路径)
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# 这段代码会自动将当前脚本所在的文件夹作为根目录
# 确保了无论您在哪里运行，路径都是正确的
ROOT = Path(__file__).resolve().parent

# 设置默认的输入和输出文件夹
SOURCE_DATA_DIR = ROOT / "source_data"
OUTPUT_DIR = ROOT / "optimized_data_for_upload"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"--- 开始优化数据文件 ---")
print(f"源目录:  {SOURCE_DATA_DIR}")
print(f"输出目录: {OUTPUT_DIR}")

# 检查必需的文件是否存在
required_files = ["card_database.pkl", "art_embeddings.npy", "lore_embeddings.npy", "meta_embeddings.npy"]
for f in required_files:
    if not (SOURCE_DATA_DIR / f).exists():
        print(f"!!! 错误: 缺少必需文件: {SOURCE_DATA_DIR / f}")
        exit(1) # 如果文件不存在，则退出

# 1. pkl -> parquet
print("1) 转换 card_database.pkl -> .parquet ...")
db_df = pd.read_pickle(SOURCE_DATA_DIR / "card_database.pkl")
db_df.to_parquet(OUTPUT_DIR / "card_database.parquet")
print("   完成")

# 2. npy -> npz(float16)
def npy2npz(src_name, dst_name):
    arr = np.load(SOURCE_DATA_DIR / src_name)
    np.savez_compressed(OUTPUT_DIR / dst_name, data=arr.astype(np.float16))
    print(f"   {src_name} -> {dst_name} 完成")

print("2) 压缩向量为 .npz ...")
npy2npz("art_embeddings.npy",  "art_embs.npz")
npy2npz("lore_embeddings.npy", "lore_embs.npz")
npy2npz("meta_embeddings.npy", "meta_embs.npz")

# 3. 生成 meta_num / meta_cat
# （这部分在我们的最终方案里已经合并到 meta_embs.npz, 此处代码仅为示例，
# 实际我们的 recommender_v2.py 直接使用 meta_embs.npz 即可）

print("\n✅ 所有数据已优化完成，输出在:", OUTPUT_DIR)