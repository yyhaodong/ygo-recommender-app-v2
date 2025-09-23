#!/usr/bin/env bash
set -euo pipefail

# 切到脚本所在目录，确保命令在正确的文件夹下执行
cd "$(dirname "$0")"

# 1) 创建并激活虚拟环境
if [ ! -d ".venv" ]; then
  echo "--- Creating virtual environment (.venv)... ---"
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "--- Virtual environment activated. ---"

# 2) 安装依赖
echo "--- Installing dependencies from requirements.txt... ---"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) 运行数据优化脚本
echo "--- Running optimize_data.py script... ---"
python optimize_data.py

echo -e "\n🎉 Setup complete! Optimized files are in ./optimized_data_for_upload/"