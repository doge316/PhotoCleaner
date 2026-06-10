#!/usr/bin/env bash
set -e

echo "========================================"
echo "  PhotoCleaner 一键部署"
echo "========================================"

# 1. 创建虚拟环境
if [ ! -d "venv" ]; then
    echo ""
    echo "[1/3] 创建虚拟环境..."
    python3 -m venv venv
else
    echo ""
    echo "[1/3] 虚拟环境已存在，跳过"
fi

# 激活
source venv/bin/activate

# 2. 安装依赖
echo ""
echo "[2/3] 安装 Python 依赖..."
pip install -r requirements.txt

# 3. 下载模型
echo ""
echo "[3/3] 下载模型文件..."
python setup_models.py

echo ""
echo "========================================"
echo "  部署完成"
echo "  启动: source venv/bin/activate && streamlit run app.py"
echo "========================================"
