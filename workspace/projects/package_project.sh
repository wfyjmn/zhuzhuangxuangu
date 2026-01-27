#!/bin/bash

echo "=========================================================="
echo "  DeepQuant 项目打包脚本"
echo "=========================================================="

# 项目名称
PROJECT_NAME="DeepQuant-System"
PACKAGE_NAME="${PROJECT_NAME}-v3.0.zip"

# 创建临时打包目录
TEMP_DIR="temp_package_${PROJECT_NAME}"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo ""
echo "[1/4] 准备打包目录..."

# 复制项目文件到临时目录
echo ""
echo "[2/4] 复制项目文件..."

# 根目录文件
cp requirements.txt "$TEMP_DIR/" 2>/dev/null
cp README.md "$TEMP_DIR/" 2>/dev/null
cp .gitignore "$TEMP_DIR/" 2>/dev/null

# src 目录
mkdir -p "$TEMP_DIR/src"
cp -r src/* "$TEMP_DIR/src/" 2>/dev/null

# assets 目录（主程序）
mkdir -p "$TEMP_DIR/assets"
cd assets

# 复制所有 Python 文件
cp *.py "$TEMP_DIR/assets/" 2>/dev/null

# 复制配置文件（不包括 .env）
cp strategy_params.json "$TEMP_DIR/assets/" 2>/dev/null
cp .env.example "$TEMP_DIR/assets/" 2>/dev/null
cp .gitignore "$TEMP_DIR/assets/" 2>/dev/null

# 复制文档文件
cp *.md "$TEMP_DIR/assets/" 2>/dev/null

# 复制 CSV 数据文件（模板）
cp validation_records.csv "$TEMP_DIR/assets/" 2>/dev/null
cp paper_trading_records.csv "$TEMP_DIR/assets/" 2>/dev/null
cp params_history.csv "$TEMP_DIR/assets/" 2>/dev/null

cd ..

echo ""
echo "[3/4] 创建压缩包..."

# 创建压缩包
cd "$TEMP_DIR"
zip -r "../$PACKAGE_NAME" ./* -x "__pycache__/*" "*.pyc" "*.pyo" "*.backup" ".DS_Store"
cd ..

echo ""
echo "[4/4] 清理临时目录..."
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================================="
echo "  打包完成！"
echo "=========================================================="
echo ""
echo "压缩包名称: $PACKAGE_NAME"
echo "压缩包大小: $(du -h "$PACKAGE_NAME" | cut -f1)"
echo "压缩包位置: $(pwd)/$PACKAGE_NAME"
echo ""
echo "包含的文件类型："
echo "  - Python 源码文件 (*.py)"
echo "  - 配置文件 (*.json, .env.example)"
echo "  - 文档文件 (*.md)"
echo "  - 数据文件模板 (*.csv)"
echo ""
echo "不包含的文件："
echo "  - .env (包含敏感 Token，需要自行创建)"
echo "  - __pycache__/ (Python 缓存)"
echo "  - *.backup (备份文件)"
echo "  - 临时文件"
echo ""
echo "下载后请执行："
echo "  1. 解压压缩包"
echo "  2. 复制 .env.example 为 .env"
echo "  3. 在 .env 中填入您的 Tushare Token"
echo "  4. 安装依赖: pip install -r requirements.txt"
echo "  5. 运行程序: python assets/main_controller.py"
echo ""
echo "=========================================================="
