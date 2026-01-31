#!/bin/bash
# 从备份恢复
# 使用方法: bash restore_from_backup.sh backup_20260131_020000.tar.gz

BACKUP_ROOT="/workspace/backups"
PROJECT_ROOT="/workspace/projects"

if [ -z "$1" ]; then
    echo "=========================================="
    echo "      从备份恢复"
    echo "=========================================="
    echo ""
    echo "用法: bash restore_from_backup.sh <备份文件名>"
    echo ""
    echo "可用的备份:"
    ls -lh $BACKUP_ROOT/backup_*.tar.gz 2>/dev/null
    exit 1
fi

BACKUP_FILE="$BACKUP_ROOT/$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "错误: 备份文件不存在: $BACKUP_FILE"
    exit 1
fi

echo "=========================================="
echo "      从备份恢复"
echo "=========================================="
echo "备份文件: $BACKUP_FILE"
echo "备份时间: $(stat -c %y "$BACKUP_FILE")"
echo ""

# 检查项目目录
echo "[警告] 此操作将覆盖以下目录中的文件:"
echo "  - $PROJECT_ROOT/assets/models/"
echo "  - $PROJECT_ROOT/data/training/"
echo ""

read -p "确认继续? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "已取消"
    exit 0
fi

# 创建临时恢复目录
TEMP_DIR="/tmp/restore_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"

echo ""
echo "正在解压备份..."
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

# 找到实际备份目录
BACKUP_CONTENT_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "temp_backup_*" | head -1)

if [ -z "$BACKUP_CONTENT_DIR" ]; then
    echo "错误: 无法找到备份内容目录"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 恢复模型文件
echo "[1/3] 恢复模型文件..."
if [ -d "$BACKUP_CONTENT_DIR/models" ]; then
    cp -r $BACKUP_CONTENT_DIR/models/*.pkl "$PROJECT_ROOT/assets/models/" 2>/dev/null
    cp -r $BACKUP_CONTENT_DIR/models/*.json "$PROJECT_ROOT/assets/models/" 2>/dev/null
    echo "  ✓ 模型文件已恢复"
else
    echo "  ⚠ 模型文件未找到"
fi

# 恢复训练数据
echo "[2/3] 恢复训练数据..."
if [ -d "$BACKUP_CONTENT_DIR/training_data" ]; then
    mkdir -p "$PROJECT_ROOT/data/training"
    cp -r $BACKUP_CONTENT_DIR/training_data/*.csv "$PROJECT_ROOT/data/training/" 2>/dev/null
    echo "  ✓ 训练数据已恢复"
else
    echo "  ⚠ 训练数据未找到"
fi

# 恢复配置文件
echo "[3/3] 恢复配置文件..."
if [ -d "$BACKUP_CONTENT_DIR/config" ]; then
    if [ -f "$BACKUP_CONTENT_DIR/config/.env" ]; then
        cp "$BACKUP_CONTENT_DIR/config/.env" "$PROJECT_ROOT/"
        echo "  ✓ .env 已恢复"
    fi
    if [ -f "$BACKUP_CONTENT_DIR/config/assets/.env" ]; then
        mkdir -p "$PROJECT_ROOT/assets"
        cp "$BACKUP_CONTENT_DIR/config/assets/.env" "$PROJECT_ROOT/assets/"
        echo "  ✓ assets/.env 已恢复"
    fi
fi

# 显示恢复清单
echo ""
echo "=========================================="
echo "      恢复完成"
echo "=========================================="
echo "恢复的文件:"
echo ""
cat "$BACKUP_CONTENT_DIR/MANIFEST.txt" 2>/dev/null || echo "（清单文件未找到）"

# 清理临时文件
rm -rf "$TEMP_DIR"

echo ""
echo "✅ 恢复完成！"
