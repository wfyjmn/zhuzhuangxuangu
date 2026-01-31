#!/bin/bash
# 完整备份脚本 - 保护所有重要数据
# 使用方法: bash backup_all.sh

BACKUP_ROOT="/workspace/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/temp_backup_$TIMESTAMP"
PROJECT_ROOT="/workspace/projects"

echo "=========================================="
echo "      开始完整备份"
echo "=========================================="
echo "备份时间: $(date)"
echo "备份目录: $BACKUP_DIR"
echo ""

# 创建临时备份目录
mkdir -p "$BACKUP_DIR"

# ============================================
# 1. 备份模型文件
# ============================================
echo "[1/6] 备份模型文件..."
MODELS_BACKUP="$BACKUP_DIR/models"
mkdir -p "$MODELS_BACKUP"

# 备份 assets/models 下的所有模型
if [ -d "$PROJECT_ROOT/assets/models" ]; then
    cp -r $PROJECT_ROOT/assets/models/*.pkl "$MODELS_BACKUP/" 2>/dev/null
    cp -r $PROJECT_ROOT/assets/models/*.json "$MODELS_BACKUP/" 2>/dev/null
    echo "  ✓ assets/models: $(ls $MODELS_BACKUP/ 2>/dev/null | wc -l) 个文件"
fi

# 备份 data/models 下的模型
if [ -d "$PROJECT_ROOT/data/models" ]; then
    find "$PROJECT_ROOT/data/models" -name "*.pkl" -exec cp {} "$MODELS_BACKUP/" \; 2>/dev/null
    echo "  ✓ data/models: $(find $MODELS_BACKUP/ -name "*.pkl" 2>/dev/null | wc -l) 个文件"
fi

# ============================================
# 2. 备份训练数据
# ============================================
echo "[2/6] 备份训练数据..."
DATA_BACKUP="$BACKUP_DIR/training_data"
mkdir -p "$DATA_BACKUP"

if [ -d "$PROJECT_ROOT/data/training" ]; then
    cp -r $PROJECT_ROOT/data/training/*.csv "$DATA_BACKUP/" 2>/dev/null
    echo "  ✓ training data: $(ls $DATA_BACKUP/ 2>/dev/null | wc -l) 个文件"
fi

# ============================================
# 3. 备份配置文件
# ============================================
echo "[3/6] 备份配置文件..."
CONFIG_BACKUP="$BACKUP_DIR/config"
mkdir -p "$CONFIG_BACKUP"

if [ -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env" "$CONFIG_BACKUP/"
    echo "  ✓ .env"
fi

if [ -f "$PROJECT_ROOT/assets/.env" ]; then
    cp "$PROJECT_ROOT/assets/.env" "$CONFIG_BACKUP/"
    echo "  ✓ assets/.env"
fi

# ============================================
# 4. 备份重要日志（最近 7 天）
# ============================================
echo "[4/6] 备份重要日志..."
LOGS_BACKUP="$BACKUP_DIR/logs"
mkdir -p "$LOGS_BACKUP"

if [ -d "$PROJECT_ROOT/logs" ]; then
    find "$PROJECT_ROOT/logs" -name "*.log" -mtime -7 -exec cp {} "$LOGS_BACKUP/" \; 2>/dev/null
    echo "  ✓ logs: $(ls $LOGS_BACKUP/ 2>/dev/null | wc -l) 个文件"
fi

if [ -d "$PROJECT_ROOT/assets/logs" ]; then
    find "$PROJECT_ROOT/assets/logs" -name "*.log" -mtime -7 -exec cp {} "$LOGS_BACKUP/" \; 2>/dev/null
fi

# ============================================
# 5. 创建备份清单
# ============================================
echo "[5/6] 创建备份清单..."
cat > "$BACKUP_DIR/MANIFEST.txt" <<EOF
==========================================
          备份清单
==========================================
备份时间: $(date)
备份目录: $BACKUP_DIR

【文件统计】
模型文件 (.pkl): $(find $MODELS_BACKUP/ -name "*.pkl" 2>/dev/null | wc -l)
配置文件 (.json): $(find $MODELS_BACKUP/ -name "*.json" 2>/dev/null | wc -l)
训练数据 (.csv): $(ls $DATA_BACKUP/ 2>/dev/null | wc -l)
日志文件 (.log): $(ls $LOGS_BACKUP/ 2>/dev/null | wc -l)

【文件列表】
EOF

echo "模型文件:" >> "$BACKUP_DIR/MANIFEST.txt"
ls -lh $MODELS_BACKUP/ 2>/dev/null >> "$BACKUP_DIR/MANIFEST.txt"

echo "" >> "$BACKUP_DIR/MANIFEST.txt"
echo "训练数据:" >> "$BACKUP_DIR/MANIFEST.txt"
ls -lh $DATA_BACKUP/ 2>/dev/null >> "$BACKUP_DIR/MANIFEST.txt"

echo "  ✓ MANIFEST.txt 已创建"

# ============================================
# 6. 压缩备份
# ============================================
echo "[6/6] 压缩备份..."
cd /workspace/backups
tar -czf "backup_$TIMESTAMP.tar.gz" -C . "temp_backup_$TIMESTAMP"
ARCHIVE_SIZE=$(du -h "backup_$TIMESTAMP.tar.gz" | cut -f1)
echo "  ✓ 压缩完成: backup_$TIMESTAMP.tar.gz ($ARCHIVE_SIZE)"

# 清理临时文件
rm -rf "temp_backup_$TIMESTAMP"

# ============================================
# 7. 清理旧备份（保留最近 7 天）
# ============================================
echo ""
echo "清理旧备份..."
OLD_COUNT=$(find $BACKUP_ROOT -name "backup_*.tar.gz" -mtime +7 -delete -print | wc -l)
echo "  ✓ 已清理 $OLD_COUNT 个旧备份"

# ============================================
# 8. 显示备份摘要
# ============================================
echo ""
echo "=========================================="
echo "      备份完成"
echo "=========================================="
echo "备份文件: /workspace/backups/backup_$TIMESTAMP.tar.gz"
echo "文件大小: $ARCHIVE_SIZE"
echo "备份清单: 请查看压缩包中的 MANIFEST.txt"
echo ""
echo "当前备份列表:"
ls -lh $BACKUP_ROOT/backup_*.tar.gz 2>/dev/null | tail -5

# ============================================
# 9. 可选：上传到云端（取消注释以启用）
# ============================================
# echo ""
# echo "上传到云端..."
# # 这里可以添加上传到 Google Drive、Dropbox、AWS S3 等的命令
# # 例如:
# # rclone copy "backup_$TIMESTAMP.tar.gz" remote:backups/
# echo "  ✓ 云端上传已跳过（需要配置）"

echo ""
echo "✅ 备份流程完成！"
