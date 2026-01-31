#!/bin/bash
# 系统健康检查和文件完整性验证
# 使用方法: bash system_health_check.sh

PROJECT_ROOT="/workspace/projects"
BACKUP_ROOT="/workspace/backups"
REPORT_FILE="/workspace/backups/health_check_report_$(date +%Y%m%d_%H%M%S).txt"

echo "=========================================="
echo "      系统健康检查"
echo "=========================================="
echo "检查时间: $(date)"
echo "报告文件: $REPORT_FILE"
echo ""

# 初始化报告
{
    echo "=========================================="
    echo "      系统健康检查报告"
    echo "=========================================="
    echo "检查时间: $(date)"
    echo ""
} > "$REPORT_FILE"

# ============================================
# 1. 检查模型文件完整性
# ============================================
echo "[1/5] 检查模型文件完整性..."
echo "" >> "$REPORT_FILE"
echo "【模型文件检查】" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

MODEL_COUNT=0
CORRUPTED_MODELS=0

for model_file in "$PROJECT_ROOT/assets/models"/*.pkl; do
    if [ -f "$model_file" ]; then
        MODEL_COUNT=$((MODEL_COUNT + 1))
        FILE_SIZE=$(stat -c%s "$model_file")
        FILE_NAME=$(basename "$model_file")
        
        if [ "$FILE_SIZE" -lt 100 ]; then
            echo "  ❌ $FILE_NAME - 文件太小 ($FILE_SIZE bytes)"
            echo "  ❌ $FILE_NAME - 文件太小 ($FILE_SIZE bytes)" >> "$REPORT_FILE"
            CORRUPTED_MODELS=$((CORRUPTED_MODELS + 1))
        else
            echo "  ✓ $FILE_NAME - 正常 ($FILE_SIZE bytes)"
            echo "  ✓ $FILE_NAME - 正常 ($FILE_SIZE bytes)" >> "$REPORT_FILE"
        fi
    fi
done

if [ $MODEL_COUNT -eq 0 ]; then
    echo "  ⚠ 未找到模型文件"
    echo "  ⚠ 未找到模型文件" >> "$REPORT_FILE"
else
    echo "  总计: $MODEL_COUNT 个模型文件, $CORRUPTED_MODELS 个可能损坏"
    echo "  总计: $MODEL_COUNT 个模型文件, $CORRUPTED_MODELS 个可能损坏" >> "$REPORT_FILE"
fi

# ============================================
# 2. 检查训练数据
# ============================================
echo ""
echo "[2/5] 检查训练数据..."
echo "" >> "$REPORT_FILE"
echo "【训练数据检查】" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

DATA_COUNT=0
if [ -d "$PROJECT_ROOT/data/training" ]; then
    DATA_COUNT=$(ls -1 "$PROJECT_ROOT/data/training"/*.csv 2>/dev/null | wc -l)
    echo "  ✓ 找到 $DATA_COUNT 个训练数据文件"
    echo "  ✓ 找到 $DATA_COUNT 个训练数据文件" >> "$REPORT_FILE"
else
    echo "  ⚠ 未找到训练数据目录"
    echo "  ⚠ 未找到训练数据目录" >> "$REPORT_FILE"
fi

# ============================================
# 3. 检查配置文件
# ============================================
echo ""
echo "[3/5] 检查配置文件..."
echo "" >> "$REPORT_FILE"
echo "【配置文件检查】" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

CONFIG_OK=true

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "  ✓ .env 存在"
    echo "  ✓ .env 存在" >> "$REPORT_FILE"
else
    echo "  ❌ .env 不存在"
    echo "  ❌ .env 不存在" >> "$REPORT_FILE"
    CONFIG_OK=false
fi

if [ -f "$PROJECT_ROOT/assets/.env" ]; then
    echo "  ✓ assets/.env 存在"
    echo "  ✓ assets/.env 存在" >> "$REPORT_FILE"
else
    echo "  ⚠ assets/.env 不存在（可选）"
    echo "  ⚠ assets/.env 不存在（可选）" >> "$REPORT_FILE"
fi

# ============================================
# 4. 检查备份状态
# ============================================
echo ""
echo "[4/5] 检查备份状态..."
echo "" >> "$REPORT_FILE"
echo "【备份状态检查】" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

LATEST_BACKUP=$(ls -t "$BACKUP_ROOT"/backup_*.tar.gz 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "  ❌ 未找到备份文件"
    echo "  ❌ 未找到备份文件" >> "$REPORT_FILE"
    BACKUP_DAYS="N/A"
else
    BACKUP_DATE=$(stat -c %y "$LATEST_BACKUP" | cut -d' ' -f1)
    BACKUP_SIZE=$(du -h "$LATEST_BACKUP" | cut -f1)
    echo "  ✓ 最新备份: $(basename $LATEST_BACKUP)"
    echo "  ✓ 备份日期: $BACKUP_DATE"
    echo "  ✓ 备份大小: $BACKUP_SIZE"
    
    echo "  ✓ 最新备份: $(basename $LATEST_BACKUP)" >> "$REPORT_FILE"
    echo "  ✓ 备份日期: $BACKUP_DATE" >> "$REPORT_FILE"
    echo "  ✓ 备份大小: $BACKUP_SIZE" >> "$REPORT_FILE"
    
    # 检查备份是否过期（超过 2 天）
    BACKUP_SECONDS=$(date -d "$BACKUP_DATE" +%s 2>/dev/null || echo 0)
    CURRENT_SECONDS=$(date +%s)
    DIFF_DAYS=$(( (CURRENT_SECONDS - BACKUP_SECONDS) / 86400 ))
    
    if [ $DIFF_DAYS -gt 2 ]; then
        echo "  ⚠ 备份已过期 ($DIFF_DAYS 天前)"
        echo "  ⚠ 备份已过期 ($DIFF_DAYS 天前)" >> "$REPORT_FILE"
    fi
fi

# 统计备份数量
BACKUP_COUNT=$(ls -1 "$BACKUP_ROOT"/backup_*.tar.gz 2>/dev/null | wc -l)
echo "  ✓ 总备份数: $BACKUP_COUNT"
echo "  ✓ 总备份数: $BACKUP_COUNT" >> "$REPORT_FILE"

# ============================================
# 5. 检查 Git 状态
# ============================================
echo ""
echo "[5/5] 检查 Git 状态..."
echo "" >> "$REPORT_FILE"
echo "【Git 状态检查】" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

cd "$PROJECT_ROOT" || exit 1

# 检查是否有未提交的更改
UNCOMMITTED=$(git status --short | wc -l)
if [ $UNCOMMITTED -gt 0 ]; then
    echo "  ⚠ 有 $UNCOMMITTED 个文件未提交"
    echo "  ⚠ 有 $UNCOMMITTED 个文件未提交" >> "$REPORT_FILE"
    git status --short >> "$REPORT_FILE"
else
    echo "  ✓ 所有更改已提交"
    echo "  ✓ 所有更改已提交" >> "$REPORT_FILE"
fi

# 检查是否与远程同步
REMOTE_STATUS=$(git status -sb | grep -o 'ahead.*behind.*')
if [ -n "$REMOTE_STATUS" ]; then
    echo "  ⚠ 与远程不同步: $REMOTE_STATUS"
    echo "  ⚠ 与远程不同步: $REMOTE_STATUS" >> "$REPORT_FILE"
else
    echo "  ✓ 与远程同步"
    echo "  ✓ 与远程同步" >> "$REPORT_FILE"
fi

# ============================================
# 6. 健康评分
# ============================================
echo ""
echo "=========================================="
echo "      健康评分"
echo "=========================================="
echo "" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "【健康评分】" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

SCORE=100

# 扣分规则
if [ $CORRUPTED_MODELS -gt 0 ]; then
    SCORE=$((SCORE - 30 * CORRUPTED_MODELS))
fi

if [ $MODEL_COUNT -eq 0 ]; then
    SCORE=$((SCORE - 20))
fi

if [ "$CONFIG_OK" = false ]; then
    SCORE=$((SCORE - 30))
fi

if [ -z "$LATEST_BACKUP" ]; then
    SCORE=$((SCORE - 50))
fi

if [ $UNCOMMITTED -gt 10 ]; then
    SCORE=$((SCORE - 10))
fi

# 确保分数不小于 0
if [ $SCORE -lt 0 ]; then
    SCORE=0
fi

echo "健康评分: $SCORE/100"
echo "健康评分: $SCORE/100" >> "$REPORT_FILE"

if [ $SCORE -ge 90 ]; then
    STATUS="优秀 ✓"
elif [ $SCORE -ge 70 ]; then
    STATUS="良好 ✓"
elif [ $SCORE -ge 50 ]; then
    STATUS="一般 ⚠"
else
    STATUS="较差 ❌"
fi

echo "状态: $STATUS"
echo "状态: $STATUS" >> "$REPORT_FILE"

# ============================================
# 7. 建议
# ============================================
echo ""
echo "【建议】"
echo "" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "【建议】" >> "$REPORT_FILE"

if [ $CORRUPTED_MODELS -gt 0 ]; then
    echo "  ⚠ 发现损坏的模型文件，建议从备份恢复"
    echo "  ⚠ 发现损坏的模型文件，建议从备份恢复" >> "$REPORT_FILE"
fi

if [ $MODEL_COUNT -eq 0 ]; then
    echo "  ⚠ 未找到模型文件，建议重新训练或从备份恢复"
    echo "  ⚠ 未找到模型文件，建议重新训练或从备份恢复" >> "$REPORT_FILE"
fi

if [ "$CONFIG_OK" = false ]; then
    echo "  ⚠ 配置文件缺失，请检查 .env 文件"
    echo "  ⚠ 配置文件缺失，请检查 .env 文件" >> "$REPORT_FILE"
fi

if [ -z "$LATEST_BACKUP" ]; then
    echo "  ⚠ 未找到备份文件，建议立即运行备份脚本"
    echo "  ⚠ 未找到备份文件，建议立即运行备份脚本" >> "$REPORT_FILE"
fi

if [ $UNCOMMITTED -gt 0 ]; then
    echo "  ⚠ 有未提交的更改，建议提交并推送到远程"
    echo "  ⚠ 有未提交的更改，建议提交并推送到远程" >> "$REPORT_FILE"
fi

if [ $SCORE -ge 90 ]; then
    echo "  ✓ 系统状态良好，继续保持"
    echo "  ✓ 系统状态良好，继续保持" >> "$REPORT_FILE"
fi

echo ""
echo "=========================================="
echo "✅ 检查完成"
echo "=========================================="
echo "报告已保存: $REPORT_FILE"
