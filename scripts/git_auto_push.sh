#!/bin/bash
# 自动推送到 GitHub
# 使用方法: bash git_auto_push.sh

PROJECT_ROOT="/workspace/projects"
LOG_FILE="/workspace/backups/git_push.log"

echo "=========================================="
echo "      自动推送到 GitHub"
echo "=========================================="
echo "时间: $(date)"
echo ""

cd "$PROJECT_ROOT" || exit 1

# 检查是否有更改
CHANGED_FILES=$(git status --short | wc -l)

if [ "$CHANGED_FILES" -eq 0 ]; then
    echo "没有更改需要推送"
    exit 0
fi

echo "检测到 $CHANGED_FILES 个文件更改"
echo ""

# 显示更改的文件
echo "更改的文件:"
git status --short
echo ""

# 添加所有更改
echo "添加所有更改..."
git add .

# 提交
COMMIT_MSG="auto: 自动备份 $(date +%Y%m%d_%H%M%S)"
echo "提交: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

# 推送到远程
echo "推送到远程..."
git push origin main

# 记录日志
{
    echo "=========================================="
    echo "时间: $(date)"
    echo "提交信息: $COMMIT_MSG"
    echo "更改文件数: $CHANGED_FILES"
    echo "=========================================="
    echo ""
} >> "$LOG_FILE"

echo ""
echo "✅ 已推送到 GitHub"
echo "日志文件: $LOG_FILE"
