#!/bin/bash

# 清理 Git 历史中的敏感信息
# ⚠️ 警告：此操作会重写 Git 历史，请谨慎使用

echo "========================================="
echo "Git 历史清理工具"
echo "========================================="
echo ""
echo "⚠️ 警告：此操作将重写 Git 历史！"
echo "⚠️ 请确保您已备份当前工作目录！"
echo ""
read -p "确认继续？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "开始清理 Git 历史..."
echo ""

# 1. 从历史中移除包含 Token 的提交
echo "步骤 1: 搜索包含 Token 的提交..."
git log --all --full-history --pretty=format:"%H" -- config/tushare_config.json | head -5

# 2. 使用 git filter-branch 清理
echo ""
echo "步骤 2: 清理历史中的敏感文件..."
git filter-branch --force --index-filter \
    'git rm --cached --ignore-unmatch config/tushare_config.json' \
    --prune-empty --tag-name-filter cat -- --all

# 3. 清理备份和引用
echo ""
echo "步骤 3: 清理备份和引用..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "========================================="
echo "✓ 清理完成！"
echo "========================================="
echo ""
echo "如果远程仓库已有敏感信息，请执行:"
echo "  git push origin --force --all"
echo "  git push origin --force --tags"
echo ""
echo "⚠️ 注意：此操作会重写历史，可能影响其他协作者"
echo "⚠️ 请通知所有协作者重新克隆仓库"
