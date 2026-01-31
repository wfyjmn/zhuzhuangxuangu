#!/bin/bash
# 自动同步脚本 - 确保所有提交都推送到远程仓库
# 添加到 crontab: */30 * * * * /workspace/projects/scripts/auto_sync.sh >> /workspace/backups/sync.log 2>&1

cd /workspace/projects

# 检查是否有未推送的提交
UNPUSHED=$(git rev-list origin/main..main | wc -l)

if [ "$UNPUSHED" -gt 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 发现 $UNPUSHED 个未推送的提交，正在推送..."
    git push origin main
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 推送成功"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 推送失败"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 没有未推送的提交"
fi

# 检查是否有未提交的更改
UNCOMMITTED=$(git status --porcelain | wc -l)

if [ "$UNCOMMITTED" -gt 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ 有 $UNCOMMITTED 个未提交的文件"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 工作区干净"
fi
