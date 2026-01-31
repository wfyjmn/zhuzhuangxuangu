#!/bin/bash
# 自动同步守护进程 - 持续监控并推送未推送的提交

cd /workspace/projects

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 自动同步守护进程启动..."

while true; do
    # 检查是否有未推送的提交
    UNPUSHED=$(git rev-list origin/main..main 2>/dev/null | wc -l)

    if [ "$UNPUSHED" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 发现 $UNPUSHED 个未推送的提交，正在推送..."
        git push origin main >> /workspace/backups/sync.log 2>&1

        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 推送成功"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 推送失败"
        fi
    fi

    # 检查是否有未提交的更改
    UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l)

    if [ "$UNCOMMITTED" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ 有 $UNCOMMITTED 个未提交的文件"
    fi

    # 等待30分钟（1800秒）
    sleep 1800
done
