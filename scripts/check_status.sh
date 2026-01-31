#!/bin/bash
# 系统状态检查脚本 - 快速验证所有保护措施

echo "=========================================="
echo "      系统状态检查"
echo "=========================================="
echo ""

# 1. 检查守护进程
echo "[1/5] 检查守护进程..."
if ps aux | grep auto_sync_daemon | grep -v grep > /dev/null; then
    echo "  ✓ 守护进程正在运行 (PID: $(ps aux | grep auto_sync_daemon | grep -v grep | awk '{print $2}'))"
else
    echo "  ✗ 守护进程未运行"
fi

# 2. 检查远程仓库
echo ""
echo "[2/5] 检查远程仓库..."
git fetch origin > /dev/null 2>&1
UNPUSHED=$(git rev-list origin/main..main 2>/dev/null | wc -l)
if [ "$UNPUSHED" -eq 0 ]; then
    echo "  ✓ 所有提交已推送到远程"
else
    echo "  ⚠ 有 $UNPUSHED 个未推送的提交"
fi

# 3. 检查备份文件
echo ""
echo "[3/5] 检查备份文件..."
if [ -f "/workspace/backups/emergency_backup_20260131_193648.tar.gz" ]; then
    SIZE=$(du -h /workspace/backups/emergency_backup_20260131_193648.tar.gz | cut -f1)
    echo "  ✓ 紧急备份存在 ($SIZE)"
else
    echo "  ✗ 紧急备份不存在"
fi

# 4. 检查模型文件
echo ""
echo "[4/5] 检查模型文件..."
MODEL_COUNT=$(ls -1 assets/models/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "  ✓ 找到 $MODEL_COUNT 个模型文件"
else
    echo "  ✗ 未找到模型文件"
fi

# 5. 检查数据文件
echo ""
echo "[5/5] 检查数据文件..."
DATA_COUNT=$(find assets/data -name "*.csv" 2>/dev/null | wc -l)
if [ "$DATA_COUNT" -gt 0 ]; then
    echo "  ✓ 找到 $DATA_COUNT 个数据文件"
else
    echo "  ✗ 未找到数据文件"
fi

echo ""
echo "=========================================="
echo "      检查完成"
echo "=========================================="
echo ""
echo "最近的提交（远程）:"
git log --oneline origin/main -5
echo ""
echo "最近的守护进程日志:"
tail -5 /workspace/backups/daemon.log 2>/dev/null | sed 's/^/  /'
