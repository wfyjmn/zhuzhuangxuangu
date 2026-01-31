#!/bin/bash
# Optuna自动化参数调优训练脚本
# 基于对话4.txt优化建议

set -e

# 1. 确保在项目根目录运行
cd "$(dirname "$0")" || exit

echo "========================================================================"
echo "Optuna自动化参数调优训练（DeepQuant V5.0）"
echo "========================================================================"
echo "配置文件: config/optuna_auto_tuned_config.json"
echo ""

# 2. 加载环境变量
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ 环境变量已加载"
else
    echo "⚠️  警告: 未找到 .env 文件"
fi

# 3. 检查依赖
echo ""
echo "检查依赖..."
python3 -c "
import sys
try:
    import optuna
    import xgboost
    import pandas
    import numpy
    print('✅ 所有依赖已安装')
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "安装依赖:"
    echo "  pip install optuna xgboost pandas numpy"
    exit 1
fi

# 4. 后台运行训练
LOG_FILE="logs/auto_tuned_training.log"
PID_FILE="auto_tuned_training.pid"

# 检查是否已有进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "❌ 已有训练任务在运行 (PID: $OLD_PID)"
        echo "   如需重新运行，请先执行: kill $OLD_PID"
        exit 1
    else
        echo "清理过期的 PID 文件"
        rm "$PID_FILE"
    fi
fi

# 创建日志目录
mkdir -p logs

echo "🚀 启动后台训练任务..."
echo "📄 日志文件: $LOG_FILE"
echo ""
echo "⚠️  预计耗时: 30-60 分钟"
echo ""

nohup python3 -u -c "
import sys
sys.path.insert(0, 'assets')
from auto_tuned_trainer import AutoTunedTrainer

try:
    trainer = AutoTunedTrainer('config/optuna_auto_tuned_config.json')
    model, threshold = trainer.train_full_pipeline()

    print('')
    print('=' * 80)
    print('✨ 训练成功完成！')
    print('=' * 80)
    print(f'最优阈值: {threshold:.2f}')
    print(f'最优参数: {trainer.best_params}')

except Exception as e:
    print('')
    print('=' * 80)
    print('❌ 训练失败')
    print('=' * 80)
    print(f'错误: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" > "$LOG_FILE" 2>&1 &

# 保存 PID
echo $! > "$PID_FILE"

echo "✅ 任务已后台启动 (PID: $!)"
echo "📄 日志文件: $LOG_FILE"
echo ""
echo "查看实时日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看任务状态:"
echo "  ps -p \$(cat $PID_FILE) || echo '任务已完成'"
echo ""
echo "停止任务:"
echo "  kill \$(cat $PID_FILE) && rm $PID_FILE"
echo ""
