#!/bin/bash
# 完整增量更新脚本 - 重新下载 2023-2024 年的所有数据
#
# 用途: 完整更新2023-2024年的所有数据（包含新特征和指数）
# 范围: 2023.01.01 ~ 2024.12.31（约 479 个交易日）
# 预计耗时: 约 240 分钟（4 小时）
# 使用场景: 首次完整数据更新，或需要彻底刷新数据

set -e  # 遇到错误立即退出

# 1. 确保在项目根目录运行
cd "$(dirname "$0")" || exit

echo "========================================================================"
echo "🚀 完整增量更新 - 2023-2024 全量数据"
echo "========================================================================"
echo "📅 目标: 2023.01.01 ~ 2024.12.31 (个股 + 指数)"
echo "⏱️  预计耗时: 3-4 小时"
echo ""

# 2. 加载环境变量
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ 环境变量已加载"
else
    echo "⚠️  警告: 未找到 .env 文件"
fi

# 3. 后台运行 Python 脚本
LOG_FILE="full_incremental_update.log"
PID_FILE="full_incremental_update.pid"

# 检查是否已有进程在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "❌ 已有更新任务在运行 (PID: $OLD_PID)"
        echo "   如需重新运行，请先执行: kill $OLD_PID"
        exit 1
    else
        echo "清理过期的 PID 文件"
        rm "$PID_FILE"
    fi
fi

echo "🚀 启动后台更新任务..."
echo "📄 日志文件: $LOG_FILE"
echo ""
echo "⚠️  注意: 此操作可能需要 3-4 小时，请耐心等待"
echo ""

nohup python3 -u -c "
import os
import sys
import time
from pathlib import Path

# 确保能导入项目根目录的模块
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

try:
    from data_warehouse import DataWarehouse
    import tushare as ts
except ImportError as e:
    print(f'❌ 导入失败: {e}')
    print(f'当前路径: {os.getcwd()}')
    sys.exit(1)

# 配置日志输出无缓冲
sys.stdout.reconfigure(encoding='utf-8')

print('=' * 80)
print('🚀 完整增量更新 - 2023-2024 全量数据')
print('=' * 80)

dw = DataWarehouse()
pro = ts.pro_api()  # 使用环境变量中的 Token

# ------------------------------------------------------------------
# 任务 1: 更新大盘指数 (用于计算相对收益 Label)
# ------------------------------------------------------------------
print('\n[任务 1/2] 更新上证指数 (000001.SH)...')
try:
    # 下载指数日线，覆盖完整的训练和测试范围
    df_index = pro.index_daily(ts_code='000001.SH', start_date='20220101', end_date='20250131')
    if not df_index.empty:
        # 保存到 data/daily 目录
        save_path = project_root / 'data' / 'daily' / '000001.SH.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_index.sort_values('trade_date', inplace=True)
        df_index.to_csv(save_path, index=False)
        print(f'  ✅ 指数数据已保存: {save_path} ({len(df_index)} 条)')
    else:
        print('  ⚠️  未获取到指数数据')
except Exception as e:
    print(f'  ❌ 指数更新失败: {e}')

# ------------------------------------------------------------------
# 任务 2: 更新个股数据 (补全 turnover_rate, pe_ttm)
# ------------------------------------------------------------------
print('\n[任务 2/2] 更新个股数据 (2023.01 ~ 2024.12)...')

# 定义时间段
dates = dw.get_trade_days('20230101', '20241231')

print(f'计划更新天数: {len(dates)} 天')
print(f'预计耗时: {len(dates) * 0.5 / 60:.1f} 分钟')

success_count = 0
fail_count = 0
start_time = time.time()

# 创建日志文件目录
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

for i, date in enumerate(dates, 1):
    try:
        # force=True 会强制重新下载并合并 daily_basic 数据
        df = dw.download_daily_data(date, force=True)

        # 检查特征是否补全
        if df is not None and 'turnover_rate' in df.columns and 'pe_ttm' in df.columns:
            # 每 20 天打印一次进度
            if i % 20 == 0 or i == 1:
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (len(dates) - i)
                print(f'  [{i}/{len(dates)}] {date} ✅ 成功 | 已用: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟')
            success_count += 1
        else:
            print(f'  [{i}/{len(dates)}] {date} ⚠️ 数据下载成功但特征仍缺失')

    except Exception as e:
        print(f'  [{i}/{len(dates)}] {date} ❌ 失败: {e}')
        fail_count += 1

    # 避免触发 Tushare 限流
    time.sleep(0.1)

elapsed = time.time() - start_time
print('\n' + '=' * 80)
print(f'🎉 更新完成！')
print(f'总耗时: {elapsed/60:.1f} 分钟 ({elapsed/3600:.1f} 小时)')
print(f'成功: {success_count}/{len(dates)} ({success_count/len(dates)*100:.1f}%)')
print(f'失败: {fail_count}/{len(dates)} ({fail_count/len(dates)*100:.1f}%)')
print('=' * 80)

if success_count > len(dates) * 0.95:
    print('\n🎊 数据更新成功！可以开始训练模型')
    print('   运行: python assets/train_real_data.py')
else:
    print('\n⚠️  部分数据更新失败，请检查日志')
    print('   可能原因:')
    print('   1. Tushare Token 权限不足')
    print('   2. 网络连接问题')
    print('   3. API 限流')
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
