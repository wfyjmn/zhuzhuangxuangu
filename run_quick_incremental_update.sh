#!/bin/bash
# 快速增量更新脚本 - 补全 2023-2024 关键特征（换手率/PE）+ 大盘指数
#
# 用途: 快速补全缺失的特征和指数数据
# 范围: 2023.07.01 ~ 2024.06.30（约 235 个交易日）
# 预计耗时: 约 5-10 分钟
# 使用场景: 数据缺失较少，快速修复

set -e  # 遇到错误立即退出

# 1. 确保在项目根目录运行
cd "$(dirname "$0")" || exit

echo "========================================================================"
echo "🚀 快速增量更新 - 补全特征与指数"
echo "========================================================================"
echo "📅 目标: 2023.07.01 ~ 2024.06.30 (个股 + 指数)"
echo "⏱️  预计耗时: 5-10 分钟"
echo ""

# 2. 加载环境变量
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ 环境变量已加载"
else
    echo "⚠️  警告: 未找到 .env 文件"
fi

# 3. 后台运行 Python 脚本
LOG_FILE="quick_incremental_update.log"
PID_FILE="quick_incremental_update.pid"

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
print('🚀 快速增量更新 - 补全特征与指数')
print('=' * 80)

dw = DataWarehouse()
pro = ts.pro_api()  # 使用环境变量中的 Token

# ------------------------------------------------------------------
# 任务 1: 更新大盘指数 (用于计算相对收益 Label)
# ------------------------------------------------------------------
print('\n[任务 1/2] 更新上证指数 (000001.SH)...')
try:
    # 下载指数日线
    df_index = pro.index_daily(ts_code='000001.SH', start_date='20230101', end_date='20241231')
    if not df_index.empty:
        # 保存到 data/daily 目录，文件名格式与其他股票一致
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
print('\n[任务 2/2] 更新个股数据 (2023.07 ~ 2024.06)...')

# 定义时间段
dates1 = dw.get_trade_days('20230701', '20231231')
dates2 = dw.get_trade_days('20240101', '20240630')
dates = dates1 + dates2

print(f'计划更新天数: {len(dates)} 天')

success_count = 0
start_time = time.time()

for i, date in enumerate(dates, 1):
    try:
        # force=True 会强制重新下载并合并 daily_basic 数据
        df = dw.download_daily_data(date, force=True)

        # 检查特征是否补全
        if df is not None and 'turnover_rate' in df.columns and 'pe_ttm' in df.columns:
            # 简单进度条，不刷屏
            if i % 10 == 0:
                print(f'  [{i}/{len(dates)}] {date} ✅ 成功 (含估值数据)')
            success_count += 1
        else:
            if i % 10 == 0:
                print(f'  [{i}/{len(dates)}] {date} ⚠️ 数据下载成功但特征仍缺失')

    except Exception as e:
        print(f'  [{i}/{len(dates)}] {date} ❌ 失败: {e}')

    # 避免触发 Tushare 限流 (视积分情况调整)
    time.sleep(0.1)

elapsed = time.time() - start_time
print('\n' + '=' * 80)
print(f'🎉 更新完成！')
print(f'耗时: {elapsed/60:.1f} 分钟')
print(f'成功: {success_count}/{len(dates)} ({success_count/len(dates)*100:.1f}%)')
print('=' * 80)

if success_count > len(dates) * 0.9:
    print('\n🎊 数据更新成功！可以重新训练模型')
else:
    print('\n⚠️  部分数据更新失败，请检查日志')
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
