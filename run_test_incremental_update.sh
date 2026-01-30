#!/bin/bash
# 测试增量更新功能（仅测试 10 天）

# 1. 确保在项目根目录运行
cd /workspace/projects || exit

# 2. 加载环境变量
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ 环境变量已加载"
else
    echo "⚠️  警告: 未找到 .env 文件"
fi

echo ""
echo "🧪 启动增量更新测试..."
echo "📅 测试范围: 2024.01.01 ~ 2024.01.10 (10 天)"

python3 -u -c "
import os
import sys
import time
from pathlib import Path

# 确保能导入项目根目录的模块
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'assets'))

try:
    from data_warehouse import DataWarehouse
    import tushare as ts
except ImportError as e:
    print(f'❌ 导入失败: {e}')
    print(f'当前路径: {os.getcwd()}')
    sys.exit(1)

print('=' * 80)
print('🧪 增量更新功能测试')
print('=' * 80)

dw = DataWarehouse()
pro = ts.pro_api()

# ------------------------------------------------------------------
# 任务 1: 下载上证指数测试
# ------------------------------------------------------------------
print('\n[测试 1/2] 下载上证指数 (000001.SH)...')
try:
    df_index = pro.index_daily(ts_code='000001.SH', start_date='20240101', end_date='20240110')
    if not df_index.empty:
        save_path = project_root / 'assets' / 'data' / 'daily' / '000001.SH.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_index.sort_values('trade_date', inplace=True)
        df_index.to_csv(save_path, index=False)
        print(f'  ✅ 指数数据已保存: {save_path} ({len(df_index)} 条)')
        print(f'  📊 包含字段: {list(df_index.columns)}')
    else:
        print('  ⚠️  未获取到指数数据')
except Exception as e:
    print(f'  ❌ 指数下载失败: {e}')

# ------------------------------------------------------------------
# 任务 2: 下载个股数据测试
# ------------------------------------------------------------------
print('\n[测试 2/2] 下载个股数据 (2024.01.01 ~ 2024.01.10)...')

test_dates = ['20240101', '20240102', '20240103', '20240104', '20240105',
              '20240108', '20240109', '20240110']

print(f'测试天数: {len(test_dates)} 天')

success_count = 0

for i, date in enumerate(test_dates, 1):
    try:
        df = dw.download_daily_data(date, force=True)

        if df is not None:
            has_turnover = 'turnover_rate' in df.columns
            has_pe = 'pe_ttm' in df.columns
            has_pb = 'pb' in df.columns

            status = '✅' if (has_turnover and has_pe and has_pb) else '⚠️ '
            print(f'  {status} [{i}/{len(test_dates)}] {date} - '
                  f'Turnover: {has_turnover}, PE: {has_pe}, PB: {has_pb}')

            if has_turnover and has_pe:
                success_count += 1
        else:
            print(f'  ❌ [{i}/{len(test_dates)}] {date} - 下载失败')

    except Exception as e:
        print(f'  ❌ [{i}/{len(test_dates)}] {date} - 错误: {e}')

print('\n' + '=' * 80)
print(f'测试完成！成功: {success_count}/{len(test_dates)}')
print('=' * 80)

if success_count == len(test_dates):
    print('\n🎊 测试通过！所有数据都包含完整特征')
else:
    print('\n⚠️  部分测试失败，请检查数据')

" > test_incremental_update.log 2>&1

echo ""
echo "✅ 测试完成"
echo "📄 日志文件: test_incremental_update.log"
echo "👀 查看命令: cat test_incremental_update.log"
