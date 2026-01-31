#!/bin/bash
# 下载测试数据 - 仅下载 2024-01-01 ~ 2024-01-31 的数据

set -e

cd /workspace/projects || exit

echo "下载测试数据 (2024-01-01 ~ 2024-01-31)"

python3 -u -c "
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from data_warehouse import DataWarehouse

dw = DataWarehouse()

# 获取交易日历
dates = dw.get_trade_days('20240101', '20240131')

print(f'计划下载 {len(dates)} 个交易日的数据')

for date in dates:
    try:
        print(f'下载 {date}...')
        df = dw.download_daily_data(date, force=True)
        if df is not None:
            print(f'  ✅ 成功 ({len(df)} 只股票)')
        else:
            print(f'  ❌ 失败')
    except Exception as e:
        print(f'  ❌ 错误: {e}')

print('\n下载完成')
"
