# -*- coding: utf-8 -*-
"""
快速下载上证指数数据
"""

import pandas as pd
import tushare as ts
from pathlib import Path

# 初始化 Tushare
pro = ts.pro_api()

# 指数代码
index_code = '000001.SH'

# 时间范围
start_date = '20220701'  # 比 20230101 早一点，用于回测
end_date = '20250120'    # 比 20241231 晚一点，用于标签计算

print(f"下载上证指数数据 {index_code}")
print(f"时间范围：{start_date} ~ {end_date}")

# 下载指数日线
df = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)

print(f"下载完成，共 {len(df)} 条记录")

# 添加 adj_factor 列（指数不需要复权，设为 1.0）
df['adj_factor'] = 1.0

# 保存到数据目录
data_dir = Path(__file__).parent / 'data' / 'daily'
data_dir.mkdir(parents=True, exist_ok=True)

# 由于是指数数据，按日期保存
unique_dates = df['trade_date'].unique()

saved_count = 0
for date in unique_dates:
    date_df = df[df['trade_date'] == date]

    # 保存为单个 CSV 文件
    filename = data_dir / f"{date}.csv"

    # 如果文件已存在，检查是否包含指数数据
    if filename.exists():
        existing_df = pd.read_csv(filename)
        # 检查是否已经包含该指数
        if index_code in existing_df['ts_code'].values:
            continue
        else:
            # 合并数据
            existing_df = pd.concat([existing_df, date_df], ignore_index=True)
            existing_df.to_csv(filename, index=False)
            saved_count += 1
    else:
        # 直接保存
        date_df.to_csv(filename, index=False)
        saved_count += 1

print(f"保存完成，共保存 {saved_count} 个文件")

# 验证
sample_date = unique_dates[-1]
sample_file = data_dir / f"{sample_date}.csv"
if sample_file.exists():
    sample_df = pd.read_csv(sample_file)
    print(f"\n验证 {sample_date}:")
    print(f"  总记录数: {len(sample_df)}")
    print(f"  是否包含指数: {'✅' if index_code in sample_df['ts_code'].values else '❌'}")
    if index_code in sample_df['ts_code'].values:
        index_data = sample_df[sample_df['ts_code'] == index_code]
        print(f"  指数数据: {index_data[['trade_date', 'open', 'high', 'low', 'close']].to_dict('records')[0]}")
