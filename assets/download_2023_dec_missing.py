# -*- coding: utf-8 -*-
"""
快速下载缺失的 2023 年 12 月数据（16 天）
"""

import os
import sys
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from data_warehouse import DataWarehouse

def main():
    print("="*80)
    print("快速下载：2023 年 12 月缺失数据")
    print("="*80 + "\n")

    # 初始化数据仓库
    warehouse = DataWarehouse(data_dir="data/daily")

    # 检查缺失的日期
    start_date = '20231201'
    end_date = '20231231'

    trade_days = warehouse.get_trade_days(start_date, end_date)

    missing_dates = []
    for date in trade_days:
        filename = os.path.join('data/daily', f'{date}.csv')
        if not os.path.exists(filename):
            missing_dates.append(date)

    print(f"待下载日期: {len(missing_dates)} 天")
    print(f"缺失日期: {missing_dates}\n")

    if len(missing_dates) == 0:
        print("✅ 数据已完整！\n")
        return

    success_count = 0
    fail_count = 0

    for i, date in enumerate(missing_dates, 1):
        print(f"[{i}/{len(missing_dates)}] 📥 下载 {date}...", end=' ', flush=True)

        try:
            df = warehouse.download_daily_data(date)
            if df is not None:
                success_count += 1
                print(f"✅ {len(df)} 只股票")
            else:
                fail_count += 1
                print(f"❌ 失败（无数据）")
        except Exception as e:
            fail_count += 1
            print(f"❌ 失败: {e}")

        time.sleep(0.3)

    print(f"\n✅ 下载完成！成功: {success_count}, 失败: {fail_count}\n")

    # 检查完整性
    print("检查 2023 年数据完整性...")
    all_dates = warehouse.get_trade_days('20230101', '20231231')
    all_missing = []

    for date in all_dates:
        filename = os.path.join('data/daily', f'{date}.csv')
        if not os.path.exists(filename):
            all_missing.append(date)

    if len(all_missing) == 0:
        print("✅ 2023 年数据完整！\n")
    else:
        print(f"⚠️  仍有 {len(all_missing)} 天缺失: {all_missing}\n")


if __name__ == "__main__":
    main()
