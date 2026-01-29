# -*- coding: utf-8 -*-
"""
下载缺失的 2023 年数据（1-11 月）
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from data_warehouse import DataWarehouse


def main():
    print("="*80)
    print(" " * 30 + "DeepQuant 数据下载")
    print(" " * 28 + "2023 年 1-11 月")
    print("="*80 + "\n")

    # 检查 Tushare Token
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        print("❌ 错误：未配置 TUSHARE_TOKEN 环境变量\n")
        sys.exit(1)

    print(f"✅ Tushare Token 已配置\n")

    # 初始化数据仓库
    try:
        warehouse = DataWarehouse(data_dir="data/daily")
        print("✅ 数据仓库初始化成功\n")
    except Exception as e:
        print(f"❌ 数据仓库初始化失败: {e}\n")
        sys.exit(1)

    # 下载范围（2023 年 1-11 月）
    start_date = "20230101"
    end_date = "20231130"

    # 获取交易日列表
    trade_days = warehouse.get_trade_days(start_date, end_date)

    print(f"📊 下载配置：")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  交易日总数: {len(trade_days)}")
    print(f"  数据目录: {warehouse.data_dir}\n")

    # 检查已下载的数据
    print("📂 检查已下载的数据...")
    missing_dates = []
    downloaded_count = 0

    for date in trade_days:
        filename = os.path.join(warehouse.data_dir, f"{date}.csv")
        if os.path.exists(filename):
            downloaded_count += 1
        else:
            missing_dates.append(date)

    print(f"  已下载数据: {downloaded_count} 天")
    print(f"  待下载数据: {len(missing_dates)} 天\n")

    if len(missing_dates) == 0:
        print("✅ 所有数据已下载完成！\n")
        return

    # 确认下载
    response = input(f"⚠️  是否开始下载 {len(missing_dates)} 天的数据？(y/n): ")
    if response.lower() != 'y':
        print("\n❌ 下载已取消\n")
        sys.exit(0)

    # 开始下载
    print("\n" + "="*80)
    print("开始下载数据...")
    print("="*80 + "\n")

    start_time = datetime.now()
    success_count = 0
    fail_count = 0
    failed_dates = []
    last_progress_time = start_time

    try:
        for i, date in enumerate(missing_dates, 1):
            filename = os.path.join(warehouse.data_dir, f"{date}.csv")

            print(f"[{i}/{len(missing_dates)}] 📥 下载 {date}...", end=' ', flush=True)

            # 下载数据（带重试）
            max_retries = 3
            for retry in range(max_retries):
                try:
                    df = warehouse.download_daily_data(date)

                    if df is not None:
                        success_count += 1
                        print(f"✅ {len(df)} 只股票")
                        break
                    else:
                        if retry < max_retries - 1:
                            print(f"⚠️  重试 {retry + 1}/{max_retries}...", end=' ', flush=True)
                            time.sleep(1)
                        else:
                            fail_count += 1
                            failed_dates.append(date)
                            print(f"❌ 失败（无数据）")

                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"⚠️  重试 {retry + 1}/{max_retries}: {e}...", end=' ', flush=True)
                        time.sleep(1)
                    else:
                        fail_count += 1
                        failed_dates.append(date)
                        print(f"❌ 失败: {e}")

            # 进度提示（每 20 天或每分钟）
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()

            if i % 20 == 0 or (current_time - last_progress_time).total_seconds() > 60:
                last_progress_time = current_time
                avg_time = elapsed / i
                remaining = (len(missing_dates) - i) * avg_time

                print(f"\n  [进度] {i}/{len(missing_dates)} ({i/len(missing_dates)*100:.1f}%)")
                print(f"  [统计] 成功: {success_count}, 失败: {fail_count}")
                print(f"  [时间] 已用: {elapsed/60:.1f} 分钟, 预计剩余: {remaining/60:.1f} 分钟\n")

            # 避免触发限流
            time.sleep(0.3)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 输出统计信息
        print("\n" + "="*80)
        print("✅ 数据下载完成！")
        print("="*80)
        print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  用时: {duration/60:.1f} 分钟")
        print(f"\n📊 下载统计：")
        print(f"  待下载数据: {len(missing_dates)}")
        print(f"  成功下载数据: {success_count}")
        print(f"  下载数据失败: {fail_count}")
        print(f"  完成率: {success_count/len(missing_dates)*100:.1f}%")

        if fail_count > 0:
            print(f"\n⚠️  失败日期列表（最多显示 20 个）：")
            for date in failed_dates[:20]:
                print(f"    - {date}")
            if len(failed_dates) > 20:
                print(f"    ... 共 {len(failed_dates)} 个失败")

        print()

        # 检查数据完整性
        print("="*80)
        print("检查数据完整性...")
        print("="*80)

        recheck_missing = []
        for date in trade_days:
            filename = os.path.join(warehouse.data_dir, f"{date}.csv")
            if not os.path.exists(filename):
                recheck_missing.append(date)

        if len(recheck_missing) == 0:
            print("✅ 2023 年 1-11 月数据完整！\n")
        else:
            print(f"⚠️  仍有 {len(recheck_missing)} 天的数据缺失")
            print(f"  缺失日期: {recheck_missing[:20]}")
            if len(recheck_missing) > 20:
                print(f"  ... 共 {len(recheck_missing)} 个缺失\n")

        print("="*80)
        print("下一步操作：")
        print("="*80)
        print("  1. 生成训练数据：python generate_training_data_2024_simple.py")
        print("  2. 训练 AI 裁判：python train_ai_referee_v4.5.py\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  下载被用户中断\n")
        print(f"  当前进度: {success_count}/{len(missing_dates)} 天已完成\n")

    except Exception as e:
        print(f"\n\n❌ 下载失败: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
