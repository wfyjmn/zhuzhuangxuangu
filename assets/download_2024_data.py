# -*- coding: utf-8 -*-
"""
下载 2024 年数据（用于快速测试）
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
    print(" " * 25 + "DeepQuant 数据下载")
    print(" " * 35 + "2024 年")
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

    # 下载范围（2024 年）
    start_date = "20240101"
    end_date = "20241231"

    # 获取交易日列表
    trade_days = warehouse.get_trade_days(start_date, end_date)

    print(f"📊 下载配置：")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  交易日总数: {len(trade_days)}")
    print(f"  数据目录: data/daily\n")

    # 检查已下载的数据
    print("📂 检查已下载的数据...")
    downloaded_dates = []
    for date in trade_days:
        filename = os.path.join("data/daily", f"{date}.csv")
        if os.path.exists(filename):
            downloaded_dates.append(date)

    print(f"  已下载数据: {len(downloaded_dates)} 天")
    print(f"  待下载数据: {len(trade_days) - len(downloaded_dates)} 天\n")

    # 确认下载
    response = input("⚠️  是否开始下载 2024 年的数据？(y/n): ")
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
    skip_count = 0
    failed_dates = []

    try:
        for i, date in enumerate(trade_days, 1):
            filename = os.path.join("data/daily", f"{date}.csv")

            # 检查是否已下载
            if os.path.exists(filename):
                print(f"[{i}/{len(trade_days)}] ⏭️  {date} 已存在，跳过")
                skip_count += 1
                continue

            print(f"[{i}/{len(trade_days)}] 📥 下载 {date}...", end=' ')

            # 下载数据
            try:
                df = warehouse.download_daily_data(date)

                if df is not None:
                    success_count += 1
                    print(f"✅ {len(df)} 只股票")
                else:
                    fail_count += 1
                    failed_dates.append(date)
                    print(f"❌ 失败（无数据）")

                # 进度提示
                if i % 20 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    avg_time = elapsed / i
                    remaining = (len(trade_days) - i) * avg_time
                    print(f"\n  [进度] {i}/{len(trade_days)} ({i/len(trade_days)*100:.1f}%)")
                    print(f"  [统计] 成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
                    print(f"  [时间] 已用: {elapsed/60:.1f} 分钟, 预计剩余: {remaining/60:.1f} 分钟\n")

                # 避免触发限流
                time.sleep(0.3)

            except KeyboardInterrupt:
                print(f"\n\n⚠️  下载被用户中断\n")
                break
            except Exception as e:
                fail_count += 1
                failed_dates.append(date)
                print(f"❌ 失败: {e}")

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
        print(f"  交易日总数: {len(trade_days)}")
        print(f"  成功下载数据: {success_count}")
        print(f"  跳过已存在: {skip_count}")
        print(f"  下载数据失败: {fail_count}")
        print(f"  总计: {success_count + skip_count}/{len(trade_days)} ({(success_count + skip_count)/len(trade_days)*100:.1f}%)")

        if fail_count > 0:
            print(f"\n⚠️  失败日期列表：")
            for date in failed_dates[:10]:
                print(f"    - {date}")
            if len(failed_dates) > 10:
                print(f"    ... 共 {len(failed_dates)} 个失败")

        print()

        # 检查数据完整性
        print("="*80)
        print("检查数据完整性...")
        print("="*80)

        missing_dates = []
        for date in trade_days:
            filename = os.path.join("data/daily", f"{date}.csv")
            if not os.path.exists(filename):
                missing_dates.append(date)

        if len(missing_dates) == 0:
            print("✅ 所有交易日数据完整！\n")
        else:
            print(f"⚠️  缺少 {len(missing_dates)} 天的数据")
            print(f"  缺失日期: {missing_dates[:10]}")
            if len(missing_dates) > 10:
                print(f"  ... 共 {len(missing_dates)} 个缺失\n")

        print("="*80)
        print("下一步操作：")
        print("="*80)
        print("  1. 生成训练数据：python generate_training_data_2024.py")
        print("  2. 训练 AI 裁判：python train_ai_referee_2024.py\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  下载被用户中断\n")
        print(f"  当前进度: {success_count + skip_count}/{len(trade_days)} 天已完成\n")

    except Exception as e:
        print(f"\n\n❌ 下载失败: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
