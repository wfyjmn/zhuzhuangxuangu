# -*- coding: utf-8 -*-
"""
下载 2023-2024 年历史数据
用于 AI 裁判训练和回测
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from data_warehouse import DataWarehouse


def download_range_with_progress(warehouse, start_date, end_date, resume=True):
    """
    下载指定日期范围的数据，带进度显示

    Args:
        warehouse: DataWarehouse 实例
        start_date: 开始日期
        end_date: 结束日期
        resume: 是否继续下载（跳过已存在的数据）
    """
    # 获取交易日列表
    trade_days = warehouse.get_trade_days(start_date, end_date)

    print(f"📊 下载配置：")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  交易日总数: {len(trade_days)}")
    print(f"  数据目录: {warehouse.data_dir}\n")

    # 检查已下载的数据
    if resume:
        print("📂 检查已下载的数据...")
        downloaded_dates = []
        missing_dates = []

        for date in trade_days:
            filename = os.path.join(warehouse.data_dir, f"{date}.csv")
            if os.path.exists(filename):
                downloaded_dates.append(date)
            else:
                missing_dates.append(date)

        print(f"  已下载数据: {len(downloaded_dates)} 天")
        print(f"  待下载数据: {len(missing_dates)} 天\n")

        if len(missing_dates) == 0:
            print("✅ 所有数据已下载完成！\n")
            return {
                'success': len(downloaded_dates),
                'fail': 0,
                'skip': 0,
                'failed_dates': []
            }

        # 只下载缺失的数据
        dates_to_download = missing_dates
    else:
        print("⚠️  强制重新下载所有数据\n")
        dates_to_download = trade_days

    # 确认下载
    if len(dates_to_download) > 10:
        response = input(f"⚠️  是否开始下载 {len(dates_to_download)} 天的数据？(y/n): ")
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
    last_progress_time = start_time

    try:
        for i, date in enumerate(dates_to_download, 1):
            filename = os.path.join(warehouse.data_dir, f"{date}.csv")

            # 检查是否已下载
            if resume and os.path.exists(filename):
                print(f"[{i}/{len(dates_to_download)}] ⏭️  {date} 已存在，跳过")
                skip_count += 1
                continue

            print(f"[{i}/{len(dates_to_download)}] 📥 下载 {date}...", end=' ', flush=True)

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
                remaining = (len(dates_to_download) - i) * avg_time

                print(f"\n  [进度] {i}/{len(dates_to_download)} ({i/len(dates_to_download)*100:.1f}%)")
                print(f"  [统计] 成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
                print(f"  [时间] 已用: {elapsed/60:.1f} 分钟, 预计剩余: {remaining/60:.1f} 分钟\n")

            # 避免触发限流（动态调整）
            sleep_time = 0.3 + (0.1 if fail_count > 10 else 0)
            time.sleep(sleep_time)

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
        print(f"  待下载数据: {len(dates_to_download)}")
        print(f"  成功下载数据: {success_count}")
        print(f"  跳过已存在: {skip_count}")
        print(f"  下载数据失败: {fail_count}")
        print(f"  完成率: {(success_count + skip_count)/len(dates_to_download)*100:.1f}%")

        if fail_count > 0:
            print(f"\n⚠️  失败日期列表（最多显示 20 个）：")
            for date in failed_dates[:20]:
                print(f"    - {date}")
            if len(failed_dates) > 20:
                print(f"    ... 共 {len(failed_dates)} 个失败")

        print()

        return {
            'success': success_count,
            'fail': fail_count,
            'skip': skip_count,
            'failed_dates': failed_dates
        }

    except KeyboardInterrupt:
        print("\n\n⚠️  下载被用户中断\n")
        print(f"  当前进度: {success_count + skip_count}/{len(dates_to_download)} 天已完成\n")

        return {
            'success': success_count,
            'fail': fail_count,
            'skip': skip_count,
            'failed_dates': failed_dates
        }

    except Exception as e:
        print(f"\n\n❌ 下载失败: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def check_data_integrity(warehouse, start_date, end_date):
    """
    检查数据完整性

    Args:
        warehouse: DataWarehouse 实例
        start_date: 开始日期
        end_date: 结束日期
    """
    print("="*80)
    print("检查数据完整性...")
    print("="*80)

    trade_days = warehouse.get_trade_days(start_date, end_date)

    missing_dates = []
    total_stocks = 0

    for date in trade_days:
        filename = os.path.join(warehouse.data_dir, f"{date}.csv")

        if not os.path.exists(filename):
            missing_dates.append(date)
        else:
            # 检查文件大小（避免空文件）
            file_size = os.path.getsize(filename)
            if file_size < 100:  # 小于 100 字节视为空文件
                missing_dates.append(f"{date} (空文件)")
            else:
                # 统计股票数量
                try:
                    df = pd.read_csv(filename)
                    total_stocks += len(df)
                except:
                    missing_dates.append(f"{date} (损坏)")

    if len(missing_dates) == 0:
        print("✅ 所有交易日数据完整！\n")
        print(f"📊 数据统计：")
        print(f"  交易日总数: {len(trade_days)}")
        print(f"  平均每日股票数: {total_stocks / len(trade_days):.0f}")
        print(f"  总记录数: {total_stocks:,}\n")
    else:
        print(f"⚠️  缺少或损坏 {len(missing_dates)} 天的数据")
        print(f"  缺失/损坏日期: {missing_dates[:20]}")
        if len(missing_dates) > 20:
            print(f"  ... 共 {len(missing_dates)} 个缺失\n")


def main():
    print("="*80)
    print(" " * 25 + "DeepQuant 数据下载")
    print(" " * 30 + "2023-2024 年")
    print("="*80 + "\n")

    # 检查 Tushare Token
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        print("❌ 错误：未配置 TUSHARE_TOKEN 环境变量\n")
        print("请先配置环境变量：")
        print("  export TUSHARE_TOKEN='your_token_here'\n")
        sys.exit(1)

    print(f"✅ Tushare Token 已配置\n")

    # 初始化数据仓库
    try:
        warehouse = DataWarehouse(data_dir="data/daily")
        print("✅ 数据仓库初始化成功\n")
    except Exception as e:
        print(f"❌ 数据仓库初始化失败: {e}\n")
        sys.exit(1)

    # 下载范围
    ranges = [
        ("20230101", "20231231"),  # 2023 年
        ("20240101", "20241231"),  # 2024 年
    ]

    total_success = 0
    total_fail = 0
    total_skip = 0

    for start_date, end_date in ranges:
        print("\n" + "="*80)
        print(f"下载 {start_date} ~ {end_date} 数据")
        print("="*80 + "\n")

        result = download_range_with_progress(warehouse, start_date, end_date, resume=True)

        total_success += result['success']
        total_fail += result['fail']
        total_skip += result['skip']

        # 检查数据完整性
        check_data_integrity(warehouse, start_date, end_date)

        # 阶段性总结
        print("="*80)
        print(f"✅ {start_date} ~ {end_date} 下载完成")
        print("="*80)
        print(f"  成功: {result['success']}")
        print(f"  跳过: {result['skip']}")
        print(f"  失败: {result['fail']}\n")

    # 总体总结
    print("="*80)
    print("🎉 全部下载完成！")
    print("="*80)
    print(f"  总成功: {total_success}")
    print(f"  总跳过: {total_skip}")
    print(f"  总失败: {total_fail}")
    print(f"  完成率: {(total_success + total_skip)/(total_success + total_skip + total_fail)*100:.1f}%\n")

    print("="*80)
    print("下一步操作：")
    print("="*80)
    print("  1. 生成训练数据：python generate_training_data_2024_simple.py")
    print("  2. 训练 AI 裁判：python train_ai_referee_v4.5.py")
    print("  3. 测试模型：python test_ai_referee_v4.5.py\n")


if __name__ == "__main__":
    main()
