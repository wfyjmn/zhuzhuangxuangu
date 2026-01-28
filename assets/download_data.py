# -*- coding: utf-8 -*-
"""
下载历史数据（2023-01-01 至 2025-12-31）
用于 AI 裁判系统训练
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from data_warehouse import DataWarehouse

def main():
    print("="*80)
    print(" " * 30 + "数据下载脚本")
    print("="*80 + "\n")

    # 检查 Tushare Token
    tushare_token = os.getenv("TUSHARE_TOKEN")
    if not tushare_token:
        print("❌ 错误：未配置 TUSHARE_TOKEN 环境变量")
        print("\n请先配置 Token：")
        print("  方法1：在 .env 文件中添加：TUSHARE_TOKEN=your_token_here")
        print("  方法2：在命令行中设置：export TUSHARE_TOKEN=your_token_here\n")
        sys.exit(1)

    print(f"✅ Tushare Token 已配置（长度: {len(tushare_token)}）\n")

    # 初始化数据仓库
    try:
        warehouse = DataWarehouse(data_dir="data/daily")
        print("✅ 数据仓库初始化成功\n")
    except Exception as e:
        print(f"❌ 数据仓库初始化失败: {e}\n")
        sys.exit(1)

    # 下载范围
    start_date = "20230101"
    end_date = "20251231"

    print(f"📊 下载配置：")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  数据目录: data/daily\n")

    # 确认下载
    response = input("⚠️  这将下载约 3 年的历史数据，可能需要较长时间。是否继续？(y/n): ")
    if response.lower() != 'y':
        print("\n❌ 下载已取消\n")
        sys.exit(0)

    # 开始下载
    print("\n" + "="*80)
    print("开始下载数据...")
    print("="*80 + "\n")

    start_time = datetime.now()

    try:
        warehouse.download_range_data(start_date, end_date)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("✅ 数据下载完成！")
        print("="*80)
        print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  用时: {duration:.1f} 秒")
        print(f"  数据目录: data/daily\n")

        # 统计已下载数据
        trade_days = warehouse.get_trade_days(start_date, end_date)
        downloaded_count = len([d for d in trade_days if os.path.exists(os.path.join("data/daily", f"{d}.csv"))])

        print(f"📊 下载统计：")
        print(f"  交易日总数: {len(trade_days)}")
        print(f"  已下载数据: {downloaded_count}")
        print(f"  成功率: {downloaded_count/len(trade_days)*100:.1f}%\n")

        # 下一步提示
        print("="*80)
        print("下一步操作：")
        print("="*80)
        print("  1. 生成训练数据：python generate_training_data.py")
        print("  2. 训练 AI 裁判：python train_ai_referee.py")
        print("  3. 测试模型：python test_ai_referee.py\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  下载被用户中断\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 下载失败: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
