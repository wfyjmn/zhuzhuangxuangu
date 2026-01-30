# -*- coding: utf-8 -*-
"""
增量更新数据脚本 - 补充缺失的特征和指数数据

功能：
1. 重新下载 2023-2024 年的数据（包含 turnover_rate, pe_ttm 等新特征）
2. 下载上证指数数据（000001.SH）
3. 更新 Turbo 仓库
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_warehouse import DataWarehouse

def update_data_with_new_features():
    """重新下载数据，补充缺失的特征（turnover_rate, pe_ttm 等）"""

    print("=" * 80)
    print("增量更新数据 - 补充缺失的特征")
    print("=" * 80)

    # 初始化数据仓库
    dw = DataWarehouse()

    # 时间范围
    start_date = '20230101'
    end_date = '20241231'

    print(f"\n[配置]")
    print(f"  时间范围：{start_date} ~ {end_date}")
    print(f"  数据目录：{dw.data_dir}")

    # 获取交易日历
    trade_days = dw.get_trade_days(start_date, end_date)

    if not trade_days:
        print("[错误] 无法获取交易日历")
        return

    print(f"\n[信息]")
    print(f"  交易日数量：{len(trade_days)}")

    # 询问是否继续
    print(f"\n[警告] 此操作将重新下载 {len(trade_days)} 个交易日的数据")
    print(f"  这可能需要较长时间（预计 {len(trade_days) * 0.5 / 60:.1f} 分钟）")
    print(f"  旧数据文件将被覆盖")

    confirm = input("\n是否继续？(yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("[取消] 操作已取消")
        return

    # 重新下载数据
    print("\n[开始] 下载数据...")
    success_count = 0
    fail_count = 0

    for i, date in enumerate(trade_days, 1):
        print(f"\n[{i}/{len(trade_days)}] 处理 {date}")

        try:
            # 强制重新下载（force=True）
            df = dw.download_daily_data(date, force=True)

            if df is not None:
                # 检查是否包含关键特征
                has_turnover = 'turnover_rate' in df.columns
                has_pe = 'pe_ttm' in df.columns
                has_adj = 'adj_factor' in df.columns

                if has_turnover and has_pe and has_adj:
                    print(f"  ✅ 成功（含 turnover_rate, pe_ttm, adj_factor）")
                    success_count += 1
                else:
                    print(f"  ⚠️  部分特征缺失")
                    print(f"     turnover_rate: {'✅' if has_turnover else '❌'}")
                    print(f"     pe_ttm: {'✅' if has_pe else '❌'}")
                    print(f"     adj_factor: {'✅' if has_adj else '❌'}")
                    success_count += 1
            else:
                print(f"  ❌ 失败")
                fail_count += 1

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            fail_count += 1

    # 打印总结
    print("\n" + "=" * 80)
    print("下载完成")
    print("=" * 80)
    print(f"  成功: {success_count} 天")
    print(f"  失败: {fail_count} 天")
    print(f"  总计: {len(trade_days)} 天")


def download_market_index():
    """下载上证指数数据"""

    print("\n" + "=" * 80)
    print("下载上证指数数据")
    print("=" * 80)

    import pandas as pd
    import tushare as ts

    # 初始化 Tushare
    pro = ts.pro_api()

    # 指数代码
    index_code = '000001.SH'

    # 时间范围
    start_date = '20220701'  # 比 20230101 早一点，用于回测
    end_date = '20250120'    # 比 20241231 晚一点，用于标签计算

    print(f"\n[配置]")
    print(f"  指数代码：{index_code}")
    print(f"  时间范围：{start_date} ~ {end_date}")

    try:
        print("\n[开始] 下载指数数据...")

        # 下载指数日线
        df = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)

        if df.empty:
            print("[错误] 下载失败，数据为空")
            return

        print(f"[成功] 下载 {len(df)} 条记录")

        # 标准化列名（与股票数据一致）
        df.rename(columns={
            'ts_code': 'ts_code',
            'trade_date': 'trade_date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'vol',
            'amount': 'amount'
        }, inplace=True)

        # 添加 adj_factor 列（指数不需要复权，设为 1.0）
        df['adj_factor'] = 1.0

        # 保存到数据目录
        data_dir = Path(__file__).parent.parent / 'data' / 'daily'
        data_dir.mkdir(parents=True, exist_ok=True)

        # 由于是指数数据，按日期保存
        unique_dates = df['trade_date'].unique()

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
            else:
                # 直接保存
                date_df.to_csv(filename, index=False)

        print(f"\n[保存] 指数数据已保存到 {data_dir}")
        print(f"  文件数：{len(unique_dates)} 个")

        # 验证
        print(f"\n[验证]")
        sample_date = unique_dates[-1]  # 最近一个交易日
        sample_file = data_dir / f"{sample_date}.csv"
        if sample_file.exists():
            sample_df = pd.read_csv(sample_file)
            if index_code in sample_df['ts_code'].values:
                print(f"  ✅ {sample_date} 包含指数数据")
            else:
                print(f"  ❌ {sample_date} 不包含指数数据")

    except Exception as e:
        print(f"[错误] 下载失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主流程"""
    print("=" * 80)
    print("         增量更新数据脚本")
    print("=" * 80)

    # 步骤 1: 下载上证指数数据
    download_market_index()

    # 步骤 2: 重新下载数据（包含新特征）
    update_data_with_new_features()

    print("\n" + "=" * 80)
    print("✅ 增量更新完成！")
    print("=" * 80)
    print("\n下一步：")
    print("  1. 验证数据是否包含 turnover_rate, pe_ttm 等特征")
    print("  2. 运行训练脚本：python train_optimized.py")
    print("  3. 检查特征重要性，确保 turnover_rate 和 pe_ttm 排名靠前")


if __name__ == '__main__':
    main()
