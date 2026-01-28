# -*- coding: utf-8 -*-
"""
生成 AI 裁判训练数据（2024 年）
使用事件驱动回测生成训练数据
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from data_warehouse import DataWarehouse
from feature_extractor import FeatureExtractor
from ai_backtest_generator import AIBacktestGenerator

def main():
    print("="*80)
    print(" " * 20 + "AI 裁判训练数据生成")
    print(" " * 30 + "2024 年")
    print("="*80 + "\n")

    # 初始化数据仓库
    print("📂 初始化数据仓库...")
    warehouse = DataWarehouse(data_dir="data/daily")
    print("✅ 数据仓库初始化成功\n")

    # 初始化特征提取器
    print("🔧 初始化特征提取器...")
    extractor = FeatureExtractor()
    print("✅ 特征提取器初始化成功\n")

    # 初始化回测生成器
    print("🎯 初始化回测生成器...")
    generator = AIBacktestGenerator()
    print("✅ 回测生成器初始化成功\n")

    # 训练数据配置
    config = {
        'start_date': '20240101',
        'end_date': '20241231',
        'holding_days': 5,  # 持有5天
        'profit_threshold': 0.03,  # 盈利阈值：3%
        'loss_threshold': -0.03,  # 止损阈值：-3%
    }

    print(f"📊 训练数据配置：")
    print(f"  起始日期: {config['start_date']}")
    print(f"  结束日期: {config['end_date']}")
    print(f"  持有天数: {config['holding_days']} 天")
    print(f"  盈利阈值: {config['profit_threshold']*100:.1f}%")
    print(f"  止损阈值: {config['loss_threshold']*100:.1f}%\n")

    # 确认生成
    response = input("⚠️  是否开始生成训练数据？(y/n): ")
    if response.lower() != 'y':
        print("\n❌ 生成已取消\n")
        sys.exit(0)

    # 开始生成训练数据
    print("\n" + "="*80)
    print("开始生成训练数据...")
    print("="*80 + "\n")

    start_time = datetime.now()

    try:
        # 获取交易日列表
        trade_days = warehouse.get_trade_days(config['start_date'], config['end_date'])
        print(f"📅 交易日数量: {len(trade_days)}\n")

        # 确保有足够的数据（需要持有天数的历史数据）
        # 实际可用的起始日期需要向后推 持有天数 + 20天（用于计算特征）
        available_start_idx = config['holding_days'] + 20
        if available_start_idx >= len(trade_days):
            print("❌ 错误：数据不足，无法生成训练数据\n")
            sys.exit(1)

        print(f"📊 可用于训练的交易日: {len(trade_days) - available_start_idx} 天")
        print(f"   （前 {available_start_idx} 天用于计算特征和历史）\n")

        # 生成训练数据
        X_list = []
        Y_list = []

        for i in range(available_start_idx, len(trade_days)):
            current_date = trade_days[i]

            # 每 50 天显示一次进度
            if (i - available_start_idx) % 50 == 0:
                progress = (i - available_start_idx) / (len(trade_days) - available_start_idx) * 100
                print(f"[进度] 处理日期 {current_date} ({i}/{len(trade_days)}, {progress:.1f}%)")

            # 加载当前日期的数据
            df_current = warehouse.load_daily_data(current_date)
            if df_current is None or df_current.empty:
                continue

            # 提取特征（批量提取，传入 None 作为 index_data 和 sector_data）
            # 注意：extract_features 需要历史数据来计算指标，不能直接用当天的数据
            # 需要先获取每只股票的历史数据

            # 获取历史数据（用于计算指标）
            history_start_idx = i - 30
            if history_start_idx < 0:
                continue

            history_dates = trade_days[history_start_idx:i+1]
            df_history_list = []
            for hist_date in history_dates:
                df_hist = warehouse.load_daily_data(hist_date)
                if df_hist is not None and not df_hist.empty:
                    df_history_list.append(df_hist)

            if not df_history_list:
                continue

            # 合并历史数据
            df_all_history = pd.concat(df_history_list, ignore_index=True)

            # 为每只股票提取特征
            features_list = []
            for ts_code in df_current['ts_code'].unique():
                df_stock = df_all_history[df_all_history['ts_code'] == ts_code].sort_values('trade_date')

                if len(df_stock) < 30:
                    continue

                # 提取特征
                features_dict = extractor.extract_features(
                    df_stock,
                    index_data=None,  # 暂不使用大盘数据
                    sector_data=None,  # 暂不使用板块数据
                    tech_score=None,
                    moneyflow_score=None,
                    new_score=None
                )

                # 添加股票代码和日期
                features_dict['ts_code'] = ts_code
                features_dict['trade_date'] = current_date

                features_list.append(features_dict)

            if not features_list:
                continue

            features_df = pd.DataFrame(features_list)
            if features_df is None or features_df.empty:
                continue

            # 获取未来数据（用于生成标签）
            future_date_idx = i + config['holding_days']
            if future_date_idx >= len(trade_days):
                continue  # 没有足够的数据

            future_date = trade_days[future_date_idx]
            df_future = warehouse.load_daily_data(future_date)
            if df_future is None or df_future.empty:
                continue

            # 生成标签（5天后是否盈利）
            features_df['trade_date'] = current_date

            # 计算每只股票的盈亏
            df_merged = features_df.merge(
                df_future[['ts_code', 'pct_chg']],
                on='ts_code',
                how='left'
            )

            # 标签：如果 5 天后涨跌幅 > 3%，则为 1（盈利），否则为 0
            df_merged['label'] = (df_merged['pct_chg'] > config['profit_threshold'] * 100).astype(int)

            # 移除不必要的列
            X = df_merged.drop(columns=['ts_code', 'trade_date', 'pct_chg', 'label'])
            Y = df_merged['label']

            X_list.append(X)
            Y_list.append(Y)

        # 合并所有数据
        print("\n正在合并数据...")
        X_all = pd.concat(X_list, ignore_index=True)
        Y_all = pd.concat(Y_list, ignore_index=True)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 输出统计信息
        print("\n" + "="*80)
        print("✅ 训练数据生成完成！")
        print("="*80)
        print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  用时: {duration/60:.1f} 分钟")

        print(f"\n📊 数据统计：")
        print(f"  特征数量: {X_all.shape[1]}")
        print(f"  样本数量: {X_all.shape[0]}")
        print(f"  特征列: {list(X_all.columns)}")

        pos_count = Y_all.sum()
        neg_count = len(Y_all) - pos_count
        pos_ratio = pos_count / len(Y_all)

        print(f"\n📈 标签分布：")
        print(f"  正样本（盈利）: {pos_count} ({pos_ratio*100:.1f}%)")
        print(f"  负样本（亏损）: {neg_count} ({(1-pos_ratio)*100:.1f}%)")
        print(f"  正负比例: {pos_count}:{neg_count} (1:{neg_count/pos_count:.1f})")

        # 保存训练数据
        os.makedirs("data/training", exist_ok=True)

        X_file = "data/training/X_2024.csv"
        Y_file = "data/training/Y_2024.csv"

        X_all.to_csv(X_file, index=False)
        Y_all.to_csv(Y_file, index=False)

        print(f"\n💾 数据已保存：")
        print(f"  特征文件: {X_file}")
        print(f"  标签文件: {Y_file}")

        print("\n" + "="*80)
        print("下一步操作：")
        print("="*80)
        print("  1. 训练 AI 裁判：python train_ai_referee_2024.py")
        print("  2. 测试模型：python test_ai_referee.py\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  生成被用户中断\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 生成失败: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
