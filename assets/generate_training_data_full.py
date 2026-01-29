# -*- coding: utf-8 -*-
"""
生成 2023-2024 年完整训练数据
"""

import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ai_backtest_generator import AIBacktestGenerator


def generate_full_training_data():
    """生成 2023-2024 年完整训练数据"""
    print("="*80)
    print(" " * 25 + "DeepQuant 训练数据生成")
    print(" " * 28 + "2023-2024 年完整版")
    print("="*80 + "\n")

    # 初始化回测生成器（V5.0）
    generator = AIBacktestGenerator()

    print("📊 配置参数：")
    print(f"  持有天数: {generator.hold_days}")
    print(f"  目标收益: {generator.target_return}%")
    print(f"  止损: {generator.stop_loss}%")
    print(f"  熊市阈值: {generator.bear_threshold}%")
    print(f"  超额收益目标: {generator.alpha_threshold}%")
    print()

    # 生成完整数据集
    print("开始生成训练数据...")
    print("="*80 + "\n")

    start_time = datetime.now()

    try:
        # 生成 2023-2024 年完整数据集
        df = generator.generate_dataset('20230101', '20241231')

        if df.empty:
            print("\n❌ 数据生成失败！数据集为空。\n")
            return None

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("✅ 训练数据生成完成！")
        print("="*80)
        print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  用时: {duration/60:.1f} 分钟")
        print(f"\n📊 数据统计：")
        print(f"  总样本数: {len(df):,}")
        print(f"  正样本: {df['label'].sum():,} ({df['label'].sum()/len(df)*100:.1f}%)")
        print(f"  负样本: {len(df) - df['label'].sum():,} ({(1-df['label'].sum()/len(df))*100:.1f}%)")
        print(f"  特征数: {len(df.columns) - 3}")  # 减去 label, ts_code, trade_date

        # 计算建议权重
        pos_count = df['label'].sum()
        neg_count = len(df) - df['label'].sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        print(f"\n  建议 scale_pos_weight: {scale_pos_weight:.2f}")

        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "data/training"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"training_data_2023_2024_{timestamp}.csv")
        df.to_csv(filename, index=False)

        print(f"\n💾 数据已保存：{filename}")

        # 保存统计信息
        stats_filename = os.path.join(output_dir, f"training_stats_{timestamp}.txt")
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write(f"训练数据统计信息\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据范围: 2023-01-01 ~ 2024-12-31\n")
            f.write(f"\n样本统计：\n")
            f.write(f"  总样本数: {len(df):,}\n")
            f.write(f"  正样本: {pos_count:,} ({pos_count/len(df)*100:.2f}%)\n")
            f.write(f"  负样本: {neg_count:,} ({neg_count/len(df)*100:.2f}%)\n")
            f.write(f"  建议 scale_pos_weight: {scale_pos_weight:.2f}\n")
            f.write(f"\n特征列表：\n")
            for col in df.columns:
                if col not in ['label', 'ts_code', 'trade_date']:
                    f.write(f"  - {col}\n")

        print(f"💾 统计信息已保存：{stats_filename}\n")

        print("="*80)
        print("下一步操作：")
        print("="*80)
        print(f"  1. 训练 AI 裁判：python train_ai_referee_full.py --data {filename}")
        print(f"  2. 测试模型：python test_ai_referee_v4.5.py\n")

        return df, filename

    except Exception as e:
        print(f"\n❌ 数据生成失败: {e}\n")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = generate_full_training_data()

    if result is not None:
        df, filename = result
        print(f"\n✅ 流程完成！训练数据已保存到：{filename}\n")
    else:
        print("\n❌ 流程失败！\n")
