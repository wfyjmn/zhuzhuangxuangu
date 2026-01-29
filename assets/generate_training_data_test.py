# -*- coding: utf-8 -*-
"""
快速测试：生成 2024 年 1 月的训练数据（测试流程）
"""

import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ai_backtest_generator import AIBacktestGenerator


def test_generate():
    """测试生成 2024 年 1 月的训练数据"""
    print("="*80)
    print("快速测试：生成 2024 年 1 月训练数据")
    print("="*80 + "\n")

    generator = AIBacktestGenerator()

    print("📊 配置参数：")
    print(f"  持有天数: {generator.hold_days}")
    print(f"  目标收益: {generator.target_return}%")
    print(f"  止损: {generator.stop_loss}%")
    print(f"  熊市阈值: {generator.bear_threshold}%")
    print(f"  超额收益目标: {generator.alpha_threshold}%")
    print()

    print("开始生成训练数据（2024年1月）...")
    print("="*80 + "\n")

    start_time = datetime.now()

    try:
        # 生成 2024 年 1 月数据（22 个交易日）
        df = generator.generate_dataset('20240101', '20240131')

        if df.empty:
            print("\n❌ 数据生成失败！数据集为空。\n")
            return

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("✅ 测试完成！")
        print("="*80)
        print(f"  用时: {duration/60:.1f} 分钟")
        print(f"\n📊 数据统计：")
        print(f"  总样本数: {len(df):,}")
        print(f"  正样本: {df['label'].sum():,} ({df['label'].sum()/len(df)*100:.1f}%)")
        print(f"  负样本: {len(df) - df['label'].sum():,} ({(1-df['label'].sum()/len(df))*100:.1f}%)")
        print(f"  特征数: {len(df.columns) - 3}")

        # 计算建议权重
        pos_count = df['label'].sum()
        neg_count = len(df) - df['label'].sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        print(f"\n  建议 scale_pos_weight: {scale_pos_weight:.2f}")

        # 保存测试数据
        output_dir = "data/training"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"test_training_202401.csv")
        df.to_csv(filename, index=False)
        print(f"\n💾 测试数据已保存：{filename}\n")

        # 估算完整数据生成时间
        print("="*80)
        print("完整数据估算")
        print("="*80)
        print(f"  测试数据: 2024年1月 (22 个交易日, {len(df):,} 样本)")
        print(f"  完整数据: 2023-2024年 (484 个交易日)")
        print(f"  预计样本数: {int(len(df) * 484 / 22):,}")
        print(f"  预计用时: {duration * 484 / 22 / 60:.1f} 分钟")
        print()

        return df

    except Exception as e:
        print(f"\n❌ 数据生成失败: {e}\n")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_generate()

    if result is not None:
        print("✅ 测试成功！可以使用 generate_training_data_full.py 生成完整数据。\n")
    else:
        print("❌ 测试失败！请检查错误信息。\n")
