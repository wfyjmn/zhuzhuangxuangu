# -*- coding: utf-8 -*-
"""
快速测试：生成 2024 年 1 月前 5 天的训练数据（超快速测试）
"""

import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ai_backtest_generator import AIBacktestGenerator


def test_generate_quick():
    """快速测试：只生成 5 天的数据"""
    print("="*80)
    print("超快速测试：生成 2024 年 1 月前 5 天训练数据")
    print("="*80 + "\n")

    generator = AIBacktestGenerator()

    print("开始生成训练数据...")
    print("="*80 + "\n")

    start_time = datetime.now()

    try:
        # 生成 2024 年 1 月前 15 天的数据（需要至少 7 天用于计算未来收益）
        # 限制样本数量为 500，加快速度
        df = generator.generate_dataset('20240102', '20240120', max_samples=500)

        if df.empty:
            print("\n❌ 数据生成失败！数据集为空。\n")
            return

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("✅ 测试完成！")
        print("="*80)
        print(f"  用时: {duration:.1f} 秒")
        print(f"\n📊 数据统计：")
        print(f"  总样本数: {len(df)}")
        print(f"  正样本: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.1f}%)")
        print(f"  负样本: {len(df) - df['label'].sum()} ({(1-df['label'].sum()/len(df))*100:.1f}%)")
        print(f"  特征数: {len(df.columns) - 3}")

        # 计算建议权重
        pos_count = df['label'].sum()
        neg_count = len(df) - df['label'].sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        print(f"\n  建议 scale_pos_weight: {scale_pos_weight:.2f}")

        # 保存测试数据
        output_dir = "data/training"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"quick_test_202401.csv")
        df.to_csv(filename, index=False)
        print(f"\n💾 测试数据已保存：{filename}\n")

        return df

    except Exception as e:
        print(f"\n❌ 数据生成失败: {e}\n")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_generate_quick()

    if result is not None:
        print("✅ 测试成功！\n")
    else:
        print("❌ 测试失败！\n")
