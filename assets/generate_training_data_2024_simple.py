# -*- coding: utf-8 -*-
"""
生成 AI 裁判训练数据（2024 年）- 简化版
使用 AIBacktestGenerator.generate_training_data() 方法
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ai_backtest_generator import AIBacktestGenerator

def main():
    print("="*80)
    print(" " * 20 + "AI 裁判训练数据生成")
    print(" " * 30 + "2024 年（简化版）")
    print("="*80 + "\n")

    # 初始化回测生成器
    print("🎯 初始化回测生成器...")
    generator = AIBacktestGenerator(data_dir="data/daily")
    print("✅ 回测生成器初始化成功\n")

    # 生成训练数据
    print("="*80)
    print("开始生成训练数据...")
    print("="*80 + "\n")

    start_time = datetime.now()

    try:
        # 生成训练数据
        X, Y = generator.generate_training_data(
            start_date='20240101',
            end_date='20241231',
            max_samples=None  # 不限制样本数量
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "="*80)
        print("✅ 训练数据生成完成！")
        print("="*80)
        print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  用时: {duration/60:.1f} 分钟")

        print(f"\n📊 数据统计：")
        print(f"  特征数量: {X.shape[1]}")
        print(f"  样本数量: {X.shape[0]}")

        pos_count = Y.sum()
        neg_count = len(Y) - pos_count
        pos_ratio = pos_count / len(Y)

        print(f"\n📈 标签分布：")
        print(f"  正样本（盈利）: {pos_count} ({pos_ratio*100:.1f}%)")
        print(f"  负样本（亏损）: {neg_count} ({(1-pos_ratio)*100:.1f}%)")
        print(f"  正负比例: {pos_count}:{neg_count} (1:{neg_count/pos_count:.1f})")

        # 保存训练数据
        os.makedirs("data/training", exist_ok=True)

        X_file = "data/training/X_2024.csv"
        Y_file = "data/training/Y_2024.csv"

        X.to_csv(X_file, index=False)
        Y.to_csv(Y_file, index=False)

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
