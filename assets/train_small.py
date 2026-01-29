"""
使用真实历史数据（小范围）训练 AI 裁判模型
优化版本，使用缓存和限制候选股票数量
"""
import os
import sys
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_warehouse_cached import DataWarehouse
from ai_backtest_generator import AIBacktestGenerator
from ai_referee import AIReferee

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def generate_small_training_data():
    """
    使用小范围真实历史数据（2024年1-3月）生成训练数据集
    优化版本，限制候选股票数量
    """
    print("\n" + "=" * 80)
    print("【步骤 1】生成小范围训练数据集（优化版）")
    print("=" * 80)

    # 初始化数据仓库
    dw = DataWarehouse()

    # 初始化回测生成器
    generator = AIBacktestGenerator()

    # [优化] 增加成交额阈值和限制候选股票数量
    generator.amount_threshold = 10000  # 1000万元 -> 1亿元
    generator.max_candidates = 50  # 每天最多50只候选股票

    # 设置时间范围：2024年1月（约20个交易日，有效约15天）
    start_date = '20240101'
    end_date = '20240131'

    print(f"\n[配置] 时间范围：{start_date} ~ {end_date}")
    print(f"[优化] 成交额阈值：{generator.amount_threshold} 千元（1000万元）")
    print(f"[优化] 最大候选：{generator.max_candidates} 只/天")

    # 生成训练数据
    print("\n[开始] 生成训练数据...")
    print("[提示] 预计耗时 5-15 分钟")

    try:
        # 使用 ai_backtest_generator 的 generate_dataset 方法
        # 限制最大样本数为 2000
        dataset = generator.generate_dataset(
            start_date=start_date,
            end_date=end_date,
            max_samples=2000
        )

        if dataset is None or len(dataset) == 0:
            print("\n[错误] 生成的训练数据为空")
            return None

        print(f"\n[成功] 生成训练数据")
        print(f"  样本数：{len(dataset)} 条")
        print(f"  正样本：{(dataset['label'] == 1).sum()} ({(dataset['label'] == 1).sum()/len(dataset)*100:.1f}%)")
        print(f"  负样本：{(dataset['label'] == 0).sum()} ({(dataset['label'] == 0).sum()/len(dataset)*100:.1f}%)")
        print(f"  特征数：{len(dataset.columns) - 1}")

        # 保存训练数据
        output_dir = project_root / 'data' / 'training'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file = output_dir / f'small_training_data_{timestamp}.csv'

        dataset.to_csv(data_file, index=False, encoding='utf-8')
        print(f"\n[保存] 训练数据已保存：{data_file}")

        return str(data_file)

    except Exception as e:
        print(f"\n[错误] 生成训练数据失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_with_small_data(data_file):
    """
    使用小数据训练 AI 裁判模型
    """
    print("\n" + "=" * 80)
    print("【步骤 2】使用小数据训练 AI 裁判模型")
    print("=" * 80)

    # 初始化 AI 裁判
    referee = AIReferee()

    # 读取训练数据
    print(f"\n[读取] 训练数据：{data_file}")
    dataset = pd.read_csv(data_file)

    # 分离特征和标签
    X = dataset.drop(['label', 'ts_code', 'trade_date'], axis=1)
    y = dataset['label']

    print(f"[信息] 特征数：{X.shape[1]}")
    print(f"[信息] 样本数：{X.shape[0]}")
    print(f"[信息] 正样本：{y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"[信息] 负样本：{(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")

    # 训练模型（时序交叉验证，3折）
    print("\n[开始] 训练模型（时序交叉验证，3折）...")

    try:
        results = referee.train_time_series(X, y, n_splits=3)

        print("\n[成功] 模型训练完成")

        # 打印交叉验证结果
        print("\n[交叉验证结果]")
        print(results['cv_results'].to_string(index=False))

        # 打印平均指标
        print("\n[平均指标]")
        for metric, value in results['avg_metrics'].items():
            print(f"  {metric}: {value:.4f}")

        # 保存模型
        output_dir = project_root / 'data' / 'training'
        output_dir.mkdir(parents=True, exist_ok=True)

        model_file = output_dir / f'ai_referee_xgboost_small_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        referee.save_model(str(model_file))
        print(f"\n[保存] 模型已保存：{model_file}")

        return True

    except Exception as e:
        print(f"\n[错误] 训练模型失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主流程
    """
    print("=" * 80)
    print("         AI 裁判 V5.0 小范围训练流程（优化版）")
    print("=" * 80)

    # 步骤 1：生成训练数据
    data_file = generate_small_training_data()

    if data_file is None:
        print("\n[错误] 无法生成训练数据，训练终止")
        return

    # 步骤 2：训练模型
    success = train_with_small_data(data_file)

    if not success:
        print("\n[错误] 模型训练失败")
        return

    print("\n" + "=" * 80)
    print("✅ 小范围训练流程完成！")
    print("\n下一步选项：")
    print("  1. 如果结果满意，可以运行 train_real_data.py 生成完整数据集（10-30分钟）")
    print("  2. 或者调整参数重新训练")
    print("=" * 80)


if __name__ == '__main__':
    main()
