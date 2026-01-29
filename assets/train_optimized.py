"""
使用真实历史数据训练 AI 裁判模型（优化版）
修复问题：
1. 保留 trade_date 列用于时序交叉验证
2. 优化内存使用（使用 float32）
3. 将日期范围提取为配置参数
4. 确保样本不平衡处理正确
"""
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
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

# ========================================
# 配置参数（可修改）
# ========================================
TRAINING_CONFIG = {
    # 时间范围
    'start_date': '20240101',   # 开始日期
    'end_date': '20240131',     # 结束日期

    # 数据生成参数
    'amount_threshold': 10000,  # 成交额阈值（千元）
    'max_candidates': 50,       # 每日最大候选股票数
    'max_samples': 2000,        # 最大样本数

    # 训练参数
    'n_splits': 3,              # 交叉验证折数
    'model_type': 'xgboost',    # 模型类型

    # 内存优化
    'use_float32': True,        # 使用 float32 节省内存
}


def generate_training_data(config: dict):
    """
    生成训练数据集

    Args:
        config: 配置参数

    Returns:
        训练数据文件路径
    """
    print("\n" + "=" * 80)
    print("【步骤 1】生成训练数据集")
    print("=" * 80)

    # 初始化数据仓库
    dw = DataWarehouse()

    # 初始化回测生成器
    generator = AIBacktestGenerator()

    # 应用配置
    generator.amount_threshold = config['amount_threshold']
    generator.max_candidates = config['max_candidates']

    print(f"\n[配置]")
    print(f"  时间范围：{config['start_date']} ~ {config['end_date']}")
    print(f"  成交额阈值：{config['amount_threshold']} 千元")
    print(f"  最大候选：{config['max_candidates']} 只/天")
    print(f"  最大样本：{config['max_samples']}")
    print(f"  内存优化：{'启用（float32）' if config['use_float32'] else '禁用'}")

    # 生成训练数据
    print("\n[开始] 生成训练数据...")

    try:
        dataset = generator.generate_dataset(
            start_date=config['start_date'],
            end_date=config['end_date'],
            max_samples=config['max_samples']
        )

        if dataset is None or len(dataset) == 0:
            print("\n[错误] 生成的训练数据为空")
            return None

        print(f"\n[成功] 生成训练数据")
        print(f"  样本数：{len(dataset)} 条")
        print(f"  正样本：{(dataset['label'] == 1).sum()} ({(dataset['label'] == 1).sum()/len(dataset)*100:.1f}%)")
        print(f"  负样本：{(dataset['label'] == 0).sum()} ({(dataset['label'] == 0).sum()/len(dataset)*100:.1f}%)")
        print(f"  特征数：{len(dataset.columns) - 3}")  # 减去 label, ts_code, trade_date

        # [优化] 使用 float32 节省内存
        if config['use_float32']:
            print("\n[优化] 转换为 float32 格式...")
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            dataset[numeric_cols] = dataset[numeric_cols].astype(np.float32)

        # 保存训练数据
        output_dir = project_root / 'data' / 'training'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file = output_dir / f'training_data_{timestamp}.csv'

        dataset.to_csv(data_file, index=False, encoding='utf-8')
        print(f"\n[保存] 训练数据已保存：{data_file}")
        print(f"       文件大小：{data_file.stat().st_size / 1024 / 1024:.2f} MB")

        return str(data_file)

    except Exception as e:
        print(f"\n[错误] 生成训练数据失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_model(data_file: str, config: dict):
    """
    训练 AI 裁判模型

    Args:
        data_file: 训练数据文件路径
        config: 配置参数

    Returns:
        是否成功
    """
    print("\n" + "=" * 80)
    print("【步骤 2】训练 AI 裁判模型")
    print("=" * 80)

    # 初始化 AI 裁判
    referee = AIReferee(model_type=config['model_type'])

    # [优化] 指定数据类型读取，节省内存
    print(f"\n[读取] 训练数据：{data_file}")
    dtype = {'label': np.int32} if config['use_float32'] else None
    dataset = pd.read_csv(data_file, dtype=dtype)

    print(f"[信息] 数据形状：{dataset.shape}")
    print(f"[信息] 内存占用：{dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # [关键修复] 保留 ts_code 和 trade_date 列
    # prepare_features() 会自动处理这些元数据列
    X = dataset.drop('label', axis=1)
    y = dataset['label'].astype(np.int32)  # 确保标签是整数类型

    print(f"[信息] 特征数（含元数据）：{X.shape[1]}")
    print(f"[信息] 实际特征数：{X.shape[1] - 2}")  # 减去 ts_code, trade_date
    print(f"[信息] 样本数：{X.shape[0]}")
    print(f"[信息] 正样本：{y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"[信息] 负样本：{(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")

    # [样本不平衡警告]
    pos_ratio = y.sum() / len(y)
    if pos_ratio < 0.05:
        print(f"\n[警告] 正样本占比过低（{pos_ratio:.1%}），可能需要更多数据或调整策略")

    # 训练模型
    print(f"\n[开始] 训练模型（{config['model_type']}，{config['n_splits']}折时序交叉验证）...")

    try:
        results = referee.train_time_series(X, y, n_splits=config['n_splits'])

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

        model_file = output_dir / f'ai_referee_{config["model_type"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        referee.save_model(str(model_file))
        print(f"\n[保存] 模型已保存：{model_file}")
        print(f"       文件大小：{Path(model_file).stat().st_size / 1024 / 1024:.2f} MB")

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
    print("         AI 裁判 V5.0 训练流程（优化版）")
    print("=" * 80)
    print("\n[配置]")
    print(f"  时间范围：{TRAINING_CONFIG['start_date']} ~ {TRAINING_CONFIG['end_date']}")
    print(f"  最大候选：{TRAINING_CONFIG['max_candidates']} 只/天")
    print(f"  最大样本：{TRAINING_CONFIG['max_samples']}")
    print(f"  交叉验证：{TRAINING_CONFIG['n_splits']} 折")
    print(f"  模型类型：{TRAINING_CONFIG['model_type']}")
    print(f"  内存优化：{TRAINING_CONFIG['use_float32']}")

    # 步骤 1：生成训练数据
    data_file = generate_training_data(TRAINING_CONFIG)

    if data_file is None:
        print("\n[错误] 无法生成训练数据，训练终止")
        return

    # 步骤 2：训练模型
    success = train_model(data_file, TRAINING_CONFIG)

    if not success:
        print("\n[错误] 模型训练失败")
        return

    print("\n" + "=" * 80)
    print("✅ 训练流程完成！")
    print("\n下一步选项：")
    print("  1. 调整 TRAINING_CONFIG 中的参数重新训练")
    print("  2. 使用更长时间范围的数据训练")
    print("  3. 集成到选股系统进行回测")
    print("=" * 80)


if __name__ == '__main__':
    main()
