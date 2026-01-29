"""
使用真实历史数据（2023-2024年）训练 AI 裁判模型 - 优化版
"""
import os
import sys
import logging
import argparse
import gc  # 引入垃圾回收
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from data_warehouse import DataWarehouse
    from ai_backtest_generator import AIBacktestGenerator
    from ai_referee import AIReferee
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确的目录下运行，且 data_warehouse 等模块存在。")
    sys.exit(1)

# 配置日志
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'train_real_data.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def optimize_dataframe(df):
    """
    内存优化：将 float64 转为 float32，int64 转为 int32
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
    return df

def generate_real_training_data(start_date, end_date):
    """
    使用真实历史数据生成训练数据集
    """
    logger.info("=" * 80)
    logger.info("【步骤 1】使用真实历史数据生成训练数据集")
    logger.info("=" * 80)

    # 路径检查
    output_dir = project_root / 'data' / 'training'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已有现成的数据文件（避免重复生成）
    # 注意：这里简单的检查文件名，实际应用可能需要更复杂的哈希检查
    existing_files = list(output_dir.glob(f'real_training_data_{start_date}_{end_date}_*.csv'))
    if existing_files:
        latest_file = sorted(existing_files)[-1]
        logger.info(f"[提示] 检测到已存在的时间段数据，跳过生成步骤：{latest_file}")
        # return str(latest_file) # 如果想强制重新生成，注释掉这行

    try:
        dw = DataWarehouse()
        generator = AIBacktestGenerator()

        logger.info(f"[配置] 时间范围：{start_date} ~ {end_date}")

        # 检查交易日历
        calendar = dw.get_trade_days(start_date, end_date)
        if not calendar or len(calendar) < 5:
            logger.error("[错误] 交易日数量不足或获取失败")
            return None

        # 生成训练数据
        logger.info("[开始] 生成训练数据 (预计耗时较长)...")
        dataset = generator.generate_dataset(start_date=start_date, end_date=end_date)

        if dataset is None or len(dataset) == 0:
            logger.error("[错误] 生成的训练数据为空")
            return None

        # 内存优化
        dataset = optimize_dataframe(dataset)

        logger.info(f"[成功] 生成训练数据")
        logger.info(f"  样本数：{len(dataset)}")

        if 'label' in dataset.columns:
            pos_samples = (dataset['label'] == 1).sum()
            neg_samples = (dataset['label'] == 0).sum()
            logger.info(f"  正样本：{pos_samples} ({pos_samples/len(dataset)*100:.2f}%)")
            logger.info(f"  负样本：{neg_samples} ({neg_samples/len(dataset)*100:.2f}%)")
        else:
            logger.error("[严重错误] 数据集中缺失 'label' 列")
            return None

        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 文件名包含日期范围，便于识别
        data_file = output_dir / f'real_training_data_{start_date}_{end_date}_{timestamp}.csv'

        dataset.to_csv(data_file, index=False, encoding='utf-8')
        logger.info(f"[保存] 训练数据已保存：{data_file}")
        logger.info(f"       文件大小：{data_file.stat().st_size / 1024 / 1024:.2f} MB")

        # 主动释放内存
        del dataset
        gc.collect()

        return str(data_file)

    except Exception as e:
        logger.error(f"生成训练数据失败: {str(e)}", exc_info=True)
        return None


def train_with_real_data(data_file):
    """
    使用真实数据训练 AI 裁判模型
    """
    logger.info("\n" + "=" * 80)
    logger.info("【步骤 2】使用真实数据训练 AI 裁判模型")
    logger.info("=" * 80)

    try:
        referee = AIReferee()

        logger.info(f"[读取] 训练数据：{data_file}")
        # 使用 chunksize 防止一次性读取爆内存（如果数据极大的话），这里演示标准读取并优化
        dataset = pd.read_csv(data_file)
        dataset = optimize_dataframe(dataset)

        # ---------------------------------------------------------
        # 核心修正：剔除元数据列
        # ---------------------------------------------------------
        # 定义不需要进入模型的列（日期、代码、名称等非特征列）
        # 请根据你的实际数据结构调整 exclude_cols
        exclude_cols = ['trade_date', 'ts_code', 'code', 'date', 'stock_code', 'name', 'industry', 'area', 'market', 'sector']

        # 确定特征列：排除 label 和 exclude_cols
        feature_cols = [c for c in dataset.columns if c not in ['label'] + exclude_cols]

        # 再次检查是否有非数值列混入 (Double Check)
        X = dataset[feature_cols]
        non_numeric = X.select_dtypes(include=['object']).columns
        if len(non_numeric) > 0:
            logger.warning(f"[警告] 发现非数值特征列，将被自动移除: {list(non_numeric)}")
            X = X.drop(columns=non_numeric)

        y = dataset['label']

        logger.info(f"[特征] 最终特征数量：{X.shape[1]}")
        logger.info(f"[特征] 特征列表示例：{list(X.columns[:5])} ...")
        logger.info(f"[样本] 训练样本数：{len(X)}")

        if 'label' not in dataset.columns:
            logger.error("[严重错误] 数据集中缺失 'label' 列")
            return False

        # 样本不平衡警告
        pos_ratio = (y == 1).sum() / len(y)
        if pos_ratio < 0.05:
            logger.warning(f"[警告] 正样本占比过低（{pos_ratio:.1%}），模型可能倾向于预测全负")
            logger.warning("[建议] 增加 max_candidates 或扩大时间范围，或在模型参数中设置 scale_pos_weight")

        # 训练模型
        logger.info("[开始] 训练模型 (Time Series CV, 5 Folds)...")
        logger.info("[提示] 这可能需要几分钟时间")

        # 调用训练
        results = referee.train_time_series(X, y, n_splits=5)

        logger.info("[成功] 模型训练完成")
        logger.info("\n[交叉验证结果]")
        if 'cv_results' in results:
            logger.info("\n" + str(results['cv_results']))

        logger.info("\n[平均指标]")
        for metric, value in results.get('avg_metrics', {}).items():
            logger.info(f"  {metric}: {value:.4f}")

        # 保存模型
        output_dir = project_root / 'data' / 'models'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = output_dir / f'ai_referee_xgboost_{timestamp}.pkl'

        referee.save_model(str(model_file))
        logger.info(f"\n[保存] 模型已保存：{model_file}")
        logger.info(f"       文件大小：{Path(model_file).stat().st_size / 1024 / 1024:.2f} MB")

        # 保存特征重要性（如果有）
        if hasattr(referee, 'model') and hasattr(referee.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': referee.model.feature_importances_
            }).sort_values('importance', ascending=False)
            imp_file = output_dir / f'feature_importance_{timestamp}.csv'
            importances.to_csv(imp_file, index=False)
            logger.info(f"[保存] 特征重要性已保存：{imp_file}")

            logger.info("\n[Top 10 重要特征]")
            for idx, row in importances.head(10).iterrows():
                logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

        return True

    except Exception as e:
        logger.error(f"训练模型失败: {str(e)}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description='AI Referee Training Pipeline')
    parser.add_argument('--start', type=str, default='20230101', help='Start Date (YYYYMMDD)')
    parser.add_argument('--end', type=str, default='20241231', help='End Date (YYYYMMDD)')
    parser.add_argument('--file', type=str, default=None, help='Directly use existing CSV file for training')

    args = parser.parse_args()

    print("=" * 80)
    print("              AI 裁判 V5.0 真实数据训练流程")
    print("=" * 80)

    data_file = args.file

    # 步骤 1：生成数据（如果未提供文件）
    if not data_file:
        data_file = generate_real_training_data(args.start, args.end)
    else:
        if not os.path.exists(data_file):
            logger.error(f"指定的文件不存在: {data_file}")
            return

    if not data_file:
        logger.error("无法获取训练数据，流程终止")
        return

    # 步骤 2：训练
    success = train_with_real_data(data_file)

    if success:
        logger.info("✅ 流程圆满完成！")
    else:
        logger.error("❌ 流程失败")

if __name__ == '__main__':
    main()
