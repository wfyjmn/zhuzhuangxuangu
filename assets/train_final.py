"""
使用真实历史数据（2023-2024年）训练 AI 裁判模型 - 终极优化版
"""
import os
import sys
import logging
import argparse
import gc
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    # 优先使用缓存版本
    from data_warehouse_cached import DataWarehouse
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


def generate_real_training_data(start_date, end_date, max_candidates=50, max_samples=None, dry_run=False):
    """
    使用真实历史数据生成训练数据集

    Args:
        start_date: 开始日期（YYYYMMDD）
        end_date: 结束日期（YYYYMMDD）
        max_candidates: 每日最大候选股票数（用于加速生成）
        max_samples: 最大样本数（防止内存溢出）
        dry_run: 是否只生成少量数据用于测试
    """
    logger.info("=" * 80)
    logger.info("【步骤 1】使用真实历史数据生成训练数据集")
    logger.info("=" * 80)

    # 路径检查
    output_dir = project_root / 'data' / 'training'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 干运行模式：限制数据量
    if dry_run:
        max_candidates = min(max_candidates, 10)
        max_samples = min(max_samples or 100, 50)
        logger.info("[干运行模式] 限制数据量用于快速测试")
        logger.info(f"  max_candidates: {max_candidates}")
        logger.info(f"  max_samples: {max_samples}")

    # 检查是否已有现成的数据文件
    existing_files = list(output_dir.glob(f'real_training_data_{start_date}_{end_date}_*.csv'))
    if existing_files:
        latest_file = sorted(existing_files)[-1]
        logger.info(f"[提示] 检测到已存在的时间段数据：{latest_file}")
        logger.info("[提示] 如需重新生成，请删除旧文件或使用 --force 参数")

        # TODO: 添加 --force 参数支持

    try:
        dw = DataWarehouse()
        generator = AIBacktestGenerator()

        # [优化] 设置性能参数
        generator.amount_threshold = 10000  # 成交额阈值（千元）
        generator.max_candidates = max_candidates  # 每日最大候选股票数

        logger.info(f"[配置] 时间范围：{start_date} ~ {end_date}")
        logger.info(f"[配置] 成交额阈值：10000 千元（1000万元）")
        logger.info(f"[配置] 最大候选股票：{max_candidates} 只/天")
        if max_samples:
            logger.info(f"[配置] 最大样本数：{max_samples}")

        # 检查交易日历
        calendar = dw.get_trade_days(start_date, end_date)
        if not calendar or len(calendar) < 5:
            logger.error("[错误] 交易日数量不足或获取失败")
            return None

        logger.info(f"[信息] 交易日数量：{len(calendar)} 个")

        # 生成训练数据
        logger.info("[开始] 生成训练数据（可能需要较长时间，请耐心等待）...")
        logger.info("[提示] 使用 max_candidates 和 max_samples 可以加速生成")

        dataset = generator.generate_dataset(
            start_date=start_date,
            end_date=end_date,
            max_samples=max_samples
        )

        if dataset is None or len(dataset) == 0:
            logger.error("[错误] 生成的训练数据为空")
            return None

        # 内存优化
        logger.info("[优化] 压缩数据类型以节省内存...")
        dataset = optimize_dataframe(dataset)

        logger.info(f"[成功] 生成训练数据")
        logger.info(f"  样本数：{len(dataset)}")

        if 'label' in dataset.columns:
            pos_samples = (dataset['label'] == 1).sum()
            neg_samples = (dataset['label'] == 0).sum()
            logger.info(f"  正样本：{pos_samples} ({pos_samples/len(dataset)*100:.2f}%)")
            logger.info(f"  负样本：{neg_samples} ({neg_samples/len(dataset)*100:.2f}%)")

            # [警告] 检查样本不平衡
            pos_ratio = pos_samples / len(dataset)
            if pos_ratio < 0.05:
                logger.warning(f"[警告] 正样本占比过低（{pos_ratio:.1%}），模型可能倾向于预测全负")
                logger.warning("[建议] 增加 max_candidates 或扩大时间范围")
        else:
            logger.error("[严重错误] 数据集中缺失 'label' 列")
            return None

        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file = output_dir / f'real_training_data_{start_date}_{end_date}_{timestamp}.csv'

        logger.info(f"[保存] 正在保存训练数据到：{data_file}")
        dataset.to_csv(data_file, index=False, encoding='utf-8')
        logger.info(f"[保存] 训练数据已保存")
        logger.info(f"       文件大小：{data_file.stat().st_size / 1024 / 1024:.2f} MB")

        # 主动释放内存
        del dataset
        gc.collect()

        return str(data_file)

    except Exception as e:
        logger.error(f"生成训练数据失败: {str(e)}", exc_info=True)
        return None


def train_with_real_data(data_file, n_splits=5):
    """
    使用真实数据训练 AI 裁判模型

    Args:
        data_file: 训练数据文件路径
        n_splits: 时序交叉验证折数
    """
    logger.info("\n" + "=" * 80)
    logger.info("【步骤 2】使用真实数据训练 AI 裁判模型")
    logger.info("=" * 80)

    try:
        referee = AIReferee(model_type='xgboost')

        logger.info(f"[读取] 训练数据：{data_file}")
        dataset = pd.read_csv(data_file)

        logger.info(f"[信息] 原始数据形状：{dataset.shape}")
        logger.info(f"[信息] 原始内存占用：{dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # 内存优化
        logger.info("[优化] 压缩数据类型...")
        dataset = optimize_dataframe(dataset)
        logger.info(f"[信息] 优化后内存占用：{dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # ---------------------------------------------------------
        # 核心修正：剔除元数据列
        # ---------------------------------------------------------
        exclude_cols = ['trade_date', 'ts_code', 'code', 'date', 'stock_code', 'name']

        # 确定特征列：排除 label 和 exclude_cols
        feature_cols = [c for c in dataset.columns if c not in ['label'] + exclude_cols]

        # 再次检查是否有非数值列混入
        X = dataset[feature_cols]
        non_numeric = X.select_dtypes(include=['object']).columns
        if len(non_numeric) > 0:
            logger.warning(f"[警告] 发现非数值特征列，将被自动移除: {list(non_numeric)}")
            X = X.drop(columns=non_numeric)
            feature_cols = X.columns.tolist()

        y = dataset['label'].astype(np.int32)

        logger.info(f"[特征] 最终特征数量：{X.shape[1]}")
        logger.info(f"[特征] 特征列表示例：{list(feature_cols[:5])} ...")
        logger.info(f"[样本] 总样本数：{len(y)}")
        logger.info(f"[样本] 正样本：{y.sum()} ({y.sum()/len(y)*100:.2f}%)")

        # 训练模型
        logger.info(f"[开始] 训练模型（{n_splits}折时序交叉验证）...")
        logger.info("[提示] 这可能需要几分钟时间")

        results = referee.train_time_series(X, y, n_splits=n_splits)

        logger.info("[成功] 模型训练完成")

        # 打印交叉验证结果
        logger.info("\n[交叉验证结果]")
        if 'cv_results' in results:
            logger.info("\n" + results['cv_results'].to_string(index=False))

        logger.info("\n[平均指标]")
        for metric, value in results.get('avg_metrics', {}).items():
            logger.info(f"  {metric}: {value:.4f}")

        # 保存模型
        output_dir = project_root / 'data' / 'models'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = output_dir / f'ai_referee_xgboost_{timestamp}.pkl'

        referee.save_model(str(model_file))
        logger.info(f"[保存] 模型已保存：{model_file}")
        logger.info(f"       文件大小：{Path(model_file).stat().st_size / 1024 / 1024:.2f} MB")

        # 保存特征重要性
        if hasattr(referee, 'model') and hasattr(referee.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': referee.model.feature_importances_
            }).sort_values('importance', ascending=False)

            imp_file = output_dir / f'feature_importance_{timestamp}.csv'
            importances.to_csv(imp_file, index=False)
            logger.info(f"[保存] 特征重要性已保存：{imp_file}")

            logger.info("\n[Top 10 特征重要性]")
            for idx, row in importances.head(10).iterrows():
                logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

        return True

    except Exception as e:
        logger.error(f"训练模型失败: {str(e)}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='AI 裁判 V5.0 真实数据训练流程',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 日期范围
    parser.add_argument('--start', type=str, default='20240101',
                        help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', type=str, default='20240131',
                        help='结束日期 (YYYYMMDD)')

    # 直接使用已有文件
    parser.add_argument('--file', type=str, default=None,
                        help='直接使用已有的 CSV 文件进行训练')

    # 性能参数
    parser.add_argument('--max-candidates', type=int, default=50,
                        help='每日最大候选股票数（越小越快，但样本越少）')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='最大样本数（防止内存溢出）')
    parser.add_argument('--n-splits', type=int, default=3,
                        help='时序交叉验证折数')

    # 测试模式
    parser.add_argument('--dry-run', action='store_true',
                        help='干运行模式：只生成少量数据用于快速测试流程')

    args = parser.parse_args()

    print("=" * 80)
    print("              AI 裁判 V5.0 真实数据训练流程（终极优化版）")
    print("=" * 80)
    print(f"[配置]")
    print(f"  时间范围：{args.start} ~ {args.end}")
    print(f"  最大候选：{args.max_candidates} 只/天")
    print(f"  最大样本：{args.max_samples}")
    print(f"  交叉验证：{args.n_splits} 折")
    print(f"  干运行：{'是' if args.dry_run else '否'}")

    data_file = args.file

    # 步骤 1：生成数据（如果未提供文件）
    if not data_file:
        data_file = generate_real_training_data(
            args.start,
            args.end,
            max_candidates=args.max_candidates,
            max_samples=args.max_samples,
            dry_run=args.dry_run
        )
    else:
        if not os.path.exists(data_file):
            logger.error(f"指定的文件不存在: {data_file}")
            return

    if not data_file:
        logger.error("无法获取训练数据，流程终止")
        return

    # 步骤 2：训练
    success = train_with_real_data(data_file, n_splits=args.n_splits)

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ 流程圆满完成！")
        logger.info("=" * 80)
        logger.info("\n[下一步]")
        logger.info("  1. 查看 feature_importance_*.csv 了解哪些特征最重要")
        logger.info("  2. 使用 --start 20240101 --end 20241231 生成更多数据")
        logger.info("  3. 集成到选股系统进行回测")
        logger.info("=" * 80)
    else:
        logger.error("❌ 流程失败")


if __name__ == '__main__':
    main()
