"""
使用真实历史数据（2023-2024年）训练 AI 裁判模型 - 终极优化版 (Fixed)
"""
import os
import sys
import logging
import argparse
import gc
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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

# -------------------------------------------------------------------------
# 尝试导入核心模块
# -------------------------------------------------------------------------
try:
    from ai_backtest_generator import AIBacktestGenerator
    from ai_referee import AIReferee

    # 尝试导入 Turbo 版本，如果不存在则使用普通版本并标记
    try:
        from data_warehouse_turbo import DataWarehouse
        IS_TURBO = True
    except ImportError:
        from data_warehouse import DataWarehouse
        IS_TURBO = False
        logger.warning("[警告] 未找到 DataWarehouseTurbo，将使用普通模式（速度较慢）")
except ImportError as e:
    logger.error(f"导入核心模块失败: {e}")
    sys.exit(1)


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
    生成训练数据集

    Args:
        start_date: 开始日期（YYYYMMDD）
        end_date: 结束日期（YYYYMMDD）
        max_candidates: 每日最大候选股票数
        max_samples: 最大样本数
        dry_run: 是否干运行模式

    Returns:
        训练数据文件路径
    """
    logger.info("=" * 80)
    logger.info("【步骤 1】使用真实历史数据生成训练数据集")
    logger.info("=" * 80)

    output_dir = project_root / 'data' / 'training'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 干运行模式
    if dry_run:
        max_candidates = min(max_candidates, 5)
        max_samples = min(max_samples or 100, 50)
        logger.info("[干运行模式] 限制数据量用于快速测试")

    try:
        dw = DataWarehouse()
        generator = AIBacktestGenerator()

        # -------------------------------------------------------------------------
        # [关键修复] Turbo 模式安全检查
        # -------------------------------------------------------------------------
        if IS_TURBO and hasattr(dw, 'preload_data'):
            logger.info("=" * 80)
            logger.info("【系统】启动 Turbo 极速模式：预加载数据到内存")
            logger.info("=" * 80)

            # 扩展结束日期以包含标签所需的未来数据
            dt_end = datetime.strptime(end_date, '%Y%m%d')
            extended_end = (dt_end + timedelta(days=20)).strftime('%Y%m%d')  # 多预留20天

            dw.preload_data(start_date, extended_end, lookback_days=120)

            if dw.memory_db is None or (hasattr(dw, 'memory_db') and dw.memory_db.empty):
                logger.error("[错误] 数据预加载失败或为空")
                return None

            # 注入 Turbo Warehouse
            generator.warehouse = dw
            logger.info("[系统] Turbo 模式已启用")
        else:
            logger.warning("【系统】使用普通模式（无内存预加载），生成速度可能较慢...")

        # 设置参数
        generator.amount_threshold = 10000
        generator.max_candidates = max_candidates

        # 检查交易日历
        calendar = dw.get_trade_days(start_date, end_date)
        if not calendar or len(calendar) < 5:
            logger.error("[错误] 交易日数量不足或获取失败")
            return None

        logger.info(f"[配置] 时间范围：{start_date} ~ {end_date}")
        logger.info(f"[配置] 成交额阈值：10000 千元（1000万元）")
        logger.info(f"[配置] 最大候选股票：{max_candidates} 只/天")
        if max_samples:
            logger.info(f"[配置] 最大样本数：{max_samples}")

        # 生成训练数据
        logger.info("[开始] 生成训练数据...")

        dataset = generator.generate_dataset(
            start_date=start_date,
            end_date=end_date,
            max_samples=max_samples
        )

        if dataset is None or len(dataset) == 0:
            logger.error("[错误] 生成的训练数据为空")
            return None

        # 内存优化
        logger.info("[优化] 压缩数据类型...")
        dataset = optimize_dataframe(dataset)

        # 检查正负样本
        if 'label' in dataset.columns:
            pos = (dataset['label'] == 1).sum()
            total = len(dataset)
            neg = total - pos
            logger.info(f"[成功] 生成训练数据")
            logger.info(f"  样本数：{total}")
            logger.info(f"  正样本：{pos} ({pos/total:.2%})")
            logger.info(f"  负样本：{neg} ({neg/total:.2%})")
            logger.info(f"  胜率：{pos/total:.2%}")

            # 样本不平衡警告
            if pos_ratio := pos / total:
                if pos_ratio < 0.05:
                    logger.warning(f"[警告] 正样本占比过低（{pos_ratio:.1%}），模型可能倾向于预测全负")
                    logger.warning("[建议] 增加 max_candidates 或扩大时间范围")
        else:
            logger.error("[错误] 数据集缺失 'label' 列")
            return None

        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file = output_dir / f'real_training_data_{start_date}_{end_date}_{timestamp}.csv'

        logger.info(f"[保存] 正在保存训练数据到：{data_file}")
        dataset.to_csv(data_file, index=False, encoding='utf-8')
        logger.info(f"[保存] 训练数据已保存")
        logger.info(f"       文件大小：{data_file.stat().st_size / 1024 / 1024:.2f} MB")

        # 释放内存
        del dataset
        gc.collect()

        return str(data_file)

    except Exception as e:
        logger.error(f"生成数据失败: {str(e)}", exc_info=True)
        return None


def train_with_real_data(data_file, n_splits=5):
    """
    训练模型

    Args:
        data_file: 训练数据文件路径
        n_splits: 交叉验证折数

    Returns:
        是否成功
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

        # -------------------------------------------------------------------------
        # [关键修复] 特征清洗：严格剔除元数据
        # -------------------------------------------------------------------------
        # 定义必须排除的非特征列
        exclude_cols = ['label', 'ts_code', 'code', 'name', 'industry', 'area', 'market', 'sector']

        # 注意：保留 trade_date 列，因为 train_time_series 需要它进行时序切分
        # prepare_features() 会自动移除这些元数据列

        # 构建 X（包含 trade_date 用于时序切分）和 y
        X = dataset.drop('label', axis=1)
        y = dataset['label'].astype(np.int32)

        # [双重保险] 确保所有特征列都是数值型
        # 排除 trade_date（字符串）和其他可能的字符串列
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # 除了 trade_date，其他非数值列都应该被剔除
        non_numeric_to_drop = [col for col in non_numeric_cols if col != 'trade_date']

        if non_numeric_to_drop:
            logger.warning(f"[警告] 以下非数值列被自动剔除: {non_numeric_to_drop}")
            X = X.drop(columns=non_numeric_to_drop)

        # 确保 trade_date 存在
        if 'trade_date' not in X.columns:
            logger.error("[错误] 数据集缺失 'trade_date' 列，无法进行时序交叉验证")
            return False

        logger.info(f"[特征] 输入特征数（含元数据）：{X.shape[1]}")
        logger.info(f"[特征] 元数据列：trade_date, ts_code（如果存在）")
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

        # 保存特征重要性（带长度检查）
        if hasattr(referee, 'model') and hasattr(referee.model, 'feature_importances_') and hasattr(referee, 'feature_names'):
            imps = referee.model.feature_importances_

            # 使用 referee.feature_names（prepare_features() 处理后的特征名）
            if len(imps) == len(referee.feature_names):
                importances = pd.DataFrame({
                    'feature': referee.feature_names,
                    'importance': imps
                }).sort_values('importance', ascending=False)

                imp_file = output_dir / f'feature_importance_{timestamp}.csv'
                importances.to_csv(imp_file, index=False)
                logger.info(f"[保存] 特征重要性已保存：{imp_file}")

                logger.info("\n[Top 10 特征重要性]")
                for idx, row in importances.head(10).iterrows():
                    logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
            else:
                logger.warning(
                    f"[警告] 特征重要性数量 ({len(imps)}) "
                    f"与 特征列数 ({len(referee.feature_names)}) 不匹配，跳过保存"
                )
        else:
            logger.warning("[跳过] 无法获取特征重要性")

        return True

    except Exception as e:
        logger.error(f"训练失败: {str(e)}", exc_info=True)
        return False


def main():
    """
    主流程
    """
    parser = argparse.ArgumentParser(
        description='AI 裁判 V5.0 真实数据训练流程（终极优化版）',
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
    print("         AI 裁判 V5.0 真实数据训练流程（终极优化版）")
    print("=" * 80)
    print(f"\n[系统信息]")
    print(f"  运行模式：{'Turbo 极速版' if IS_TURBO else '普通版'}")
    print(f"\n[配置]")
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
        logger.info("  2. 使用 --start 20240101 --end 20240401 生成更多数据（1季度）")
        logger.info("  3. 使用 --start 20230101 --end 20241231 生成完整数据（全量）")
        logger.info("  4. 集成到选股系统进行回测")
        logger.info("=" * 80)
    else:
        logger.error("❌ 流程失败")


if __name__ == '__main__':
    main()
