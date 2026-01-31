# -*- coding: utf-8 -*-
"""
使用真实历史数据（2023-2024年）训练 AI 裁判模型 - 终极优化版 (Fixed)
基于对话3.txt中的改进建议，包含：
1. Turbo模式支持（IS_TURBO标志位）
2. 时间扩展逻辑（extended_end）
3. 严格的特征清洗（select_dtypes）
4. 特征重要性保存（带长度检查）
5. 内存优化（float32）
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

# 尝试导入
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
    """内存优化：将 float64 转为 float32，int64 转为 int32"""
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
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        max_candidates: 每天最多选多少只股票
        max_samples: 最多生成多少个样本（None表示不限制）
        dry_run: 是否干运行（仅生成少量数据用于测试）
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
        logger.info(f"  max_candidates={max_candidates}, max_samples={max_samples}")

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

            if dw.memory_db is None or dw.memory_db.empty:
                logger.error("[错误] 数据预加载失败或为空")
                return None

            # 注入 Turbo Warehouse
            generator.warehouse = dw
        else:
            logger.warning("【系统】使用普通模式（无内存预加载），生成速度可能较慢...")

        # 设置参数
        generator.amount_threshold = 10000
        generator.max_candidates = max_candidates

        # 生成训练数据
        logger.info(f"[开始] 生成训练数据 ({start_date} ~ {end_date})...")
        logger.info(f"  每日候选数: {max_candidates}")
        logger.info(f"  总样本上限: {max_samples if max_samples else '无限制'}")

        dataset = generator.generate_dataset(
            start_date=start_date,
            end_date=end_date,
            max_samples=max_samples
        )

        if dataset is None or len(dataset) == 0:
            logger.error("[错误] 生成的训练数据为空")
            return None

        # 内存优化
        dataset = optimize_dataframe(dataset)

        # 检查正负样本
        if 'label' in dataset.columns:
            pos = (dataset['label'] == 1).sum()
            total = len(dataset)
            logger.info(f"  样本统计：总数 {total} | 正样本 {pos} ({pos/total:.2%})")
        else:
            logger.error("[错误] 数据集缺失 'label' 列")
            return None

        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file = output_dir / f'real_training_data_{start_date}_{end_date}_{timestamp}.csv'
        dataset.to_csv(data_file, index=False, encoding='utf-8')
        logger.info(f"[保存] {data_file}")
        logger.info(f"       文件大小：{data_file.stat().st_size / 1024 / 1024:.2f} MB")

        del dataset
        gc.collect()
        return str(data_file)

    except Exception as e:
        logger.error(f"生成数据失败: {str(e)}", exc_info=True)
        return None


def train_with_real_data(data_file, n_splits=5):
    """
    使用真实数据训练 AI 裁判模型

    Args:
        data_file: 训练数据文件路径
        n_splits: 时序交叉验证的折数
    """
    logger.info("\n" + "=" * 80)
    logger.info("【步骤 2】使用真实数据训练 AI 裁判模型")
    logger.info("=" * 80)

    try:
        referee = AIReferee(model_type='xgboost')

        logger.info(f"[读取] {data_file}")
        dataset = pd.read_csv(data_file)
        dataset = optimize_dataframe(dataset)

        # -------------------------------------------------------------------------
        # [关键修复] 特征清洗：严格剔除元数据
        # -------------------------------------------------------------------------
        # 定义必须排除的非特征列
        exclude_cols = ['label', 'trade_date', 'ts_code', 'code', 'name', 'industry', 'area', 'market']

        # 1. 获取所有列
        all_cols = dataset.columns.tolist()

        # 2. 筛选出特征列（不在排除列表中）
        feature_cols = [c for c in all_cols if c not in exclude_cols]

        # 3. 构建 X (特征) 和 y (标签)
        X = dataset[feature_cols]
        y = dataset['label'].astype(np.int32)

        # 4. [双重保险] 仅保留数值型列，剔除漏网的字符串列
        X_numeric = X.select_dtypes(include=[np.number])

        # 检查是否有列被意外剔除
        dropped_cols = set(X.columns) - set(X_numeric.columns)
        if dropped_cols:
            logger.warning(f"[警告] 以下非数值列被自动剔除: {dropped_cols}")

        X = X_numeric  # 最终使用的特征集

        logger.info(f"[特征] 最终特征数：{X.shape[1]}")
        logger.info(f"[特征] 特征列预览：{list(X.columns[:5])} ...")

        # 训练模型
        logger.info(f"[开始] 训练模型 (Time Series CV, {n_splits} Folds)...")
        logger.info("[提示] 这可能需要几分钟时间")

        results = referee.train_time_series(X, y, n_splits=n_splits)

        logger.info("\n[评估结果]")
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
        logger.info(f"[保存] 模型: {model_file}")

        # 保存特征重要性 (带长度检查)
        if hasattr(referee, 'model') and hasattr(referee.model, 'feature_importances_'):
            imps = referee.model.feature_importances_
            if len(imps) == len(X.columns):
                importances = pd.DataFrame({
                    'feature': X.columns,
                    'importance': imps
                }).sort_values('importance', ascending=False)

                imp_file = output_dir / f'feature_importance_{timestamp}.csv'
                importances.to_csv(imp_file, index=False)
                logger.info(f"[保存] 特征重要性: {imp_file}")

                logger.info("\n[Top 10 重要特征]")
                for idx, row in importances.head(10).iterrows():
                    logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
            else:
                logger.warning(f"[警告] 特征重要性数量 ({len(imps)}) 与 特征列数 ({len(X.columns)}) 不匹配，跳过保存。")

        return True

    except Exception as e:
        logger.error(f"训练失败: {str(e)}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='AI 裁判 V5.0 真实数据训练流程')
    parser.add_argument('--start', type=str, default='20230101', help='Start Date (YYYYMMDD)')
    parser.add_argument('--end', type=str, default='20241231', help='End Date (YYYYMMDD)')
    parser.add_argument('--file', type=str, default=None, help='Existing CSV File')
    parser.add_argument('--max-candidates', type=int, default=50, help='Stocks per day')
    parser.add_argument('--max-samples', type=int, default=2000, help='Max samples total')
    parser.add_argument('--n-splits', type=int, default=3, help='CV splits')
    parser.add_argument('--dry-run', action='store_true', help='Test run with small data')

    args = parser.parse_args()

    print("=" * 80)
    print("              AI 裁判 V5.0 真实数据训练流程")
    print("=" * 80)
    print(f"模式: {'Turbo 极速版' if IS_TURBO else '普通版'}")
    if args.dry_run:
        print("⚠️  干运行模式：仅生成少量测试数据")

    data_file = args.file
    if not data_file:
        data_file = generate_real_training_data(
            args.start, args.end,
            max_candidates=args.max_candidates,
            max_samples=args.max_samples,
            dry_run=args.dry_run
        )

    if data_file and os.path.exists(data_file):
        success = train_with_real_data(data_file, n_splits=args.n_splits)
        if success:
            logger.info("✅ 流程圆满完成！")
        else:
            logger.error("❌ 流程失败")
    else:
        logger.error("无有效数据文件，流程结束")


if __name__ == '__main__':
    main()
