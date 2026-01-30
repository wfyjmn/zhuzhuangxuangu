# -*- coding: utf-8 -*-
"""
使用真实历史数据训练 AI 裁判模型（终极优化版）
优化点：
1. 集成 DataWarehouseTurbo 实现极速数据生成
2. 自动导出特征重要性 (Feature Importance)
3. 增强内存管理与垃圾回收
4. 修正 CSV 读取时的日期格式问题
"""
import os
import sys
import logging
import gc
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入核心模块
try:
    from ai_backtest_generator import AIBacktestGenerator
    from ai_referee import AIReferee

    # [优化] 尝试导入 Turbo 版本
    try:
        from data_warehouse_turbo import DataWarehouse
        IS_TURBO = True
    except ImportError:
        from data_warehouse import DataWarehouse
        IS_TURBO = False
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 配置日志
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'train_optimized.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ========================================
# 配置参数（可修改）
# ========================================
TRAINING_CONFIG = {
    # 时间范围 (2023-2024 年完整数据)
    'start_date': '20230101',
    'end_date': '20241231',

    # 数据生成参数
    'amount_threshold': 10000,  # 成交额阈值（千元）
    'max_candidates': 100,      # 每日最大候选股票数（增加以获得更多样本）
    'max_samples': 500000,      # 【手术一】彻底放开样本限制（50万），跑完 2023-2024 全年

    # 训练参数
    'n_splits': 5,              # 交叉验证折数
    'model_type': 'xgboost',    # 模型类型

    # 内存优化
    'use_float32': True,        # 使用 float32 节省内存
}


def generate_training_data(config: dict):
    """
    生成训练数据集
    """
    logger.info("=" * 80)
    logger.info("【步骤 1】生成训练数据集")
    logger.info("=" * 80)

    # 初始化数据仓库
    dw = DataWarehouse()
    generator = AIBacktestGenerator()

    # [优化] Turbo 模式预加载
    if IS_TURBO and hasattr(dw, 'preload_data'):
        logger.info("[系统] 启动 Turbo 极速模式：预加载数据到内存")
        # 扩展结束日期以包含标签所需的未来数据 (Labeling 需要未来5-10天数据)
        dt_end = datetime.strptime(config['end_date'], '%Y%m%d')
        extended_end = (dt_end + timedelta(days=20)).strftime('%Y%m%d')

        dw.preload_data(config['start_date'], extended_end, lookback_days=120)

        # 注入 Turbo Warehouse
        generator.warehouse = dw
    else:
        logger.warning("[系统] 使用普通模式（无内存预加载），速度较慢")

    # 应用配置
    generator.amount_threshold = config['amount_threshold']
    generator.max_candidates = config['max_candidates']

    logger.info(f"\n[配置]")
    logger.info(f"  时间范围：{config['start_date']} ~ {config['end_date']}")
    logger.info(f"  成交额阈值：{config['amount_threshold']} 千元")
    logger.info(f"  最大候选：{config['max_candidates']} 只/天")

    # 生成训练数据
    logger.info("\n[开始] 生成训练数据...")

    try:
        dataset = generator.generate_dataset(
            start_date=config['start_date'],
            end_date=config['end_date'],
            max_samples=config['max_samples']
        )

        if dataset is None or len(dataset) == 0:
            logger.error("\n[错误] 生成的训练数据为空")
            return None

        # 统计信息
        pos_samples = (dataset['label'] == 1).sum()
        neg_samples = (dataset['label'] == 0).sum()
        total_samples = len(dataset)

        logger.info(f"\n[成功] 生成训练数据")
        logger.info(f"  样本数：{total_samples} 条")
        logger.info(f"  正样本：{pos_samples} ({pos_samples/total_samples*100:.2f}%)")
        logger.info(f"  负样本：{neg_samples} ({neg_samples/total_samples*100:.2f}%)")

        # [优化] 使用 float32 节省内存
        if config['use_float32']:
            logger.info("[优化] 转换为 float32 格式...")
            numeric_cols = dataset.select_dtypes(include=[np.float64]).columns
            dataset[numeric_cols] = dataset[numeric_cols].astype(np.float32)

        # 保存训练数据
        output_dir = project_root / 'data' / 'training'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_file = output_dir / f'training_data_{timestamp}.csv'

        dataset.to_csv(data_file, index=False, encoding='utf-8')
        logger.info(f"\n[保存] 训练数据已保存：{data_file}")
        logger.info(f"       文件大小：{data_file.stat().st_size / 1024 / 1024:.2f} MB")

        # 主动释放内存
        del dataset
        if IS_TURBO:
            dw.clear_memory()  # 如果是Turbo，释放大内存块
        gc.collect()

        return str(data_file)

    except Exception as e:
        logger.error(f"\n[错误] 生成训练数据失败：{str(e)}", exc_info=True)
        return None


def train_model(data_file: str, config: dict):
    """
    训练 AI 裁判模型
    """
    logger.info("\n" + "=" * 80)
    logger.info("【步骤 2】训练 AI 裁判模型")
    logger.info("=" * 80)

    try:
        referee = AIReferee(model_type=config['model_type'])

        logger.info(f"\n[读取] 训练数据：{data_file}")

        # [优化] 指定数据类型读取，防止 CSV 将日期读成整数
        dtype_dict = {'label': np.int32, 'trade_date': str, 'ts_code': str}
        if config['use_float32']:
            # 这里的逻辑稍微复杂，无法预知所有列名，所以只指定关键列
            pass

        dataset = pd.read_csv(data_file, dtype=dtype_dict)

        # [优化] 再次强制转换 float32 (Pandas read_csv 默认是 float64)
        if config['use_float32']:
            float_cols = dataset.select_dtypes(include=['float64']).columns
            dataset[float_cols] = dataset[float_cols].astype('float32')

        # [关键] 确保 trade_date 是字符串或 datetime，以便 TimeSeriesSplit 正确排序
        dataset['trade_date'] = dataset['trade_date'].astype(str)

        logger.info(f"[信息] 原始数据形状：{dataset.shape}")
        logger.info(f"[信息] 内存占用：{dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # 准备数据
        # AIReferee.train_time_series 需要 trade_date 列进行排序和切分
        # prepare_features 内部会自动处理它，所以这里传入包含 trade_date 的 X
        X = dataset.drop('label', axis=1)
        y = dataset['label'].astype(np.int32)

        logger.info(f"[信息] 样本数：{X.shape[0]}")
        logger.info(f"[信息] 正样本占比：{y.sum()/len(y)*100:.2f}%")

        # 样本不平衡警告
        if y.sum() / len(y) < 0.05:
            logger.warning(f"[警告] 正样本极少，模型可能倾向于预测全负！")

        # 训练模型
        logger.info(f"\n[开始] 训练模型（{config['n_splits']}折时序交叉验证）...")
        logger.info("[提示] 这可能需要几分钟时间")

        results = referee.train_time_series(X, y, n_splits=config['n_splits'])

        logger.info("\n[成功] 模型训练完成")

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
        model_file = output_dir / f'ai_referee_{config["model_type"]}_{timestamp}.pkl'

        referee.save_model(str(model_file))
        logger.info(f"\n[保存] 模型已保存：{model_file}")
        logger.info(f"       文件大小：{Path(model_file).stat().st_size / 1024 / 1024:.2f} MB")

        # [新增] 保存特征重要性
        # 这对于理解模型逻辑至关重要
        if hasattr(referee, 'get_feature_importance'):
            imp_df = referee.get_feature_importance()
            if not imp_df.empty:
                imp_file = output_dir / f'feature_importance_{timestamp}.csv'
                imp_df.to_csv(imp_file, index=False)
                logger.info(f"[保存] 特征重要性已保存：{imp_file}")

                logger.info("\n[Top 10 重要特征]")
                for idx, row in imp_df.head(10).iterrows():
                    logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        else:
            # 如果 AIReferee 没有 get_feature_importance 方法，手动提取
            if hasattr(referee, 'model') and hasattr(referee.model, 'feature_importances_'):
                imps = referee.model.feature_importances_
                if hasattr(referee, 'feature_names'):
                    feature_names = referee.feature_names
                    if len(imps) == len(feature_names):
                        importances = pd.DataFrame({
                            'feature': feature_names,
                            'importance': imps
                        }).sort_values('importance', ascending=False)

                        imp_file = output_dir / f'feature_importance_{timestamp}.csv'
                        importances.to_csv(imp_file, index=False)
                        logger.info(f"[保存] 特征重要性已保存：{imp_file}")

                        logger.info("\n[Top 10 重要特征]")
                        for idx, row in importances.head(10).iterrows():
                            logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

        return True

    except Exception as e:
        logger.error(f"\n[错误] 训练模型失败：{str(e)}", exc_info=True)
        return False


def main():
    """主流程"""
    print("=" * 80)
    print("         AI 裁判 V5.0 训练流程（Turbo 增强版）")
    print("=" * 80)

    # 打印当前使用的仓库模式
    mode = "🚀 Turbo 极速模式" if IS_TURBO else "🐢 普通硬盘模式"
    print(f"当前运行模式: {mode}")

    # 步骤 1：生成
    data_file = generate_training_data(TRAINING_CONFIG)
    if not data_file: return

    # 步骤 2：训练
    success = train_model(data_file, TRAINING_CONFIG)
    if not success: return

    print("\n" + "=" * 80)
    print("✅ 训练全流程完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
