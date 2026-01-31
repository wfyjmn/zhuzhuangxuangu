# -*- coding: utf-8 -*-
"""
自动化参数调优训练脚本（DeepQuant V5.0）
使用 Optuna 贝叶斯优化自动搜索最优超参数

优化改进（基于对话4.txt建议）：
1. 内存优化：Float32降维，节省50%内存
2. 垃圾回收：激进的GC策略，防止内存泄漏
3. XGBoost内存优化：使用max_bin参数减少内存占用
4. 优化目标调整：使用AUC代替F1，避免"宁滥勿缺"
5. 大盘环境特征：加入上证指数涨跌幅作为环境特征
6. 阈值选择逻辑：硬性约束精确率≥60%

作者: Coze Coding
日期: 2026-01-31
"""

import os
import sys
import json
import pickle
import gc
import warnings
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

import optuna
import xgboost as xgb

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'assets'))

try:
    from data_warehouse import DataWarehouse
    from ai_backtest_generator import AIBacktestGenerator
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


class AutoTunedTrainer:
    """
    自动化参数调优训练器

    特点：
    1. 使用Optuna进行贝叶斯优化
    2. 自动内存优化（Float32 + GC）
    3. 支持时序交叉验证
    4. 自动阈值优化（精确率约束）
    5. 大盘环境特征集成
    """

    def __init__(self, config_path: str = "config/precision_priority_config.json"):
        self.config_path = config_path
        self.config = self._load_config()

        # 数据仓库（用于采集数据）
        self.warehouse = DataWarehouse()

        # 回测生成器（用于特征工程和标签计算）
        self.backtest_generator = AIBacktestGenerator()

        # 模型相关
        self.best_model = None
        self.best_params = None
        self.feature_names = None
        self.study = None

        # 大盘数据缓存
        self.market_data = None

        print("=" * 80)
        print("AutoTunedTrainer 初始化")
        print("=" * 80)
        print(f"配置文件: {config_path}")
        print(f"优化目标: {self.config['optuna']['metric']}")
        print(f"目标精确率: {self.config['threshold']['target_precision']}")
        print()

    def _load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            "strategy_name": "自动化参数调优训练器",
            "version": "3.0",

            "data": {
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "min_return_threshold": 0.04,  # 放宽：5% → 4%
                "prediction_days": [3, 4, 5],
                "n_stocks": 150,  # 减少：300 → 150（避免OOM）
                "max_samples": 5000  # 最大样本数限制
            },

            "optuna": {
                "n_trials": 50,  # 减少：100 → 50（避免OOM）
                "timeout": 3600,
                "direction": "maximize",
                "metric": "auc",  # 改进：f1 → auc
                "cv_folds": 3,
                "early_stopping_rounds": 50
            },

            "threshold": {
                "target_precision": 0.60,  # 改进：精确率目标≥60%
                "threshold_range": [0.15, 0.45],
                "threshold_step": 0.01
            },

            "model": {
                "tree_method": "hist",
                "max_bin": 128,  # 新增：内存优化参数
                "use_label_encoder": False
            }
        }

        # 合并用户配置
        config_path = project_root / self.config_path
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 递归合并配置
                self._deep_merge(default_config, user_config)
            except Exception as e:
                print(f"⚠️  加载用户配置失败，使用默认配置: {e}")

        return default_config

    def _deep_merge(self, base: Dict, override: Dict):
        """递归合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _convert_numpy_to_python(self, obj):
        """将 numpy 类型转换为原生 Python 类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [优化1] 内存优化：将 float64 转换为 float32
        这可以节省 50% 的内存占用
        """
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'float64':
                df[col] = df[col].astype('float32')
            elif col_type == 'int64':
                df[col] = df[col].astype('int32')
        return df

    def _force_gc(self):
        """[优化2] 激进的垃圾回收"""
        gc.collect()
        gc.collect()

    def get_stock_list(self, n_stocks: int = 150) -> List[str]:
        """获取股票列表"""
        print("=" * 80)
        print(f"【步骤1】获取{n_stocks}只股票列表")
        print("=" * 80)

        try:
            # 使用数据仓库的缓存
            all_stocks = list(self.warehouse.basic_info_cache['ts_code'].values)

            # 过滤 ST 股
            all_stocks = [s for s in all_stocks if not any(x in s for x in ['ST', '退'])]

            # 选取前 n_stocks 只
            selected_stocks = all_stocks[:n_stocks]

            print(f"✓ 获取到 {len(selected_stocks)} 只股票")
            print()

            return selected_stocks

        except Exception as e:
            print(f"❌ 获取股票列表失败: {e}")
            return []

    def collect_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """采集股票数据（使用 DataWarehouse）"""
        print("=" * 80)
        print("【步骤2】采集股票历史数据")
        print("=" * 80)

        try:
            # 获取交易日历
            trade_days = self.warehouse.get_trade_days(
                start_date.replace('-', ''),
                end_date.replace('-', '')
            )

            if not trade_days:
                raise ValueError("无法获取交易日历")

            print(f"  交易日数量: {len(trade_days)}")

            # 加载所有股票数据
            all_data = []
            for date in trade_days[:-5]:  # 排除最后5天（用于标签计算）
                df_daily = self.warehouse.load_daily_data(date)

                if df_daily is not None and not df_daily.empty:
                    all_data.append(df_daily)

                if len(all_data) % 50 == 0:
                    print(f"  已加载 {len(all_data)} 个交易日数据")

            if not all_data:
                raise ValueError("未能采集到任何数据")

            # 合并数据
            df_all = pd.concat(all_data, ignore_index=False)
            df_all = df_all.sort_index()

            # [优化1] 内存优化
            df_all = self._optimize_memory(df_all)

            print(f"\n✓ 数据采集完成:")
            print(f"  总样本数: {len(df_all)}")
            print(f"  股票数: {df_all.index.get_level_values(0).nunique()}")
            print(f"  时间范围: {df_all.index.get_level_values(1).min()} 至 {df_all.index.get_level_values(1).max()}")
            print()

            return df_all

        except Exception as e:
            print(f"❌ 数据采集失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间序列划分数据集"""
        print("=" * 80)
        print("【步骤3】时间序列划分数据集")
        print("=" * 80)

        # 确保按时间排序
        df = df.sort_index()

        n = len(df)
        n_train = int(n * 0.60)
        n_val = int(n * 0.20)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()

        print(f"✓ 数据集划分（按时间）:")
        print(f"  训练集: {len(train_df)}条")
        print(f"  验证集: {len(val_df)}条")
        print(f"  测试集: {len(test_df)}条")
        print()

        return train_df, val_df, test_df

    def add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [优化5] 添加大盘环境特征
        包括：大盘涨跌幅、板块平均涨跌幅等
        """
        print("  添加大盘环境特征...")

        try:
            # 获取上证指数数据
            if self.market_data is None:
                index_code = '000001.SH'
                self.market_data = self.warehouse.get_stock_data(index_code, '20241231', days=3650)

            if self.market_data is not None and not self.market_data.empty:
                # 计算大盘涨跌幅
                self.market_data['market_pct_chg'] = self.market_data['close'].pct_change()

                # 对齐日期
                market_dict = {}
                for date in df.index.get_level_values(1):
                    if date in self.market_data.index:
                        market_dict[date] = self.market_data.loc[date, 'market_pct_chg']

                # 添加到 DataFrame
                market_series = pd.Series(market_dict)
                df = df.copy()
                df['market_pct_chg'] = market_series.values
                df['market_pct_chg'] = df['market_pct_chg'].fillna(0)

                print(f"    ✓ 大盘涨跌幅特征已添加")

        except Exception as e:
            print(f"    ⚠️  大盘特征添加失败: {e}")

        return df

    def create_features_and_labels(self, train_df: pd.DataFrame,
                                    val_df: pd.DataFrame,
                                    test_df: pd.DataFrame) -> Tuple:
        """创建特征和标签"""
        print("=" * 80)
        print("【步骤4】创建特征和标签")
        print("=" * 80)

        datasets = {'训练集': train_df, '验证集': val_df, '测试集': test_df}
        processed_datasets = {}

        for name, df in datasets.items():
            print(f"  处理{name}...")

            # [优化5] 添加大盘特征
            df = self.add_market_features(df)

            # 使用回测生成器计算特征
            # 注意：这里需要适配现有的回测生成器接口
            # 为了简化，我们直接计算一些基础技术指标
            df = self._calculate_technical_features(df)

            # 创建标签
            df = self._create_labels(df)

            # [优化1] 内存优化
            df = self._optimize_memory(df)

            processed_datasets[name] = df

            # 统计信息
            if 'label' in df.columns:
                pos_samples = (df['label'] == 1).sum()
                total = len(df)
                print(f"    样本数: {total}, 正样本: {pos_samples} ({pos_samples/total:.2%})")

        print()

        return processed_datasets

    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术特征（简化版）"""
        df = df.copy()

        # 按股票分组计算
        def calc_features(group):
            # 基础价格特征
            group['ma5'] = group['close'].rolling(5).mean()
            group['ma10'] = group['close'].rolling(10).mean()
            group['ma20'] = group['close'].rolling(20).mean()

            # 动量特征
            group['momentum_5'] = group['close'] / group['close'].shift(5) - 1
            group['momentum_10'] = group['close'] / group['close'].shift(10) - 1

            # 波动率特征
            group['volatility_10'] = group['close'].pct_change().rolling(10).std()

            # 成交量特征
            group['volume_ratio'] = group['vol'] / group['vol'].rolling(5).mean()

            # 技术指标
            group['rsi'] = self._calculate_rsi(group['close'], 14)

            return group

        df = df.groupby(level=0).apply(calc_features)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建标签（3-5天涨幅≥4%）"""
        min_return = self.config['data']['min_return_threshold']
        prediction_days = self.config['data']['prediction_days']

        # 计算未来N天涨幅
        for days in prediction_days:
            df[f'future_return_{days}d'] = df.groupby(level=0)['close'].pct_change(days).shift(-days)

        # 标签：3-5天内任意一天涨幅≥目标
        future_returns = [f'future_return_{days}d' for days in prediction_days]
        df['max_future_return'] = df[future_returns].max(axis=1)
        df['label'] = (df['max_future_return'] >= min_return).astype(int)

        # 移除未来数据列
        future_cols = [col for col in df.columns if 'future_return' in col or col == 'max_future_return']
        df = df.drop(columns=future_cols)

        # 移除无法计算的样本
        df = df.dropna(subset=['label'])

        return df

    def extract_features_and_labels(self, processed_datasets: Dict) -> Tuple:
        """提取特征和标签"""
        print("=" * 80)
        print("【步骤5】提取特征和标签")
        print("=" * 80)

        # 定义排除列
        exclude_cols = [
            'open', 'high', 'low', 'close', 'vol', 'amount',
            'pre_close', 'pct_chg', 'label'
        ]

        # 获取特征列
        train_df = processed_datasets['训练集']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        # 检查未来数据列
        future_keywords = ['future', 'return_', 'max_return']
        for col in feature_cols:
            if any(keyword in col.lower() for keyword in future_keywords):
                raise ValueError(f"发现未来数据列: {col}，必须完全移除！")

        # 提取特征矩阵
        X_train = processed_datasets['训练集'][feature_cols].values
        y_train = processed_datasets['训练集']['label'].values

        X_val = processed_datasets['验证集'][feature_cols].values
        y_val = processed_datasets['验证集']['label'].values

        X_test = processed_datasets['测试集'][feature_cols].values
        y_test = processed_datasets['测试集']['label'].values

        # [优化1] 内存优化
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_test = X_test.astype(np.float32)

        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_test = y_test.astype(np.int32)

        print(f"✓ 特征和标签提取完成:")
        print(f"  特征数: {len(feature_cols)}")
        print(f"  训练集: {X_train.shape}, 正样本: {y_train.sum()} ({y_train.mean():.2%})")
        print(f"  验证集: {X_val.shape}, 正样本: {y_val.sum()} ({y_val.mean():.2%})")
        print(f"  测试集: {X_test.shape}, 正样本: {y_test.sum()} ({y_test.mean():.2%})")
        print()

        self.feature_names = feature_cols

        # [优化2] 清理原始DataFrame，释放内存
        del processed_datasets
        self._force_gc()

        return X_train, y_train, X_val, y_val, X_test, y_test

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """定义 Optuna 参数搜索空间"""
        params = {
            # 学习率
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),

            # 树结构
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),

            # 采样
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

            # 正则化
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5),

            # 样本权重（[改进] 移除 scale_pos_weight，避免"宁滥勿缺"）
            # 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),

            # 模型参数
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'random_state': 42,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',

            # [优化3] 内存优化参数
            'tree_method': self.config['model']['tree_method'],
            'max_bin': self.config['model']['max_bin'],
            'use_label_encoder': self.config['model']['use_label_encoder']
        }

        return params

    def objective(self, trial: optuna.Trial,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Optuna 优化目标函数
        [改进] 使用 AUC 作为优化目标，而不是 F1
        """
        try:
            # 从试验中采样参数
            params = self.define_search_space(trial)

            # 创建模型
            model = xgb.XGBClassifier(**params)

            # 训练模型
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 预测
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # [改进] 使用 AUC 作为优化目标
            metric = self.config['optuna']['metric']

            if metric == 'auc':
                # 优化 AUC（分离度）
                return roc_auc_score(y_val, y_pred_proba)
            elif metric == 'f1':
                # 优化 F1（需要先找到最优阈值）
                best_f1 = 0
                for threshold in np.arange(0.15, 0.45, 0.01):
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    if y_pred.sum() > 0:
                        f1 = f1_score(y_val, y_pred)
                        if f1 > best_f1:
                            best_f1 = f1
                return best_f1
            elif metric == 'precision':
                # 优化精确率
                best_precision = 0
                for threshold in np.arange(0.15, 0.45, 0.01):
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    if y_pred.sum() > 0:
                        precision = precision_score(y_val, y_pred)
                        if precision > best_precision:
                            best_precision = precision
                return best_precision
            else:
                # 默认优化 AUC
                return roc_auc_score(y_val, y_pred_proba)

        except Exception as e:
            # [优化2] 清理临时对象
            try:
                del model
                del y_pred_proba
            except:
                pass
            self._force_gc()

            print(f"  ⚠️  试验失败: {e}")
            return -1e6

    def run_optuna_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray):
        """运行 Optuna 优化"""
        print("=" * 80)
        print("【步骤6】Optuna 自动化参数调优")
        print("=" * 80)

        print(f"优化配置:")
        print(f"  试验次数: {self.config['optuna']['n_trials']}")
        print(f"  超时时间: {self.config['optuna']['timeout']}秒")
        print(f"  优化指标: {self.config['optuna']['metric']}")
        print(f"  优化方向: {self.config['optuna']['direction']}")
        print()

        # 创建研究
        self.study = optuna.create_study(direction=self.config['optuna']['direction'])

        # 定义进度回调
        def progress_callback(study, trial):
            if trial.number % 5 == 0 or trial.number == self.config['optuna']['n_trials'] - 1:
                print(f"  试验 {trial.number + 1}/{self.config['optuna']['n_trials']}: "
                      f"最佳分数 = {study.best_value:.4f}")

        # 运行优化
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.config['optuna']['n_trials'],
            timeout=self.config['optuna']['timeout'],
            callbacks=[progress_callback],
            show_progress_bar=False
        )

        # 输出最优结果
        print(f"\n✓ 优化完成！")
        print(f"  最优分数: {self.study.best_value:.4f}")
        print(f"  最优参数:")
        for key, value in self.study.best_params.items():
            print(f"    {key}: {value}")
        print()

        # 训练最优模型
        self.best_params = self.study.best_params.copy()

        # 创建并训练最优模型
        self.best_model = xgb.XGBClassifier(**self.best_params)

        # 训练模型
        self.best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # 特征重要性
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("✓ Top 10 特征重要性:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        print()

    def optimize_threshold(self, y_val: np.ndarray, y_val_pred_proba: np.ndarray) -> float:
        """
        优化决策阈值
        [改进6] 硬性约束精确率≥60%
        """
        print("=" * 80)
        print("【步骤7】优化决策阈值（硬性约束精确率≥60%）")
        print("=" * 80)

        target_precision = self.config['threshold']['target_precision']
        threshold_range = self.config['threshold']['threshold_range']
        threshold_step = self.config['threshold']['threshold_step']

        thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_step)
        valid_thresholds = []

        print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1':<10} {'预测数'}")
        print("-" * 60)

        for threshold in thresholds:
            y_pred = (y_val_pred_proba >= threshold).astype(int)

            if y_pred.sum() > 0:
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                # [改进6] 只收集满足精确率要求的阈值
                if precision >= target_precision:
                    valid_thresholds.append({
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })

                # 打印部分阈值
                if abs(threshold - 0.25) < 0.02 or (valid_thresholds and abs(threshold - valid_thresholds[-1]['threshold']) < 0.01):
                    print(f"{threshold:.2f}     {precision:.2%}      {recall:.2%}      {f1:.3f}     {y_pred.sum()}")

        # 在满足精确率的前提下，选择召回率最高的阈值
        if valid_thresholds:
            best_threshold_info = max(valid_thresholds, key=lambda x: x['recall'])
            best_threshold = best_threshold_info['threshold']
            print(f"\n✓ 最优阈值: {best_threshold:.2f}")
            print(f"  精确率: {best_threshold_info['precision']:.2%} (目标: {target_precision:.0%})")
            print(f"  召回率: {best_threshold_info['recall']:.2%}")
            print(f"  F1分数: {best_threshold_info['f1']:.3f}")
        else:
            print(f"\n⚠️  警告：没有任何阈值能达到 {target_precision:.0%} 精确率")
            print(f"  将使用默认阈值: {threshold_range[1]}")
            best_threshold = threshold_range[1]

        print()
        return best_threshold

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                       threshold: float = 0.5) -> Dict:
        """评估模型性能"""
        print("=" * 80)
        print("【步骤8】模型性能评估（测试集）")
        print("=" * 80)

        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'n_predictions': y_pred.sum(),
            'n_positive_samples': y_test.sum()
        }

        cm = confusion_matrix(y_test, y_pred)

        print(f"✓ 模型性能指标（阈值={threshold:.2f})")
        print(f"  准确率: {metrics['accuracy']:.2%}")
        print(f"  精确率: {metrics['precision']:.2%}")
        print(f"  召回率: {metrics['recall']:.2%}")
        print(f"  F1分数: {metrics['f1']:.3f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"\n✓ 混淆矩阵:")
        print(f"  预测为负（实际负样本: {cm[0, 0]}, 实际正样本: {cm[1, 0]})")
        print(f"  预测为正（实际负样本: {cm[0, 1]}, 实际正样本: {cm[1, 1]})")
        print(f"\n✓ 预测统计:")
        print(f"  预测为正的样本数: {metrics['n_predictions']}")
        print(f"  实际正样本数: {metrics['n_positive_samples']}")
        print()

        return metrics

    def save_model(self, threshold: float, metrics: Dict):
        """保存模型和配置"""
        print("=" * 80)
        print("【步骤9】保存模型和配置")
        print("=" * 80)

        # 创建模型目录
        model_dir = project_root / 'assets' / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        model_path = model_dir / 'auto_tuned_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'feature_names': self.feature_names
            }, f)

        print(f"✓ 模型已保存: {model_path}")

        # 保存元数据
        metadata = {
            'model_name': 'auto_tuned',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_count': len(self.feature_names),
            'best_params': self.best_params,
            'decision_threshold': float(threshold),
            'metrics': self._convert_numpy_to_python(metrics),
            'config': self.config,
            'optimization': {
                'n_trials': self.config['optuna']['n_trials'],
                'best_score': float(self.study.best_value) if self.study else None,
                'metric': self.config['optuna']['metric']
            }
        }

        metadata_path = model_dir / 'auto_tuned_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ 元数据已保存: {metadata_path}")
        print()

        # 保存 Optuna 研究结果
        if self.study:
            study_path = model_dir / 'optuna_study.pkl'
            with open(study_path, 'wb') as f:
                pickle.dump(self.study, f)
            print(f"✓ Optuna 研究已保存: {study_path}")
            print()

    def train_full_pipeline(self):
        """完整训练流程"""
        print("\n" + "=" * 80)
        print("自动化参数调优训练流程（DeepQuant V5.0）")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 1. 采集数据
        df_all = self.collect_stock_data(
            self.config['data']['start_date'],
            self.config['data']['end_date']
        )

        if df_all is None:
            raise ValueError("数据采集失败")

        # 2. 划分数据集
        train_df, val_df, test_df = self.split_data(df_all)

        # 3. 创建特征和标签
        processed_datasets = self.create_features_and_labels(train_df, val_df, test_df)

        # 4. 提取特征和标签
        X_train, y_train, X_val, y_val, X_test, y_test = self.extract_features_and_labels(
            processed_datasets
        )

        # 5. Optuna 自动化参数调优
        self.run_optuna_optimization(X_train, y_train, X_val, y_val)

        # 6. 优化阈值
        y_val_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
        optimal_threshold = self.optimize_threshold(y_val, y_val_pred_proba)

        # 7. 评估模型
        metrics = self.evaluate_model(X_test, y_test, optimal_threshold)

        # 8. 保存模型
        self.save_model(optimal_threshold, metrics)

        print("=" * 80)
        print("✓ 训练完成！")
        print("=" * 80)
        print(f"\n模型文件: assets/models/auto_tuned_model.pkl")
        print(f"测试集精确率: {metrics['precision']:.2%}")
        print(f"测试集AUC: {metrics['auc']:.4f}")
        print(f"测试集F1: {metrics['f1']:.3f}")
        print(f"测试集召回率: {metrics['recall']:.2%}")
        print(f"\n最优参数已自动搜索并保存！")
        print()

        return self.best_model, optimal_threshold


def main():
    """主函数"""
    try:
        trainer = AutoTunedTrainer()
        model, threshold = trainer.train_full_pipeline()

        print("\n✨ 自动化参数调优训练完成！")
        print(f"   最优阈值: {threshold:.2f}")
        print(f"   最优参数: {trainer.best_params}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
