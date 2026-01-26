#!/usr/bin/env python3
"""
完全自动化参数调优训练脚本
使用 Optuna 贝叶斯优化自动搜索最优超参数
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import json
import pickle
import warnings
from datetime import datetime
import optuna
import akshare as ak
import xgboost as xgb

warnings.filterwarnings('ignore')

# 添加src和scripts到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))
sys.path.insert(0, os.path.join(workspace_path, "scripts"))

from stock_system.enhanced_features import EnhancedFeatureEngineer


class AutoTunedTrainer:
    """自动化参数调优训练器"""

    def __init__(self, config_path: str = "config/precision_priority_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.feature_engineer = EnhancedFeatureEngineer(config_path)
        self.best_model = None
        self.best_params = None
        self.feature_names = None
        self.study = None

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

    def _load_config(self) -> Dict:
        """加载配置"""
        config = {
            "data": {
                "start_date": "2020-01-01",
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "min_return_threshold": 0.04,
                "prediction_days": [3, 4, 5],
                "n_stocks": 150
            },
            "optuna": {
                "n_trials": 50,  # 优化试验次数
                "timeout": 3600,  # 最长优化时间（秒）
                "direction": "maximize",  # 优化方向
                "metric": "f1",  # 优化指标: f1, precision, recall, auc
                "cv_folds": 3,  # 交叉验证折数
                "early_stopping_rounds": 50
            },
            "threshold": {
                "target_precision": 0.60,
                "threshold_range": [0.15, 0.45],
                "threshold_step": 0.01
            }
        }

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    external_config = json.load(f)
                config.update(external_config)
            except Exception as e:
                print(f"⚠ 加载外部配置失败，使用默认配置: {e}")

        return config

    def get_stock_list(self, n_stocks: int = 150) -> List[str]:
        """获取股票列表"""
        print("=" * 70)
        print(f"【步骤1】获取{n_stocks}只股票列表")
        print("=" * 70)

        try:
            stock_list = ak.stock_info_a_code_name()
            print(f"✓ 获取到 {len(stock_list)} 只A股股票")

            # 过滤股票
            stock_list = stock_list[~stock_list['name'].str.contains('ST|退|暂停', na=False)]
            print(f"✓ 过滤ST、退市后: {len(stock_list)} 只")

            stock_list = stock_list.head(n_stocks * 2)
            print(f"✓ 选取前 {len(stock_list)} 只股票用于数据采集")
            print()

            return stock_list['code'].tolist()[:n_stocks]

        except Exception as e:
            print(f"❌ 获取股票列表失败: {e}")
            return []

    def collect_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """采集单只股票历史数据"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"
            )

            df['stock_code'] = stock_code
            df['date'] = pd.to_datetime(df['日期'])
            df = df.drop(columns=['日期'])
            df = df.set_index('date')

            # 重命名列
            df = df.rename(columns={
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate'
            })

            return df

        except Exception as e:
            print(f"  ⚠ {stock_code} 数据获取失败: {e}")
            return None

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间序列划分数据集"""
        print("=" * 70)
        print("【步骤2】时间序列划分数据集")
        print("=" * 70)

        df = df.sort_index()

        n = len(df)
        n_train = int(n * 0.60)
        n_val = int(n * 0.20)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()

        print(f"✓ 数据集划分（按时间）:")
        print(f"  训练集: {len(train_df)}条 ({train_df.index.min()} 至 {train_df.index.max()})")
        print(f"  验证集: {len(val_df)}条 ({val_df.index.min()} 至 {val_df.index.max()})")
        print(f"  测试集: {len(test_df)}条 ({test_df.index.min()} 至 {test_df.index.max()})")
        print()

        return train_df, val_df, test_df

    def create_features_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算特征"""
        print("=" * 70)
        print("【步骤3】计算特征（增强版）")
        print("=" * 70)

        required_cols = ['open', 'high', 'low', 'close', 'volume', 'stock_code']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"数据缺少必要列: {col}")

        df_features = self.feature_engineer.create_all_features(df)

        print(f"✓ 特征计算完成:")
        print(f"  样本数: {len(df_features)}")
        print(f"  特征数: {len(self.feature_engineer.get_feature_names())}")
        print()

        return df_features

    def create_labels_separately(self, train_df: pd.DataFrame,
                                   val_df: pd.DataFrame,
                                   test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """创建标签（目标：≥4%，3-5天）"""
        print("=" * 70)
        print("【步骤4】创建标签（目标：≥4%，3-5天）")
        print("=" * 70)

        min_return = self.config['data']['min_return_threshold']
        prediction_days = self.config['data']['prediction_days']

        for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
            # 计算未来3-5天的涨幅
            for days in prediction_days:
                df[f'future_return_{days}d'] = df.groupby('stock_code')['close'].pct_change(days).shift(-days)

            # 标签：3-5天内任意一天涨幅≥4%
            future_returns = [f'future_return_{days}d' for days in prediction_days]
            df['max_future_return'] = df[future_returns].max(axis=1)
            df['label'] = (df['max_future_return'] >= min_return).astype(int)

            # 移除所有未来数据列
            future_cols = [col for col in df.columns if 'future_return' in col or col == 'max_future_return']
            df.drop(columns=future_cols, inplace=True)

            # 移除无法计算的样本
            df.dropna(subset=['label'], inplace=True)

            print(f"{name}:")
            print(f"  样本数: {len(df)}")
            print(f"  正样本（≥{min_return * 100:.0f}%）: {df['label'].sum()} ({df['label'].mean():.2%})")

        print()
        return train_df, val_df, test_df

    def extract_features_and_labels(self, train_df: pd.DataFrame,
                                      val_df: pd.DataFrame,
                                      test_df: pd.DataFrame) -> Tuple:
        """提取特征和标签"""
        print("=" * 70)
        print("【步骤5】提取特征和标签")
        print("=" * 70)

        try:
            feature_names = self.feature_engineer.get_feature_names()
        except Exception as e:
            print(f"ERROR: 获取feature_names失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            exclude_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'amount', 'amplitude', 'pct_change', 'change_amount',
                'turnover_rate', 'stock_code', '股票代码',
                'future_return', 'max_future_return', 'label',
                'date', 'returns', 'daily_return', '日期'
            ]

            available_features = [col for col in feature_names if col not in exclude_cols]
            existing_features = [col for col in available_features if col in train_df.columns]

            # 检查未来数据列
            future_keywords = ['future', 'return_', 'max_return']
            for col in existing_features:
                if any(keyword in col.lower() for keyword in future_keywords):
                    raise ValueError(f"发现未来数据列: {col}，必须完全移除！")

            print(f"DEBUG: 开始提取特征矩阵...")
            X_train = train_df[existing_features].values
            X_val = val_df[existing_features].values
            X_test = test_df[existing_features].values

            y_train = train_df['label'].values
            y_val = val_df['label'].values
            y_test = test_df['label'].values

            print(f"✓ 特征和标签提取完成:")
            print(f"  特征数: {len(existing_features)}")
            print(f"  训练集: {X_train.shape}, 正样本: {y_train.sum()} ({y_train.mean():.2%})")
            print(f"  验证集: {X_val.shape}, 正样本: {y_val.sum()} ({y_val.mean():.2%})")
            print(f"  测试集: {X_test.shape}, 正样本: {y_test.sum()} ({y_test.mean():.2%})")
            print()

        except Exception as e:
            print(f"ERROR: 提取特征和标签失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        self.feature_names = available_features
        return X_train, y_train, X_val, y_val, X_test, y_test

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        定义 Optuna 参数搜索空间
        """
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5, log=True),  # 修复：low 必须 > 0 当 log=True
            'gamma': trial.suggest_float('gamma', 0.0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'random_state': 42,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'tree_method': 'hist'
        }

        return params

    def objective(self, trial: optuna.Trial,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Optuna 优化目标函数
        使用验证集评估模型性能
        """
        try:
            # 从试验中采样参数
            params = self.define_search_space(trial)

            # 计算样本权重
            scale_pos_weight = params['scale_pos_weight']
            sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

            # 创建模型
            model = xgb.XGBClassifier(**params)

            # 训练模型（XGBoost 3.x 使用 early_stopping_rounds 参数）
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 预测
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # 优化目标（从配置中读取）
            metric = self.config['optuna']['metric']

            if metric == 'f1':
                # 优化 F1 分数（需要先找到最优阈值）
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

            elif metric == 'auc':
                # 优化 AUC
                return roc_auc_score(y_val, y_pred_proba)

            else:
                # 默认优化 F1
                best_f1 = 0
                for threshold in np.arange(0.15, 0.45, 0.01):
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    if y_pred.sum() > 0:
                        f1 = f1_score(y_val, y_pred)
                        if f1 > best_f1:
                            best_f1 = f1
                return best_f1

        except Exception as e:
            # 返回极小值作为惩罚
            print(f"  ⚠ 试验失败: {e}")
            return -1e6

    def optimize_threshold(self, y_val: np.ndarray, y_val_pred_proba: np.ndarray) -> float:
        """优化决策阈值"""
        print("=" * 70)
        print("【步骤7】优化决策阈值（目标精确率≥60%）")
        print("=" * 70)

        target_precision = self.config['threshold']['target_precision']
        threshold_range = self.config['threshold']['threshold_range']
        threshold_step = self.config['threshold']['threshold_step']

        thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_step)
        best_threshold = threshold_range[0]
        best_f1 = 0

        print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1':<10} {'预测数'}")
        print("-" * 60)

        for threshold in thresholds:
            y_pred = (y_val_pred_proba >= threshold).astype(int)

            if y_pred.sum() > 0:
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                if precision >= target_precision and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

                # 打印部分阈值
                if abs(threshold - 0.25) < 0.02 or abs(threshold - best_threshold) < 0.01:
                    print(f"{threshold:.2f}     {precision:.2%}      {recall:.2%}      {f1:.3f}     {y_pred.sum()}")

        print(f"\n✓ 最优阈值: {best_threshold:.2f} (精确率目标: {target_precision:.0%})")
        print()
        return best_threshold

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                       threshold: float = 0.5) -> Dict:
        """评估模型性能"""
        print("=" * 70)
        print("【步骤8】模型性能评估（测试集）")
        print("=" * 70)

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
        print("=" * 70)
        print("【步骤9】保存模型和配置")
        print("=" * 70)

        # 创建模型目录
        model_dir = os.path.join(workspace_path, "assets", "models")
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(model_dir, "auto_tuned_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'feature_names': self.feature_names,
                'feature_engineer': self.feature_engineer
            }, f)

        print(f"✓ 模型已保存: {model_path}")

        # 保存元数据（转换 numpy 类型为原生 Python 类型）
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

        metadata_path = os.path.join(model_dir, "auto_tuned_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ 元数据已保存: {metadata_path}")
        print()

        # 保存 Optuna 研究结果
        if self.study:
            study_path = os.path.join(model_dir, "optuna_study.pkl")
            with open(study_path, 'wb') as f:
                pickle.dump(self.study, f)
            print(f"✓ Optuna 研究已保存: {study_path}")
            print()

    def run_optuna_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray):
        """运行 Optuna 优化"""
        print("=" * 70)
        print("【步骤6】Optuna 自动化参数调优")
        print("=" * 70)

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
            if trial.number % 10 == 0 or trial.number == self.config['optuna']['n_trials'] - 1:
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

        # 计算样本权重
        scale_pos_weight = self.best_params['scale_pos_weight']
        sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

        # 创建并训练最优模型
        self.best_model = xgb.XGBClassifier(**self.best_params)

        # 训练模型
        self.best_model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
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

    def train_full_pipeline(self):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("自动化参数调优训练流程")
        print("=" * 70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 1. 获取股票列表
        stock_codes = self.get_stock_list(self.config['data']['n_stocks'])
        if not stock_codes:
            raise ValueError("未能获取股票列表")

        # 2. 采集数据
        print("=" * 70)
        print("【步骤2】采集股票历史数据")
        print("=" * 70)

        all_data = []
        for i, stock_code in enumerate(stock_codes):
            if (i + 1) % 20 == 0:
                print(f"  进度: {i + 1}/{len(stock_codes)}")

            df = self.collect_stock_data(
                stock_code,
                self.config['data']['start_date'],
                self.config['data']['end_date']
            )

            if df is not None and len(df) > 0:
                all_data.append(df)

        if not all_data:
            raise ValueError("未能采集到任何数据")

        df_all = pd.concat(all_data, ignore_index=False)
        df_all = df_all.sort_index()

        print(f"\n✓ 数据采集完成:")
        print(f"  总样本数: {len(df_all)}")
        print(f"  股票数: {df_all['stock_code'].nunique()}")
        print(f"  时间范围: {df_all.index.min()} 至 {df_all.index.max()}")
        print()

        # 3. 划分数据集
        train_df, val_df, test_df = self.split_data(df_all)

        # 4. 计算特征
        train_df_features = self.create_features_only(train_df)
        val_df_features = self.create_features_only(val_df)
        test_df_features = self.create_features_only(test_df)

        # 5. 创建标签
        train_df_labeled, val_df_labeled, test_df_labeled = self.create_labels_separately(
            train_df_features, val_df_features, test_df_features
        )

        # 6. 提取特征和标签
        X_train, y_train, X_val, y_val, X_test, y_test = self.extract_features_and_labels(
            train_df_labeled, val_df_labeled, test_df_labeled
        )

        # 7. Optuna 自动化参数调优
        self.run_optuna_optimization(X_train, y_train, X_val, y_val)

        # 8. 优化阈值
        y_val_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
        optimal_threshold = self.optimize_threshold(y_val, y_val_pred_proba)

        # 9. 评估模型
        metrics = self.evaluate_model(X_test, y_test, optimal_threshold)

        # 10. 保存模型
        self.save_model(optimal_threshold, metrics)

        print("=" * 70)
        print("✓ 训练完成！")
        print("=" * 70)
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
