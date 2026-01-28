# -*- coding: utf-8 -*-
"""
DeepQuant AI裁判 (AI Referee)
功能：
1. 使用XGBoost/LightGBM训练分类器
2. 预测股票未来5天的盈利概率
3. 替代传统的线性评分规则
4. 支持模型保存和加载

核心能力：
- 二分类：盈利（1）/ 亏损（0）
- 输出概率：Probability（0~1）
- 可解释性：特征重要性分析
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from feature_extractor import FeatureExtractor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 尝试导入XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[警告] XGBoost 未安装，将使用其他模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[警告] LightGBM 未安装，将使用其他模型")


class AIReferee:
    """AI裁判类"""

    def __init__(self, model_type: str = 'xgboost', model_params: Dict = None):
        """
        初始化AI裁判

        Args:
            model_type: 模型类型（xgboost/lightgbm）
            model_params: 模型参数
        """
        self.model_type = model_type
        self.model = None
        # [关键修复1] 删除 StandardScaler，树模型不需要标准化
        self.feature_names = None
        self.model_params = model_params or {}

        # 初始化模型
        self._init_model()

        # 训练历史
        self.training_history = {}

    def _get_default_params(self) -> Dict:
        """获取默认参数（优化版）"""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 200,  # 增加树的数量
                'max_depth': 5,       # 降低深度，避免过拟合
                'learning_rate': 0.05,  # 降低学习率，更稳健
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                # [关键修复4] 添加 scale_pos_weight 处理不平衡样本
                'scale_pos_weight': 1.0  # 将在训练时动态设置
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1,
                # [关键修复4] 添加 is_unbalance 处理不平衡样本
                'is_unbalance': True
            }
        else:
            return {}

    def _get_model_instance(self, params_override: Dict = None):
        """
        获取模型实例（动态创建）

        Args:
            params_override: 参数覆盖

        Returns:
            模型实例
        """
        params = self._get_default_params().copy()
        if self.model_params:
            params.update(self.model_params)
        if params_override:
            params.update(params_override)

        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(**params)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(**params)
        else:
            # 使用简单的逻辑回归作为后备
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # 处理不平衡样本
            )

    def _init_model(self):
        """初始化模型（保留向后兼容）"""
        self.model = self._get_model_instance()

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据

        Args:
            X: 原始特征DataFrame

        Returns:
            处理后的特征DataFrame
        """
        # 移除非特征列
        feature_cols = [col for col in X.columns if col not in ['ts_code', 'trade_date']]
        X_features = X[feature_cols].copy()

        # [关键修复3] 不做 fillna(0)，保留 NaN
        # XGBoost 和 LightGBM 原生支持缺失值，会自动学习缺失值的含义
        # 简单粗暴填 0 会引入噪音

        # 记录特征名称
        self.feature_names = feature_cols

        return X_features

    def train(self, X: pd.DataFrame, Y: pd.Series, validation_split: float = 0.2):
        """
        训练模型

        Args:
            X: 特征DataFrame（必须包含 trade_date 列用于时序切分）
            Y: 标签Series
            validation_split: 验证集比例（取最后 N% 的数据作为验证集）
        """
        print(f"\n[AI裁判] 开始训练模型")
        print(f"  模型类型: {self.model_type}")
        print(f"  训练样本: {len(X)}")
        print(f"  验证比例: {validation_split}")

        # 准备特征
        X_features = self.prepare_features(X)

        # [关键修复2] 使用时序切分，避免数据泄露
        # 训练集必须在时间上早于验证集
        if 'trade_date' in X.columns:
            # 按时间排序
            X_sorted = X.sort_values('trade_date').reset_index(drop=True)
            Y_sorted = Y.loc[X_sorted.index].reset_index(drop=True)

            # 计算切分点（最后 N% 作为验证集）
            split_idx = int(len(X_sorted) * (1 - validation_split))

            X_train = X_features.loc[:split_idx].reset_index(drop=True)
            X_val = X_features.loc[split_idx:].reset_index(drop=True)
            y_train = Y_sorted.loc[:split_idx].reset_index(drop=True)
            y_val = Y_sorted.loc[split_idx:].reset_index(drop=True)

            print(f"  [时序切分] 训练集: {X_train['trade_date'].min()} ~ {X_train['trade_date'].max()}")
            print(f"  [时序切分] 验证集: {X_val['trade_date'].min()} ~ {X_val['trade_date'].max()}")

            # 从特征中移除 trade_date 列（只用于切分，不用于训练）
            X_train = X_train.drop(columns=['trade_date'])
            X_val = X_val.drop(columns=['trade_date'])
        else:
            print(f"  [警告] 缺少 trade_date 列，使用随机切分（可能导致数据泄露）")
            X_train, X_val, y_train, y_val = train_test_split(
                X_features, Y,
                test_size=validation_split,
                random_state=42,
                stratify=Y
            )

        # [关键修复4] 计算正负样本权重比
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        print(f"  [样本统计] 正样本: {pos_count}, 负样本: {neg_count}")
        print(f"  [样本权重] scale_pos_weight: {scale_pos_weight:.2f}")

        # 更新 XGBoost 的 scale_pos_weight
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model.set_params(scale_pos_weight=scale_pos_weight)

        # [关键修复1] 删除特征标准化，树模型不需要
        # X_train_scaled = self.scaler.fit_transform(X_train)
        # X_val_scaled = self.scaler.transform(X_val)

        # 训练模型
        print(f"  开始训练...")
        self.model.fit(X_train, y_train)

        # 验证模型
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]

        # 计算评估指标
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)

        # 记录训练历史
        self.training_history = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'pos_samples': int(pos_count),
            'neg_samples': int(neg_count),
            'scale_pos_weight': scale_pos_weight,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'feature_count': len(self.feature_names)
        }

        print(f"\n  [训练完成] 评估指标:")
        print(f"    准确率（Accuracy）: {accuracy:.4f}")
        print(f"    精确率（Precision）: {precision:.4f}")
        print(f"    召回率（Recall）: {recall:.4f}")
        print(f"    F1分数: {f1:.4f}")
        print(f"    AUC分数: {auc:.4f}")

        # 打印混淆矩阵
        cm = confusion_matrix(y_val, y_pred)
        print(f"\n    混淆矩阵:")
        print(f"      {cm}")

    def train_time_series(self, X: pd.DataFrame, Y: pd.Series, n_splits: int = 5):
        """
        [新增] 时序交叉验证训练（推荐使用）

        使用 TimeSeriesSplit 进行多折时序交叉验证，比单次切分更稳健。
        保留最后一个 Fold 的模型，因为它看过的历史数据最多。

        Args:
            X: 特征DataFrame（必须包含 trade_date 列）
            Y: 标签Series
            n_splits: 交叉验证折数

        Returns:
            交叉验证结果（DataFrame）
        """
        print(f"\n{'='*80}")
        print(f"[AI裁判] 时序交叉验证训练 (n_splits={n_splits})")
        print(f"{'='*80}")

        # 准备特征
        X_features = self.prepare_features(X)

        # [关键修复2] 使用时序切分，避免数据泄露
        if 'trade_date' not in X.columns:
            raise ValueError("缺少 trade_date 列，无法进行时序交叉验证")

        # 按时间排序
        X_sorted = X.sort_values('trade_date').reset_index(drop=True)

        # 确保 Y 是 pandas.Series
        if isinstance(Y, np.ndarray):
            Y = pd.Series(Y, index=X.index)
        Y_sorted = Y.loc[X_sorted.index].reset_index(drop=True)

        # 从特征中移除 trade_date 列（只用于切分，不用于训练）
        X_features_sorted = X_features.loc[X_sorted.index].reset_index(drop=True)

        # [关键修复4] 计算正负样本权重比（全局）
        pos_count = Y_sorted.sum()
        neg_count = len(Y_sorted) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        pos_ratio = pos_count / len(Y_sorted)

        print(f"[样本统计] 总样本: {len(Y_sorted)}, 正样本: {pos_count}, 负样本: {neg_count}")
        print(f"[样本平衡] 正样本占比: {pos_ratio:.1%}, scale_pos_weight: {scale_pos_weight:.2f}")

        # 使用 TimeSeriesSplit 进行时序交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_results = []
        best_model = None
        best_fold = None
        best_score = 0

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_features_sorted), 1):
            print(f"\n{'='*80}")
            print(f"[Fold {fold}/{n_splits}] 时序交叉验证")
            print(f"{'='*80}")

            # 切分数据
            X_train = X_features_sorted.iloc[train_idx]
            X_val = X_features_sorted.iloc[val_idx]
            y_train = Y_sorted.iloc[train_idx]
            y_val = Y_sorted.iloc[val_idx]

            # 计算时间范围
            train_date_start = X_sorted['trade_date'].iloc[train_idx].min()
            train_date_end = X_sorted['trade_date'].iloc[train_idx].max()
            val_date_start = X_sorted['trade_date'].iloc[val_idx].min()
            val_date_end = X_sorted['trade_date'].iloc[val_idx].max()

            print(f"[时间范围]")
            print(f"  训练集: {train_date_start} ~ {train_date_end} ({len(X_train)} 样本)")
            print(f"  验证集: {val_date_start} ~ {val_date_end} ({len(X_val)} 样本)")

            # [关键] 动态创建模型实例（每个 Fold 独立）
            if self.model_type == 'xgboost':
                model = self._get_model_instance({'scale_pos_weight': scale_pos_weight})
            else:
                model = self._get_model_instance()

            # 训练模型
            print(f"[训练中]...")
            model.fit(X_train, y_train)

            # 验证模型
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            # 计算评估指标
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_prob)

            # 记录 Fold 结果
            fold_result = {
                'fold': fold,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'train_date_start': str(train_date_start),
                'train_date_end': str(train_date_end),
                'val_date_start': str(val_date_start),
                'val_date_end': str(val_date_end)
            }
            fold_results.append(fold_result)

            # 打印评估指标
            print(f"[评估指标]")
            print(f"  准确率（Accuracy）: {accuracy:.4f}")
            print(f"  精确率（Precision）: {precision:.4f}")
            print(f"  召回率（Recall）: {recall:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  AUC分数: {auc:.4f}")

            # 打印混淆矩阵
            cm = confusion_matrix(y_val, y_pred)
            print(f"\n  混淆矩阵:")
            print(f"    TN={cm[0,0]:3d} | FP={cm[0,1]:3d}")
            print(f"    FN={cm[1,0]:3d} | TP={cm[1,1]:3d}")

            # 保留最后一个 Fold 的模型（因为它看过的历史数据最多）
            if fold == n_splits:
                best_model = model
                best_fold = fold
                best_score = auc

        # 保存最终模型
        self.model = best_model

        # 记录训练历史
        df_fold_results = pd.DataFrame(fold_results)

        # 计算平均指标
        avg_metrics = {
            'avg_accuracy': df_fold_results['accuracy'].mean(),
            'std_accuracy': df_fold_results['accuracy'].std(),
            'avg_precision': df_fold_results['precision'].mean(),
            'std_precision': df_fold_results['precision'].std(),
            'avg_recall': df_fold_results['recall'].mean(),
            'std_recall': df_fold_results['recall'].std(),
            'avg_f1': df_fold_results['f1_score'].mean(),
            'std_f1': df_fold_results['f1_score'].std(),
            'avg_auc': df_fold_results['auc_score'].mean(),
            'std_auc': df_fold_results['auc_score'].std(),
        }

        self.training_history = {
            'method': 'time_series_cv',
            'n_splits': n_splits,
            'total_samples': len(Y_sorted),
            'pos_samples': int(pos_count),
            'neg_samples': int(neg_count),
            'scale_pos_weight': scale_pos_weight,
            'best_fold': best_fold,
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'feature_count': len(self.feature_names)
        }

        # 打印汇总
        print(f"\n{'='*80}")
        print(f"[训练完成] 交叉验证汇总")
        print(f"{'='*80}")
        print(f"[平均指标]")
        print(f"  准确率（Accuracy）: {avg_metrics['avg_accuracy']:.4f} (+/- {avg_metrics['std_accuracy']:.4f})")
        print(f"  精确率（Precision）: {avg_metrics['avg_precision']:.4f} (+/- {avg_metrics['std_precision']:.4f})")
        print(f"  召回率（Recall）: {avg_metrics['avg_recall']:.4f} (+/- {avg_metrics['std_recall']:.4f})")
        print(f"  F1分数: {avg_metrics['avg_f1']:.4f} (+/- {avg_metrics['std_f1']:.4f})")
        print(f"  AUC分数: {avg_metrics['avg_auc']:.4f} (+/- {avg_metrics['std_auc']:.4f})")

        print(f"\n[模型选择]")
        print(f"  已保存第 {best_fold} Fold 的模型（看过的历史数据最多）")
        print(f"  AUC分数: {best_score:.4f}")

        # 打印详细结果
        print(f"\n[详细结果]")
        print(df_fold_results.to_string(index=False))

        return df_fold_results

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        预测股票盈利概率

        Args:
            X: 特征DataFrame

        Returns:
            预测概率Series（0~1）
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        # 准备特征
        X_features = self.prepare_features(X)

        # [关键修复1] 删除特征标准化
        # X_scaled = self.scaler.transform(X_features)

        # 预测概率
        y_prob = self.model.predict_proba(X_features)[:, 1]

        # 转换为Series
        prob_series = pd.Series(y_prob, index=X.index, name='probability')

        return prob_series

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性

        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练")

        # 获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            raise ValueError("模型不支持特征重要性分析")

        # 创建DataFrame
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })

        # 排序
        df_importance = df_importance.sort_values('importance', ascending=False)

        return df_importance

    def save_model(self, model_dir: str = "models"):
        """
        保存模型

        Args:
            model_dir: 模型目录
        """
        os.makedirs(model_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(model_dir, f"ai_referee_{self.model_type}_{timestamp}.pkl")

        # 保存模型、特征名称、训练历史
        # [关键修复1] 删除 scaler，树模型不需要标准化
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_history': self.training_history
        }

        joblib.dump(model_data, model_file)

        print(f"\n[保存] 模型已保存: {model_file}")

        return model_file

    def load_model(self, model_file: str):
        """
        加载模型

        Args:
            model_file: 模型文件路径
        """
        print(f"\n[加载] 正在加载模型: {model_file}")

        model_data = joblib.load(model_file)

        self.model = model_data['model']
        # [关键修复1] 删除 scaler，树模型不需要标准化
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        self.training_history = model_data['training_history']

        print(f"  模型类型: {self.model_type}")
        print(f"  特征数量: {len(self.feature_names)}")
        print(f"  训练历史: {self.training_history}")

        return self

    def cross_validate(self, X: pd.DataFrame, Y: pd.Series, cv: int = 5) -> pd.DataFrame:
        """
        [已废弃] 请使用 train_time_series() 方法进行时序交叉验证训练

        Args:
            X: 特征DataFrame（必须包含 trade_date 列）
            Y: 标签Series
            cv: 折数

        Returns:
            交叉验证结果（DataFrame）
        """
        print(f"\n[警告] cross_validate() 方法已废弃，请使用 train_time_series() 方法")
        print(f"        train_time_series() 提供更详细的交叉验证结果和模型保存功能\n")

        # 直接调用 train_time_series
        return self.train_time_series(X, Y, n_splits=cv)


def main():
    """测试函数"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant AI裁判")
    print(" " * 30 + "测试运行")
    print("="*80 + "\n")

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000

    features = {
        'vol_ratio': np.random.randn(n_samples) + 1.5,
        'turnover_rate': np.random.rand(n_samples) * 10,
        'pe_ttm': np.random.rand(n_samples) * 50 + 10,
        'bias_5': np.random.randn(n_samples) * 5,
        'bias_10': np.random.randn(n_samples) * 8,
        'bias_20': np.random.randn(n_samples) * 10,
        'pct_chg_5d': np.random.randn(n_samples) * 5,
        'pct_chg_10d': np.random.randn(n_samples) * 8,
        'pct_chg_20d': np.random.randn(n_samples) * 12,
        'ma5_slope': np.random.randn(n_samples) * 0.5,
        'ma10_slope': np.random.randn(n_samples) * 0.3,
        'ma20_slope': np.random.randn(n_samples) * 0.2,
        'rsi': np.random.rand(n_samples) * 100,
        'macd_dif': np.random.randn(n_samples) * 0.5,
        'macd_dea': np.random.randn(n_samples) * 0.3,
        'index_pct_chg': np.random.randn(n_samples) * 3,
        'sector_pct_chg': np.random.randn(n_samples) * 5,
        'moneyflow_score': np.random.rand(n_samples) * 100,
        'tech_score': np.random.rand(n_samples) * 100,
    }

    X = pd.DataFrame(features)
    # 模拟标签（根据部分特征生成）
    Y = ((features['bias_5'] > 0) & (features['rsi'] > 50)).astype(int)

    # 初始化AI裁判
    referee = AIReferee(model_type='xgboost')

    # 训练模型（单次切分）
    print("\n[测试1] 单次时序切分训练（保留向后兼容）")
    referee.train(X, Y)

    # 预测
    test_X = X[:10]
    probabilities = referee.predict(test_X)

    print(f"\n[测试1] 预测结果（前10个样本）:")
    for i, prob in enumerate(probabilities):
        label = Y.iloc[i]
        print(f"  样本 {i+1}: 概率={prob:.4f}, 真实标签={label}")

    # 特征重要性
    print(f"\n[测试1] 特征重要性 Top 10:")
    importance_df = referee.get_feature_importance()
    print(importance_df.head(10))

    # 保存模型
    model_file = referee.save_model()

    # 测试加载模型
    print(f"\n[测试1] 加载模型...")
    new_referee = AIReferee()
    new_referee.load_model(model_file)

    # 验证预测结果一致
    new_probabilities = new_referee.predict(test_X)
    print(f"  预测结果一致: {all(probabilities == new_probabilities)}")

    # ========== 测试2：时序交叉验证训练（推荐） ==========
    print(f"\n{'='*80}")
    print(f"[测试2] 时序交叉验证训练（推荐使用）")
    print(f"{'='*80}")

    # 创建带时间序列的模拟数据
    dates = pd.date_range('20230101', periods=n_samples)
    X_with_date = X.copy()
    X_with_date['trade_date'] = dates

    # 初始化新的AI裁判
    referee_ts = AIReferee(model_type='xgboost')

    # 使用时序交叉验证训练
    fold_results = referee_ts.train_time_series(X_with_date, Y, n_splits=5)

    # 预测
    probabilities_ts = referee_ts.predict(test_X)

    print(f"\n[测试2] 预测结果（前10个样本）:")
    for i, prob in enumerate(probabilities_ts):
        label = Y.iloc[i]
        print(f"  样本 {i+1}: 概率={prob:.4f}, 真实标签={label}")

    # 特征重要性
    print(f"\n[测试2] 特征重要性 Top 10:")
    importance_df_ts = referee_ts.get_feature_importance()
    print(importance_df_ts.head(10))

    # 保存模型
    model_file_ts = referee_ts.save_model()

    # 对比两种方法的结果
    print(f"\n{'='*80}")
    print(f"[对比] 单次切分 vs 时序交叉验证")
    print(f"{'='*80}")
    print(f"单次切分 AUC: {referee.training_history['auc_score']:.4f}")
    print(f"时序交叉 AUC: {referee_ts.training_history['avg_metrics']['avg_auc']:.4f} (+/- {referee_ts.training_history['avg_metrics']['std_auc']:.4f})")


    # 特征重要性
    print(f"\n[特征重要性] Top 10:")
    importance_df = referee.get_feature_importance()
    print(importance_df.head(10))

    # 保存模型
    model_file = referee.save_model()

    # 测试加载模型
    print(f"\n[测试] 加载模型...")
    new_referee = AIReferee()
    new_referee.load_model(model_file)

    # 验证预测结果一致
    new_probabilities = new_referee.predict(test_X)
    print(f"  预测结果一致: {all(probabilities == new_probabilities)}")

    print("\n[完成] AI裁判测试完成\n")


if __name__ == "__main__":
    main()
