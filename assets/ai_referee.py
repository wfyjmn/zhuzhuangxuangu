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

    def _init_model(self):
        """初始化模型"""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost分类器
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                # [关键修复4] 添加 scale_pos_weight 处理不平衡样本
                'scale_pos_weight': 1.0  # 将在 train() 中动态设置
            }
            params.update(self.model_params)

            self.model = xgb.XGBClassifier(**params)

        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # LightGBM分类器
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1,
                # [关键修复4] 添加 is_unbalance 处理不平衡样本
                'is_unbalance': True
            }
            params.update(self.model_params)

            self.model = lgb.LGBMClassifier(**params)

        else:
            # 使用简单的逻辑回归作为后备
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # 处理不平衡样本
            )
            print(f"[警告] 使用 LogisticRegression 作为后备模型")

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

    def cross_validate(self, X: pd.DataFrame, Y: pd.Series, cv: int = 5) -> Dict:
        """
        [关键修复2] 时序交叉验证

        Args:
            X: 特征DataFrame（必须包含 trade_date 列）
            Y: 标签Series
            cv: 折数

        Returns:
            交叉验证结果
        """
        print(f"\n[交叉验证] {cv}折时序交叉验证")

        X_features = self.prepare_features(X)

        # [关键修复2] 使用 TimeSeriesSplit 进行时序交叉验证
        # [关键修复1] 删除标准化
        # X_scaled = self.scaler.fit_transform(X_features)

        if 'trade_date' in X.columns:
            # 使用时序交叉验证
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = cross_val_score(self.model, X_features, Y, cv=tscv, scoring='accuracy')
        else:
            print(f"  [警告] 缺少 trade_date 列，使用普通交叉验证")
            scores = cross_val_score(self.model, X_features, Y, cv=cv, scoring='accuracy')

        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }

        print(f"  平均准确率: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")

        return results


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

    # 训练模型
    referee.train(X, Y)

    # 预测
    test_X = X[:10]
    probabilities = referee.predict(test_X)

    print(f"\n[测试] 预测结果（前10个样本）:")
    for i, prob in enumerate(probabilities):
        label = Y.iloc[i]
        print(f"  样本 {i+1}: 概率={prob:.4f}, 真实标签={label}")

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
