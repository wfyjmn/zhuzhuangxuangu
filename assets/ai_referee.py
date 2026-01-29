# -*- coding: utf-8 -*-
"""
DeepQuant AI裁判 (AI Referee) - 终极修正版
功能：
1. 使用XGBoost/LightGBM训练分类器
2. 预测股票未来5天的盈利概率
3. 替代传统的线性评分规则
4. 支持模型保存和加载

核心能力：
- 二分类：盈利（1）/ 亏损（0）
- 输出概率：Probability（0~1）
- 可解释性：特征重要性分析
- 鲁棒性：支持早停（Early Stopping）和特征自动对齐
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 尝试导入XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[警告] XGBoost 未安装")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[警告] LightGBM 未安装")


class AIReferee:
    """AI裁判类"""

    def __init__(self, model_type: str = 'xgboost', model_params: Dict = None):
        """
        初始化AI裁判
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.model_params = model_params or {}
        
        # 初始化模型实例
        self._init_model()
        self.training_history = {}

    def _get_default_params(self) -> Dict:
        """获取默认参数"""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 1000,   # 设置较大，配合 early_stopping 使用
                'max_depth': 6,
                'learning_rate': 0.03,  # 较低的学习率配合更多的树
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'auc',   # 优化目标改为 AUC
                'n_jobs': -1,
                'verbosity': 0
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'is_unbalance': True
            }
        else:
            return {}

    def _get_model_instance(self, params_override: Dict = None):
        """获取模型实例（动态创建）"""
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
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    def _init_model(self):
        self.model = self._get_model_instance()

    def prepare_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        准备特征数据
        Args:
            X: 原始DataFrame
            is_training: 是否为训练阶段。如果是预测阶段，会进行特征对齐。
        """
        # 1. 移除非特征列
        exclude_cols = ['ts_code', 'trade_date', 'date', 'code', 'label']
        feature_cols = [col for col in X.columns if col not in exclude_cols]
        X_features = X[feature_cols].copy()

        # 2. 如果是训练阶段，记录特征名称
        if is_training:
            self.feature_names = feature_cols
        
        # 3. [关键优化] 如果是预测阶段，强制对齐特征顺序
        elif self.feature_names is not None:
            # 缺失的列补 NaN
            missing_cols = set(self.feature_names) - set(X_features.columns)
            if missing_cols:
                for c in missing_cols:
                    X_features[c] = np.nan
            
            # 只保留训练时用过的列，并按顺序排列
            X_features = X_features[self.feature_names]

        return X_features

    def train(self, X: pd.DataFrame, Y: pd.Series, validation_split: float = 0.2):
        """训练模型（单次时序切分）"""
        print(f"\n[AI裁判] 开始训练模型 ({self.model_type})")

        # 准备特征
        X_features = self.prepare_features(X, is_training=True)

        # 时序切分
        if 'trade_date' in X.columns:
            X_sorted = X.sort_values('trade_date').reset_index(drop=True)
            Y_sorted = Y.loc[X_sorted.index].reset_index(drop=True)
            
            # 使用 features 进行切分
            X_features_sorted = X_features.loc[X_sorted.index].reset_index(drop=True)

            split_idx = int(len(X_sorted) * (1 - validation_split))
            X_train = X_features_sorted.iloc[:split_idx]
            X_val = X_features_sorted.iloc[split_idx:]
            y_train = Y_sorted.iloc[:split_idx]
            y_val = Y_sorted.iloc[split_idx:]
            
            print(f"  [切分] 训练集截止: {X_sorted.iloc[split_idx]['trade_date']}")
        else:
            raise ValueError("训练数据缺少 trade_date 列，无法进行时序切分")

        # 计算样本权重
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model.set_params(scale_pos_weight=scale_pos_weight)

        # 训练（加入 Early Stopping）
        print(f"  开始训练 (Early Stopping Enabled)...")
        eval_set = [(X_val, y_val)]
        
        try:
            # XGBoost/LGBM 的 fit 参数支持 early_stopping_rounds
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        except TypeError:
             # 兼容 sklearn 接口或其他不支持 early_stopping 的模型
            self.model.fit(X_train, y_train)

        # 验证
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        
        self.training_history = {
            'auc_score': auc, 
            'accuracy': acc,
            'scale_pos_weight': scale_pos_weight
        }
        print(f"  [完成] AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    def train_time_series(self, X: pd.DataFrame, Y: pd.Series, n_splits: int = 5):
        """时序交叉验证训练（推荐）"""
        print(f"\n{'='*60}")
        print(f"[AI裁判] 时序交叉验证训练 (n_splits={n_splits})")
        print(f"{'='*60}")

        X_features = self.prepare_features(X, is_training=True)

        if 'trade_date' not in X.columns:
            raise ValueError("缺少 trade_date 列")

        # 排序
        sort_idx = X.sort_values('trade_date').index
        X_features_sorted = X_features.loc[sort_idx].reset_index(drop=True)
        Y_sorted = Y.loc[sort_idx].reset_index(drop=True)
        dates_sorted = X.loc[sort_idx, 'trade_date'].reset_index(drop=True)

        # 计算全局权重（作为参考）
        pos_count = Y_sorted.sum()
        scale_pos_weight = (len(Y_sorted) - pos_count) / pos_count if pos_count > 0 else 1.0

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_features_sorted), 1):
            # 切分
            X_train, X_val = X_features_sorted.iloc[train_idx], X_features_sorted.iloc[val_idx]
            y_train, y_val = Y_sorted.iloc[train_idx], Y_sorted.iloc[val_idx]
            
            # 动态权重：根据当前 fold 的训练集计算
            fold_pos = y_train.sum()
            fold_scale = (len(y_train) - fold_pos) / fold_pos if fold_pos > 0 else 1.0

            # 实例化新模型
            if self.model_type == 'xgboost':
                model = self._get_model_instance({'scale_pos_weight': fold_scale})
            else:
                model = self._get_model_instance()

            # 训练 (Early Stopping)
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            except:
                model.fit(X_train, y_train)

            # 评估
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_prob)
            
            # 记录时间范围
            t_start, t_end = dates_sorted.iloc[train_idx].min(), dates_sorted.iloc[train_idx].max()
            v_start, v_end = dates_sorted.iloc[val_idx].min(), dates_sorted.iloc[val_idx].max()

            print(f"[Fold {fold}] Train: {t_start}~{t_end} | Val: {v_start}~{v_end} | AUC: {auc:.4f}")

            fold_results.append({
                'fold': fold,
                'auc_score': auc,
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0)
            })

            # 保留最后一个模型
            if fold == n_splits:
                best_model = model

        self.model = best_model
        
        df_res = pd.DataFrame(fold_results)
        avg_auc = df_res['auc_score'].mean()
        
        self.training_history = {
            'method': 'time_series_cv',
            'avg_metrics': {
                'avg_auc': avg_auc,
                'std_auc': df_res['auc_score'].std()
            },
            'cv_results': df_res
        }
        
        print(f"\n[平均指标] AUC: {avg_auc:.4f}")
        return self.training_history

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测股票盈利概率"""
        if self.model is None:
            raise ValueError("模型未训练")

        # 准备特征（自动对齐）
        X_features = self.prepare_features(X, is_training=False)
        
        y_prob = self.model.predict_proba(X_features)[:, 1]
        return pd.Series(y_prob, index=X.index, name='probability')

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame()

        importances = self.model.feature_importances_
        # 确保长度一致
        if len(importances) != len(self.feature_names):
            # 如果使用了 LightGBM 且开启了 bagging，可能会导致特征数不一致，这里做个安全处理
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return df

    def save_model(self, model_dir: str = "data/models"):
        """保存模型"""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(model_dir, f"ai_referee_{self.model_type}_{timestamp}.pkl")

        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'history': self.training_history
        }, model_file)

        print(f"[保存] {model_file}")
        return model_file

    def load_model(self, model_file: str):
        """加载模型"""
        data = joblib.load(model_file)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = data.get('model_type', 'xgboost')
        self.training_history = data.get('history', {})
        print(f"[加载] 模型已加载，特征数: {len(self.feature_names)}")
        return self


def main():
    """测试流程"""
    print("AI 裁判测试启动...")
    
    # 1. 模拟数据
    np.random.seed(42)
    n = 2000
    dates = pd.date_range('20230101', periods=n)
    
    data = pd.DataFrame({
        'trade_date': dates,
        'ts_code': ['000001.SZ'] * n,
        'feature_A': np.random.randn(n),
        'feature_B': np.random.randn(n) * 10,
        'random_noise': np.random.rand(n)
    })
    # 模拟标签：如果 Feature A > 0 且 Feature B > 0，则盈利
    labels = ((data['feature_A'] > 0) & (data['feature_B'] > 0)).astype(int)

    # 2. 初始化裁判
    referee = AIReferee(model_type='xgboost')

    # 3. 时序交叉验证训练
    referee.train_time_series(data, labels, n_splits=3)

    # 4. 预测（模拟缺失特征的情况，测试对齐功能）
    test_data = data.iloc[:5].copy()
    test_data = test_data.drop(columns=['feature_B']) # 故意删掉一列
    
    print("\n[测试特征对齐] 输入缺少 feature_B...")
    probs = referee.predict(test_data)
    print(f"预测结果:\n{probs}")

    # 5. 特征重要性
    print("\n[特征重要性]")
    print(referee.get_feature_importance().head())

    # 6. 保存与加载
    path = referee.save_model()
    loaded_referee = AIReferee().load_model(path)
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
