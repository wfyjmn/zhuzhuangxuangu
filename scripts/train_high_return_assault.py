#!/usr/bin/env python3
"""
高涨幅精准突击训练脚本
核心目标：
1. 预测3-5天涨幅>=8%（高涨幅）
2. 精准筛选（精确率>70%）
3. 数量少但必中（每天精选5-10只）
4. 高回报率（平均收益>6%）
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, log_loss, confusion_matrix
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# 添加src到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.assault_features import AssaultFeatureEngineer
from stock_system.triple_confirmation import TripleConfirmation
import xgboost as xgb


class HighReturnAssaultTrainer:
    """高涨幅精准突击训练器"""

    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.feature_engineer = AssaultFeatureEngineer(config_path)
        self.model = None
        self.feature_names = None

    def _load_config(self) -> Dict:
        """加载配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def create_high_return_labels(self, df: pd.DataFrame, 
                                lookforward_days: int = 3,
                                min_return: float = 0.08) -> pd.DataFrame:
        """
        创建高涨幅标签
        
        Args:
            df: 股票数据
            lookforward_days: 预测天数（3-5天）
            min_return: 最小涨幅阈值（8%）
        
        Returns:
            添加了标签的DataFrame
        """
        df = df.copy()
        
        # 计算未来3-5天的涨幅
        for days in [3, 4, 5]:
            df[f'future_return_{days}d'] = df['close'].pct_change(days).shift(-days)
        
        # 标签：3-5天内任意一天涨幅>=8%
        df['max_future_return'] = df[[f'future_return_{days}d' for days in [3, 4, 5]]].max(axis=1)
        df['label'] = (df['max_future_return'] >= min_return).astype(int)
        
        # 额外的盈利目标标签
        df['target_return'] = df['max_future_return']  # 目标是最大化收益
        
        # 移除无法计算的数据
        df = df.dropna(subset=['label', 'max_future_return'])
        
        print(f"标签创建完成:")
        print(f"  总样本数: {len(df)}")
        print(f"  正样本数（涨幅≥{min_return*100:.0f}%）: {df['label'].sum()} ({df['label'].mean():.2%})")
        print(f"  正样本平均涨幅: {df[df['label']==1]['max_future_return'].mean()*100:.2f}%")
        
        return df

    def generate_mock_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        生成模拟股票数据（更贴近高涨幅场景）
        """
        np.random.seed(42)
        
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # 模拟更剧烈的波动（高涨幅场景）
        returns = np.random.normal(0.003, 0.025, n_samples)  # 略高的波动
        prices = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': np.roll(prices, 1) * (1 + np.random.normal(0, 0.015, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_samples))),
            'volume': np.random.lognormal(11, 0.6, n_samples)  # 成交量
        })
        
        df.loc[0, 'open'] = df.loc[0, 'close']
        
        # 模拟资金流向数据
        df['main_net_inflow'] = np.random.normal(0, 0.5, n_samples)  # 主力净流入（百万）
        df['big_buy_amount'] = np.random.exponential(50, n_samples)  # 大单买入额（百万）
        df['total_amount'] = df['close'] * df['volume'] / 10000  # 成交额（万）
        
        print(f"✓ 生成了{len(df)}条模拟数据")
        
        return df.set_index('date')

    def split_data(self, df: pd.DataFrame, 
                   test_size: float = 0.2,
                   val_size: float = 0.15) -> Tuple:
        """划分数据集"""
        n = len(df)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val
        
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train+n_val]
        test_df = df.iloc[n_train+n_val:]
        
        print(f"✓ 数据集划分:")
        print(f"  训练集: {len(train_df)}条 ({len(train_df)/n:.1%})")
        print(f"  验证集: {len(val_df)}条 ({len(val_df)/n:.1%})")
        print(f"  测试集: {len(test_df)}条 ({len(test_df)/n:.1%})")
        
        return train_df, val_df, test_df

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray):
        """训练XGBoost模型（精准优化）"""
        
        print("\n" + "=" * 70)
        print("【模型训练】高涨幅精准突击模型")
        print("=" * 70)
        
        # 精准优化参数（更保守，避免假阳性）
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss', 'error'],
            'learning_rate': 0.01,      # 更小的学习率，更精准
            'max_depth': 4,             # 更浅的树，避免过拟合
            'min_child_weight': 10,      # 更高的最小子节点权重
            'subsample': 0.7,           # 采样比例
            'colsample_bytree': 0.7,    # 特征采样比例
            'reg_lambda': 5,             # 更强的L2正则化
            'reg_alpha': 1,              # 更强的L1正则化
            'gamma': 1,                  # 更高的分裂阈值
            'scale_pos_weight': 2,      # 正样本权重（针对高涨幅样本）
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': 42
        }
        
        print(f"模型参数（精准优化）:")
        for key, value in model_params.items():
            print(f"  {key}: {value}")
        print()
        
        self.model = xgb.XGBClassifier(**model_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100
        )
        
        print("\n✓ 模型训练完成")
        return self.model

    def optimize_threshold(self, y_true: np.ndarray, 
                        y_pred_proba: np.ndarray,
                        target_precision: float = 0.70) -> float:
        """
        优化决策阈值（以精确率为目标）
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            target_precision: 目标准确率（70%）
        
        Returns:
            最优阈值
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        print(f"\n【阈值优化】目标精确率: {target_precision:.0%}")
        print("-" * 70)
        print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
        print("-" * 70)
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if precision >= target_precision:
                # 在满足精确率的前提下，最大化F1
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            if threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                print(f"{threshold:.2f}     {precision:.2%}      {recall:.2%}      {f1:.4f}")
        
        print(f"\n最优阈值: {best_threshold:.2f} (精确率={target_precision:.0%})")
        
        return best_threshold

    def evaluate_model(self, y_true: np.ndarray,
                     y_pred_proba: np.ndarray,
                     threshold: float = 0.5,
                     dataset_name: str = "测试集") -> Dict:
        """评估模型性能"""
        
        print(f"\n【{dataset_name}性能评估】")
        print("-" * 70)
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 基础指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        # 精准指标
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 预测为正的样本中，实际为正的比例（关键指标）
        predicted_positive = tp + fp
        if predicted_positive > 0:
            hit_rate = tp / predicted_positive
        else:
            hit_rate = 0
        
        print(f"精确率（Precision）:    {precision:.4f} ({'✓' if precision >= 0.70 else '✗'})")
        print(f"召回率（Recall）:       {recall:.4f}")
        print(f"F1分数:               {f1:.4f}")
        print(f"AUC:                  {auc:.4f}")
        print(f"\n混淆矩阵:")
        print(f"  TN={tn:4d}  FP={fp:4d}")
        print(f"  FN={fn:4d}  TP={tp:4d}")
        print(f"\n预测为正的样本: {predicted_positive}个")
        print(f"实际为正的样本: {tp}个")
        print(f"命中率（Hit Rate）:   {hit_rate:.2%}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'hit_rate': hit_rate,
            'predicted_positive': int(predicted_positive),
            'true_positive': int(tp)
        }

    def train_full_pipeline(self):
        """完整的训练流程"""
        print("=" * 70)
        print("高涨幅精准突击训练系统")
        print("=" * 70)
        print("核心目标：")
        print("  1. 预测3-5天涨幅≥8%（高涨幅）")
        print("  2. 精准筛选（精确率>70%）")
        print("  3. 数量少但必中（每天精选5-10只）")
        print("  4. 高回报率（平均收益>6%）")
        print("=" * 70)
        
        # 1. 生成数据
        print("\n【步骤1】生成模拟数据")
        print("-" * 70)
        df = self.generate_mock_data(n_samples=10000)
        
        # 2. 创建标签（高涨幅）
        print("\n【步骤2】创建高涨幅标签")
        print("-" * 70)
        df_labeled = self.create_high_return_labels(df, lookforward_days=3, min_return=0.08)
        
        # 3. 特征工程
        print("\n【步骤3】特征工程")
        print("-" * 70)
        df_with_features = self.feature_engineer.create_all_features(df_labeled)
        self.feature_names = self.feature_engineer.get_feature_names()
        
        print(f"生成特征数量: {len(self.feature_names)}")
        
        # 4. 划分数据集
        print("\n【步骤4】划分数据集")
        print("-" * 70)
        train_df, val_df, test_df = self.split_data(df_with_features)
        
        # 5. 准备训练数据
        print("\n【步骤5】准备训练数据")
        print("-" * 70)
        
        X_train = train_df[self.feature_names].values
        y_train = train_df['label'].values
        X_val = val_df[self.feature_names].values
        y_val = val_df['label'].values
        X_test = test_df[self.feature_names].values
        y_test = test_df['label'].values
        
        print(f"训练集形状: {X_train.shape}")
        print(f"验证集形状: {X_val.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        # 6. 训练模型
        model = self.train_model(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # 7. 预测
        print("\n【步骤7】生成预测")
        print("-" * 70)
        
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 8. 优化阈值
        optimal_threshold = self.optimize_threshold(y_val, val_pred_proba, target_precision=0.70)
        
        # 9. 评估
        print("\n" + "=" * 70)
        print("模型性能评估（阈值={:.2f})".format(optimal_threshold))
        print("=" * 70)
        
        train_metrics = self.evaluate_model(y_train, train_pred_proba, optimal_threshold, "训练集")
        val_metrics = self.evaluate_model(y_val, val_pred_proba, optimal_threshold, "验证集")
        test_metrics = self.evaluate_model(y_test, test_pred_proba, optimal_threshold, "测试集")
        
        # 10. 保存模型
        print("\n【步骤10】保存模型")
        print("-" * 70)
        
        model_dir = Path(workspace_path) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = model_dir / "high_return_assault_model.pkl"
        pickle.dump(model, open(model_path, 'wb'))
        print(f"✓ 模型已保存: {model_path}")
        
        # 保存特征名称
        feature_path = model_dir / "high_return_assault_features.pkl"
        pickle.dump(self.feature_names, open(feature_path, 'wb'))
        print(f"✓ 特征名称已保存: {feature_path}")
        
        # 保存元数据
        metadata = {
            'model_name': '高涨幅精准突击模型',
            'version': '1.0',
            'target_return': 0.08,  # 8%目标涨幅
            'prediction_horizon': [3, 4, 5],  # 3-5天预测
            'optimal_threshold': optimal_threshold,
            'target_precision': 0.70,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'config': model.get_params()
        }
        
        metadata_path = model_dir / "high_return_assault_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {metadata_path}")
        
        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        
        return model, optimal_threshold


def main():
    """主函数"""
    trainer = HighReturnAssaultTrainer()
    model, threshold = trainer.train_full_pipeline()
    
    print("\n✨ 模型已成功训练并保存！")
    print(f"   最优阈值: {threshold:.2f}")
    print(f"   目标精确率: 70%")
    print(f"   预测周期: 3-5天")
    print(f"   目标涨幅: ≥8%")


if __name__ == '__main__':
    main()
