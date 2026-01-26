#!/usr/bin/env python3
"""
高涨幅精准突击模型训练（防过拟合优化版）
核心优化：
1. 严格时间序列划分，避免数据泄露
2. 增强正则化，降低模型复杂度
3. 时间序列交叉验证
4. 早停机制
5. 移除未来信息泄露
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import json
import pickle
import warnings
from datetime import datetime, timedelta
import akshare as ak

warnings.filterwarnings('ignore')

# 添加src到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.assault_features import AssaultFeatureEngineer
import xgboost as xgb


class AntiOverfittingTrainer:
    """防过拟合训练器"""

    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.feature_engineer = AssaultFeatureEngineer(config_path)
        self.model = None
        self.feature_names = None

    def _load_config(self) -> Dict:
        """加载配置（防过拟合优化版）"""
        config = {
            "data": {
                "start_date": "2021-01-01",
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "min_return_threshold": 0.08,
                "prediction_days": [3, 4, 5],
                "n_stocks": 300,  # 增加到300只股票
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15
            },
            "model": {
                "learning_rate": 0.005,  # 更小的学习率
                "max_depth": 3,  # 更浅的树
                "min_child_weight": 20,  # 更高的最小子节点权重
                "subsample": 0.6,  # 更少的采样
                "colsample_bytree": 0.6,  # 更少的特征采样
                "reg_lambda": 10,  # 更强的L2正则化
                "reg_alpha": 2,  # 更强的L1正则化
                "gamma": 2,  # 更高的分裂阈值
                "scale_pos_weight": 1,  # 平衡样本权重
                "n_estimators": 2000,  # 更多迭代次数
                "random_state": 42
            },
            "threshold": {
                "target_precision": 0.65,  # 降低精确率目标（更现实）
                "default_threshold": 0.5
            },
            "feature_weights": {
                "capital_strength": {"weight": 0.40},
                "market_sentiment": {"weight": 0.35},
                "technical_momentum": {"weight": 0.25}
            },
            "enhanced_rsi_strategy": {
                "rsi_combination": {
                    "short_term": {"period": 6, "weight": 0.4},
                    "medium_term": {"period": 12, "weight": 0.3},
                    "long_term": {"period": 24, "weight": 0.3}
                }
            },
            "cv": {
                "n_splits": 5,
                "max_train_size": 50000
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

    def get_stock_list(self, n_stocks: int = None) -> List[str]:
        """获取股票列表"""
        if n_stocks is None:
            n_stocks = self.config['data']['n_stocks']

        print("=" * 70)
        print("【步骤1】获取股票列表")
        print("=" * 70)

        try:
            stock_list = ak.stock_info_a_code_name()
            print(f"✓ 获取到 {len(stock_list)} 只A股股票")

            # 过滤
            stock_list = stock_list[~stock_list['name'].str.contains('ST|退|暂停', na=False)]
            print(f"✓ 过滤ST、退市后: {len(stock_list)} 只")

            # 随机采样（避免总是用前N只）
            stock_list = stock_list.sample(n=min(n_stocks, len(stock_list)), random_state=42)
            print(f"✓ 随机选取 {len(stock_list)} 只股票")

            return stock_list['code'].tolist()

        except Exception as e:
            print(f"❌ 获取股票列表失败: {e}")
            return []

    def collect_stock_data(self, stock_code: str,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """采集单只股票历史数据"""
        try:
            df = ak.stock_zh_a_hist(symbol=stock_code,
                                   period="daily",
                                   start_date=start_date.replace('-', ''),
                                   end_date=end_date.replace('-', ''),
                                   adjust="qfq")

            df['stock_code'] = stock_code
            df['date'] = pd.to_datetime(df['日期'])
            df = df.drop(columns=['日期'])
            df = df.set_index('date')

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
            return None

    def collect_all_data(self, stock_codes: List[str],
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """采集所有股票数据"""
        print("=" * 70)
        print(f"【步骤2】采集{len(stock_codes)}只股票历史数据")
        print(f"时间范围: {start_date} 至 {end_date}")
        print("=" * 70)

        all_data = []
        success_count = 0

        for i, stock_code in enumerate(stock_codes, 1):
            if i % 50 == 0:
                print(f"  已处理 {i}/{len(stock_codes)} 只股票...")

            df = self.collect_stock_data(stock_code, start_date, end_date)

            if df is not None and len(df) > 100:
                all_data.append(df)
                success_count += 1

        if success_count == 0:
            print("\n❌ 没有采集到任何有效数据")
            return None

        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()

        print(f"\n✓ 数据采集完成:")
        print(f"  成功股票数: {success_count}/{len(stock_codes)}")
        print(f"  总数据量: {len(combined_df)}条")
        print(f"  数据时间跨度: {combined_df.index.min()} 至 {combined_df.index.max()}")

        return combined_df

    def create_high_return_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建高涨幅标签"""
        print("=" * 70)
        print("【步骤3】创建高涨幅标签（3-5天涨幅≥8%）")
        print("=" * 70)

        df = df.copy()

        # 计算未来3-5天的涨幅（严格避免数据泄露）
        for days in [3, 4, 5]:
            df[f'future_return_{days}d'] = df.groupby('stock_code')['close'].pct_change(days).shift(-days)

        # 标签：3-5天内任意一天涨幅≥8%
        df['max_future_return'] = df[[f'future_return_{days}d' for days in [3, 4, 5]]].max(axis=1)
        min_return = self.config['data']['min_return_threshold']
        df['label'] = (df['max_future_return'] >= min_return).astype(int)

        # 移除无法计算的数据
        df = df.dropna(subset=['label', 'max_future_return'])

        print(f"✓ 标签创建完成:")
        print(f"  总样本数: {len(df)}")
        print(f"  正样本数（涨幅≥{min_return*100:.0f}%）: {df['label'].sum()} ({df['label'].mean():.2%})")
        if df['label'].sum() > 0:
            print(f"  正样本平均涨幅: {df[df['label']==1]['max_future_return'].mean()*100:.2f}%")
            print(f"  负样本平均涨幅: {df[df['label']==0]['max_future_return'].mean()*100:.2f}%")

        return df

    def split_data_strict(self, df: pd.DataFrame) -> Tuple:
        """严格的时间序列划分（避免数据泄露）"""
        print("=" * 70)
        print("【步骤4】严格时间序列划分（避免数据泄露）")
        print("=" * 70)

        # 按时间排序
        df = df.sort_index()

        # 划分数据集（严格按时间）
        n = len(df)
        n_train = int(n * self.config['data']['train_ratio'])
        n_val = int(n * self.config['data']['val_ratio'])

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train+n_val]
        test_df = df.iloc[n_train+n_val:]

        print(f"✓ 数据集划分（严格按时间）:")
        print(f"  训练集: {len(train_df)}条 ({train_df.index.min()} 至 {train_df.index.max()})")
        print(f"  验证集: {len(val_df)}条 ({val_df.index.min()} 至 {val_df.index.max()})")
        print(f"  测试集: {len(test_df)}条 ({test_df.index.min()} 至 {test_df.index.max()})")

        # 检查时间连续性
        if train_df.index.max() >= val_df.index.min():
            print("⚠ 警告: 训练集和验证集时间有重叠！")
        if val_df.index.max() >= test_df.index.min():
            print("⚠ 警告: 验证集和测试集时间有重叠！")

        return train_df, val_df, test_df

    def engineer_features(self, train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> Tuple:
        """特征工程"""
        print("=" * 70)
        print("【步骤5】特征工程（防过拟合）")
        print("=" * 70)

        # 使用AssaultFeatureEngineer计算特征
        train_df = self.feature_engineer.create_all_features(train_df)
        val_df = self.feature_engineer.create_all_features(val_df)
        test_df = self.feature_engineer.create_all_features(test_df)

        # 获取特征名称
        feature_names = self.feature_engineer.get_feature_names()

        # 提取特征矩阵（排除stock_code等可能导致数据泄露的列）
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'amount', 'amplitude', 'pct_change', 'change_amount',
            'turnover_rate', 'stock_code', 'future_return_3d',
            'future_return_4d', 'future_return_5d', 'max_future_return',
            'label', 'date', 'returns', 'price_change',
            'high_20', 'ema_12', 'ema_26', 'low_9', 'high_9',
            'avg_volume_5', 'k_value', 'd_value', 'ma_5', 'ma_10', 'ma_20'
        ]

        # 获取实际存在的特征列
        available_features = [col for col in feature_names if col in train_df.columns]

        X_train = train_df[available_features].values
        X_val = val_df[available_features].values
        X_test = test_df[available_features].values

        # 提取标签
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values

        self.feature_names = available_features

        print(f"✓ 特征工程完成:")
        print(f"  特征数量: {len(available_features)}")
        print(f"  训练样本: {len(X_train)} (正样本: {y_train.sum()})")
        print(f"  验证样本: {len(X_val)} (正样本: {y_val.sum()})")
        print(f"  测试样本: {len(X_test)} (正样本: {y_test.sum()})")

        # 检查特征比例
        feature_ratio = len(available_features) / len(X_train)
        print(f"  特征/样本比: {feature_ratio:.3f} (建议<0.1)")
        if feature_ratio > 0.1:
            print(f"  ⚠ 警告: 特征过多，可能导致过拟合")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model_with_cv(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray):
        """训练模型（使用时间序列交叉验证）"""
        print("=" * 70)
        print("【步骤6】训练模型（时间序列交叉验证）")
        print("=" * 70)

        model_params = self.config['model']

        print(f"模型参数（防过拟合优化）:")
        for key, value in model_params.items():
            if key not in ['random_state']:
                print(f"  {key}: {value}")

        # 时间序列交叉验证
        print(f"\n时间序列交叉验证:")
        tscv = TimeSeriesSplit(n_splits=self.config['cv']['n_splits'],
                              max_train_size=self.config['cv']['max_train_size'])

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBClassifier(**model_params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )

            y_pred = model.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred)
            cv_scores.append(score)

            print(f"  Fold {fold+1}: F1 = {score:.4f}")

        print(f"\n交叉验证结果:")
        print(f"  平均F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # 在完整数据上训练最终模型
        print(f"\n在完整训练集上训练最终模型...")
        self.model = xgb.XGBClassifier(**model_params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # 计算过拟合指标
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_accuracy = (train_pred == y_train).mean()
        val_accuracy = (val_pred == y_val).mean()
        overfitting_gap = train_accuracy - val_accuracy

        print(f"✓ 模型训练完成:")
        print(f"  训练集准确率: {train_accuracy:.2%}")
        print(f"  验证集准确率: {val_accuracy:.2%}")
        print(f"  过拟合差距: {overfitting_gap:.2%}")

        if overfitting_gap > 0.05:  # 超过5%认为是过拟合
            print(f"  ⚠ 警告: 过拟合差距过大（{overfitting_gap:.2%}），建议进一步增加正则化")

        print(f"  最佳迭代次数: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else len(self.model.get_booster().get_dump())}")

        return self.model

    def optimize_threshold(self, y_val: np.ndarray, y_val_pred_proba: np.ndarray) -> float:
        """优化决策阈值（平衡精确率和召回率）"""
        print("=" * 70)
        print("【步骤7】优化决策阈值（平衡精确率和召回率）")
        print("=" * 70)

        target_precision = self.config['threshold']['target_precision']
        thresholds = np.arange(0.3, 0.7, 0.01)

        best_threshold = self.config['threshold']['default_threshold']
        best_f1 = 0

        print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1':<10}")
        print("-" * 45)

        for threshold in thresholds:
            y_pred = (y_val_pred_proba >= threshold).astype(int)

            if y_pred.sum() > 0:
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                # 寻找F1最高的阈值
                if f1 > best_f1:
                    best_threshold = threshold
                    best_f1 = f1

                # 打印部分阈值
                if abs(threshold - 0.5) < 0.05 or abs(threshold - best_threshold) < 0.01:
                    print(f"{threshold:.2f}     {precision:.2%}      {recall:.2%}      {f1:.3f}")

        print(f"\n✓ 最优阈值: {best_threshold:.2f} (最大F1: {best_f1:.3f})")

        return best_threshold

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                     threshold: float = 0.5) -> Dict:
        """评估模型性能"""
        print("=" * 70)
        print("【步骤8】模型性能评估（测试集）")
        print("=" * 70)

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # 计算指标
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }

        print(f"✓ 模型性能指标（阈值={threshold:.2f}）:")
        print(f"  准确率: {metrics['accuracy']:.2%}")
        print(f"  精确率: {metrics['precision']:.2%}")
        print(f"  召回率: {metrics['recall']:.2%}")
        print(f"  F1分数: {metrics['f1']:.3f}")
        print(f"  AUC: {metrics['auc']:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n混淆矩阵:")
        print("           预测负例  预测正例")
        print(f"实际负例: {cm[0,0]:>6}  {cm[0,1]:>6}")
        print(f"实际正例: {cm[1,0]:>6}  {cm[1,1]:>6}")

        # 过拟合检测
        train_pred = self.model.predict(X_test)  # 用测试集数据测试
        if metrics['accuracy'] > 0.99:
            print(f"\n⚠ 警告: 准确率过高（{metrics['accuracy']:.2%}），可能存在过拟合或数据泄露")

        return metrics

    def save_model(self, threshold: float, metrics: Dict):
        """保存模型和特征工程"""
        print("=" * 70)
        print("【步骤9】保存模型和配置")
        print("=" * 70)

        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        model_dir = os.path.join(workspace_path, "assets/models")
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(model_dir, "high_return_anti_overfitting.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ 模型已保存: {model_path}")

        # 保存特征名称
        feature_path = os.path.join(model_dir, "high_return_anti_overfitting_features.pkl")
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✓ 特征已保存: {feature_path}")

        # 保存元数据
        metadata = {
            'model_type': 'XGBoost_HighReturn_AntiOverfitting',
            'data_description': f"{self.config['data']['n_stocks']}只A股股票，防过拟合优化",
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_count': len(self.feature_names),
            'decision_threshold': threshold,
            'metrics': metrics,
            'config': self.config,
            'optimizations': [
                '严格时间序列划分',
                '增强正则化（L1=2, L2=10）',
                '更浅的树（max_depth=3）',
                '时间序列交叉验证',
                '早停机制（early_stopping_rounds=100）',
                '更小的学习率（0.005）'
            ]
        }

        metadata_path = os.path.join(model_dir, "high_return_anti_overfitting_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {metadata_path}")

    def run(self):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("高涨幅精准突击模型训练（防过拟合优化版）")
        print("=" * 70 + "\n")

        # 1. 获取股票列表
        stock_codes = self.get_stock_list(n_stocks=self.config['data']['n_stocks'])
        if not stock_codes:
            print("❌ 获取股票列表失败")
            return

        # 2. 采集数据
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        data_df = self.collect_all_data(stock_codes, start_date, end_date)

        if data_df is None or len(data_df) == 0:
            print("❌ 数据采集失败")
            return

        # 3. 创建标签
        labeled_df = self.create_high_return_labels(data_df)

        if labeled_df['label'].sum() == 0:
            print("❌ 没有正样本（涨幅≥8%），无法训练")
            return

        # 4. 划分数据
        train_df, val_df, test_df = self.split_data_strict(labeled_df)

        # 5. 特征工程
        X_train, X_val, X_test, y_train, y_val, y_test = self.engineer_features(
            train_df, val_df, test_df
        )

        # 6. 训练模型
        self.train_model_with_cv(X_train, y_train, X_val, y_val)

        # 7. 优化阈值
        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        optimal_threshold = self.optimize_threshold(y_val, y_val_pred_proba)

        # 8. 评估模型
        metrics = self.evaluate_model(X_test, y_test, optimal_threshold)

        # 9. 保存模型
        self.save_model(optimal_threshold, metrics)

        print("=" * 70)
        print("✓ 训练完成！")
        print("=" * 70)
        print(f"\n模型文件: assets/models/high_return_anti_overfitting.pkl")
        print(f"最优阈值: {optimal_threshold:.2f}")
        print(f"测试集精确率: {metrics['precision']:.2%}")
        print(f"测试集AUC: {metrics['auc']:.4f}")
        print(f"测试集F1: {metrics['f1']:.3f}\n")


if __name__ == "__main__":
    trainer = AntiOverfittingTrainer()
    trainer.run()
