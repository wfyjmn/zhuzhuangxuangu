#!/usr/bin/env python3
"""
修复数据泄露的高涨幅模型训练脚本
关键修复：
1. 特征计算与标签分离
2. 严格的时间序列划分
3. 完全移除未来数据
4. 按股票分组处理
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import json
import pickle
import warnings
from datetime import datetime, timedelta
import akshare as ak

warnings.filterwarnings('ignore')

# 添加src和scripts到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))
sys.path.insert(0, os.path.join(workspace_path, "scripts"))

from stock_system.assault_features import AssaultFeatureEngineer
from data_leakage_detector import DataLeakageDetector
import xgboost as xgb


class AntiLeakageTrainer:
    """防数据泄露的高涨幅模型训练器"""

    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.feature_engineer = AssaultFeatureEngineer(config_path)
        self.model = None
        self.feature_names = None
        self.detector = DataLeakageDetector()

    def _load_config(self) -> Dict:
        """加载配置"""
        # 防泄露配置
        config = {
            "data": {
                "start_date": "2022-01-01",
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "min_return_threshold": 0.08,
                "prediction_days": [3, 4, 5]
            },
            "model": {
                "learning_rate": 0.005,  # 降低学习率
                "max_depth": 3,          # 减小深度
                "min_child_weight": 15,  # 增加最小子节点权重
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "reg_lambda": 10,        # 增强L2正则化
                "reg_alpha": 2,          # 增加L1正则化
                "gamma": 2,
                "scale_pos_weight": 2,
                "n_estimators": 500,
                "early_stopping_rounds": 30,
                "random_state": 42
            },
            "threshold": {
                "target_precision": 0.70,
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
            }
        }

        # 尝试加载外部配置文件（如果存在）
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    external_config = json.load(f)
                # 合并配置（外部配置覆盖默认配置）
                config.update(external_config)
            except Exception as e:
                print(f"⚠ 加载外部配置失败，使用默认配置: {e}")

        return config

    def get_stock_list(self, n_stocks: int = 100) -> List[str]:
        """获取股票列表"""
        print("=" * 70)
        print("【步骤1】获取股票列表")
        print("=" * 70)

        try:
            stock_list = ak.stock_info_a_code_name()
            print(f"✓ 获取到 {len(stock_list)} 只A股股票")

            # 过滤股票
            stock_list = stock_list[~stock_list['name'].str.contains('ST|退|暂停', na=False)]
            print(f"✓ 过滤ST、退市后: {len(stock_list)} 只")

            # 选择主板股票
            stock_list = stock_list.head(n_stocks * 2)

            print(f"✓ 选取前 {len(stock_list)} 只股票用于数据采集")
            print()

            return stock_list['code'].tolist()[:n_stocks]

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
            print(f"  ⚠ 获取{stock_code}数据失败: {e}")
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
            print(f"  [{i}/{len(stock_codes)}] 采集 {stock_code}...", end=" ")

            df = self.collect_stock_data(stock_code, start_date, end_date)

            if df is not None and len(df) > 100:
                all_data.append(df)
                success_count += 1
                print(f"✓ ({len(df)}条)")
            else:
                print("✗ 数据不足")

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
        print()

        return combined_df

    def split_data_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间划分数据集（在特征计算之前）

        关键修复：先划分数据，再计算特征和标签
        """
        print("=" * 70)
        print("【步骤3】按时间划分数据集（特征计算之前）")
        print("=" * 70)

        # 按时间划分：70%训练，15%验证，15%测试
        n = len(df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train+n_val].copy()
        test_df = df.iloc[n_train+n_val:].copy()

        print(f"✓ 数据集划分（按时间）:")
        print(f"  训练集: {len(train_df)}条 ({train_df.index.min()} 至 {train_df.index.max()})")
        print(f"  验证集: {len(val_df)}条 ({val_df.index.min()} 至 {val_df.index.max()})")
        print(f"  测试集: {len(test_df)}条 ({test_df.index.min()} 至 {test_df.index.max()})")
        print()

        return train_df, val_df, test_df

    def create_features_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        只计算特征（不包含任何未来数据）

        关键修复：特征计算与标签完全分离
        """
        print("=" * 70)
        print("【步骤4】计算特征（仅使用历史数据）")
        print("=" * 70)

        # 确保原始数据中只包含历史列
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'stock_code']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"数据缺少必要列: {col}")

        # 使用特征工程计算特征
        df_features = self.feature_engineer.create_all_features(df)

        print(f"✓ 特征计算完成（未使用任何未来数据）:")
        print(f"  样本数: {len(df_features)}")
        print(f"  特征数: {len(self.feature_engineer.get_feature_names())}")
        print()

        return df_features

    def create_labels_separately(self, train_df: pd.DataFrame,
                                val_df: pd.DataFrame,
                                test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分别创建标签（在特征计算之后）

        关键修复：标签计算与特征计算完全分离
        """
        print("=" * 70)
        print("【步骤5】创建标签（在特征计算之后）")
        print("=" * 70)

        for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
            # 计算未来3-5天的涨幅
            for days in [3, 4, 5]:
                df[f'future_return_{days}d'] = df.groupby('stock_code')['close'].pct_change(days).shift(-days)

            # 标签：3-5天内任意一天涨幅≥8%
            df['max_future_return'] = df[[f'future_return_{days}d' for days in [3, 4, 5]]].max(axis=1)
            min_return = self.config['data']['min_return_threshold']
            df['label'] = (df['max_future_return'] >= min_return).astype(int)

            # 移除所有未来数据列（关键修复！）
            future_cols = [col for col in df.columns if 'future_return' in col or col == 'max_future_return']
            df.drop(columns=future_cols, inplace=True)

            # 移除无法计算的样本
            df.dropna(subset=['label'], inplace=True)

            print(f"{name}:")
            print(f"  样本数: {len(df)}")
            print(f"  正样本: {df['label'].sum()} ({df['label'].mean():.2%})")

        print()
        return train_df, val_df, test_df

    def extract_features_and_labels(self, train_df: pd.DataFrame,
                                   val_df: pd.DataFrame,
                                   test_df: pd.DataFrame) -> Tuple:
        """
        提取特征和标签（完全移除未来数据）

        关键修复：严格移除所有未来数据列
        """
        print("=" * 70)
        print("【步骤6】提取特征和标签")
        print("=" * 70)

        # 获取特征名称
        feature_names = self.feature_engineer.get_feature_names()

        # 严格排除所有包含未来信息的列
        exclude_cols = [
            # 原始数据列
            'open', 'high', 'low', 'close', 'volume',
            'amount', 'amplitude', 'pct_change', 'change_amount',
            'turnover_rate', 'stock_code',
            # 未来数据列（完全移除）
            'future_return_3d', 'future_return_4d', 'future_return_5d',
            'max_future_return',
            # 标签列
            'label',
            # 临时计算列
            'date', 'returns', 'price_change',
            # 其他非特征列
            'high_20', 'ema_12', 'ema_26', 'low_9', 'high_9',
            'avg_volume_5', 'k_value', 'd_value', 'ma_5', 'ma_10', 'ma_20',
            'macd', 'macd_signal', 'macd_hist', 'macd_golden_cross',
            'kdj_golden_cross', 'price_breakout_20', 'volume_ratio_5',
            'ma_bullish_arrangement', 'volatility_20'
        ]

        # 获取实际存在的特征列
        available_features = [col for col in feature_names if col in train_df.columns]

        # 再次检查：确保没有未来数据列
        future_keywords = ['future', 'return_', 'max_return']
        for col in available_features:
            if any(keyword in col.lower() for keyword in future_keywords):
                raise ValueError(f"发现未来数据列: {col}，必须完全移除！")

        # 提取特征矩阵
        X_train = train_df[available_features].values
        X_val = val_df[available_features].values
        X_test = test_df[available_features].values

        # 提取标签
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values

        self.feature_names = available_features

        print(f"✓ 特征和标签提取完成:")
        print(f"  特征数量: {len(available_features)}")
        print(f"  训练样本: {len(X_train)} (正样本: {y_train.sum()})")
        print(f"  验证样本: {len(X_val)} (正样本: {y_val.sum()})")
        print(f"  测试样本: {len(X_test)} (正样本: {y_test.sum()})")
        print()

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray):
        """训练模型（防过拟合配置）"""
        print("=" * 70)
        print("【步骤7】训练防泄露模型")
        print("=" * 70)

        model_params = self.config['model']

        print(f"模型参数（防过拟合配置）:")
        for key, value in model_params.items():
            if key not in ['random_state']:
                print(f"  {key}: {value}")
        print()

        self.model = xgb.XGBClassifier(**model_params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # 打印训练结果
        try:
            results = self.model.evals_result()
            available_metrics = list(results['validation_0'].keys())
            if available_metrics:
                metric_name = available_metrics[0]
                train_score = results['validation_0'][metric_name][-1]
                val_score = results['validation_1'][metric_name][-1]
                overfitting_gap = train_score - val_score

                print(f"✓ 模型训练完成:")
                print(f"  训练集{metric_name}: {train_score:.4f}")
                print(f"  验证集{metric_name}: {val_score:.4f}")
                print(f"  过拟合差距: {overfitting_gap:.4f}")

                # 检测过拟合
                if overfitting_gap > 0.05:
                    print(f"  ⚠️ 警告：过拟合差距 > 0.05")
                elif overfitting_gap < 0:
                    print(f"  ✓ 过拟合差距为负（模型欠拟合或欠优化）")
                else:
                    print(f"  ✓ 过拟合差距在合理范围内")

            else:
                print(f"✓ 模型训练完成")
        except Exception as e:
            print(f"✓ 模型训练完成 (无法获取训练日志: {e})")

        print()

        return self.model

    def optimize_threshold(self, y_val: np.ndarray, y_val_pred_proba: np.ndarray) -> float:
        """优化决策阈值（以精确率为目标）"""
        print("=" * 70)
        print("【步骤8】优化决策阈值（目标精确率≥70%）")
        print("=" * 70)

        target_precision = self.config['threshold']['target_precision']
        thresholds = np.arange(0.3, 0.8, 0.01)

        best_threshold = self.config['threshold']['default_threshold']
        best_precision = 0

        print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1':<10}")
        print("-" * 45)

        for threshold in thresholds:
            y_pred = (y_val_pred_proba >= threshold).astype(int)

            if y_pred.sum() > 0:
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                if precision >= target_precision and f1 > best_precision:
                    best_threshold = threshold
                    best_precision = f1

                # 打印部分阈值
                if abs(threshold - 0.5) < 0.05 or abs(threshold - best_threshold) < 0.01:
                    print(f"{threshold:.2f}     {precision:.2%}      {recall:.2%}      {f1:.3f}")

        print(f"\n✓ 最优阈值: {best_threshold:.2f} (精确率目标: {target_precision:.0%})")
        print()

        return best_threshold

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                     threshold: float = 0.5) -> Dict:
        """评估模型性能"""
        print("=" * 70)
        print("【步骤9】模型性能评估（测试集）")
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

        # 检测过拟合信号
        print(f"\n过拟合检测:")
        if metrics['precision'] >= 0.95:
            print(f"  ⚠️ 警告：精确率 {metrics['precision']:.2%} 过高，可能存在数据泄露")
        elif metrics['precision'] >= 0.85:
            print(f"  ⚠️ 警告：精确率 {metrics['precision']:.2%} 偏高，需要进一步检查")
        else:
            print(f"  ✓ 精确率 {metrics['precision']:.2%} 在合理范围内")

        if metrics['auc'] >= 0.95:
            print(f"  ⚠️ 警告：AUC {metrics['auc']:.4f} 过高，可能存在数据泄露")
        elif metrics['auc'] >= 0.85:
            print(f"  ⚠️ 警告：AUC {metrics['auc']:.4f} 偏高，需要进一步检查")
        else:
            print(f"  ✓ AUC {metrics['auc']:.4f} 在合理范围内")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n混淆矩阵:")
        print(f"           预测负例  预测正例")
        print(f"实际负例: {cm[0,0]:>6}  {cm[0,1]:>6}")
        print(f"实际正例: {cm[1,0]:>6}  {cm[1,1]:>6}")

        # 计算预测样本的平均涨幅
        pred_positive = y_pred == 1
        if pred_positive.sum() > 0:
            avg_return = y_test[pred_positive].mean()
            print(f"\n✓ 预测为正的样本平均标签值: {avg_return:.4f}")
        print()

        return metrics

    def save_model(self, threshold: float, metrics: Dict):
        """保存模型和配置"""
        print("=" * 70)
        print("【步骤10】保存模型和配置")
        print("=" * 70)

        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        model_dir = os.path.join(workspace_path, "assets/models")
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(model_dir, "anti_leakage_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ 模型已保存: {model_path}")

        # 保存特征名称
        feature_path = os.path.join(model_dir, "anti_leakage_features.pkl")
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✓ 特征已保存: {feature_path}")

        # 保存元数据
        metadata = {
            'model_type': 'XGBoost_AntiLeakage',
            'data_description': '防数据泄露版本',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_count': len(self.feature_names),
            'decision_threshold': threshold,
            'metrics': metrics,
            'config': self.config,
            'leakage_prevention': {
                'features_computed_before_split': False,
                'future_columns_removed': True,
                'temporal_split': True
            }
        }

        metadata_path = os.path.join(model_dir, "anti_leakage_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {metadata_path}")
        print()

    def run(self):
        """完整训练流程（防数据泄露）"""
        print("\n" + "=" * 70)
        print("防数据泄露高涨幅模型训练")
        print("=" * 70 + "\n")

        # 1. 获取股票列表
        stock_codes = self.get_stock_list(n_stocks=100)
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

        # 3. 【关键修复】先划分数据（在特征计算之前）
        train_df, val_df, test_df = self.split_data_by_time(data_df)

        # 4. 计算特征（只使用历史数据）
        train_df = self.create_features_only(train_df)
        val_df = self.create_features_only(val_df)
        test_df = self.create_features_only(test_df)

        # 5. 创建标签（在特征计算之后，且立即移除未来数据列）
        train_df, val_df, test_df = self.create_labels_separately(train_df, val_df, test_df)

        # 6. 提取特征和标签（严格移除所有未来数据列）
        X_train, X_val, X_test, y_train, y_val, y_test = self.extract_features_and_labels(
            train_df, val_df, test_df
        )

        # 7. 训练模型（防过拟合配置）
        self.train_model(X_train, y_train, X_val, y_val)

        # 8. 优化阈值
        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        optimal_threshold = self.optimize_threshold(y_val, y_val_pred_proba)

        # 9. 评估模型
        metrics = self.evaluate_model(X_test, y_test, optimal_threshold)

        # 10. 保存模型
        self.save_model(optimal_threshold, metrics)

        print("=" * 70)
        print("✓ 训练完成！")
        print("=" * 70)
        print(f"\n模型文件: assets/models/anti_leakage_model.pkl")
        print(f"最优阈值: {optimal_threshold:.2f}")
        print(f"测试集精确率: {metrics['precision']:.2%}")
        print(f"测试集AUC: {metrics['auc']:.4f}\n")


if __name__ == "__main__":
    trainer = AntiLeakageTrainer()
    trainer.run()
