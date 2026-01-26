#!/usr/bin/env python3
"""
使用100只股票3年真实数据训练高涨幅精准突击模型
核心目标：
1. 数据源：100只A股股票，3年历史数据
2. 预测目标：3-5天涨幅≥8%
3. 模型优化：高精准率（≥70%）、高回报（平均收益≥6%）
4. 输出：每天精选5-10只股票
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

# 添加src到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.assault_features import AssaultFeatureEngineer
from stock_system.triple_confirmation import TripleConfirmation
import xgboost as xgb


class RealDataHighReturnTrainer:
    """使用真实数据训练高涨幅模型"""

    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.feature_engineer = AssaultFeatureEngineer(config_path)
        self.model = None
        self.feature_names = None

    def _load_config(self) -> Dict:
        """加载配置"""
        # 内置默认配置
        config = {
            "data": {
                "start_date": "2022-01-01",  # 3年前
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "min_return_threshold": 0.08,
                "prediction_days": [3, 4, 5]
            },
            "model": {
                "learning_rate": 0.01,
                "max_depth": 4,
                "min_child_weight": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_lambda": 5,
                "reg_alpha": 1,
                "gamma": 1,
                "scale_pos_weight": 2,
                "n_estimators": 1000,
                "early_stopping_rounds": 50,
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
        """
        获取股票列表（排除ST、退市股票）
        
        Args:
            n_stocks: 股票数量
            
        Returns:
            股票代码列表
        """
        print("=" * 70)
        print("【步骤1】获取股票列表")
        print("=" * 70)
        
        try:
            # 获取A股所有股票
            stock_list = ak.stock_info_a_code_name()
            print(f"✓ 获取到 {len(stock_list)} 只A股股票")
            
            # 过滤股票
            stock_list = stock_list[~stock_list['name'].str.contains('ST|退|暂停', na=False)]
            print(f"✓ 过滤ST、退市后: {len(stock_list)} 只")
            
            # 选择主板股票（优先沪深300成分股）
            stock_list = stock_list.head(n_stocks * 2)  # 多选一些，确保有足够数据
            
            print(f"✓ 选取前 {len(stock_list)} 只股票用于数据采集")
            print()
            
            return stock_list['code'].tolist()[:n_stocks]
            
        except Exception as e:
            print(f"❌ 获取股票列表失败: {e}")
            return []

    def collect_stock_data(self, stock_code: str, 
                          start_date: str, 
                          end_date: str) -> pd.DataFrame:
        """
        采集单只股票历史数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票历史数据
        """
        try:
            # AkShare获取历史行情
            df = ak.stock_zh_a_hist(symbol=stock_code, 
                                   period="daily",
                                   start_date=start_date.replace('-', ''),
                                   end_date=end_date.replace('-', ''),
                                   adjust="qfq")  # 前复权
            
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
        """
        采集所有股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            所有股票的合并数据
        """
        print("=" * 70)
        print(f"【步骤2】采集{len(stock_codes)}只股票历史数据")
        print(f"时间范围: {start_date} 至 {end_date}")
        print("=" * 70)
        
        all_data = []
        success_count = 0
        
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"  [{i}/{len(stock_codes)}] 采集 {stock_code}...", end=" ")
            
            df = self.collect_stock_data(stock_code, start_date, end_date)
            
            if df is not None and len(df) > 100:  # 至少100条数据
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

    def create_high_return_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建高涨幅标签
        
        Args:
            df: 股票数据
            
        Returns:
            添加了标签的DataFrame
        """
        print("=" * 70)
        print("【步骤3】创建高涨幅标签（3-5天涨幅≥8%）")
        print("=" * 70)
        
        df = df.copy()
        
        # 计算未来3-5天的涨幅
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
        print()
        
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple:
        """划分数据集（时间序列分割）"""
        print("=" * 70)
        print("【步骤4】划分训练集/验证集/测试集")
        print("=" * 70)
        
        # 按时间划分：70%训练，15%验证，15%测试
        n = len(df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train+n_val]
        test_df = df.iloc[n_train+n_val:]
        
        print(f"✓ 数据集划分（按时间）:")
        print(f"  训练集: {len(train_df)}条 ({train_df.index.min()} 至 {train_df.index.max()})")
        print(f"  验证集: {len(val_df)}条 ({val_df.index.min()} 至 {val_df.index.max()})")
        print(f"  测试集: {len(test_df)}条 ({test_df.index.min()} 至 {test_df.index.max()})")
        print()
        
        return train_df, val_df, test_df

    def engineer_features(self, train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> Tuple:
        """特征工程"""
        print("=" * 70)
        print("【步骤5】特征工程")
        print("=" * 70)

        # 使用AssaultFeatureEngineer计算特征
        train_df = self.feature_engineer.create_all_features(train_df)
        val_df = self.feature_engineer.create_all_features(val_df)
        test_df = self.feature_engineer.create_all_features(test_df)

        # 获取特征名称
        feature_names = self.feature_engineer.get_feature_names()

        # 提取特征矩阵
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                        'amount', 'amplitude', 'pct_change', 'change_amount',
                        'turnover_rate', 'stock_code', 'future_return_3d',
                        'future_return_4d', 'future_return_5d', 'max_future_return',
                        'label', 'date', 'returns', 'price_change',
                        'high_20', 'ema_12', 'ema_26', 'low_9', 'high_9',
                        'avg_volume_5', 'k_value', 'd_value', 'ma_5', 'ma_10', 'ma_20']

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
        print()

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray):
        """训练模型"""
        print("=" * 70)
        print("【步骤6】训练高涨幅精准突击模型")
        print("=" * 70)

        model_params = self.config['model']

        print(f"模型参数:")
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
            # 获取可用的评估指标
            available_metrics = list(results['validation_0'].keys())
            if available_metrics:
                metric_name = available_metrics[0]  # 使用第一个可用的指标
                train_score = results['validation_0'][metric_name][-1]
                val_score = results['validation_1'][metric_name][-1]
                print(f"✓ 模型训练完成:")
                print(f"  训练集{metric_name}: {train_score:.4f}")
                print(f"  验证集{metric_name}: {val_score:.4f}")
                print(f"  最佳迭代次数: {self.model.best_iteration + 1}")
            else:
                print(f"✓ 模型训练完成:")
                print(f"  最佳迭代次数: {self.model.best_iteration + 1}")
        except Exception as e:
            print(f"✓ 模型训练完成 (无法获取训练日志: {e})")
            print(f"  最佳迭代次数: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else len(self.model.get_booster().get_dump())}")
        print()

        return self.model

    def optimize_threshold(self, y_val: np.ndarray, y_val_pred_proba: np.ndarray) -> float:
        """优化决策阈值（以精确率为目标）"""
        print("=" * 70)
        print("【步骤7】优化决策阈值（目标精确率≥70%）")
        print("=" * 70)
        
        target_precision = self.config['threshold']['target_precision']
        thresholds = np.arange(0.3, 0.8, 0.01)
        
        best_threshold = self.config['threshold']['default_threshold']
        best_precision = 0
        
        print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1':<10}")
        print("-" * 45)
        
        for threshold in thresholds:
            y_pred = (y_val_pred_proba >= threshold).astype(int)
            
            if y_pred.sum() > 0:  # 有预测为正的样本
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                # 寻找满足精确率要求且F1最高的阈值
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
        print()
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("混淆矩阵:")
        print("           预测负例  预测正例")
        print(f"实际负例: {cm[0,0]:>6}  {cm[0,1]:>6}")
        print(f"实际正例: {cm[1,0]:>6}  {cm[1,1]:>6}")
        print()
        
        # 计算预测样本的平均涨幅
        pred_positive = y_pred == 1
        if pred_positive.sum() > 0:
            avg_return = y_test[pred_positive].mean()  # 正样本的平均涨幅
            print(f"✓ 预测为正的样本平均涨幅: {avg_return*100:.2f}%")
        print()
        
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
        model_path = os.path.join(model_dir, "high_return_100stocks_3years.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ 模型已保存: {model_path}")
        
        # 保存特征名称
        feature_path = os.path.join(model_dir, "high_return_100stocks_3years_features.pkl")
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✓ 特征已保存: {feature_path}")
        
        # 保存元数据
        metadata = {
            'model_type': 'XGBoost_HighReturn_Assault',
            'data_description': '100只A股股票，3年历史数据',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_count': len(self.feature_names),
            'decision_threshold': threshold,
            'metrics': metrics,
            'config': self.config
        }
        
        metadata_path = os.path.join(model_dir, "high_return_100stocks_3years_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {metadata_path}")
        print()

    def run(self):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("高涨幅精准突击模型训练（100股×3年）")
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
        
        # 3. 创建标签
        labeled_df = self.create_high_return_labels(data_df)
        
        if labeled_df['label'].sum() == 0:
            print("❌ 没有正样本（涨幅≥8%），无法训练")
            return
        
        # 4. 划分数据
        train_df, val_df, test_df = self.split_data(labeled_df)
        
        # 5. 特征工程
        X_train, X_val, X_test, y_train, y_val, y_test = self.engineer_features(
            train_df, val_df, test_df
        )
        
        # 6. 训练模型
        self.train_model(X_train, y_train, X_val, y_val)
        
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
        print(f"\n模型文件: assets/models/high_return_100stocks_3years.pkl")
        print(f"最优阈值: {optimal_threshold:.2f}")
        print(f"测试集精确率: {metrics['precision']:.2%}")
        print(f"测试集AUC: {metrics['auc']:.4f}\n")


if __name__ == "__main__":
    trainer = RealDataHighReturnTrainer()
    trainer.run()
