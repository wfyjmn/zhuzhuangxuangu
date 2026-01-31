#!/usr/bin/env python3
"""
自动化参数调优训练脚本 - 基于Optuna
功能：
1. 加载配置和数据
2. 特征工程（包含真实资金流和复权价格）
3. 使用Optuna自动调参
4. 评估模型性能
5. 保存最佳模型
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import optuna
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 添加路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.enhanced_features import EnhancedFeatureEngineer
from stock_system.data_collector import MarketDataCollector


class OptunaTrainer:
    """Optuna自动调参训练器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.collector = MarketDataCollector()
        self.engineer = EnhancedFeatureEngineer()
        self.best_params = None
        self.best_score = 0.0
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ 配置文件加载成功: {self.config_path}")
        print(f"  策略名称: {config['strategy_name']}")
        print(f"  版本: {config['version']}")
        return config
    
    def prepare_data(self) -> pd.DataFrame:
        """准备训练数据"""
        print("\n" + "=" * 60)
        print("步骤1: 准备数据")
        print("=" * 60)
        
        # 获取股票池（排除北交所BJ、科创板688、创业板300/301）
        stock_codes = self.collector.get_stock_pool_tree(
            pool_size=self.config['data']['n_stocks'],
            exclude_markets=['BJ'],  # 排除北交所股票（数据不足）
            exclude_board_types=['688', '300', '301']  # 排除科创板（688）、创业板（300/301）
        )
        print(f"✓ 股票池: {len(stock_codes)} 只股票")
        
        # 采集数据
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        all_data = []
        print(f"⏳ 正在采集数据并计算特征 (预计耗时 5-10 分钟)...")
        print(f"  日期范围: {start_date} ~ {end_date}")
        
        for idx, code in enumerate(stock_codes):
            try:
                # 获取数据（包含资金流和复权因子）
                df = self.collector.get_daily_data(code, start_date, end_date)
                
                if df is None or len(df) < 60:
                    continue
                
                # 特征工程
                df_feat = self.engineer.create_all_features(df)
                
                # 生成标签
                df_feat = self._create_labels(df_feat)
                
                if not df_feat.empty:
                    all_data.append(df_feat)
                
                # 进度提示
                if (idx + 1) % 50 == 0:
                    print(f"  进度: {idx + 1}/{len(stock_codes)}")
                    
            except Exception as e:
                continue
        
        if not all_data:
            print("❌ 数据准备失败：没有获取到有效数据")
            return None
        
        # 合并所有数据
        full_df = pd.concat(all_data, ignore_index=True)
        
        # 提取配置中的特征列表
        feature_list = self._get_feature_list(full_df)
        
        # 选择特征列和标签列
        cols_to_keep = ['label'] + feature_list
        full_df = full_df[cols_to_keep].dropna()
        
        print(f"\n✓ 数据集构建完成:")
        print(f"  总样本数: {len(full_df)}")
        print(f"  特征数: {len(feature_list)}")
        print(f"  正样本数: {(full_df['label'] == 1).sum()}")
        print(f"  负样本数: {(full_df['label'] == 0).sum()}")
        print(f"  正样本比例: {full_df['label'].mean():.2%}")
        
        return full_df
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建标签"""
        df = df.copy()
        
        # 计算未来收益率
        prediction_days = self.config['data']['prediction_days']
        min_return = self.config['data']['min_return_threshold']
        
        # 使用3日和5日预测
        df['future_return_3d'] = df['close'].pct_change(3).shift(-3)
        df['future_return_5d'] = df['close'].pct_change(5).shift(-5)
        
        # 使用最大未来收益作为目标
        df['max_future_return'] = df[['future_return_3d', 'future_return_5d']].max(axis=1)
        
        # 基础标签：收益率达标
        mask = (df['max_future_return'] >= min_return)
        
        # 额外过滤：换手率在合理区间（如果有换手率数据）
        if 'turnover_rate' in df.columns:
            mask = mask & (df['turnover_rate'] > 1.0) & (df['turnover_rate'] < 20.0)
        
        # 如果有主力资金流特征，要求有资金流入
        if 'main_net_inflow' in df.columns:
            mask = mask & (df['main_net_inflow'] > 0)
        
        df['label'] = mask.astype(int)
        
        # 删除未来数据列
        df = df.drop(columns=[c for c in df.columns if 'future' in c])
        
        return df
    
    def _get_feature_list(self, df: pd.DataFrame) -> list:
        """获取特征列表"""
        # 从配置中读取特征列表
        config_features = self.config.get('train_features', [])
        
        # 过滤掉注释行
        actual_features = [f for f in config_features if not f.startswith('---')]
        
        # 检查哪些特征在数据中实际存在
        available_features = [f for f in actual_features if f in df.columns]
        
        if len(available_features) < len(actual_features):
            missing = set(actual_features) - set(available_features)
            print(f"⚠️ 警告：{len(missing)} 个特征在数据中不存在: {missing}")
        
        return available_features
    
    def define_search_space(self, trial):
        """定义Optuna搜索空间"""
        params_space = self.config['xgboost_params_space']
        
        params = {
            'learning_rate': trial.suggest_float(
                'learning_rate', 
                params_space['learning_rate']['low'], 
                params_space['learning_rate']['high'],
                log=params_space['learning_rate']['log']
            ),
            'max_depth': trial.suggest_int(
                'max_depth',
                params_space['max_depth']['low'],
                params_space['max_depth']['high']
            ),
            'min_child_weight': trial.suggest_int(
                'min_child_weight',
                params_space['min_child_weight']['low'],
                params_space['min_child_weight']['high']
            ),
            'n_estimators': trial.suggest_int(
                'n_estimators',
                params_space['n_estimators']['low'],
                params_space['n_estimators']['high']
            ),
            'scale_pos_weight': trial.suggest_float(
                'scale_pos_weight',
                params_space['scale_pos_weight']['low'],
                params_space['scale_pos_weight']['high']
            ),
            'subsample': trial.suggest_float(
                'subsample',
                params_space['subsample']['low'],
                params_space['subsample']['high']
            ),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                params_space['colsample_bytree']['low'],
                params_space['colsample_bytree']['high']
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                params_space['reg_alpha']['low'],
                params_space['reg_alpha']['high'],
                log=params_space['reg_alpha']['log']
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                params_space['reg_lambda']['low'],
                params_space['reg_lambda']['high'],
                log=params_space['reg_lambda']['log']
            ),
            'gamma': trial.suggest_float(
                'gamma',
                params_space['gamma']['low'],
                params_space['gamma']['high']
            ),
        }
        
        # 添加固定参数
        model_config = self.config['model_config']
        params.update({
            k: v for k, v in model_config.items() 
            if k not in ['eval_metric']
        })
        
        return params
    
    def objective(self, trial, X_train, y_train, feature_names):
        """Optuna优化目标函数"""
        # 定义参数
        params = self.define_search_space(trial)
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.config['optuna']['cv_folds'])
        
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 训练模型
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # 预测
            y_pred = model.predict(X_val)
            
            # 计算F1分数
            score = f1_score(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def train_and_tune(self):
        """训练并调优模型"""
        print("\n" + "=" * 60)
        print("步骤2: 准备训练数据")
        print("=" * 60)
        
        # 准备数据
        full_df = self.prepare_data()
        if full_df is None:
            return False
        
        # 划分训练集和测试集（时间序列划分）
        split_idx = int(len(full_df) * 0.85)
        train_df = full_df.iloc[:split_idx]
        test_df = full_df.iloc[split_idx:]
        
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']
        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']
        
        feature_names = X_train.columns.tolist()
        
        print(f"\n✓ 数据集划分完成:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        # Optuna优化
        print("\n" + "=" * 60)
        print("步骤3: Optuna超参数优化")
        print("=" * 60)
        print(f"试验次数: {self.config['optuna']['n_trials']}")
        print(f"优化目标: {self.config['optuna']['metric']}")
        print(f"超时时间: {self.config['optuna']['timeout']} 秒")
        
        def objective_wrapper(trial):
            return self.objective(trial, X_train, y_train, feature_names)
        
        study = optuna.create_study(
            direction=self.config['optuna']['direction'],
            study_name=f"{self.config['strategy_name']}_optimization"
        )
        
        study.optimize(
            objective_wrapper,
            n_trials=self.config['optuna']['n_trials'],
            timeout=self.config['optuna']['timeout'],
            show_progress_bar=True
        )
        
        # 获取最佳参数
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\n✓ 优化完成!")
        print(f"  最佳F1分数: {self.best_score:.4f}")
        print(f"  最佳参数:")
        for key, value in self.best_params.items():
            print(f"    {key}: {value}")
        
        # 训练最终模型
        print("\n" + "=" * 60)
        print("步骤4: 训练最终模型")
        print("=" * 60)
        
        # 添加固定参数
        model_config = self.config['model_config']
        final_params = self.best_params.copy()
        final_params.update({
            k: v for k, v in model_config.items() 
            if k not in ['eval_metric']
        })
        
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # 评估模型
        print("\n" + "=" * 60)
        print("步骤5: 评估模型性能")
        print("=" * 60)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"测试集性能:")
        print(f"  精确率: {precision:.2%}")
        print(f"  召回率: {recall:.2%}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n混淆矩阵:")
        print(f"  真负例: {cm[0, 0]}  假正例: {cm[0, 1]}")
        print(f"  假负例: {cm[1, 0]}  真正例: {cm[1, 1]}")
        
        # 特征重要性
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 重要特征:")
        print(importance.head(10).to_string(index=False))
        
        # 保存模型
        print("\n" + "=" * 60)
        print("步骤6: 保存模型")
        print("=" * 60)
        
        model_dir = os.path.join(workspace_path, "assets/models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{self.config['strategy_name']}_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'params': final_params,
                'feature_names': feature_names,
                'config': self.config,
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                },
                'importance': importance
            }, f)
        
        print(f"✓ 模型已保存: {model_path}")
        
        return True
    
    def run(self):
        """运行完整训练流程"""
        print("=" * 60)
        print(f"开始训练: {self.config['strategy_name']}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            success = self.train_and_tune()
            
            if success:
                print("\n" + "=" * 60)
                print("✓ 训练成功完成!")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("✗ 训练失败!")
                print("=" * 60)
                
        except Exception as e:
            print(f"\n✗ 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 配置文件路径
    config_path = os.path.join(
        workspace_path, 
        "config/precision_priority_v4_config.json"
    )
    
    # 创建训练器并运行
    trainer = OptunaTrainer(config_path)
    trainer.run()
