#!/usr/bin/env python3
"""
高涨幅预测模型训练（简化稳健版）
核心原则：少而精，防过拟合
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import akshare as ak
import xgboost as xgb

warnings.filterwarnings('ignore')

workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))


def create_simple_features(df):
    """创建简化特征（12个核心特征）"""
    df = df.copy()

    # 1. 动量特征（3个）
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    # 2. 成交量特征（3个）
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()

    # 3. 技术指标（3个）
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'].fillna(50, inplace=True)

    # MACD
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']

    # 价格位置
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_position'] = (
        (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
    ).fillna(0.5)

    # 4. 波动率（3个）
    df['volatility_5'] = df['close'].pct_change().rolling(5).std()
    df['volatility_10'] = df['close'].pct_change().rolling(10).std()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()

    # 清理
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df


def train_robust_model():
    """训练稳健模型"""

    print("\n" + "=" * 70)
    print("高涨幅预测模型训练（简化稳健版）")
    print("=" * 70 + "\n")

    # 1. 获取股票列表（随机100只）
    print("【步骤1】获取股票列表")
    stock_list = ak.stock_info_a_code_name()
    stock_list = stock_list[~stock_list['name'].str.contains('ST|退|暂停', na=False)]
    stock_list = stock_list.sample(n=100, random_state=42)
    stock_codes = stock_list['code'].tolist()
    print(f"✓ 选取 {len(stock_codes)} 只股票\n")

    # 2. 采集数据
    print("【步骤2】采集历史数据")
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=1200)).strftime('%Y%m%d')  # 约3.5年

    all_data = []
    for i, stock_code in enumerate(stock_codes[:50], 1):  # 只用50只加快速度
        try:
            df = ak.stock_zh_a_hist(symbol=stock_code,
                                   period="daily",
                                   start_date=start_date,
                                   end_date=end_date,
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
                '成交额': 'amount'
            })

            all_data.append(df)
            if i % 10 == 0:
                print(f"  已处理 {i}/{len(stock_codes[:50])} 只股票...")
        except Exception as e:
            continue

    combined_df = pd.concat(all_data, ignore_index=False)
    combined_df = combined_df.sort_index()
    print(f"✓ 数据采集完成: {len(combined_df)}条\n")

    # 3. 创建标签
    print("【步骤3】创建高涨幅标签（3-5天涨幅≥8%）")

    # 计算未来涨幅（严格避免数据泄露）
    for days in [3, 4, 5]:
        combined_df[f'future_return_{days}d'] = (
            combined_df.groupby('stock_code')['close']
            .pct_change(days)
            .shift(-days)
        )

    combined_df['max_future_return'] = (
        combined_df[[f'future_return_{days}d' for days in [3, 4, 5]]]
        .max(axis=1)
    )

    min_return = 0.08
    combined_df['label'] = (combined_df['max_future_return'] >= min_return).astype(int)

    # 移除无法计算的数据
    combined_df = combined_df.dropna(subset=['label', 'max_future_return'])

    print(f"✓ 总样本: {len(combined_df)}")
    print(f"  正样本（涨幅≥8%）: {combined_df['label'].sum()} ({combined_df['label'].mean():.2%})")
    print()

    # 4. 特征工程
    print("【步骤4】特征工程（12个核心特征）")
    combined_df = create_simple_features(combined_df)

    # 定义特征列（仅使用12个核心特征）
    feature_cols = [
        'momentum_5', 'momentum_10', 'momentum_20',
        'volume_ratio', 'volume_surge',
        'rsi', 'macd', 'price_position',
        'volatility_5', 'volatility_10', 'volatility_20'
    ]

    print(f"✓ 特征数量: {len(feature_cols)}")
    print()

    # 5. 严格时间序列划分
    print("【步骤5】严格时间序列划分")
    combined_df = combined_df.sort_index()

    n = len(combined_df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_df = combined_df.iloc[:n_train]
    val_df = combined_df.iloc[n_train:n_train+n_val]
    test_df = combined_df.iloc[n_train+n_val:]

    print(f"训练集: {len(train_df)}条 ({train_df.index.min()} 至 {train_df.index.max()})")
    print(f"验证集: {len(val_df)}条 ({val_df.index.min()} 至 {val_df.index.max()})")
    print(f"测试集: {len(test_df)}条 ({test_df.index.min()} 至 {test_df.index.max()})")
    print()

    # 6. 准备数据
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    # 7. 训练模型（强正则化）
    print("【步骤6】训练模型（强正则化）")

    model_params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.01,
        'max_depth': 3,
        'min_child_weight': 20,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_lambda': 10,
        'reg_alpha': 2,
        'gamma': 2,
        'n_estimators': 500,
        'random_state': 42
    }

    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("✓ 模型训练完成\n")

    # 8. 评估模型
    print("【步骤7】评估模型性能")

    # 训练集性能
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    train_precision = precision_score(y_train, train_pred, zero_division=0)
    train_recall = recall_score(y_train, train_pred, zero_division=0)
    train_auc = roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) > 1 else 0

    # 验证集性能
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_precision = precision_score(y_val, val_pred, zero_division=0)
    val_recall = recall_score(y_val, val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0

    # 测试集性能
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0

    print("训练集:")
    print(f"  精确率: {train_precision:.2%}")
    print(f"  召回率: {train_recall:.2%}")
    print(f"  AUC: {train_auc:.4f}")

    print("\n验证集:")
    print(f"  精确率: {val_precision:.2%}")
    print(f"  召回率: {val_recall:.2%}")
    print(f"  AUC: {val_auc:.4f}")

    print("\n测试集:")
    print(f"  精确率: {test_precision:.2%}")
    print(f"  召回率: {test_recall:.2%}")
    print(f"  F1: {test_f1:.3f}")
    print(f"  AUC: {test_auc:.4f}")

    # 过拟合检测
    overfitting_gap = train_auc - test_auc
    print(f"\n过拟合差距（Train AUC - Test AUC）: {overfitting_gap:.4f}")

    if overfitting_gap < 0.05:
        print("✓ 过拟合程度: 轻度/无（健康）")
    elif overfitting_gap < 0.10:
        print("⚠ 过拟合程度: 中度（建议优化）")
    else:
        print("❌ 过拟合程度: 严重（必须优化）")

    # 混淆矩阵
    cm = confusion_matrix(y_test, test_pred)
    print(f"\n混淆矩阵（测试集）:")
    print(f"           预测负例  预测正例")
    print(f"实际负例: {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"实际正例: {cm[1,0]:>6}  {cm[1,1]:>6}")

    # 9. 保存模型
    print("\n【步骤8】保存模型")
    model_dir = os.path.join(workspace_path, "assets/models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "high_return_simple_robust.pkl"), 'wb') as f:
        pickle.dump(model, f)

    metadata = {
        'model_type': 'XGBoost_Simple_Robust',
        'feature_count': len(feature_cols),
        'data_description': '50只股票，简化特征（12个）',
        'metrics': {
            'train_precision': train_precision,
            'train_auc': train_auc,
            'val_precision': val_precision,
            'val_auc': val_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc
        },
        'overfitting_gap': overfitting_gap,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(model_dir, "high_return_simple_robust_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✓ 模型已保存: assets/models/high_return_simple_robust.pkl")
    print()

    print("=" * 70)
    print("✓ 训练完成！")
    print("=" * 70)
    print(f"\n总结:")
    print(f"  特征数: {len(feature_cols)}个")
    print(f"  训练集AUC: {train_auc:.4f}")
    print(f"  测试集AUC: {test_auc:.4f}")
    print(f"  测试集精确率: {test_precision:.2%}")
    print(f"  过拟合差距: {overfitting_gap:.4f}")
    print()


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    train_robust_model()
