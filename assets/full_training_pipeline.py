# -*- coding: utf-8 -*-
"""
完整训练流程：使用模拟数据进行演示
由于真实数据生成速度太慢（约23秒/股票），这里使用模拟数据演示完整流程
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ai_referee import AIReferee


def generate_mock_data(n_samples=1000):
    """生成模拟训练数据"""
    print("="*80)
    print("生成模拟训练数据")
    print("="*80 + "\n")

    np.random.seed(42)

    # 生成 22 个特征
    feature_names = [
        'vol_ratio', 'turnover_rate', 'pe_ttm',
        'pct_chg_1d', 'pct_chg_5d', 'pct_chg_20d',
        'ma5_slope', 'ma20_slope',
        'bias_5', 'bias_20',
        'rsi_14', 'std_20_ratio',
        'position_20d', 'position_250d',
        'macd_dif', 'macd_dea', 'macd_hist',
        'index_pct_chg', 'sector_pct_chg',
        'moneyflow_score', 'tech_score'
    ]

    data = {}

    for feature in feature_names:
        if 'ratio' in feature or 'pct' in feature:
            data[feature] = np.random.randn(n_samples) * 0.5 + 1.0
        elif feature in ['rsi_14', 'position_20d', 'position_250d']:
            data[feature] = np.random.rand(n_samples) * 100
        elif 'score' in feature:
            data[feature] = np.random.rand(n_samples) * 100
        else:
            data[feature] = np.random.randn(n_samples)

    # 生成标签（15% 正样本）
    labels = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    df = pd.DataFrame(data)
    df['label'] = labels

    # 添加 ts_code 和 trade_date（仅用于标识）
    df['ts_code'] = [f'60{i:04d}.SH' for i in range(n_samples)]
    df['trade_date'] = np.random.choice(['20240102', '20240103', '20240104', '20240105'], n_samples)

    print(f"✅ 生成 {n_samples} 条模拟数据")
    print(f"  正样本: {labels.sum()} ({labels.sum()/n_samples*100:.1f}%)")
    print(f"  负样本: {n_samples - labels.sum()} ({(1-labels.sum()/n_samples)*100:.1f}%)")
    print(f"  特征数: {len(feature_names)}\n")

    return df


def full_training_pipeline():
    """完整训练流程（使用模拟数据）"""
    print("="*80)
    print(" " * 25 + "DeepQuant 完整训练流程")
    print(" " * 28 + "（模拟数据演示）")
    print("="*80 + "\n")

    # 步骤 1：生成训练数据
    print("【步骤 1】生成训练数据")
    print("="*80 + "\n")

    df = generate_mock_data(n_samples=5000)

    # 保存数据
    output_dir = "data/training"
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(output_dir, "mock_training_data.csv")
    df.to_csv(data_file, index=False)
    print(f"💾 数据已保存：{data_file}\n")

    # 步骤 2：训练模型
    print("\n" + "="*80)
    print("【步骤 2】训练 AI 裁判模型")
    print("="*80 + "\n")

    feature_cols = [col for col in df.columns if col not in ['label', 'ts_code', 'trade_date']]
    X = df[feature_cols]
    y = df['label']

    # 保留 trade_date 列用于时序交叉验证
    X_with_date = X.copy()
    X_with_date['trade_date'] = df['trade_date']

    print(f"特征数: {len(feature_cols)}")
    print(f"训练样本: {len(X)}")
    print(f"正样本: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"负样本: {len(y) - y.sum()} ({(1-y.sum()/len(y))*100:.1f}%)\n")

    # 初始化模型
    print("初始化 XGBoost 模型...")
    referee = AIReferee(model_type='xgboost')

    start_time = datetime.now()

    # 训练模型（使用时序交叉验证）
    print("开始训练（时序交叉验证，5折）...")
    referee.train_time_series(X_with_date, y, n_splits=5)

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ 模型训练完成！用时: {duration/60:.1f} 分钟")

    # 保存模型
    model_file = os.path.join(output_dir, "ai_referee_model.pkl")
    referee.save_model(model_file)
    print(f"💾 模型已保存：{model_file}\n")

    # 步骤 3：评估模型
    print("\n" + "="*80)
    print("【步骤 3】评估模型")
    print("="*80 + "\n")

    y_pred = referee.model.predict(X)
    y_prob = referee.model.predict_proba(X)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)

    print("训练集评估指标：")
    print(f"  准确率（Accuracy）: {accuracy:.4f}")
    print(f"  精确率（Precision）: {precision:.4f}")
    print(f"  召回率（Recall）: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  AUC分数: {auc:.4f}")

    print("\n混淆矩阵：")
    cm = confusion_matrix(y, y_pred)
    print(f"  预测负样本: TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  预测正样本: FN={cm[1,0]}, TP={cm[1,1]}")

    print("\n详细分类报告：")
    print(classification_report(y, y_pred, digits=4))

    # 显示预测概率分布
    print("\n预测概率分布：")
    print(f"  平均概率: {y_prob.mean():.4f}")
    print(f"  正样本概率: {y_prob[y==1].mean():.4f}")
    print(f"  负样本概率: {y_prob[y==0].mean():.4f}")
    print(f"  概率中位数: {np.median(y_prob):.4f}")

    # 特征重要性
    print("\n" + "="*80)
    print("特征重要性（Top 10）")
    print("="*80 + "\n")

    # 根据模型类型选择特征重要性计算方式
    if hasattr(referee.model, 'feature_importances_'):
        # 树模型（XGBoost, LightGBM）
        importances = referee.model.feature_importances_
    elif hasattr(referee.model, 'coef_'):
        # 线性模型（LogisticRegression）
        importances = np.abs(referee.model.coef_[0])
    else:
        print("  [警告] 当前模型不支持特征重要性分析")
        importances = np.zeros(len(feature_cols))

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    if importances.sum() > 0:
        print(feature_importance.head(10).to_string(index=False))
    else:
        print("  模型不提供特征重要性")

    print("\n" + "="*80)
    print("✅ 完整训练流程完成！")
    print("="*80 + "\n")

    print("使用说明：")
    print("  1. 训练数据已保存：data/training/mock_training_data.csv")
    print("  2. 模型已保存：data/training/ai_referee_model.pkl")
    print("  3. 可以使用以下代码加载模型：")
    print("     from ai_referee import AIReferee")
    print("     referee = AIReferee()")
    print("     referee.load_model('data/training/ai_referee_model.pkl')")
    print("\n")


if __name__ == "__main__":
    try:
        full_training_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  流程被用户中断\n")
    except Exception as e:
        print(f"\n\n❌ 流程失败: {e}\n")
        import traceback
        traceback.print_exc()
