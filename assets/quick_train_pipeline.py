# -*- coding: utf-8 -*-
"""
快速训练流程：生成少量数据 -> 训练模型 -> 测试效果
"""

import pandas as pd
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ai_backtest_generator import AIBacktestGenerator
from ai_referee import AIReferee


def quick_train_pipeline():
    """快速训练流程"""
    print("="*80)
    print(" " * 30 + "DeepQuant 快速训练流程")
    print("="*80 + "\n")

    # 步骤 1：生成训练数据
    print("【步骤 1】生成训练数据")
    print("="*80 + "\n")

    generator = AIBacktestGenerator()

    start_time = datetime.now()

    # 生成 2024 年 1 月的数据（限制样本数）
    print("开始生成训练数据（2024年1月，最多 1000 样本）...")
    df = generator.generate_dataset('20240102', '20240131', max_samples=1000)

    if df.empty:
        print("\n❌ 数据生成失败！\n")
        return

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ 数据生成完成！用时: {duration/60:.1f} 分钟")
    print(f"  总样本数: {len(df)}")
    print(f"  正样本: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.1f}%)")
    print(f"  负样本: {len(df) - df['label'].sum()} ({(1-df['label'].sum()/len(df))*100:.1f}%)")

    # 保存数据
    output_dir = "data/training"
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(output_dir, "quick_train_data.csv")
    df.to_csv(data_file, index=False)
    print(f"💾 数据已保存：{data_file}\n")

    # 步骤 2：训练模型
    print("\n" + "="*80)
    print("【步骤 2】训练 AI 裁判模型")
    print("="*80 + "\n")

    # 准备数据
    feature_cols = [col for col in df.columns if col not in ['label', 'ts_code', 'trade_date']]
    X = df[feature_cols]
    y = df['label']

    print(f"特征数: {len(feature_cols)}")
    print(f"训练样本: {len(X)}\n")

    # 初始化模型
    referee = AIReferee(model_type='xgboost')

    start_time = datetime.now()

    # 训练模型（使用时序交叉验证）
    referee.train_time_series(X, y, n_splits=3)

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ 模型训练完成！用时: {duration/60:.1f} 分钟")

    # 保存模型
    model_file = os.path.join(output_dir, "quick_ai_referee.pkl")
    referee.save_model(model_file)
    print(f"💾 模型已保存：{model_file}\n")

    # 步骤 3：测试模型
    print("\n" + "="*80)
    print("【步骤 3】测试模型")
    print("="*80 + "\n")

    # 使用训练数据进行预测测试
    y_pred = referee.model.predict(X)
    y_prob = referee.model.predict_proba(X)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)

    print("测试集评估指标：")
    print(f"  准确率（Accuracy）: {accuracy:.4f}")
    print(f"  精确率（Precision）: {precision:.4f}")
    print(f"  召回率（Recall）: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  AUC分数: {auc:.4f}")

    print("\n混淆矩阵：")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # 显示预测概率分布
    print("\n预测概率分布：")
    print(f"  平均概率: {y_prob.mean():.4f}")
    print(f"  正样本概率: {y_prob[y==1].mean():.4f}")
    print(f"  负样本概率: {y_prob[y==0].mean():.4f}")

    print("\n" + "="*80)
    print("✅ 快速训练流程完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        quick_train_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  流程被用户中断\n")
    except Exception as e:
        print(f"\n\n❌ 流程失败: {e}\n")
        import traceback
        traceback.print_exc()
