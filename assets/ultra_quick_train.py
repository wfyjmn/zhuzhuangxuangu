# -*- coding: utf-8 -*-
"""
超快速训练流程：只生成 2 天的数据
"""

import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from ai_backtest_generator import AIBacktestGenerator
from ai_referee import AIReferee


def ultra_quick_train():
    """超快速训练：只生成 2 天的数据"""
    print("="*80)
    print(" " * 30 + "DeepQuant 超快速训练")
    print("="*80 + "\n")

    generator = AIBacktestGenerator()

    # 生成 2024年1月前半个月的数据（需要至少 8 天）
    print("【步骤 1】生成训练数据（2024年1月前半月）")
    print("="*80 + "\n")

    start_time = datetime.now()

    df = generator.generate_dataset('20240102', '20240115', max_samples=200)

    if df.empty:
        print("\n❌ 数据生成失败！\n")
        return

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ 数据生成完成！用时: {duration:.1f} 秒")
    print(f"  总样本数: {len(df)}")
    print(f"  正样本: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.1f}%)")
    print(f"  负样本: {len(df) - df['label'].sum()} ({(1-df['label'].sum()/len(df))*100:.1f}%)")

    # 保存数据
    output_dir = "data/training"
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(output_dir, "ultra_quick_train_data.csv")
    df.to_csv(data_file, index=False)
    print(f"💾 数据已保存：{data_file}\n")

    # 步骤 2：训练模型
    print("\n" + "="*80)
    print("【步骤 2】训练模型")
    print("="*80 + "\n")

    feature_cols = [col for col in df.columns if col not in ['label', 'ts_code', 'trade_date']]
    X = df[feature_cols]
    y = df['label']

    print(f"特征数: {len(feature_cols)}")
    print(f"训练样本: {len(X)}\n")

    if len(X) < 50:
        print("⚠️  样本数过少，跳过训练\n")
        return

    # 初始化模型
    referee = AIReferee(model_type='xgboost')

    start_time = datetime.now()

    # 训练模型（使用时序交叉验证）
    referee.train_time_series(X, y, n_splits=2)

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ 模型训练完成！用时: {duration:.1f} 秒")

    # 保存模型
    model_file = os.path.join(output_dir, "ultra_quick_ai_referee.pkl")
    referee.save_model(model_file)
    print(f"💾 模型已保存：{model_file}\n")

    # 步骤 3：测试模型
    print("\n" + "="*80)
    print("【步骤 3】测试模型")
    print("="*80 + "\n")

    y_pred = referee.model.predict(X)
    y_prob = referee.model.predict_proba(X)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)

    print("评估指标：")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  AUC分数: {auc:.4f}")

    print("\n" + "="*80)
    print("✅ 超快速训练完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        ultra_quick_train()
    except KeyboardInterrupt:
        print("\n\n⚠️  流程被用户中断\n")
    except Exception as e:
        print(f"\n\n❌ 流程失败: {e}\n")
        import traceback
        traceback.print_exc()
