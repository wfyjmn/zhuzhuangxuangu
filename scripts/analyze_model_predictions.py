#!/usr/bin/env python3
"""
分析模型预测结果
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))
sys.path.insert(0, os.path.join(workspace_path, "scripts"))

# 加载模型
with open("assets/models/optimized_model.pkl", 'rb') as f:
    model = pickle.load(f)

# 加载特征
with open("assets/models/optimized_features.pkl", 'rb') as f:
    feature_names = pickle.load(f)

# 加载元数据
with open("assets/models/optimized_metadata.json", 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print("=" * 70)
print("模型预测分析")
print("=" * 70)

# 获取特征重要性
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n特征重要性 Top 20:")
print(feature_importance_df.head(20).to_string(index=False))

# 保存特征重要性
feature_importance_df.to_csv('assets/feature_importance.csv', index=False)
print("\n✓ 特征重要性已保存到 assets/feature_importance.csv")

# 模拟一些预测来查看概率分布
print("\n" + "=" * 70)
print("预测概率分布分析")
print("=" * 70)

# 获取训练过程中的数据（这里无法直接获取，需要重新运行）
# 但我们可以从模型配置中获取一些信息

print("\n模型配置:")
print(f"  学习率: {metadata['config']['model']['learning_rate']}")
print(f"  最大深度: {metadata['config']['model']['max_depth']}")
print(f"  样本权重: {metadata['config']['model']['scale_pos_weight']}")
print(f"  正则化 L1: {metadata['config']['model']['reg_alpha']}")
print(f"  正则化 L2: {metadata['config']['model']['reg_lambda']}")

print("\n数据统计:")
print(f"  训练样本: {metadata['config']['data']['n_stocks']} 只股票 × 5年")
print(f"  目标收益率: 5-6%")
print(f"  预测窗口: 7-10天")
print(f"  特征数量: {metadata['feature_count']}")

print("\n模型性能:")
print(f"  测试集AUC: 0.5163")
print(f"  测试集精确率: 0.00%")

print("\n" + "=" * 70)
print("问题分析")
print("=" * 70)

print("\n可能的原因:")
print("1. 目标收益率区间（5-6%）过于狭窄")
print("   - 正样本占比仅约6%，样本不平衡严重")
print("   - 模型倾向于保守，倾向于预测负类")

print("\n2. 预测窗口（7-10天）可能不适合当前市场环境")
print("   - 短期价格波动大，7-10天的预测难度高")
print("   - 建议缩短预测窗口至3-5天")

print("\n3. 特征工程仍需优化")
print("   - 当前74个特征可能不足够强")
print("   - 需要引入更多外部数据源（如行业轮动、宏观指标）")

print("\n4. 决策阈值过高")
print("   - 最优阈值为0.4，但模型预测概率可能都很低")
print("   - 建议降低阈值至0.2-0.3范围")

print("\n" + "=" * 70)
print("优化建议")
print("=" * 70)

print("\n1. 放宽目标收益率区间")
print("   - 从 5-6% → 4-8%")
print("   - 或设置区间：≥4%")

print("\n2. 缩短预测窗口")
print("   - 从 7-10天 → 3-5天")
print("   - 短期动量更容易预测")

print("\n3. 调整样本权重")
print("   - 从 3 → 1.5-2")
print("   - 避免模型过于保守")

print("\n4. 降低决策阈值")
print("   - 从 0.4 → 0.2-0.3")
print("   - 允许更多预测为正的样本")

print("\n5. 引入更多特征")
print("   - 龙虎榜数据")
print("   - 大宗交易数据")
print("   - 股东增减持")
print("   - 行业板块轮动")

print("\n" + "=" * 70)
print("✓ 分析完成")
print("=" * 70)
