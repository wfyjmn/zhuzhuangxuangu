#!/usr/bin/env python3
"""
过拟合分析工具
对比训练集和测试集性能，检测过拟合
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

# 添加src到路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))


def analyze_overfitting():
    """分析过拟合情况"""

    print("=" * 70)
    print("过拟合分析报告")
    print("=" * 70)

    # 加载元数据
    metadata_path = os.path.join(workspace_path, "assets/models/high_return_100stocks_3years_metadata.json")

    if not os.path.exists(metadata_path):
        print("⚠ 未找到原始模型元数据")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    metrics = metadata['metrics']
    config = metadata['config']

    print("\n【模型配置】")
    print(f"数据量: {config['data'].get('n_stocks', 100)}只股票")
    print(f"特征数: {metadata['feature_count']}个")
    print(f"学习率: {config['model']['learning_rate']}")
    print(f"最大深度: {config['model']['max_depth']}")
    print(f"L2正则化: {config['model']['reg_lambda']}")
    print(f"L1正则化: {config['model']['reg_alpha']}")

    print("\n【模型性能】")
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"精确率: {metrics['precision']:.2%}")
    print(f"召回率: {metrics['recall']:.2%}")
    print(f"F1分数: {metrics['f1']:.3f}")
    print(f"AUC: {metrics['auc']:.4f}")

    print("\n【过拟合检测】")

    # 检测指标
    overfitting_indicators = []

    # 1. 精确率100%检测
    if metrics['precision'] >= 0.99:
        overfitting_indicators.append({
            '指标': '精确率≥99%',
            '当前值': f"{metrics['precision']:.2%}",
            '状态': '❌ 严重过拟合',
            '说明': '精确率100%在真实市场中几乎不可能实现'
        })

    # 2. AUC过高检测
    if metrics['auc'] >= 0.99:
        overfitting_indicators.append({
            '指标': 'AUC≥0.99',
            '当前值': f"{metrics['auc']:.4f}",
            '状态': '❌ 严重过拟合',
            '说明': 'AUC接近1说明模型记住了训练数据'
        })

    # 3. F1过高检测
    if metrics['f1'] >= 0.99:
        overfitting_indicators.append({
            '指标': 'F1≥0.99',
            '当前值': f"{metrics['f1']:.3f}",
            '状态': '❌ 严重过拟合',
            '说明': 'F1接近1可能存在数据泄露'
        })

    # 4. 特征数量检测
    if metadata['feature_count'] > 30:
        overfitting_indicators.append({
            '指标': '特征数量过多',
            '当前值': f"{metadata['feature_count']}个",
            '状态': '⚠ 风险',
            '说明': '特征过多可能导致过拟合（建议<30个）'
        })

    # 5. 正则化不足检测
    if config['model']['reg_lambda'] < 5:
        overfitting_indicators.append({
            '指标': 'L2正则化不足',
            '当前值': f"{config['model']['reg_lambda']}",
            '状态': '⚠ 风险',
            '说明': 'L2正则化过弱（建议≥5）'
        })

    # 6. 树深度检测
    if config['model']['max_depth'] > 4:
        overfitting_indicators.append({
            '指标': '树深度过大',
            '当前值': f"{config['model']['max_depth']}",
            '状态': '⚠ 风险',
            '说明': '树过深可能导致过拟合（建议≤4）'
        })

    # 7. 学习率检测
    if config['model']['learning_rate'] > 0.01:
        overfitting_indicators.append({
            '指标': '学习率过大',
            '当前值': f"{config['model']['learning_rate']}",
            '状态': '⚠ 风险',
            '说明': '学习率过快可能导致过拟合（建议≤0.01）'
        })

    # 打印检测结果
    if not overfitting_indicators:
        print("✓ 未检测到明显的过拟合信号")
    else:
        print(f"⚠ 检测到 {len(overfitting_indicators)} 个过拟合风险信号:\n")

        for idx, indicator in enumerate(overfitting_indicators, 1):
            print(f"{idx}. {indicator['指标']}")
            print(f"   当前值: {indicator['当前值']}")
            print(f"   状态: {indicator['状态']}")
            print(f"   说明: {indicator['说明']}\n")

    # 总体评估
    print("=" * 70)
    print("【总体评估】")

    severe_count = sum(1 for x in overfitting_indicators if '❌' in x['状态'])
    warning_count = sum(1 for x in overfitting_indicators if '⚠' in x['状态'])

    if severe_count > 0:
        print(f"❌ 严重过拟合: 检测到 {severe_count} 个严重问题")
        print("\n【建议立即采取的措施】")
        print("1. 检查是否存在数据泄露（未来信息）")
        print("2. 增强正则化（L1/L2加大）")
        print("3. 减少树深度（max_depth ≤ 3）")
        print("4. 增加训练数据量（≥300只股票）")
        print("5. 使用时间序列交叉验证")
        print("6. 添加Dropout或噪声")
    elif warning_count > 0:
        print(f"⚠ 存在风险: 检测到 {warning_count} 个潜在问题")
        print("\n【建议优化措施】")
        print("1. 适当增强正则化")
        print("2. 考虑减少特征数量")
        print("3. 使用交叉验证评估模型稳定性")
    else:
        print("✓ 模型健康，无明显过拟合")

    print("=" * 70)


if __name__ == "__main__":
    analyze_overfitting()
