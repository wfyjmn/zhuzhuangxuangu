"""
短期突击模型训练脚本
基于"少错过，不犯错，全身而退"的核心理念
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, log_loss
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stock_system.assault_features import AssaultFeatureEngineer
from stock_system.triple_confirmation import TripleConfirmation
from stock_system.assault_trading import AssaultTradingSystem


def generate_mock_data(n_samples=5000):
    """生成模拟股票数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    returns = np.random.normal(0.002, 0.02, n_samples)
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': np.roll(prices, 1) * (1 + np.random.normal(0, 0.01, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'volume': np.random.lognormal(10, 0.5, n_samples)
    })
    
    df.loc[0, 'open'] = df.loc[0, 'close']
    
    # 生成标签（未来3天涨幅>5%）
    df['future_return'] = df['close'].pct_change(3).shift(-3)
    df['label'] = (df['future_return'] > 0.05).astype(int)
    
    df = df.dropna()
    df.set_index('date', inplace=True)
    
    print(f"✓ 生成了{len(df)}条模拟数据")
    print(f"  正样本比例: {df['label'].mean():.2%}")
    
    return df


def split_data(df: pd.DataFrame, 
               test_size: float = 0.2,
               val_size: float = 0.15) -> Tuple:
    """划分数据集"""
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    
    print(f"✓ 数据集划分:")
    print(f"  训练集: {len(train_df)}条 ({len(train_df)/n:.1%})")
    print(f"  验证集: {len(val_df)}条 ({len(val_df)/n:.1%})")
    print(f"  测试集: {len(test_df)}条 ({len(test_df)/n:.1%})")
    
    return train_df, val_df, test_df


def train_assault_model():
    """训练短期突击模型"""
    print("=" * 70)
    print("短期突击特征权重体系 - 模型训练")
    print("=" * 70)
    print("核心理念：少错过，不犯错，全身而退")
    print("=" * 70)
    print()
    
    # 加载配置
    config_path = "config/short_term_assault_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 1. 加载数据
    print("【步骤1】加载数据")
    print("-" * 70)
    df = generate_mock_data(n_samples=5000)
    print()
    
    # 2. 划分数据集
    print("【步骤2】划分数据集")
    print("-" * 70)
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.15)
    print()
    
    # 3. 特征工程
    print("【步骤3】短期突击特征工程")
    print("-" * 70)
    feature_engineer = AssaultFeatureEngineer(config_path)
    
    train_with_features = feature_engineer.create_all_features(train_df)
    val_with_features = feature_engineer.create_all_features(val_df)
    test_with_features = feature_engineer.create_all_features(test_df)
    
    feature_names = feature_engineer.get_feature_names()
    print(f"\n特征列表（前20个）: {feature_names[:20]}")
    print()
    
    # 4. 准备训练数据
    print("【步骤4】准备训练数据")
    print("-" * 70)
    
    X_train = train_with_features[feature_names].values
    y_train = train_with_features['label'].values
    
    X_val = val_with_features[feature_names].values
    y_val = val_with_features['label'].values
    
    X_test = test_with_features[feature_names].values
    y_test = test_with_features['label'].values
    
    print(f"训练集特征形状: {X_train.shape}")
    print(f"验证集特征形状: {X_val.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集正样本比例: {y_train.mean():.2%}")
    print()
    
    # 5. 训练模型
    print("【步骤5】训练XGBoost模型")
    print("-" * 70)
    
    import xgboost as xgb
    
    model_params = config['model_params']['xgboost']
    print(f"模型参数:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    print()
    
    model = xgb.XGBClassifier(**model_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    print("✓ 模型训练完成")
    print()
    
    # 6. 评估模型性能
    print("【步骤6】评估模型性能")
    print("-" * 70)
    
    # 预测概率
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 使用最优阈值预测（默认0.3）
    threshold = 0.3
    y_train_pred = (y_train_pred_proba >= threshold).astype(int)
    y_val_pred = (y_val_pred_proba >= threshold).astype(int)
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)
    
    # 计算指标
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # 过拟合差距
    overfitting_auc = abs(train_auc - val_auc)
    
    # 目标检查
    goals = config['optimization_goals']
    
    print(f"精确率（Precision）- 目标: {goals['precision']['target']:.2%}")
    print(f"  训练集: {train_precision:.4f}")
    print(f"  验证集: {val_precision:.4f} {'✓' if val_precision >= goals['precision']['target'] else '✗'}")
    print(f"  测试集: {test_precision:.4f} {'✓' if test_precision >= goals['precision']['target'] else '✗'}")
    print()
    
    print(f"召回率（Recall）- 目标: {goals['recall']['target']:.2%}")
    print(f"  训练集: {train_recall:.4f}")
    print(f"  验证集: {val_recall:.4f} {'✓' if val_recall >= goals['recall']['target'] else '✗'}")
    print(f"  测试集: {test_recall:.4f} {'✓' if test_recall >= goals['recall']['target'] else '✗'}")
    print()
    
    print(f"AUC - 参考指标: 0.65")
    print(f"  训练集: {train_auc:.4f}")
    print(f"  验证集: {val_auc:.4f}")
    print(f"  测试集: {test_auc:.4f}")
    print()
    
    print(f"过拟合差距 - 目标: <{goals['overfitting_gap']['target']:.2%}")
    print(f"  AUC差距: {overfitting_auc:.4f} {'✓' if overfitting_auc < goals['overfitting_gap']['target'] else '✗'}")
    print()
    
    print(f"F1-Score:")
    print(f"  训练集: {train_f1:.4f}")
    print(f"  验证集: {val_f1:.4f}")
    print(f"  测试集: {test_f1:.4f}")
    print()
    
    # 计算综合得分
    precision_score_weighted = min(val_precision / goals['precision']['target'], 1.0) * goals['precision']['weight']
    recall_score_weighted = min(val_recall / goals['recall']['target'], 1.0) * goals['recall']['weight']
    overfitting_score_weighted = max(1 - overfitting_auc / goals['overfitting_gap']['target'], 0) * goals['overfitting_gap']['weight']
    
    total_score = precision_score_weighted + recall_score_weighted + overfitting_score_weighted
    
    print(f"综合得分: {total_score:.4f}/1.00")
    print(f"  - 精确率得分: {precision_score_weighted:.4f}")
    print(f"  - 召回率得分: {recall_score_weighted:.4f}")
    print(f"  - 过拟合得分: {overfitting_score_weighted:.4f}")
    print()
    
    # 7. 三重确认演示
    print("【步骤7】三重确认机制演示")
    print("-" * 70)
    
    triple_confirmation = TripleConfirmation(config_path)
    
    # 选择几个样本进行验证
    sample_indices = np.random.choice(len(val_df), min(5, len(val_df)), replace=False)
    
    print("样本信号验证:")
    for idx in sample_indices:
        result = triple_confirmation.validate_all_confirmations(val_with_features, idx)
        signal_grade = triple_confirmation.get_signal_grade(result)
        
        print(f"\n样本 {idx}:")
        print(f"  资金确认: {'✓' if result['capital']['confirmed'] else '✗'} ({result['capital']['score']:.2f})")
        print(f"  情绪确认: {'✓' if result['sentiment']['confirmed'] else '✗'} ({result['sentiment']['score']:.2f})")
        print(f"  技术确认: {'✓' if result['technical']['confirmed'] else '✗'} ({result['technical']['score']:.2f})")
        print(f"  确认数量: {result['confirmed_count']}/3")
        print(f"  信号等级: {signal_grade}级")
        print(f"  最终确认: {'买入' if result['final_confirmed'] else '不买入'}")
    print()
    
    # 8. 信号分级统计
    print("【步骤8】信号分级统计")
    print("-" * 70)
    
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    # 统计验证集的信号等级
    start_idx = max(20, 0)
    for idx in range(start_idx, min(len(val_df), 200)):  # 限制样本数量
        result = triple_confirmation.validate_all_confirmations(val_with_features, idx)
        signal_grade = triple_confirmation.get_signal_grade(result)
        grade_counts[signal_grade] += 1
    
    total = sum(grade_counts.values())
    print(f"验证集信号分布（样本数: {total}）:")
    for grade, count in grade_counts.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {grade}级: {count}次 ({percentage:.1f}%)")
    print()
    
    # 9. 保存模型
    print("【步骤9】保存模型")
    print("-" * 70)
    
    os.makedirs("models", exist_ok=True)
    
    # 保存模型
    model_path = "models/assault_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ 模型已保存到: {model_path}")
    
    # 保存特征名称
    feature_names_path = "models/assault_feature_names.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ 特征名称已保存到: {feature_names_path}")
    print()
    
    # 10. 生成评估报告
    print("【步骤10】生成评估报告")
    print("-" * 70)
    
    report = f"""# 短期突击模型评估报告

## 一、核心理念

**少错过**：召回率>{goals['recall']['target']:.0%}，抓住绝大多数上涨机会
**不犯错**：精确率>{goals['precision']['target']:.0%}，出击必中，减少无效交易
**全身而退**：过拟合差距<{goals['overfitting_gap']['target']:.0%}，夏普比率>{goals['sharpe_ratio']['target']:.0f}

## 二、模型性能

### 2.1 核心指标

| 指标 | 训练集 | 验证集 | 测试集 | 目标 | 达成 |
|------|--------|--------|--------|------|------|
| 精确率 | {train_precision:.4f} | {val_precision:.4f} | {test_precision:.4f} | {goals['precision']['target']:.2%} | {'✓' if val_precision >= goals['precision']['target'] else '✗'} |
| 召回率 | {train_recall:.4f} | {val_recall:.4f} | {test_recall:.4f} | {goals['recall']['target']:.2%} | {'✓' if val_recall >= goals['recall']['target'] else '✗'} |
| AUC | {train_auc:.4f} | {val_auc:.4f} | {test_auc:.4f} | 0.65 | - |
| F1-Score | {train_f1:.4f} | {val_f1:.4f} | {test_f1:.4f} | - | - |

### 2.2 过拟合差距

- AUC差距: {overfitting_auc:.4f} (目标: <{goals['overfitting_gap']['target']:.2%}) {'✓' if overfitting_auc < goals['overfitting_gap']['target'] else '✗'}

### 2.3 综合得分

**综合得分**: {total_score:.4f}/1.00

- 精确率得分: {precision_score_weighted:.4f} (权重{goals['precision']['weight']:.0%})
- 召回率得分: {recall_score_weighted:.4f} (权重{goals['recall']['weight']:.0%})
- 过拟合得分: {overfitting_score_weighted:.4f} (权重{goals['overfitting_gap']['weight']:.0%})

## 三、信号分级统计

验证集信号分布（样本数: {total}）:

| 等级 | 次数 | 占比 | 说明 |
|------|------|------|------|
| A级 | {grade_counts['A']} | {grade_counts['A']/total*100 if total > 0 else 0:.1f}% | 三重确认全部满足 |
| B级 | {grade_counts['B']} | {grade_counts['B']/total*100 if total > 0 else 0:.1f}% | 满足两重确认 |
| C级 | {grade_counts['C']} | {grade_counts['C']/total*100 if total > 0 else 0:.1f}% | 仅满足资金强度 |
| D级 | {grade_counts['D']} | {grade_counts['D']/total*100 if total > 0 else 0:.1f}% | 不满足任何确认 |

## 四、特征工程

### 4.1 特征权重分布

- 资金强度（40%）：{feature_engineer.get_feature_weights()['capital_strength']:.0%}
- 市场情绪（35%）：{feature_engineer.get_feature_weights()['market_sentiment']:.0%}
- 技术动量（25%）：{feature_engineer.get_feature_weights()['technical_momentum']:.0%}

### 4.2 特征总数

总特征数: {len(feature_names)}个

## 五、总结

### 达成目标

- ✅ 少错过（召回率{val_recall:.2%}，目标{goals['recall']['target']:.0%}）{' ✓' if val_recall >= goals['recall']['target'] else ' ✗'}
- ✅ 不犯错（精确率{val_precision:.2%}，目标{goals['precision']['target']:.0%}）{' ✓' if val_precision >= goals['precision']['target'] else ' ✗'}
- ✅ 全身而退（过拟合差距{overfitting_auc:.2%}，目标<{goals['overfitting_gap']['target']:.0%}）{' ✓' if overfitting_auc < goals['overfitting_gap']['target'] else ' ✗'}

---

**报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: 3.0
**状态**: {'优秀' if total_score >= 0.8 else '良好' if total_score >= 0.6 else '需改进'}
"""
    
    # 保存报告
    os.makedirs("assets", exist_ok=True)
    report_path = "assets/assault_model_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 报告已保存到: {report_path}")
    print()
    
    # 11. 总结
    print("=" * 70)
    print("训练完成总结")
    print("=" * 70)
    print(f"✓ 短期突击模型训练完成")
    print(f"✓ 综合得分: {total_score:.4f} ({'优秀' if total_score >= 0.8 else '良好' if total_score >= 0.6 else '需改进'})")
    print(f"✓ 验证集召回率: {val_recall:.4f} (目标: {goals['recall']['target']:.2f}) {'✓' if val_recall >= goals['recall']['target'] else '✗'}")
    print(f"✓ 验证集精确率: {val_precision:.4f} (目标: {goals['precision']['target']:.2f}) {'✓' if val_precision >= goals['precision']['target'] else '✗'}")
    print(f"✓ 过拟合差距: {overfitting_auc:.4f} (目标: <{goals['overfitting_gap']['target']:.2f}) {'✓' if overfitting_auc < goals['overfitting_gap']['target'] else '✗'}")
    print()
    
    return {
        'model': model,
        'feature_names': feature_names,
        'train_metrics': {
            'precision': train_precision,
            'recall': train_recall,
            'auc': train_auc,
            'f1': train_f1
        },
        'val_metrics': {
            'precision': val_precision,
            'recall': val_recall,
            'auc': val_auc,
            'f1': val_f1
        },
        'test_metrics': {
            'precision': test_precision,
            'recall': test_recall,
            'auc': test_auc,
            'f1': test_f1
        },
        'overfitting_gap': overfitting_auc,
        'total_score': total_score,
        'grade_counts': grade_counts
    }


if __name__ == "__main__":
    results = train_assault_model()
