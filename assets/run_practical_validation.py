"""
自动化阈值优化系统实战验证
基于模拟A股数据的完整验证流程
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, '/workspace/projects/src')

# 导入优化模块
from stock_system.auto_threshold_optimizer import AutoThresholdOptimizer
from stock_system.rsi_threshold_optimizer import RSIThresholdOptimizer, calculate_rsi
from stock_system.capital_threshold_optimizer import (
    CapitalIntensityThresholdOptimizer,
    calculate_capital_persistence,
    calculate_capital_momentum
)
from stock_system.constrained_optimizer import ConstrainedOptimizer
from stock_system.dynamic_threshold_adjuster import (
    DynamicThresholdAdjuster,
    MarketConditionAnalyzer
)
from stock_system.multi_objective_optimizer import MultiObjectiveOptimizer


def generate_realistic_a_stock_data(n_days=1000):
    """
    生成贴近真实A股特征的模拟数据
    
    Args:
        n_days: 数据天数
    
    Returns:
        包含价格、资金流、技术指标的DataFrame
    """
    np.random.seed(42)
    
    # 生成日期索引（3年左右）
    start_date = datetime(2021, 1, 1)
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # 生成价格序列（带趋势和波动）
    trend = np.linspace(100, 120, n_days)  # 上升趋势
    volatility = 0.02  # 2%日波动率
    random_walk = np.cumsum(np.random.randn(n_days) * volatility * 100)
    
    # 价格 = 趋势 + 随机游走
    prices_array = trend + random_walk
    prices = pd.Series(prices_array, index=dates, name='close')
    
    # 计算收益率
    returns = prices.pct_change()
    returns = returns.fillna(0)  # 第一天收益率为0
    
    # 生成成交量（带周期性）
    base_volume = 1000000
    volume_pattern = np.sin(np.linspace(0, 4*np.pi, n_days)) * 0.3 + 1  # 周期性波动
    volume_noise = np.random.lognormal(0, 0.1, n_days)
    volume = pd.Series(base_volume * volume_pattern * volume_noise, index=dates)
    
    # 生成主力资金净流入（与价格涨跌相关）
    capital_base = returns * 1000000  # 资金与涨跌正相关
    capital_noise = np.random.randn(n_days) * 100000
    main_capital_inflow = pd.Series(capital_base.values + capital_noise, index=dates)
    main_capital_inflow.iloc[0] = 0
    
    # 生成大单占比（与资金流相关）
    large_order_ratio = np.abs(main_capital_inflow) / (volume * 0.5) * 100
    large_order_ratio = np.clip(large_order_ratio, 5, 40)  # 5%-40%之间
    large_order_ratio = pd.Series(large_order_ratio.values, index=dates)
    
    # 生成北向资金（独立但相关）
    northbound_flow = pd.Series(
        np.random.randn(n_days) * 50000 + main_capital_inflow.values * 0.1,
        index=dates
    )
    
    # 计算资金持续性
    capital_persistence = calculate_capital_persistence(
        pd.Series(main_capital_inflow, index=dates),
        window=5
    )
    
    # 计算资金动量
    capital_momentum = calculate_capital_momentum(
        pd.Series(main_capital_inflow, index=dates)
    )
    
    # 生成市场情绪指标
    sentiment_base = (prices / prices.rolling(20).mean() - 1) * 100  # 相对20日均线的位置
    sentiment_noise = np.random.randn(n_days) * 10
    stock_sentiment_score = pd.Series(sentiment_base.values + sentiment_noise, index=dates)
    stock_sentiment_score = stock_sentiment_score.clip(30, 90)
    
    # 生成板块热度
    sector_heat = pd.Series(np.random.randint(40, 80, n_days), index=dates)
    
    # 生成市场广度
    market_breadth = pd.Series(np.random.uniform(0.4, 0.8, n_days), index=dates)
    
    # 生成情绪周期（模拟牛熊转换）
    sentiment_cycle = np.sin(np.linspace(0, 2*np.pi, n_days)) * 0.5 + 0.5
    sentiment_cycle = pd.Series(sentiment_cycle, index=dates)
    
    # 计算RSI
    rsi_6 = calculate_rsi(prices, 6)
    rsi_12 = calculate_rsi(prices, 12)
    rsi_14 = calculate_rsi(prices, 14)
    rsi_24 = calculate_rsi(prices, 24)
    
    # 生成成交量比率
    volume_ma5 = volume.rolling(5).mean()
    volume_ratio = volume / volume_ma5
    
    # 生成攻击形态（价格突破）
    ma5 = prices.rolling(5).mean()
    ma10 = prices.rolling(10).mean()
    attack_pattern = ((ma5 > ma10) & (ma5.shift(1) <= ma10.shift(1))).astype(int)
    
    # 生成目标变量（未来1日上涨）
    target_returns = (returns.shift(-1) > 0).astype(int)
    target_returns = target_returns.fillna(0)
    
    # 组装所有特征
    data = pd.DataFrame({
        # 价格相关
        'close': prices,
        'volume': volume,
        'returns': returns,
        
        # 资金强度特征（40%权重）
        'main_capital_inflow_ratio': main_capital_inflow / (volume * 0.01),  # 资金净流入占成交额比例
        'large_order_ratio': large_order_ratio,
        'capital_persistence': capital_persistence,
        'northbound_flow_ratio': northbound_flow / (volume * 0.01),
        'capital_momentum': capital_momentum,
        
        # 市场情绪特征（35%权重）
        'sector_heat_index': sector_heat,
        'stock_sentiment_score': stock_sentiment_score,
        'market_breadth': market_breadth,
        'sentiment_cycle': sentiment_cycle,
        
        # 技术动量特征（25%权重）
        'rsi_6': rsi_6,
        'rsi_12': rsi_12,
        'rsi_14': rsi_14,
        'rsi_24': rsi_24,
        'volume_ratio': volume_ratio,
        'attack_pattern': attack_pattern,
        
        # 目标变量
        'target': target_returns
    })
    
    # 去除包含NaN的行
    data = data.dropna()
    
    print(f"✓ 生成模拟A股数据：{len(data)}天")
    print(f"  价格范围: {data['close'].min():.2f} ~ {data['close'].max():.2f}")
    print(f"  目标上涨比例: {data['target'].mean():.2%}")
    
    return data


def calculate_features(data):
    """计算三维度特征"""
    print("\n=== 计算三维度特征 ===")
    
    # 资金强度特征（40%）
    capital_features = [
        'main_capital_inflow_ratio',
        'large_order_ratio',
        'capital_persistence',
        'northbound_flow_ratio',
        'capital_momentum'
    ]
    
    # 市场情绪特征（35%）
    sentiment_features = [
        'sector_heat_index',
        'stock_sentiment_score',
        'market_breadth',
        'sentiment_cycle'
    ]
    
    # 技术动量特征（25%）
    momentum_features = [
        'rsi_6', 'rsi_12', 'rsi_14', 'rsi_24',
        'volume_ratio',
        'attack_pattern'
    ]
    
    all_features = {
        'capital_intensity': {'features': capital_features, 'weight': 0.40},
        'market_sentiment': {'features': sentiment_features, 'weight': 0.35},
        'technical_momentum': {'features': momentum_features, 'weight': 0.25}
    }
    
    print(f"  资金强度特征: {len(capital_features)}个")
    print(f"  市场情绪特征: {len(sentiment_features)}个")
    print(f"  技术动量特征: {len(momentum_features)}个")
    
    return all_features


def optimize_single_features(data, features_dict):
    """优化单特征阈值"""
    print("\n=== 单特征阈值优化 ===")
    
    all_features = []
    for category, info in features_dict.items():
        all_features.extend(info['features'])
    
    optimizer = AutoThresholdOptimizer(data, 'target', all_features)
    thresholds = optimizer.calculate_optimal_thresholds(method='ensemble')
    
    print(f"✓ 优化完成，共 {len(thresholds)} 个特征")
    
    # 打印部分结果
    for feature in all_features[:6]:
        if feature in thresholds:
            print(f"  {feature}: {thresholds[feature]['optimal']:.4f}")
    
    return thresholds


def optimize_rsi_thresholds_full(data):
    """优化RSI多周期阈值"""
    print("\n=== RSI阈值优化 ===")
    
    price_data = data['close']
    target_returns = data['target']
    
    config = {
        'periods': [6, 12, 14, 24],
        'recall_constraint': 0.7
    }
    
    optimizer = RSIThresholdOptimizer(price_data, target_returns, config)
    results = optimizer.optimize_rsi_thresholds()
    
    print(f"✓ RSI阈值优化完成")
    
    # 打印结果
    for rsi_key, result in results.items():
        if 'optimal_threshold' in result:
            print(f"  {rsi_key}: 阈值={result['optimal_threshold']:.1f}, "
                  f"精确率={result.get('precision_at_threshold', 0):.3f}, "
                  f"召回率={result.get('recall_at_threshold', 0):.3f}, "
                  f"F1={result['best_f1']:.3f}")
    
    return results


def optimize_capital_thresholds_full(data):
    """优化资金强度动态阈值"""
    print("\n=== 资金强度阈值优化 ===")
    
    config = {'lookback_days': 60}
    optimizer = CapitalIntensityThresholdOptimizer(data, config)
    
    capital_features = [
        'main_capital_inflow_ratio',
        'large_order_ratio',
        'capital_persistence',
        'northbound_flow_ratio'
    ]
    
    results = optimizer.optimize_capital_features(
        capital_features,
        data['target'],
        recall_constraint=0.7
    )
    
    print(f"✓ 资金强度阈值优化完成")
    
    # 打印结果
    for feature, result in results.items():
        if 'optimal_threshold' in result:
            print(f"  {feature}: 阈值={result['optimal_threshold']:.4f}, "
                  f"F1={result['performance'].get('f1', 0):.3f}")
    
    return results


def train_and_evaluate_model(data, features_dict):
    """训练XGBoost模型并评估"""
    print("\n=== 模型训练与评估 ===")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    # 准备特征
    all_features = []
    for category, info in features_dict.items():
        all_features.extend(info['features'])
    
    # 划分数据
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    test_data = data.iloc[train_size+val_size:]
    
    X_train = train_data[all_features].fillna(0)
    y_train = train_data['target']
    
    X_val = val_data[all_features].fillna(0)
    y_val = val_data['target']
    
    X_test = test_data[all_features].fillna(0)
    y_test = test_data['target']
    
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # 评估
    def calculate_metrics(y_true, y_pred):
        return {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print(f"✓ 模型训练完成")
    print(f"  训练集: 精确率={train_metrics['precision']:.3f}, "
          f"召回率={train_metrics['recall']:.3f}, F1={train_metrics['f1']:.3f}")
    print(f"  验证集: 精确率={val_metrics['precision']:.3f}, "
          f"召回率={val_metrics['recall']:.3f}, F1={val_metrics['f1']:.3f}")
    print(f"  测试集: 精确率={test_metrics['precision']:.3f}, "
          f"召回率={test_metrics['recall']:.3f}, F1={test_metrics['f1']:.3f}")
    
    # 约束优化
    print("\n--- 约束优化（召回率≥70%）---")
    from stock_system.constrained_optimizer import ConstrainedOptimizer
    
    constrained_optimizer = ConstrainedOptimizer(
        X_val.values,
        y_val.values,
        model,
        config={'min_recall': 0.70, 'threshold_bounds': [0.3, 0.7]}
    )
    
    optimal_threshold, optimized_metrics = constrained_optimizer.optimize_threshold_for_f1()
    
    print(f"  最优阈值: {optimal_threshold:.3f}")
    print(f"  优化后验证集: 精确率={optimized_metrics['precision']:.3f}, "
          f"召回率={optimized_metrics['recall']:.3f}, F1={optimized_metrics['f1']:.3f}")
    
    # 使用最优阈值在测试集上评估
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred_opt = (y_test_proba > optimal_threshold).astype(int)
    test_opt_metrics = calculate_metrics(y_test, y_test_pred_opt)
    
    print(f"  优化后测试集: 精确率={test_opt_metrics['precision']:.3f}, "
          f"召回率={test_opt_metrics['recall']:.3f}, F1={test_opt_metrics['f1']:.3f}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'optimized_metrics': optimized_metrics,
        'test_optimized_metrics': test_opt_metrics,
        'optimal_threshold': optimal_threshold
    }


def test_dynamic_threshold_adjustment(data):
    """测试动态阈值调整机制"""
    print("\n=== 动态阈值调整验证 ===")
    
    # 创建调整器
    adjuster = DynamicThresholdAdjuster(base_threshold=0.5)
    
    # 分析市场条件
    analyzer = MarketConditionAnalyzer(data['close'], window=20)
    
    # 测试不同时期
    test_periods = [
        ('早期', 100),
        ('中期', len(data)//2),
        ('近期', len(data)-20)
    ]
    
    for period_name, idx in test_periods:
        if idx < 20:
            continue
            
        conditions = analyzer.analyze_market_conditions(idx-20, idx)
        
        # 动态调整阈值
        adjusted = adjuster.adjust_threshold(
            {
                'volatility': conditions['volatility'],
                'trend': conditions['trend'],
                'recent_precision': 0.45,
                'recent_recall': 0.72
            }
        )
        
        print(f"  {period_name} (日期: {data.index[idx].date()}): "
              f"趋势={conditions['trend']}, 波动率={conditions['volatility']:.4f}, "
              f"阈值调整: 0.5 -> {adjusted:.3f}")
    
    print(f"✓ 动态阈值调整验证完成")


def run_complete_validation():
    """运行完整的实战验证流程"""
    print("="*80)
    print("自动化阈值优化系统 - 实战验证")
    print("="*80)
    
    # 1. 生成模拟A股数据
    print("\n【步骤1】生成模拟A股历史数据")
    data = generate_realistic_a_stock_data(n_days=1000)
    
    # 2. 计算三维度特征
    print("\n【步骤2】计算三维度特征")
    features_dict = calculate_features(data)
    
    # 3. 优化单特征阈值
    print("\n【步骤3】优化单特征阈值")
    single_thresholds = optimize_single_features(data, features_dict)
    
    # 4. 优化RSI阈值
    print("\n【步骤4】优化RSI多周期阈值")
    rsi_results = optimize_rsi_thresholds_full(data)
    
    # 5. 优化资金强度阈值
    print("\n【步骤5】优化资金强度阈值")
    capital_results = optimize_capital_thresholds_full(data)
    
    # 6. 训练模型和约束优化
    print("\n【步骤6】模型训练与约束优化")
    model_results = train_and_evaluate_model(data, features_dict)
    
    # 7. 动态阈值调整验证
    print("\n【步骤7】动态阈值调整验证")
    test_dynamic_threshold_adjustment(data)
    
    # 8. 生成对比报告
    print("\n【步骤8】生成性能对比报告")
    print("\n" + "="*80)
    print("优化前后性能对比")
    print("="*80)
    
    baseline_metrics = model_results['val_metrics']
    optimized_metrics = model_results['optimized_metrics']
    test_opt_metrics = model_results['test_optimized_metrics']
    
    comparison = pd.DataFrame({
        '指标': ['精确率', '召回率', 'F1分数', '准确率'],
        '基准模型': [
            baseline_metrics['precision'],
            baseline_metrics['recall'],
            baseline_metrics['f1'],
            baseline_metrics['accuracy']
        ],
        '约束优化(验证集)': [
            optimized_metrics['precision'],
            optimized_metrics['recall'],
            optimized_metrics['f1'],
            '-'
        ],
        '约束优化(测试集)': [
            test_opt_metrics['precision'],
            test_opt_metrics['recall'],
            test_opt_metrics['f1'],
            test_opt_metrics['accuracy']
        ],
        '目标值': ['0.45-0.50', '0.70-0.75', '0.55-0.60', '-']
    })
    
    print(comparison.to_string(index=False))
    
    # 计算提升幅度
    print("\n性能提升幅度:")
    precision_improvement = (test_opt_metrics['precision'] - baseline_metrics['precision'])
    f1_improvement = (test_opt_metrics['f1'] - baseline_metrics['f1'])
    
    print(f"  精确率提升: {precision_improvement:+.2%}")
    print(f"  F1分数提升: {f1_improvement:+.2%}")
    
    # 检查是否达到目标
    print("\n目标达成情况:")
    precision_target = test_opt_metrics['precision'] >= 0.45
    recall_target = 0.70 <= test_opt_metrics['recall'] <= 0.75
    f1_target = test_opt_metrics['f1'] >= 0.55
    
    print(f"  精确率目标(≥45%): {'✓ 达成' if precision_target else '✗ 未达成'} "
          f"({test_opt_metrics['precision']:.2%})")
    print(f"  召回率目标(70-75%): {'✓ 达成' if recall_target else '✗ 未达成'} "
          f"({test_opt_metrics['recall']:.2%})")
    print(f"  F1分数目标(≥55%): {'✓ 达成' if f1_target else '✗ 未达成'} "
          f"({test_opt_metrics['f1']:.2%})")
    
    # 保存结果
    results = {
        'data_stats': {
            'total_days': len(data),
            'target_positive_rate': float(data['target'].mean()),
            'price_range': [float(data['close'].min()), float(data['close'].max())]
        },
        'feature_thresholds': single_thresholds,
        'rsi_optimization': rsi_results,
        'capital_optimization': capital_results,
        'model_performance': {
            'baseline': model_results['val_metrics'],
            'optimized_val': model_results['optimized_metrics'],
            'optimized_test': model_results['test_optimized_metrics']
        },
        'optimal_threshold': float(model_results['optimal_threshold'])
    }
    
    import json
    with open('assets/validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✓ 验证结果已保存到 assets/validation_results.json")
    
    print("\n" + "="*80)
    print("实战验证完成！")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = run_complete_validation()
