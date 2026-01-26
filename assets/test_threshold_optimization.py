"""
自动化阈值优化系统测试脚本
验证所有优化模块的功能
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/workspace/projects/src')

# 导入测试模块
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


def generate_test_data(n_samples=1000):
    """生成测试数据"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'main_capital_inflow_ratio': np.random.uniform(-0.1, 0.2, n_samples),
        'large_order_ratio': np.random.uniform(0.1, 0.5, n_samples),
        'capital_persistence': np.random.uniform(0.3, 0.9, n_samples),
        'sector_heat_index': np.random.uniform(50, 90, n_samples),
        'stock_sentiment_score': np.random.uniform(40, 95, n_samples),
        'market_breadth': np.random.uniform(0.4, 0.8, n_samples),
        'rsi_6': np.random.uniform(20, 80, n_samples),
        'rsi_12': np.random.uniform(25, 75, n_samples),
        'rsi_24': np.random.uniform(30, 70, n_samples),
        'volume_ratio': np.random.uniform(0.5, 3.0, n_samples),
    })
    
    # 生成目标变量（基于特征的简单规则）
    target = (
        (data['main_capital_inflow_ratio'] > 0.05) |
        (data['rsi_6'] > 60) |
        (data['stock_sentiment_score'] > 70)
    ).astype(int)
    
    data['target'] = target
    
    return data


def test_auto_threshold_optimizer():
    """测试自动化阈值优化器"""
    print("\n=== 测试 AutoThresholdOptimizer ===")
    
    data = generate_test_data()
    features = ['main_capital_inflow_ratio', 'rsi_6', 'volume_ratio']
    
    optimizer = AutoThresholdOptimizer(data, 'target', features)
    thresholds = optimizer.calculate_optimal_thresholds(method='ensemble')
    
    print(f"✓ 成功优化 {len(thresholds)} 个特征的阈值")
    
    # 打印部分结果
    for feature, info in thresholds.items():
        print(f"  {feature}: 最优阈值 = {info['optimal']:.4f} (方法: {info['method']})")
    
    # 获取摘要
    summary = optimizer.get_threshold_summary()
    print(f"✓ 生成阈值优化摘要，共 {len(summary)} 行")
    
    return True


def test_rsi_threshold_optimizer():
    """测试RSI阈值优化器"""
    print("\n=== 测试 RSIThresholdOptimizer ===")
    
    # 生成价格数据
    np.random.seed(42)
    price_data = pd.Series(
        100 + np.cumsum(np.random.randn(1000) * 0.5),
        index=pd.date_range('2020-01-01', periods=1000)
    )
    
    # 计算目标收益率（价格上涨为1，下跌为0）
    returns = price_data.pct_change()
    target_returns = (returns > 0).shift(-1).fillna(0).astype(int)
    
    # 配置
    config = {
        'periods': [6, 12, 14],
        'recall_constraint': 0.7
    }
    
    optimizer = RSIThresholdOptimizer(price_data, target_returns, config)
    results = optimizer.optimize_rsi_thresholds()
    
    print(f"✓ 成功优化 {len(results)} 个RSI周期的阈值")
    
    # 打印结果
    for rsi_key, result in results.items():
        if 'optimal_threshold' in result:
            print(f"  {rsi_key}: 阈值={result['optimal_threshold']:.1f}, F1={result['best_f1']:.3f}")
    
    # 测试动态阈值
    rsi_values = calculate_rsi(price_data, 6)
    dynamic_threshold = optimizer.calculate_dynamic_threshold(rsi_values, 6, lookback=60)
    print(f"✓ 计算动态阈值，序列长度: {len(dynamic_threshold)}")
    
    return True


def test_capital_threshold_optimizer():
    """测试资金强度阈值优化器"""
    print("\n=== 测试 CapitalIntensityThresholdOptimizer ===")
    
    data = generate_test_data()
    data['market_capital_inflow'] = np.random.uniform(-0.05, 0.15, len(data))
    
    config = {'lookback_days': 60}
    optimizer = CapitalIntensityThresholdOptimizer(data, config)
    
    # 测试单个特征优化
    thresholds = optimizer.learn_capital_thresholds(
        'main_capital_inflow_ratio',
        'market_capital_inflow'
    )
    
    print(f"✓ 成功学习资金强度阈值")
    print(f"  推荐阈值: {thresholds['recommended']:.4f}")
    print(f"  方法: rolling_q85, zscore, relative")
    
    # 测试资金持续性
    capital_persistence = calculate_capital_persistence(
        data['main_capital_inflow_ratio'],
        window=5
    )
    print(f"✓ 计算资金持续性，均值: {capital_persistence.mean():.3f}")
    
    # 测试资金动量
    capital_momentum = calculate_capital_momentum(
        data['main_capital_inflow_ratio']
    )
    print(f"✓ 计算资金动量，均值: {capital_momentum.mean():.3f}")
    
    return True


def test_constrained_optimizer():
    """测试约束优化器"""
    print("\n=== 测试 ConstrainedOptimizer ===")
    
    # 生成模拟数据
    np.random.seed(42)
    X_val = np.random.rand(200, 5)
    y_val = (X_val[:, 0] + X_val[:, 1] > 1.0).astype(int)
    
    # 创建一个简单的模型
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_val, y_val)
    
    # 创建优化器
    config = {
        'min_recall': 0.7,
        'threshold_bounds': [0.3, 0.7]
    }
    optimizer = ConstrainedOptimizer(X_val, y_val, model, config)
    
    # 测试F1优化
    threshold, metrics = optimizer.optimize_threshold_for_f1()
    print(f"✓ 优化F1分数:")
    print(f"  最优阈值: {threshold:.3f}")
    print(f"  精确率: {metrics['precision']:.3f}")
    print(f"  召回率: {metrics['recall']:.3f}")
    print(f"  F1分数: {metrics['f1']:.3f}")
    
    # 测试精确率优化
    threshold, metrics = optimizer.optimize_threshold_for_precision()
    print(f"✓ 优化精确率:")
    print(f"  最优阈值: {threshold:.3f}")
    print(f"  精确率: {metrics['precision']:.3f}")
    
    return True


def test_dynamic_threshold_adjuster():
    """测试动态阈值调整器"""
    print("\n=== 测试 DynamicThresholdAdjuster ===")
    
    adjuster = DynamicThresholdAdjuster(base_threshold=0.5)
    
    # 测试不同市场条件
    test_cases = [
        {'volatility': 0.03, 'trend': 'bearish', 'recent_precision': 0.4, 'recent_recall': 0.8},
        {'volatility': 0.003, 'trend': 'bullish', 'recent_precision': 0.6, 'recent_recall': 0.72},
        {'volatility': 0.01, 'trend': 'neutral', 'recent_precision': 0.5, 'recent_recall': 0.7},
    ]
    
    for i, conditions in enumerate(test_cases):
        adjusted = adjuster.adjust_threshold(conditions)
        print(f"  案例{i+1}: {conditions['trend']}市场, 波动率={conditions['volatility']:.3f}")
        print(f"    基础阈值: 0.5 -> 调整后: {adjusted:.3f}")
    
    # 测试市场条件分析器
    np.random.seed(42)
    price_data = pd.Series(
        100 + np.cumsum(np.random.randn(200) * 0.5),
        index=pd.date_range('2023-01-01', periods=200)
    )
    
    analyzer = MarketConditionAnalyzer(price_data, window=20)
    conditions = analyzer.analyze_market_conditions()
    print(f"✓ 分析市场条件:")
    print(f"  波动率: {conditions['volatility']:.3f}")
    print(f"  趋势: {conditions['trend']}")
    print(f"  价格变化: {conditions['price_change']:.2%}")
    
    return True


def test_multi_objective_optimizer():
    """测试多目标优化器"""
    print("\n=== 测试 MultiObjectiveOptimizer ===")
    
    data = generate_test_data()
    features = ['main_capital_inflow_ratio', 'rsi_6', 'volume_ratio']
    
    # 创建简单的信号生成函数
    def generate_signals(data, features, params):
        signals = pd.Series(0, index=data.index)
        
        if 'main_capital_inflow_ratio' in data.columns:
            signal1 = (data['main_capital_inflow_ratio'] > params.get('capital_threshold', 0.05)).astype(int)
            signals += signal1
        
        if 'rsi_6' in data.columns:
            signal2 = (data['rsi_6'] > params.get('rsi_threshold', 50)).astype(int)
            signals += signal2
        
        if 'volume_ratio' in data.columns:
            signal3 = (data['volume_ratio'] > params.get('volume_threshold', 2.0)).astype(int)
            signals += signal3
        
        return (signals > 0).astype(int)
    
    optimizer = MultiObjectiveOptimizer(
        data, features, 'target',
        config={'n_trials': 20, 'recall_constraint': 0.7}
    )
    
    best_params, best_score = optimizer.optimize(generate_signals)
    
    print(f"✓ 多目标优化完成:")
    print(f"  最佳得分: {best_score:.4f}")
    print(f"  最佳参数:")
    for key, value in best_params.items():
        print(f"    {key}: {value:.3f}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始自动化阈值优化系统测试")
    print("=" * 60)
    
    tests = [
        ("自动化阈值优化器", test_auto_threshold_optimizer),
        ("RSI阈值优化器", test_rsi_threshold_optimizer),
        ("资金强度阈值优化器", test_capital_threshold_optimizer),
        ("约束优化器", test_constrained_optimizer),
        ("动态阈值调整器", test_dynamic_threshold_adjuster),
        ("多目标优化器", test_multi_objective_optimizer),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✓ {name} 测试通过")
        except Exception as e:
            failed += 1
            print(f"✗ {name} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
