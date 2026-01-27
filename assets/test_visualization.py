"""
模型可视化监控测试脚本
演示如何生成完整的模型训练报告
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 设置环境变量
os.environ['TUSHARE_TOKEN'] = 'your_tushare_token_here'

from src.stock_system.model_reporter import ModelReporter
from src.stock_system.data_collector import MarketDataCollector
from sklearn.model_selection import train_test_split
import xgboost as xgb

def create_sample_data(n_samples=1000, n_features=10):
    """创建示例数据"""
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + 
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    feature_names = [f'特征{i+1}' for i in range(n_features)]
    
    return X, y, feature_names

def test_basic_reporting():
    """测试基础报告生成"""
    print("=" * 60)
    print("测试1: 基础模型报告生成")
    print("=" * 60)
    
    # 创建报告生成器
    reporter = ModelReporter()
    
    # 创建示例数据
    X, y, feature_names = create_sample_data(n_samples=1000)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    print("\n正在训练XGBoost模型...")
    parameters = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    model = xgb.XGBClassifier(**parameters)
    model.fit(X_train, y_train)
    print("模型训练完成")
    
    # 生成完整报告
    print("\n正在生成完整报告...")
    report_info = reporter.generate_full_report(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        parameters=parameters,
        stock_pool=None,
        industries=None
    )
    
    # 打印摘要
    reporter.print_summary(report_info)
    
    print("\n✓ 测试1完成")
    return report_info

def test_overfitting_detection():
    """测试过拟合检测"""
    print("\n" + "=" * 60)
    print("测试2: 过拟合检测")
    print("=" * 60)
    
    reporter = ModelReporter()
    
    # 创建数据（故意让训练集简单，验证集难）
    X_train, y_train, feature_names = create_sample_data(n_samples=500)
    X_val, y_val, _ = create_sample_data(n_samples=500)
    
    # 创建过拟合模型（深度过大，学习率过小）
    parameters = {
        'n_estimators': 500,
        'max_depth': 15,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    print("\n训练过拟合模型...")
    model = xgb.XGBClassifier(**parameters)
    model.fit(X_train, y_train)
    
    # 生成报告
    report_info = reporter.generate_full_report(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        parameters=parameters
    )
    
    # 检查是否检测到过拟合
    if report_info['overfitting']['is_overfitting']:
        print("\n✓ 成功检测到过拟合")
        print(f"  严重程度: {report_info['overfitting']['severity']}")
        print("  警告信息:")
        for warning in report_info['overfitting']['warnings']:
            print(f"    - {warning}")
    else:
        print("\n⚠ 未检测到过拟合（模型可能未过拟合）")
    
    reporter.print_summary(report_info)
    print("\n✓ 测试2完成")

def test_industry_sampling():
    """测试行业采样可视化"""
    print("\n" + "=" * 60)
    print("测试3: 行业采样可视化")
    print("=" * 60)
    
    # 使用真实数据采集器
    collector = MarketDataCollector()
    
    # 获取股票池
    print("\n正在获取股票池...")
    stock_pool = collector.get_stock_pool_tree(
        pool_size=50,
        market='SSE',
        exclude_st=True,
        min_days_listed=30
    )
    
    # 获取股票列表（获取行业信息）
    stock_list = collector.get_stock_list(use_cache=True)
    
    # 获取行业信息
    industries = []
    for stock_code in stock_pool:
        stock_info = stock_list[stock_list['ts_code'] == stock_code]
        if not stock_info.empty:
            industries.append(stock_info.iloc[0]['industry'])
        else:
            industries.append('未知')
    
    # 生成报告
    print(f"\n股票池大小: {len(stock_pool)}")
    print(f"行业数量: {len(set(industries))}")
    print("\n行业分布:")
    industry_counts = pd.Series(industries).value_counts()
    print(industry_counts.head(10))
    
    # 使用可视化器生成行业采样图
    from src.stock_system.visualizer import ModelVisualizer
    visualizer = ModelVisualizer()
    
    print("\n生成行业采样分布图...")
    industry_path = visualizer.plot_industry_sampling(stock_pool, industries)
    print(f"✓ 行业采样分布图已保存: {industry_path}")
    
    print("\n✓ 测试3完成")

def test_completion_flag():
    """测试训练完成标识文件"""
    print("\n" + "=" * 60)
    print("测试4: 训练完成标识文件")
    print("=" * 60)
    
    reporter = ModelReporter()
    
    # 检查是否有之前的训练
    if reporter.is_training_completed():
        print("\n✓ 检测到训练完成标识文件")
        
        # 加载最新报告
        latest_report = reporter.load_latest_report()
        
        if latest_report:
            print("\n最新训练信息:")
            print(f"  训练时间: {latest_report['timestamp']}")
            print(f"  训练轮次: {latest_report['training_epochs']}")
            print(f"  AUC: {latest_report['metrics']['auc']:.4f}")
            print(f"  准确率: {latest_report['metrics']['accuracy']:.4f}")
            print(f"  过拟合: {'是' if latest_report['overfitting']['is_overfitting'] else '否'}")
    else:
        print("\n⚠ 未找到训练完成标识文件")
        print("请先运行 test_basic_reporting() 生成训练报告")
    
    print("\n✓ 测试4完成")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("A股模型实盘对比系统 - 可视化监控测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行所有测试
        test_basic_reporting()
        test_overfitting_detection()
        test_industry_sampling()
        test_completion_flag()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        print("\n查看报告:")
        print("  - HTML报告: assets/reports/training_report_latest.html")
        print("  - 可视化图表: assets/reports/*.png")
        print("  - 标识文件: assets/reports/training_complete_latest.flag")
        print("\n可以在浏览器中打开HTML报告查看完整的训练结果和可视化图表！")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
