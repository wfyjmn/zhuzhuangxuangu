#!/usr/bin/env python3
"""
系统测试脚本
用于测试各个模块的功能
"""
import os
import sys
import logging
from datetime import datetime

# 添加src到Python路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
src_path = os.path.join(workspace_path, "src")
sys.path.insert(0, src_path)

# 配置测试日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_collector():
    """测试数据采集器"""
    print("\n" + "=" * 60)
    print("测试1: 数据采集器")
    print("=" * 60)
    
    try:
        from stock_system.data_collector import MarketDataCollector
        
        collector = MarketDataCollector()
        
        # 测试获取股票池
        print("\n1.1 获取股票池...")
        stock_pool = collector.get_stock_pool(pool_size=5)
        print(f"   ✓ 获取股票池成功，数量: {len(stock_pool)}")
        
        if stock_pool:
            # 测试获取单只股票数据
            print(f"\n1.2 获取股票数据 ({stock_pool[0]})...")
            daily_data = collector.get_daily_data(stock_pool[0], '20240101', '20241231')
            if not daily_data.empty:
                print(f"   ✓ 获取数据成功，记录数: {len(daily_data)}")
            else:
                print(f"   ⚠ 数据为空（可能需要配置tushare token）")
        
        print("\n✅ 数据采集器测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 数据采集器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictor():
    """测试预测器"""
    print("\n" + "=" * 60)
    print("测试2: 预测器")
    print("=" * 60)
    
    try:
        from stock_system.predictor import StockPredictor
        import numpy as np
        import pandas as pd
        
        predictor = StockPredictor()
        
        # 创建测试数据
        print("\n2.1 创建测试数据...")
        test_data = pd.DataFrame({
            'ts_code': ['600000.SH'],
            'trade_date': ['20241231']
        })
        
        for feat in predictor.features:
            test_data[feat] = np.random.randn()
        
        print(f"   ✓ 测试数据创建成功，特征数: {len(predictor.features)}")
        
        # 测试预测
        print("\n2.2 执行预测...")
        result = predictor.predict(test_data)
        if not result.empty:
            print(f"   ✓ 预测成功")
            print(f"      预测标签: {result['predicted_label'].values[0]}")
            print(f"      预测概率: {result['predicted_prob'].values[0]:.4f}")
        else:
            print(f"   ⚠ 预测结果为空")
        
        print("\n✅ 预测器测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 预测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyzer():
    """测试分析器"""
    print("\n" + "=" * 60)
    print("测试3: 分析器")
    print("=" * 60)
    
    try:
        from stock_system.analyzer import PerformanceAnalyzer
        import numpy as np
        import pandas as pd
        
        analyzer = PerformanceAnalyzer()
        
        # 创建模拟的对齐数据
        print("\n3.1 创建模拟数据...")
        np.random.seed(42)
        
        aligned_df = pd.DataFrame({
            'ts_code': [f'60000{i}.SH' for i in range(20)],
            'predicted_label': np.random.randint(0, 2, 20),
            'predicted_prob': np.random.random(20),
            'actual_label': np.random.randint(0, 2, 20),
            'actual_change': np.random.randn(20) * 0.05,
            'predict_correct': [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
        })
        
        print(f"   ✓ 模拟数据创建成功，记录数: {len(aligned_df)}")
        
        # 测试指标计算
        print("\n3.2 计算指标...")
        metrics = analyzer.calculate_metrics(aligned_df)
        print(f"   ✓ 指标计算成功")
        print(f"      Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"      Precision: {metrics.get('precision', 0):.4f}")
        print(f"      Recall:    {metrics.get('recall', 0):.4f}")
        print(f"      F1 Score:  {metrics.get('f1', 0):.4f}")
        
        # 测试混淆矩阵
        print("\n3.3 生成混淆矩阵...")
        confusion_mat = analyzer.generate_confusion_matrix(aligned_df)
        print(f"   ✓ 混淆矩阵生成成功")
        
        # 测试调整判断
        print("\n3.4 判断是否需要调整...")
        should_adjust, reason = analyzer.should_trigger_adjustment(metrics)
        print(f"   ✓ 判断完成")
        print(f"      需要调整: {should_adjust}")
        if should_adjust:
            print(f"      原因: {reason}")
        
        print("\n✅ 分析器测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_tracker():
    """测试误差追踪器"""
    print("\n" + "=" * 60)
    print("测试4: 误差追踪器")
    print("=" * 60)
    
    try:
        from stock_system.error_tracker import ErrorTracker
        import numpy as np
        import pandas as pd
        
        tracker = ErrorTracker()
        
        # 创建模拟数据
        print("\n4.1 创建模拟数据...")
        aligned_df = pd.DataFrame({
            'predicted_label': np.random.randint(0, 2, 30),
            'predicted_prob': np.random.random(30),
            'actual_label': np.random.randint(0, 2, 30),
            'actual_change': np.random.randn(30) * 0.05,
            'predict_correct': [1, 1, 1, 0, 0] * 6
        })
        
        print(f"   ✓ 模拟数据创建成功")
        
        # 测试误差分析
        print("\n4.2 分析误差...")
        error_analysis = tracker.analyze_errors(aligned_df)
        print(f"   ✓ 误差分析成功")
        print(f"      误差率: {error_analysis.get('error_rate', 0)*100:.2f}%")
        
        print("\n✅ 误差追踪器测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 误差追踪器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_tuner():
    """测试参数调整器"""
    print("\n" + "=" * 60)
    print("测试5: 参数调整器")
    print("=" * 60)
    
    try:
        from stock_system.parameter_tuner import ParameterTuner
        
        tuner = ParameterTuner()
        
        # 模拟误差分析
        print("\n5.1 创建模拟数据...")
        error_analysis = {
            'false_positive_rate': 0.25,
            'false_negative_rate': 0.15
        }
        
        metrics = {
            'precision': 0.22,
            'recall': 0.72,
            'f1': 0.34
        }
        
        print(f"   ✓ 模拟数据创建成功")
        
        # 测试自动调整
        print("\n5.2 执行自动调整...")
        new_params, new_threshold, strategy = tuner.auto_adjust(
            error_analysis, metrics
        )
        
        print(f"   ✓ 自动调整成功")
        print(f"      新阈值: {new_threshold:.4f}")
        print(f"      调整策略: {strategy.get('adjust_threshold', False)}")
        
        print("\n✅ 参数调整器测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 参数调整器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_updater():
    """测试模型更新器"""
    print("\n" + "=" * 60)
    print("测试6: 模型更新器")
    print("=" * 60)
    
    try:
        from stock_system.model_updater import ModelUpdater
        import xgboost as xgb
        import pandas as pd
        import numpy as np
        
        updater = ModelUpdater()
        
        # 创建测试模型
        print("\n6.1 创建测试模型...")
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        dtrain = xgb.DMatrix(X, label=y)
        params = {'objective': 'binary:logistic', 'max_depth': 3}
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        print(f"   ✓ 测试模型创建成功")
        
        # 保存模型
        print("\n6.2 保存模型...")
        metadata = {
            'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_score': 0.75,
            'params': params
        }
        save_path = updater.save_model(model, metadata)
        print(f"   ✓ 模型保存成功: {save_path}")
        
        # 加载模型
        print("\n6.3 加载模型...")
        loaded_model, loaded_metadata = updater.load_model()
        print(f"   ✓ 模型加载成功")
        
        print("\n✅ 模型更新器测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型更新器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_manager():
    """测试缓存管理器"""
    print("\n" + "=" * 60)
    print("测试7: 缓存管理器")
    print("=" * 60)
    
    try:
        from stock_system.cache_manager import CacheManager
        
        cache = CacheManager()
        
        # 测试保存和加载
        print("\n7.1 测试缓存保存和加载...")
        test_data = {'test': 'data', 'timestamp': str(datetime.now())}
        cache_path = cache.save(test_data, 'logs', 'test_cache')
        loaded_data = cache.load('logs', 'test_cache')
        
        if loaded_data:
            print(f"   ✓ 缓存保存和加载成功")
        else:
            print(f"   ⚠ 缓存加载失败")
        
        # 测试统计
        print("\n7.2 获取缓存统计...")
        stats = cache.get_cache_stats()
        print(f"   ✓ 缓存统计: {stats.get('total_files', 0)} 个文件")
        
        print("\n✅ 缓存管理器测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 缓存管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("A股模型实盘对比系统 - 系统测试")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有测试
    test_results = []
    
    test_results.append(("数据采集器", test_data_collector()))
    test_results.append(("预测器", test_predictor()))
    test_results.append(("分析器", test_analyzer()))
    test_results.append(("误差追踪器", test_error_tracker()))
    test_results.append(("参数调整器", test_parameter_tuner()))
    test_results.append(("模型更新器", test_model_updater()))
    test_results.append(("缓存管理器", test_cache_manager()))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s} {status}")
    
    success_count = sum(1 for _, result in test_results if result)
    total_count = len(test_results)
    
    print("-" * 60)
    print(f"总计: {success_count}/{total_count} 通过")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 返回状态码
    sys.exit(0 if success_count == total_count else 1)


if __name__ == '__main__':
    main()
