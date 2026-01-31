import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, auc

# 添加路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.data_collector import MarketDataCollector
from stock_system.enhanced_features import EnhancedFeatureEngineer

def find_atm_threshold():
    print("=" * 60)
    print("🚀 启动自动提款机模式：阈值优选程序")
    print("=" * 60)

    # 1. 加载模型
    model_path = os.path.join(workspace_path, "assets/models/主力资金驱动-高置信度策略_model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}")
        return

    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        feature_names = saved_data['feature_names']
        config = saved_data['config']
    
    print(f"✓ 模型加载成功")
    print(f"  当前默认阈值: 0.5")
    print(f"  当前精确率 (Default): {saved_data['metrics']['precision']:.2%}")
    print(f"  当前召回率 (Default): {saved_data['metrics']['recall']:.2%}")

    # 2. 获取验证数据 (为了演示，这里重新获取一小部分近期数据作为验证集)
    # 注意：在实际生产中，应该使用独立的测试集或保留的验证集
    collector = MarketDataCollector()
    engineer = EnhancedFeatureEngineer()
    
    print("\n⏳正在获取验证数据 (使用最近2个月数据进行校准)...")
    # 获取一部分数据用于寻找阈值
    stock_codes = collector.get_stock_pool_tree(pool_size=100) 
    df_list = []
    
    # 动态计算日期
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime('%Y%m%d')

    for code in stock_codes[:50]: # 采样50只股票做快速验证
        try:
            df = collector.get_daily_data(code, start_date, end_date)
            if df is not None and len(df) > 30:
                df = engineer.create_all_features(df)
                # 重新构建标签逻辑以保持一致
                df['future_return'] = df['close'].pct_change(5).shift(-5)
                df['label'] = (df['future_return'] >= 0.05).astype(int)
                df = df.dropna()
                df_list.append(df)
        except:
            continue
            
    if not df_list:
        print("❌ 无法获取验证数据")
        return

    val_df = pd.concat(df_list)
    X_val = val_df[feature_names]
    y_val = val_df['label']

    # 3. 预测概率
    y_scores = model.predict_proba(X_val)[:, 1]

    # 4. 计算 PR 曲线
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)

    # 5. 寻找最佳阈值 (目标：精确率 > 60%)
    target_precision = 0.60
    optimal_idx = np.argmax(precisions >= target_precision)
    
    # 如果找不到 60% 精确率的，就找 F1 最高的
    if precisions[optimal_idx] < target_precision:
        print("⚠️ 警告：无法达到 60% 精确率，切换为最大 F1 分数模式")
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)

    best_threshold = thresholds[optimal_idx]
    best_precision = precisions[optimal_idx]
    best_recall = recalls[optimal_idx]

    print("\n" + "=" * 60)
    print("🏆 提款机模式 - 最佳参数结果")
    print("=" * 60)
    print(f"🔑 最佳置信度阈值 (Threshold): {best_threshold:.4f}")
    print(f"📈 预期精确率 (Precision):      {best_precision:.2%} (每买10只，{int(best_precision*10)}只大涨)")
    print(f"🎯 预期召回率 (Recall):         {best_recall:.2%} (能抓住市场上 {best_recall*100:.1f}% 的机会)")
    
    # 6. 保存这个阈值配置
    config_path = os.path.join(workspace_path, "config/atm_strategy_config.json")
    atm_config = {
        "model_path": model_path,
        "prediction_threshold": float(best_threshold),
        "expected_precision": float(best_precision),
        "updated_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(config_path, 'w') as f:
        import json
        json.dump(atm_config, f, indent=4)
        
    print(f"\n✅ 配置文件已生成: {config_path}")
    print("下一步：运行 scripts/daily_prediction.py 使用此配置进行选股。")

if __name__ == "__main__":
    find_atm_threshold()
