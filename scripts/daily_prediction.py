import sys
import os
import pickle
import json
import pandas as pd
from datetime import datetime

# 添加路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.data_collector import MarketDataCollector
from stock_system.enhanced_features import EnhancedFeatureEngineer

def daily_prediction():
    print("=" * 80)
    print("🏦 自动提款机模式 - 每日选股程序")
    print("=" * 80)

    # 1. 加载配置
    config_path = os.path.join(workspace_path, "config/atm_strategy_config.json")
    if not os.path.exists(config_path):
        print(f"❌ 未找到配置文件: {config_path}")
        print("请先运行: python3 scripts/optimize_threshold.py")
        return

    with open(config_path, 'r') as f:
        atm_config = json.load(f)

    model_path = atm_config['model_path']
    threshold = atm_config['prediction_threshold']
    expected_precision = atm_config['expected_precision']

    print(f"📋 配置信息:")
    print(f"  模型路径: {model_path}")
    print(f"  阈值: {threshold:.4f}")
    print(f"  预期精确率: {expected_precision:.2%}")

    # 2. 加载模型
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}")
        return

    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        feature_names = saved_data['feature_names']

    print(f"✓ 模型加载成功")
    print(f"  特征数量: {len(feature_names)}")

    # 3. 获取股票池和数据
    collector = MarketDataCollector()
    engineer = EnhancedFeatureEngineer()

    print("\n⏳ 正在获取股票池...")
    stock_codes = collector.get_stock_pool_tree(pool_size=200)
    print(f"  股票池大小: {len(stock_codes)} 只")

    # 获取最新数据
    print("\n⏳ 正在获取最新数据...")
    predictions = []

    for idx, code in enumerate(stock_codes, 1):
        try:
            # 获取最近 90 天数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y%m%d')

            df = collector.get_daily_data(code, start_date, end_date)

            if df is None or len(df) < 30:
                continue

            # 创建特征
            df = engineer.create_all_features(df)

            # 只保留最后一行（最新数据）
            if len(df) == 0:
                continue

            latest = df.iloc[-1:][feature_names]

            # 预测
            prob = model.predict_proba(latest)[0, 1]

            predictions.append({
                'stock_code': code,
                'trade_date': df.iloc[-1]['trade_date'],
                'close': df.iloc[-1]['close'],
                'probability': prob,
                'is_signal': prob >= threshold
            })

            if idx % 20 == 0:
                print(f"  已处理: {idx}/{len(stock_codes)}")

        except Exception as e:
            continue

    # 4. 筛选结果
    pred_df = pd.DataFrame(predictions)
    signal_df = pred_df[pred_df['is_signal']].sort_values('probability', ascending=False)

    print("\n" + "=" * 80)
    print("🎯 选股结果")
    print("=" * 80)
    print(f"总预测股票数: {len(pred_df)}")
    print(f"符合阈值股票数: {len(signal_df)} (阈值 {threshold:.4f})")

    if len(signal_df) == 0:
        print("\n⚠️ 今日无符合条件的股票")
        print("提示: 当前市场可能处于调整期，建议耐心等待")
        return

    print(f"\n📊 推荐买入股票（按置信度排序）:")
    print("-" * 80)
    for i, row in signal_df.iterrows():
        print(f"  {row['stock_code']} | 日期: {row['trade_date']} | "
              f"收盘价: {row['close']:.2f} | 置信度: {row['probability']:.4f}")

    # 5. 保存结果
    output_path = os.path.join(workspace_path, f"assets/daily_prediction_{datetime.now().strftime('%Y%m%d')}.csv")
    signal_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存: {output_path}")

    # 6. 风险提示
    print("\n" + "=" * 80)
    print("⚠️  风险提示")
    print("=" * 80)
    print("1. 本策略基于历史数据训练，不保证未来收益")
    print("2. 建议结合基本面分析和市场情绪")
    print("3. 严格控制仓位，单只股票建议仓位不超过 5%")
    print("4. 设置止损点（建议 -8% 至 -10%）")
    print(f"5. 当前阈值精确率: {expected_precision:.2%}，仍有约 {100-expected_precision*100:.0f}% 的失败概率")

if __name__ == "__main__":
    daily_prediction()
