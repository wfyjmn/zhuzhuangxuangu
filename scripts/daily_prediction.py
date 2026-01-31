import sys
import os
import json
import pickle
import pandas as pd
from datetime import datetime

workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))

from stock_system.data_collector import MarketDataCollector
from stock_system.enhanced_features import EnhancedFeatureEngineer

def run_atm_prediction():
    print("=" * 60)
    print("🏧 自动提款机模式 - 每日选股")
    print("=" * 60)

    # 1. 加载 ATM 配置
    atm_config_path = os.path.join(workspace_path, "config/atm_strategy_config.json")
    if not os.path.exists(atm_config_path):
        print("❌ 未找到 ATM 配置文件，请先运行 optimize_threshold.py")
        return
        
    with open(atm_config_path, 'r') as f:
        atm_config = json.load(f)
        threshold = atm_config['prediction_threshold']
        
    print(f"⚙️  加载策略配置: 强力过滤阈值 > {threshold:.4f}")

    # 2. 加载模型
    with open(atm_config['model_path'], 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        feature_names = saved_data['feature_names']

    # 3. 获取全市场股票（或指定池子）
    collector = MarketDataCollector()
    engineer = EnhancedFeatureEngineer()
    
    # 示例：获取沪深300或自定义池子，这里演示取前100只活跃股
    # 实际使用建议遍历 collector.get_stock_pool_tree() 获取的全部股票
    stock_codes = collector.get_stock_pool_tree(
        pool_size=200,
        exclude_markets=['BJ'],
        exclude_board_types=['688', '300', '301']  # 排除科创板（688）、创业板（300/301）
    )
    print(f"📥 正在分析 {len(stock_codes)} 只潜力股票...")

    results = []
    
    # 获取最近数据（需要足够的历史数据来计算特征，至少60天）
    start_date = (datetime.now() - pd.Timedelta(days=100)).strftime('%Y%m%d')
    end_date = datetime.now().strftime('%Y%m%d')

    for idx, code in enumerate(stock_codes):
        try:
            # 获取数据
            df = collector.get_daily_data(code, start_date, end_date)
            if df is None or len(df) < 60:
                continue
                
            # 特征工程
            # 注意：我们需要预测的是"明天"，所以我们取最新的一行数据作为输入
            df_feat = engineer.create_all_features(df)
            
            # 取最后一行（最新交易日）
            last_row = df_feat.iloc[[-1]].copy()
            last_date = last_row['trade_date'].values[0]
            last_close = last_row['close'].values[0]
            
            # 检查是否有停牌或数据过旧
            # check_date_logic_here...
            
            # 预测
            X_input = last_row[feature_names]
            
            # 关键：获取概率，而不是直接获取 0/1
            prob = model.predict_proba(X_input)[0, 1]
            
            # 记录结果
            results.append({
                'code': code,
                'date': last_date,
                'price': last_close,
                'probability': prob,
                'main_flow': last_row['main_net_inflow'].values[0] if 'main_net_inflow' in last_row else 0,
                'turnover': last_row['turnover_rate'].values[0] if 'turnover_rate' in last_row else 0
            })
            
            print(f"\r  进度: {idx+1}/{len(stock_codes)} - 发现目标: {code} 概率: {prob:.4f}", end="")
            
        except Exception as e:
            continue

    print("\n\n" + "-" * 60)
    print("📊 分析完成，正在筛选真龙...")
    print("-" * 60)

    # 4. 筛选与排序
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        print("未获取到有效数据。")
        return

    # 核心过滤：只看概率大于阈值的
    dragons = df_res[df_res['probability'] >= threshold].copy()
    
    # 二次排序：按概率从高到低
    dragons = dragons.sort_values('probability', ascending=False)

    # 5. 输出结果
    print(f"🔍 原始推荐数: {len(df_res)}")
    print(f"🦁 过滤后真龙数: {len(dragons)} (过滤率: {1 - len(dragons)/len(df_res):.2%})")
    print("\n🏆 今日【自动提款机】精选推荐:")
    print("=" * 80)
    print(f"{'代码':<10} {'日期':<10} {'现价':<8} {'上涨概率':<10} {'主力净流':<12} {'换手率':<8}")
    print("-" * 80)
    
    for _, row in dragons.head(10).iterrows():
        star = "⭐" if row['probability'] > 0.9 else ""
        print(f"{row['code']:<10} {row['date']} {row['price']:<8.2f} {row['probability']:<10.4f} {row['main_flow']:<12.2f} {row['turnover']:<8.2f}% {star}")
    
    print("=" * 80)
    print("💡 操盘建议:")
    print("1. 概率 > 0.90 (⭐): 极高置信度，重点关注，资金驱动明显。")
    print("2. 建议结合K线形态，剔除处于明显下降通道的股票。")
    print("3. 严格止损 -5%，即使是高概率也可能失败。")
    
    # 6. 保存结果
    output_path = os.path.join(workspace_path, f"assets/atm_prediction_{datetime.now().strftime('%Y%m%d')}.csv")
    dragons.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存: {output_path}")

if __name__ == "__main__":
    run_atm_prediction()
