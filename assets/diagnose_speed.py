# -*- coding: utf-8 -*-
"""诊断数据生成速度"""

import time
from data_warehouse import DataWarehouse
from ai_backtest_generator import AIBacktestGenerator
from feature_extractor import FeatureExtractor

warehouse = DataWarehouse()
generator = AIBacktestGenerator()
extractor = FeatureExtractor()

print("诊断数据生成速度")
print("="*80)

# 测试单只股票的完整流程
test_stock = '601598.SH'
test_date = '20240102'

print(f"\n测试股票: {test_stock}")
print(f"测试日期: {test_date}\n")

# 1. 加载历史数据
print("1. 加载历史数据（60天）...")
t1 = time.time()
hist_df = warehouse.get_stock_data(test_stock, test_date, days=60)
t2 = time.time()
print(f"   用时: {t2-t1:.2f} 秒")
print(f"   数据量: {len(hist_df)} 行")

# 2. 提取特征
print("\n2. 提取特征（22个特征）...")
t1 = time.time()
features = extractor.extract_features(hist_df)
t2 = time.time()
print(f"   用时: {t2-t1:.2f} 秒")
print(f"   特征数: {len(features)}")

# 3. 获取未来数据
print("\n3. 获取未来数据（5天）...")
t1 = time.time()
future_df = generator._get_future_data(test_stock, test_date, 5)
t2 = time.time()
print(f"   用时: {t2-t1:.2f} 秒")
print(f"   数据量: {len(future_df) if future_df is not None else 0} 行")

# 4. 计算标签
print("\n4. 计算标签...")
t1 = time.time()
buy_price = hist_df.iloc[-1]['close_qfq' if 'close_qfq' in hist_df.columns else 'close']
index_start_price = None
index_future_df = None

if index_future_df is not None:
    label = generator.calculate_label(future_df, buy_price, index_start_price, index_future_df)
else:
    # 简化版标签计算
    if future_df is not None and len(future_df) > 0:
        final_price = future_df.iloc[-1]['close_qfq' if 'close_qfq' in future_df.columns else 'close']
        final_return = (final_price - buy_price) / buy_price * 100
        label = 1 if final_return > 0 else 0
    else:
        label = 0

t2 = time.time()
print(f"   用时: {t2-t1:.2f} 秒")
print(f"   标签: {label}")

# 总计
print("\n" + "="*80)
print(f"单只股票完整流程用时: 约 {(t2-t1)*3:.2f} 秒（估算）")
print("\n如果生成 1000 只股票，预计用时: {(t2-t1)*3*1000/60:.1f} 分钟")
print("="*80)
