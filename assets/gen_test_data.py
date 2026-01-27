# -*- coding: utf-8 -*-
"""
DeepQuant 测试数据生成器
生成包含分项得分的模拟验证记录，用于测试遗传算法
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data(num_records=200, output_file="validation_records.csv"):
    print(f"[生成] 开始生成 {num_records} 条高保真测试数据...")
    np.random.seed(42) # 固定种子，保证结果可复现

    strategies = ['强攻', '洗盘', '梯量', '正常']
    # 稍微偏向强攻和洗盘
    strategy_weights = [0.3, 0.3, 0.2, 0.2]

    data = []
    base_date = datetime.now() - timedelta(days=100)

    for i in range(num_records):
        strategy = np.random.choice(strategies, p=strategy_weights)
        
        # 模拟不同策略的得分分布特征
        if strategy == '强攻':
            s_safe = np.random.uniform(10, 20) # 强攻通常位置稍高，安全分略低
            s_off = np.random.uniform(25, 35)  # 进攻分很高
            pct_5d = np.random.normal(3.0, 8.0) # 波动大，平均收益高
        elif strategy == '洗盘':
            s_safe = np.random.uniform(20, 25) # 位置低，安全分高
            s_off = np.random.uniform(10, 20)  # 缩量回调，进攻分低
            pct_5d = np.random.normal(2.0, 4.0) # 稳健
        else:
            s_safe = np.random.uniform(15, 25)
            s_off = np.random.uniform(15, 30)
            pct_5d = np.random.normal(0.5, 6.0)

        # 确定性和配合度随机
        s_cert = np.random.uniform(10, 25)
        s_match = np.random.uniform(5, 15)
        
        # 总分
        score = s_safe + s_off + s_cert + s_match
        
        # 添加一些人为的规律供AI发现：
        # 规律1：如果安全分极低，大概率大跌
        if s_safe < 12: pct_5d -= 5.0
        # 规律2：如果洗盘策略且进攻分有补偿(即>15)，大概率上涨
        if strategy == '洗盘' and s_off > 15: pct_5d += 3.0

        ts_code = f"{600000 + i:06d}.SH"
        trade_date = (base_date + timedelta(days=int(i/5))).strftime('%Y%m%d')

        data.append({
            'ts_code': ts_code,
            'trade_date': trade_date,
            'strategy': strategy,
            'score': round(score, 2),
            # 关键：必须保存分项得分，遗传算法才能重新计算总分
            's_safe': round(s_safe, 2),
            's_off': round(s_off, 2),
            's_cert': round(s_cert, 2),
            's_match': round(s_match, 2),
            'close': round(np.random.uniform(10, 50), 2),
            'pct_chg': round(np.random.uniform(-3, 9), 2),
            'pct_1d': round(np.random.normal(0, 3), 2),
            'pct_3d': round(np.random.normal(1, 4), 2),
            'pct_5d': round(pct_5d, 2), # 遗传算法主要优化目标
            'status': 'completed'
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"[完成] 数据已保存至 {output_file}")
    
    # 简单统计
    win_rate = (df['pct_5d'] > 0).sum() / len(df) * 100
    print(f"[统计] 初始5日胜率: {win_rate:.2f}% (这是基准线)")

if __name__ == "__main__":
    generate_test_data()
