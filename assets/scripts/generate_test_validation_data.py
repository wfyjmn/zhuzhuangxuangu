#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据生成脚本
生成模拟的验证记录数据，用于测试遗传算法优化器
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_test_validation_records(num_records: int = 100, output_path: str = "validation_records.csv"):
    """
    生成模拟的验证记录数据
    
    Args:
        num_records: 生成记录数量
        output_path: 输出文件路径
    """
    print(f"[生成] 开始生成 {num_records} 条测试验证记录...")
    
    # 设置随机种子以保证可复现性
    np.random.seed(42)
    
    # 策略类型
    strategies = ['强攻', '洗盘', '梯量', '正常']
    strategy_weights = [0.35, 0.30, 0.25, 0.10]
    
    # 生成数据
    data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(num_records):
        # 随机选择策略
        strategy = np.random.choice(strategies, p=strategy_weights)
        
        # 生成评分（强攻策略通常分数较高，洗盘策略分数稍低）
        if strategy == '强攻':
            score = np.random.uniform(70, 95)
        elif strategy == '洗盘':
            score = np.random.uniform(60, 85)
        else:
            score = np.random.uniform(65, 90)
        
        # 生成分项得分
        s_safe = np.random.uniform(10, 25)
        s_off = np.random.uniform(15, 35)
        s_cert = np.random.uniform(10, 25)
        s_match = np.random.uniform(5, 15)
        
        # 根据策略生成不同的收益率特征
        if strategy == '强攻':
            # 强攻策略：短期高收益，但波动大
            pct_1d = np.random.normal(2, 5)
            pct_3d = np.random.normal(4, 8)
            pct_5d = np.random.normal(5, 12)
        elif strategy == '洗盘':
            # 洗盘策略：短期低收益，但后续反弹概率高
            pct_1d = np.random.normal(-1, 3)
            pct_3d = np.random.normal(2, 5)
            pct_5d = np.random.normal(5, 8)
        elif strategy == '梯量':
            # 梯量策略：稳健上涨
            pct_1d = np.random.normal(1, 2)
            pct_3d = np.random.normal(3, 4)
            pct_5d = np.random.normal(5, 6)
        else:
            # 正常策略：随机
            pct_1d = np.random.normal(0, 4)
            pct_3d = np.random.normal(1, 6)
            pct_5d = np.random.normal(2, 8)
        
        # 生成股票代码
        ts_code = f"{np.random.randint(600000, 605000)}.SH"
        
        # 生成交易日期
        trade_date = (base_date + timedelta(days=i*2)).strftime('%Y%m%d')
        
        data.append({
            'ts_code': ts_code,
            'trade_date': trade_date,
            'strategy': strategy,
            'score': round(score, 2),
            's_safe': round(s_safe, 2),
            's_off': round(s_off, 2),
            's_cert': round(s_cert, 2),
            's_match': round(s_match, 2),
            'close': round(np.random.uniform(10, 100), 2),
            'pct_chg': round(np.random.uniform(-5, 8), 2),
            'vol_ratio': round(np.random.uniform(0.5, 3.0), 2),
            'pos_ratio': round(np.random.uniform(0.1, 0.8), 2),
            'pct_1d': round(pct_1d, 2),
            'pct_3d': round(pct_3d, 2),
            'pct_5d': round(pct_5d, 2),
            'validated': True
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"[完成] 已生成 {len(df)} 条测试数据")
    print(f"[保存] 文件已保存到: {output_path}")
    
    # 打印统计信息
    print(f"\n[统计] 数据分布:")
    print(f"  - 平均1日收益: {df['pct_1d'].mean():.2f}%")
    print(f"  - 平均3日收益: {df['pct_3d'].mean():.2f}%")
    print(f"  - 平均5日收益: {df['pct_5d'].mean():.2f}%")
    print(f"  - 5日胜率: {(df['pct_5d'] > 0).sum() / len(df) * 100:.2f}%")
    print(f"\n[策略分布]:")
    for strategy in strategies:
        count = (df['strategy'] == strategy).sum()
        print(f"  - {strategy}: {count} 条 ({count/len(df)*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    # 生成100条测试数据
    generate_test_validation_records(num_records=100, output_path="validation_records.csv")
    
    print("\n[提示] 测试数据已生成，可以运行遗传算法优化器进行测试")
    print("  python genetic_optimizer.py")
