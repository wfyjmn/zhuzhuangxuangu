#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成验证跟踪系统格式的测试数据
用于演示验证跟踪功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_validation_test_records(num_records: int = 50, output_path: str = "validation_records.csv"):
    """
    生成验证跟踪系统格式的测试数据
    
    Args:
        num_records: 生成记录数量
        output_path: 输出文件路径
    """
    print(f"[生成] 开始生成 {num_records} 条验证记录...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 基础日期
    base_date = datetime(2026, 1, 20)
    
    # 生成数据
    data = []
    
    for i in range(num_records):
        # 生成交易日期（回溯到前几天）
        trade_date = (base_date - timedelta(days=i*2)).strftime('%Y-%m-%d')
        
        # 股票代码
        ts_code = f"{np.random.randint(600000, 605000)}.SH"
        
        # 策略类型
        strategies = ['强攻', '洗盘', '梯量']
        strategy = np.random.choice(strategies)
        
        # 买入价格
        buy_price = round(np.random.uniform(10, 100), 2)
        
        # 计算收益率
        day1_return = round(np.random.uniform(-5, 8), 2)
        day3_return = round(day1_return + np.random.uniform(-3, 10), 2)
        day5_return = round(day3_return + np.random.uniform(-5, 12), 2)
        
        # 根据收益率计算后续价格
        day1_price = round(buy_price * (1 + day1_return / 100), 2)
        day3_price = round(buy_price * (1 + day3_return / 100), 2)
        day5_price = round(buy_price * (1 + day5_return / 100), 2)
        
        # 最高价和最低价
        max_price = round(buy_price * (1 + np.random.uniform(0, 0.15)), 2)
        min_price = round(buy_price * (1 - np.random.uniform(0, 0.15)), 2)
        
        # 最大回撤
        max_drawdown = round((min_price - buy_price) / buy_price * 100, 2)
        
        # 状态
        status = 'completed'
        
        # 创建时间
        create_time = (base_date - timedelta(days=i*2)).strftime('%Y-%m-%d %H:%M:%S')
        update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        data.append({
            'record_id': f"{ts_code}_{trade_date}",
            'ts_code': ts_code,
            'pick_date': trade_date,
            'strategy': strategy,
            'buy_price': buy_price,
            'status': status,
            'day1_price': day1_price,
            'day1_return': day1_return,
            'day3_price': day3_price,
            'day3_return': day3_return,
            'day5_price': day5_price,
            'day5_return': day5_return,
            'max_drawdown': max_drawdown,
            'max_price': max_price,
            'min_price': min_price,
            'validation_start_date': trade_date,
            'validation_end_date': (datetime.strptime(trade_date, '%Y-%m-%d') + timedelta(days=5)).strftime('%Y-%m-%d'),
            'create_time': create_time,
            'update_time': update_time
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"[完成] 已生成 {len(df)} 条验证记录")
    print(f"[保存] 文件已保存到: {output_path}")
    
    # 统计信息
    print(f"\n[统计] 数据分析:")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 平均1日收益: {df['day1_return'].mean():.2f}%")
    print(f"  - 平均3日收益: {df['day3_return'].mean():.2f}%")
    print(f"  - 平均5日收益: {df['day5_return'].mean():.2f}%")
    print(f"  - 5日胜率: {(df['day5_return'] > 0).sum() / len(df) * 100:.2f}%")
    print(f"  - 平均最大回撤: {df['max_drawdown'].mean():.2f}%")
    
    print(f"\n[策略分布]:")
    for strategy in df['strategy'].unique():
        count = (df['strategy'] == strategy).sum()
        print(f"  - {strategy}: {count} 条 ({count/len(df)*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    generate_validation_test_records(num_records=50, output_path="validation_records.csv")
    
    print("\n[提示] 验证记录已生成，可以运行以下命令:")
    print("  python validation_track.py report   # 查看验证报告")
    print("  python main_controller.py full     # 运行完整流程")
