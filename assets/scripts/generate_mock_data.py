#!/usr/bin/env python3
"""
生成模拟股票数据脚本
用于演示和测试训练流程，无需真实 Tushare Token
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加src到Python路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
src_path = os.path.join(workspace_path, "src")
sys.path.insert(0, src_path)

import json


def generate_stock_data(ts_code: str, start_date: str, end_date: str, 
                       base_price: float = 10.0, trend: float = 0.001):
    """
    生成单只股票的模拟日线数据
    
    Args:
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        base_price: 基准价格
        trend: 趋势因子（正数为上涨趋势，负数为下跌趋势）
        
    Returns:
        DataFrame 包含日线数据
    """
    # 生成日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # 只保留工作日
    
    # 初始化数据
    data = []
    price = base_price
    np.random.seed(hash(ts_code) % 2**32)  # 使用股票代码作为随机种子，确保数据可重现
    
    for date in dates:
        # 生成随机价格波动（带趋势）
        change_pct = np.random.normal(trend, 0.02)  # 2%的标准差
        
        # 涨跌幅限制在-10%到10%之间
        change_pct = np.clip(change_pct, -0.1, 0.1)
        
        # 计算当日价格
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price * (1 + change_pct)
        
        # 确保价格合理
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # 成交量和成交额
        volume = np.random.randint(100000, 10000000)
        amount = volume * close_price * np.random.uniform(0.9, 1.1)
        
        data.append({
            'ts_code': ts_code,
            'trade_date': date.strftime('%Y%m%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'pre_close': round(price, 2),
            'change': round(change_pct * price, 2),
            'pct_chg': round(change_pct * 100, 2),
            'vol': volume,
            'amount': round(amount, 2)
        })
        
        price = close_price
    
    df = pd.DataFrame(data)
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成技术指标特征
    
    Args:
        df: 原始日线数据
        
    Returns:
        包含特征的DataFrame
    """
    df = df.copy()
    df = df.sort_values('trade_date')
    
    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    
    # 成交量比率
    df['vol_ma5'] = df['vol'].rolling(window=5).mean()
    df['volume_ratio'] = df['vol'] / df['vol_ma5']
    
    # 换手率（模拟）
    df['turnover_rate'] = df['vol'] / 100000000 * np.random.uniform(1, 10)
    df['turnover_ratio'] = df['turnover_rate'] / df['turnover_rate'].rolling(window=5).mean()
    
    # RSI（简化版）
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD（简化版）
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    
    # KDJ（简化版）
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 布林带
    df['boll_mid'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['close'].rolling(window=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
    
    # 价格变化
    df['price_change_5d'] = df['close'].pct_change(5)
    df['price_change_10d'] = df['close'].pct_change(10)
    
    # 成交量变化
    df['volume_change_5d'] = df['vol'].pct_change(5)
    df['volume_change_10d'] = df['vol'].pct_change(10)
    
    # 振幅
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close'] * 100
    
    # 高低比
    df['high_low_ratio'] = df['high'] / df['low']
    
    # 金额比率
    df['amount_ma5'] = df['amount'].rolling(window=5).mean()
    df['amount_ratio'] = df['amount'] / df['amount_ma5']
    
    # 删除包含NaN的行
    df = df.dropna()
    
    return df


def create_target(df: pd.DataFrame, prediction_days: int = 5) -> pd.DataFrame:
    """
    创建预测目标：未来N天的涨跌
    
    Args:
        df: 包含特征的DataFrame
        prediction_days: 预测天数
        
    Returns:
        包含目标的DataFrame
    """
    df = df.copy()
    
    # 计算未来N天的收益率
    df['future_return'] = df['close'].shift(-prediction_days) / df['close'] - 1
    
    # 创建标签：未来收益率>0为1（上涨），否则为0（下跌）
    df['target'] = (df['future_return'] > 0).astype(int)
    
    # 删除最后prediction_days行（没有目标值）
    df = df[:-prediction_days]
    
    return df


def main():
    """主函数：生成完整的训练数据集"""
    print("=" * 80)
    print("生成模拟股票数据")
    print("=" * 80)
    
    # 配置参数
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    
    # 加载配置
    config_path = os.path.join(workspace_path, "config/model_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    stock_pool_size = config['data']['stock_pool_size']
    prediction_days = config['data']['prediction_days']
    
    # 日期范围：生成过去2年的数据
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=730)
    
    print(f"日期范围: {start_date.strftime('%Y%m%d')} - {end_date.strftime('%Y%m%d')}")
    print(f"股票池大小: {stock_pool_size}")
    print(f"预测天数: {prediction_days}")
    print()
    
    # 生成股票列表
    stock_codes = []
    for i in range(stock_pool_size):
        if i < stock_pool_size // 2:
            ts_code = f"{600000 + i}.SH"  # 上海股票
        else:
            ts_code = f"{000000 + (i - stock_pool_size // 2):06d}.SZ"  # 深圳股票
        stock_codes.append(ts_code)
    
    print(f"生成 {len(stock_codes)} 只股票的数据...")
    
    # 生成每只股票的数据
    all_data = []
    for i, ts_code in enumerate(stock_codes):
        print(f"\r进度: {i+1}/{len(stock_codes)} ({(i+1)/len(stock_codes)*100:.1f}%)", end='')
        
        # 随机生成不同的股票特征
        base_price = np.random.uniform(5, 50)
        trend = np.random.normal(0.0005, 0.002)  # 大部分股票有小幅上涨趋势
        
        # 生成日线数据
        df = generate_stock_data(
            ts_code=ts_code,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            base_price=base_price,
            trend=trend
        )
        
        # 生成特征
        df = generate_features(df)
        
        # 创建目标
        df = create_target(df, prediction_days)
        
        all_data.append(df)
    
    print("\n✅ 数据生成完成")
    print()
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"总数据量: {len(combined_df)} 行")
    print(f"正样本（上涨）: {(combined_df['target'] == 1).sum()} ({(combined_df['target'] == 1).sum()/len(combined_df)*100:.1f}%)")
    print(f"负样本（下跌）: {(combined_df['target'] == 0).sum()} ({(combined_df['target'] == 0).sum()/len(combined_df)*100:.1f}%)")
    print()
    
    # 保存数据
    output_dir = os.path.join(workspace_path, "assets/data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整数据集
    output_file = os.path.join(output_dir, "stock_training_data.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"✅ 完整数据集已保存: {output_file}")
    
    # 分割训练集和测试集（按时间分割）
    split_date = end_date - timedelta(days=180)  # 最后180天作为测试集
    split_date_str = split_date.strftime('%Y%m%d')
    
    train_df = combined_df[combined_df['trade_date'] <= split_date_str].copy()
    test_df = combined_df[combined_df['trade_date'] > split_date_str].copy()
    
    train_file = os.path.join(output_dir, "stock_train.csv")
    test_file = os.path.join(output_dir, "stock_test.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"✅ 训练集已保存: {train_file} ({len(train_df)} 行)")
    print(f"✅ 测试集已保存: {test_file} ({len(test_df)} 行)")
    
    # 保存数据统计信息
    stats = {
        'total_samples': len(combined_df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'positive_ratio': float(combined_df['target'].mean()),
        'date_range': {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        },
        'stock_count': stock_pool_size
    }
    
    stats_file = os.path.join(output_dir, "data_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ 数据统计已保存: {stats_file}")
    
    print()
    print("=" * 80)
    print("模拟数据生成完成！")
    print("现在可以使用这些数据训练模型了。")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
