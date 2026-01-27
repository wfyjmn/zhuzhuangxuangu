# -*- coding: utf-8 -*-
"""
验证跟踪系统 (Validation Tracker)
功能：
1. 读取选股结果，创建验证记录
2. 跟踪选股后1天、3天、5天的表现
3. 计算收益率和最大回撤
4. 记录模拟交易
5. 生成验证报告
"""

import tushare as ts
from config import TUSHARE_TOKEN
import pandas as pd
import numpy as np
import time
import datetime
import os
import json
from tqdm import tqdm

# ================= 配置区域 =================


# 文件路径
PARAMS_FILE = 'strategy_params.json'
VALIDATION_RECORDS_FILE = 'validation_records.csv'
PAPER_TRADING_FILE = 'paper_trading_records.csv'

# 选股结果文件前缀
PICK_RESULT_PREFIX = 'DeepQuant_TopPicks_'
# ===========================================

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(timeout=30)


def load_params():
    """加载参数配置"""
    try:
        with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
            params = json.load(f)
        return params['params']
    except Exception as e:
        print(f"[警告] 无法加载参数配置文件: {e}")
        return None


def get_trade_context():
    """获取交易日历"""
    try:
        now_date = datetime.datetime.now().strftime('%Y%m%d')
        cal_df = pro.trade_cal(exchange='', start_date='20200101', end_date=now_date, is_open='1')
        if cal_df.empty:
            return None, None
        cal_df = cal_df.sort_values('cal_date', ascending=True).reset_index(drop=True)
        last_trade_day = cal_df['cal_date'].values[-1]
        trade_dates = cal_df['cal_date'].tolist()
        return last_trade_day, trade_dates
    except Exception as e:
        print(f"[错误] 获取交易日历失败: {e}")
        return None, None


def get_future_trade_days(start_date, trade_dates, days=5):
    """获取指定日期后的交易日"""
    try:
        if start_date not in trade_dates:
            return []
        start_idx = trade_dates.index(start_date)
        return trade_dates[start_idx+1:start_idx+1+days]
    except:
        return []


def load_validation_records():
    """加载验证记录"""
    if not os.path.exists(VALIDATION_RECORDS_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(VALIDATION_RECORDS_FILE, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"[错误] 加载验证记录失败: {e}")
        return pd.DataFrame()


def save_validation_records(df):
    """保存验证记录"""
    # 使用 utf-8-sig 编码，防止 Excel 打开乱码
    df.to_csv(VALIDATION_RECORDS_FILE, index=False, encoding='utf-8-sig')


def find_pick_result_files():
    """查找所有选股结果文件"""
    files = []
    current_dir = os.getcwd()

    for filename in os.listdir(current_dir):
        if filename.startswith(PICK_RESULT_PREFIX) and filename.endswith('.csv'):
            try:
                # 从文件名中提取日期
                date_str = filename.replace(PICK_RESULT_PREFIX, '').replace('.csv', '')
                if len(date_str) == 8 and date_str.isdigit():
                    files.append((filename, date_str))
            except:
                continue

    # 按日期排序
    files.sort(key=lambda x: x[1])
    return files


def create_validation_record(pick_df, pick_date, trade_dates):
    """为选股结果创建验证记录"""
    params = load_params()
    track_days = params.get('validation', {}).get('TRACK_DAYS', [1, 3, 5])

    records = []
    create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for _, row in pick_df.iterrows():
        ts_code = row['ts_code']
        strategy = row['strategy']
        buy_price = row['close']

        # 获取后续交易日
        future_dates = get_future_trade_days(pick_date, trade_dates, max(track_days))

        record = {
            'record_id': f"{ts_code}_{pick_date}",
            'ts_code': ts_code,
            'pick_date': pick_date,
            'strategy': strategy,
            'buy_price': buy_price,
            'status': 'validating',
            'day1_price': '',
            'day1_return': '',
            'day3_price': '',
            'day3_return': '',
            'day5_price': '',
            'day5_return': '',
            'max_drawdown': '',
            'max_price': '',
            'min_price': '',
            'validation_start_date': pick_date,
            'validation_end_date': future_dates[-1] if future_dates else '',
            'create_time': create_time,
            'update_time': create_time
        }

        records.append(record)

    return pd.DataFrame(records)


def update_validation_records(last_trade_day, trade_dates):
    """更新验证记录（获取最新数据）"""
    df = load_validation_records()
    if df.empty:
        print("[信息] 没有需要更新的验证记录")
        return

    print(f"[系统] 更新验证记录，共 {len(df)} 条")

    update_count = 0
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="更新进度"):
        ts_code = row['ts_code']
        # 兼容不同格式的日期字段
        pick_date = row['pick_date'] if 'pick_date' in row.index else row.get('trade_date')
        buy_price = float(row['buy_price'])

        # 获取从选股日到最新交易日的所有数据
        try:
            df_daily = pro.daily(ts_code=ts_code, start_date=pick_date, end_date=last_trade_day)

            if df_daily.empty:
                continue

            df_daily = df_daily.sort_values('trade_date').reset_index(drop=True)

            # 找到选股日的索引
            pick_idx = df_daily[df_daily['trade_date'] == pick_date].index
            if len(pick_idx) == 0:
                continue
            pick_idx = pick_idx[0]

            # 获取后续交易日数据（最多取到今天）
            future_data = df_daily.iloc[pick_idx+1:].reset_index(drop=True)

            # 计算1天、3天、5天收益率
            updates = {}

            if len(future_data) >= 1:
                day1_price = future_data.iloc[0]['close']
                day1_return = (day1_price - buy_price) / buy_price * 100
                updates['day1_price'] = day1_price
                updates['day1_return'] = round(day1_return, 2)

            if len(future_data) >= 3:
                day3_price = future_data.iloc[2]['close']
                day3_return = (day3_price - buy_price) / buy_price * 100
                updates['day3_price'] = day3_price
                updates['day3_return'] = round(day3_return, 2)

            if len(future_data) >= 5:
                day5_price = future_data.iloc[4]['close']
                day5_return = (day5_price - buy_price) / buy_price * 100
                updates['day5_price'] = day5_price
                updates['day5_return'] = round(day5_return, 2)

            # 计算最大回撤和最高价、最低价
            if len(future_data) > 0:
                max_price = future_data['high'].max()
                min_price = future_data['low'].min()
                max_drawdown = (min_price - buy_price) / buy_price * 100
                updates['max_price'] = max_price
                updates['min_price'] = min_price
                updates['max_drawdown'] = round(max_drawdown, 2)

            # 更新数据
            for key, value in updates.items():
                df.loc[idx, key] = value

            df.loc[idx, 'update_time'] = current_time

            # 判断是否完成验证
            track_days = [1, 3, 5]
            if not pd.isna(df.loc[idx, 'day5_return']):
                df.loc[idx, 'status'] = 'completed'
            elif not pd.isna(df.loc[idx, 'day3_return']):
                df.loc[idx, 'status'] = 'validating_3days'
            elif not pd.isna(df.loc[idx, 'day1_return']):
                df.loc[idx, 'status'] = 'validating_1day'

            update_count += 1
            time.sleep(0.1)  # 防止请求过快

        except Exception as e:
            print(f"[警告] 更新股票 {ts_code} 失败: {e}")
            continue

    if update_count > 0:
        save_validation_records(df)
        print(f"[系统] 成功更新 {update_count} 条验证记录")
    else:
        print("[信息] 没有数据需要更新")


def create_paper_trade_record(pick_df, pick_date):
    """创建模拟交易记录"""
    if not os.path.exists(PAPER_TRADING_FILE):
        # 创建空文件
        pd.DataFrame(columns=['trade_date', 'ts_code', 'name', 'strategy', 'action', 'price',
                            'quantity', 'amount', 'commission', 'total_amount', 'stop_loss',
                            'take_profit', 'reason', 'status', 'create_time']).to_csv(
                            PAPER_TRADING_FILE, index=False, encoding='utf-8-sig')

    # 读取现有记录
    df = pd.read_csv(PAPER_TRADING_FILE, encoding='utf-8-sig')

    params = load_params()
    position_ratio = params.get('validation', {}).get('MAX_POSITION_PER_STOCK', 10)
    stop_loss = params.get('validation', {}).get('STOP_LOSS_RATIO', -0.08)

    new_records = []
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for _, row in pick_df.iterrows():
        ts_code = row['ts_code']
        name = row.get('name', '')
        strategy = row['strategy']
        buy_price = row['close']

        # 假设每次买入固定金额，根据价格计算数量
        # 这里简化为固定数量，实际可以根据资金管理调整
        quantity = 1000  # 固定买入1000股
        amount = buy_price * quantity
        commission = amount * 0.0003  # 假设佣金为0.03%
        total_amount = amount + commission

        stop_loss_price = buy_price * (1 + stop_loss)

        record = {
            'trade_date': pick_date,
            'ts_code': ts_code,
            'name': name,
            'strategy': strategy,
            'action': 'BUY',
            'price': buy_price,
            'quantity': quantity,
            'amount': amount,
            'commission': commission,
            'total_amount': total_amount,
            'stop_loss': stop_loss_price,
            'take_profit': '',
            'reason': f"策略选中：{strategy}",
            'status': 'open',
            'create_time': current_time
        }

        new_records.append(record)

    # 添加新记录
    df = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)
    df.to_csv(PAPER_TRADING_FILE, index=False, encoding='utf-8-sig')

    print(f"[系统] 创建 {len(new_records)} 条模拟交易记录")


def generate_validation_report():
    """生成验证报告"""
    df = load_validation_records()
    if df.empty:
        print("[信息] 没有验证记录，无法生成报告")
        return

    print("\n" + "="*80)
    print("【📊 验证报告】")
    print("="*80)

    # 总体统计
    total = len(df)
    completed = len(df[df['status'] == 'completed'])
    validating = len(df[df['status'].str.contains('validating', na=False)])

    print(f"\n[总体概况]")
    print(f"  总记录数: {total}")
    print(f"  已完成验证: {completed}")
    print(f"  验证中: {validating}")

    # 按策略统计
    if completed > 0:
        df_completed = df[df['status'] == 'completed'].copy()

        print(f"\n[策略表现（5天收益率）]")

        for strategy in df_completed['strategy'].unique():
            df_strategy = df_completed[df_completed['strategy'] == strategy]
            avg_return = df_completed['day5_return'].mean()
            win_rate = (df_completed['day5_return'] > 0).sum() / len(df_completed) * 100
            max_return = df_completed['day5_return'].max()
            min_return = df_completed['day5_return'].min()

            print(f"  {strategy}:")
            print(f"    平均收益: {avg_return:.2f}%")
            print(f"    胜率: {win_rate:.1f}%")
            print(f"    最大收益: {max_return:.2f}%")
            print(f"    最小收益: {min_return:.2f}%")

    # 最新记录
    print(f"\n[最新选股记录]")
    date_column = 'pick_date' if 'pick_date' in df.columns else 'trade_date'
    latest_records = df.sort_values(date_column, ascending=False).head(10)
    cols = [date_column if c == 'pick_date' else c for c in ['pick_date', 'ts_code', 'strategy', 'buy_price', 'day1_return', 'day3_return', 'day5_return', 'status']]
    # 过滤存在的列
    available_cols = [col for col in cols if col in latest_records.columns]
    print(latest_records[available_cols].to_string(index=False))

    print("="*80)


def run_validation_tracker(mode='all'):
    """
    运行验证跟踪系统

    参数:
        mode: 运行模式
            - 'scan': 扫描新的选股结果文件，创建验证记录
            - 'update': 更新现有验证记录
            - 'report': 生成验证报告
            - 'all': 执行全部流程（默认）
    """
    print("="*80)
    print("   DeepQuant 验证跟踪系统")
    print("="*80)

    # 获取交易日历
    last_trade_day, trade_dates = get_trade_context()
    if not last_trade_day:
        print("[错误] 无法获取交易日历")
        return

    print(f"[系统] 最新交易日: {last_trade_day}")

    if mode in ['scan', 'all']:
        print("\n[步骤 1] 扫描选股结果文件...")
        pick_files = find_pick_result_files()
        print(f"[系统] 找到 {len(pick_files)} 个选股结果文件")

        df_records = load_validation_records()
        # 兼容不同格式的验证记录文件（pick_date 或 trade_date）
        date_column = 'pick_date' if 'pick_date' in df_records.columns else 'trade_date'
        existing_picks = set(df_records[date_column].tolist()) if not df_records.empty else set()

        for filename, date_str in pick_files:
            if date_str in existing_picks:
                print(f"[跳过] {filename} 已存在验证记录")
                continue

            try:
                df_pick = pd.read_csv(filename, encoding='utf-8-sig')
                print(f"[处理] 读取 {filename}，共 {len(df_pick)} 只股票")

                # 创建验证记录
                df_new_records = create_validation_record(df_pick, date_str, trade_dates)

                # 合并记录
                if df_records.empty:
                    df_records = df_new_records
                else:
                    df_records = pd.concat([df_records, df_new_records], ignore_index=True)

                # 创建模拟交易记录
                create_paper_trade_record(df_pick, date_str)

                print(f"[完成] 为 {filename} 创建验证记录")

            except Exception as e:
                print(f"[错误] 处理 {filename} 失败: {e}")
                continue

        # 保存验证记录
        if not df_records.empty:
            save_validation_records(df_records)
            print(f"[系统] 验证记录已保存，总计 {len(df_records)} 条")

    if mode in ['update', 'all']:
        print("\n[步骤 2] 更新验证数据...")
        update_validation_records(last_trade_day, trade_dates)

    if mode in ['report', 'all']:
        print("\n[步骤 3] 生成验证报告...")
        generate_validation_report()

    print("\n[系统] 验证跟踪系统运行完成")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'all'

    run_validation_tracker(mode=mode)
