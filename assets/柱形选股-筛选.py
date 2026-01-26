# -*- coding: utf-8 -*-
"""
第一轮筛选：智能精选 (宽进严出版)
修复：放宽缩量洗盘的准入门槛，解决"源头无水"的问题
"""

import tushare as ts
import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm

# ================= 用户配置区域 =================
MY_TOKEN = '8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7'

# 策略阈值
YEAR_WINDOW = 250
BATCH_SIZE = 50
REQUEST_INTERVAL = 0.3

# [关键] 严选阈值 (此处稍微放宽，把过滤留给第二轮)
HIGH_RISK_POS = 0.8  # 放宽到 0.8，防止漏掉强势突破的票
STRONG_CHG_PCT = 2.5  # 强攻：涨幅 > 2.5% 即可纳入观察
# ==============================================

ts.set_token(MY_TOKEN)
pro = ts.pro_api(timeout=15)


def get_trade_context():
    try:
        now_date = datetime.datetime.now().strftime('%Y%m%d')
        cal_df = pro.trade_cal(exchange='', start_date='20200101', end_date=now_date, is_open='1')
        if cal_df.empty: return None, None
        cal_df = cal_df.sort_values('cal_date', ascending=True).reset_index(drop=True)
        last_trade_day = cal_df['cal_date'].values[-1]
        start_date = cal_df['cal_date'].values[max(0, len(cal_df) - 400)]
        print(f"[系统] 锁定最新交易日: {last_trade_day}")
        return last_trade_day, start_date
    except:
        return None, None


def get_basic_pool():
    print("-" * 30)
    print("[1] 全市场扫描与板块剔除...")
    for _ in range(3):
        try:
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,market,industry')
            break
        except:
            time.sleep(1)
    else:
        return []

    # 基础剔除
    df = df[~df['name'].str.contains('ST')]
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[~df['ts_code'].str.startswith('30')]
    df = df[~df['ts_code'].str.endswith('.BJ')]

    return df['ts_code'].tolist()


def calculate_single_stock(sub_df, target_date):
    # 至少需要 30 天数据
    df = sub_df.sort_values('trade_date', ascending=True).reset_index(drop=True)
    if len(df) < 30: return None
    if df.iloc[-1]['trade_date'] != target_date: return None

    # === 数据准备 ===
    curr = df.iloc[-1]
    prev_1 = df.iloc[-2]
    prev_2 = df.iloc[-3]

    vol = curr['vol']
    vol_ref1 = prev_1['vol']
    vol_ref2 = prev_2['vol']

    # 均量
    vol_ma5 = df['vol'].iloc[-5:].mean()
    if vol_ma5 == 0: return None

    # 量比 (当前量 / 5日均量)
    vol_ratio = vol / vol_ma5

    # 高量柱参考 (前3天均量)
    vol_ma3_pre = df['vol'].iloc[-4:-1].mean()

    # 5日最低量
    min_vol_5 = df['vol'].iloc[-5:].min()

    # 均线计算
    ma5 = df['close'].iloc[-5:].mean()
    ma20 = df['close'].iloc[-20:].mean()
    ma60 = df['close'].iloc[-60:].mean()

    # 位置计算
    long_term_df = df.iloc[-YEAR_WINDOW:]
    hhv = long_term_df['high'].max()
    llv = long_term_df['low'].min()
    pos_ratio = 0.5
    if hhv != llv:
        pos_ratio = (curr['close'] - llv) / (hhv - llv)

    # === [关键过滤 1] 剔除绝对高位 ===
    if pos_ratio > HIGH_RISK_POS: return None

    # === 形态识别 (打标签) ===
    patterns = []

    # 1. 倍量
    if vol_ref1 > 0 and (vol / vol_ref1) >= 1.9: patterns.append('倍量')
    # 2. 缩量 (严谨定义：连续3天递减)
    if vol_ref2 > 0 and vol < vol_ref1 and vol_ref1 < vol_ref2: patterns.append('连续缩量')
    # 3. 梯量 (连续3天递增)
    if vol_ref2 > 0 and vol > vol_ref1 and vol_ref1 > vol_ref2: patterns.append('梯量')
    # 4. 高量
    if vol_ma3_pre > 0 and vol > (vol_ma3_pre * 1.5): patterns.append('高量')
    # 5. 低量 (5日地量)
    if vol == min_vol_5: patterns.append('地量')
    # 6. 平量
    if vol_ref1 > 0 and abs(vol - vol_ref1) / vol_ref1 <= 0.05: patterns.append('平量')

    # [关键修复] 增加"普通缩量"标签，只要比5日均量小就算，不要求连续3天
    if vol_ratio < 0.9: patterns.append('缩量')

    # [致命BUG修复] 删除 "if not patterns: return None"
    # 我们依靠下面的 Strategy 逻辑来决定去留，而不是依靠死板的形态标签

    # === [关键过滤 2] 策略精选逻辑 ===
    strategy_type = None
    reason = ""

    # 策略 A: 【低位强攻】
    if pos_ratio < 0.40:  # 放宽到 0.4
        if vol_ratio >= 1.5:  # 只要放量
            if curr['pct_chg'] > STRONG_CHG_PCT:  # 且大涨
                if curr['close'] > curr['open']:  # 阳线
                    if curr['close'] > ma20:
                        strategy_type = "★低位强攻"
                        reason = f"底部放量,涨幅{curr['pct_chg']:.1f}%"

    # 策略 B: 【缩量洗盘】 (大幅优化，更容易选中)
    if not strategy_type and pos_ratio < 0.65:  # 中位也可以洗盘
        # 核心条件1：趋势向上 (60日线向上，或者20>60)
        if ma20 > ma60:
            # 核心条件2：缩量 (量比 < 1.0 即可，不用非得地量)
            if vol_ratio < 1.05:
                # 核心条件3：价格支撑 (允许瞬间击穿20日线，但不能偏离太远)
                # 只要收盘价 > 20日线 * 0.98 就算有效支撑
                if curr['close'] > ma20 * 0.98 and curr['close'] > ma60:
                    # 核心条件4：未出现暴跌
                    if -5.0 < curr['pct_chg'] < 3.5:
                        strategy_type = "☆缩量洗盘"
                        reason = "上升趋势中缩量回调,支撑有效"

    # 策略 C: 【梯量上行】
    if not strategy_type and pos_ratio < 0.6:
        if '梯量' in patterns:
            # 必须是红盘或者微跌
            if curr['pct_chg'] > -1 and curr['close'] > ma5:
                strategy_type = "▲梯量上行"
                reason = "成交量温和放大,主力推升"

    # 如果不符合任何精选策略，直接丢弃
    if not strategy_type:
        return None

    return {
        'ts_code': curr['ts_code'],
        'trade_date': target_date,
        'close': curr['close'],
        'pct_chg': curr['pct_chg'],
        'vol_ratio': round(vol_ratio, 2),
        'strategy': strategy_type,
        'patterns': '+'.join(patterns),
        'position_ratio': round(pos_ratio, 2),
        'reason': reason
    }


def process_batch(batch_codes, start_date, end_date):
    results = []
    batch_ts_codes = ",".join(batch_codes)
    try:
        df_batch = pro.daily(ts_code=batch_ts_codes, start_date=start_date, end_date=end_date)
    except:
        return []

    if df_batch.empty: return []
    grouped = df_batch.groupby('ts_code')

    for ts_code in batch_codes:
        if ts_code in grouped.groups:
            res = calculate_single_stock(grouped.get_group(ts_code), end_date)
            if res: results.append(res)
    return results


def run_screener():
    target_date, start_date = get_trade_context()
    if not target_date: return

    stock_list = get_basic_pool()
    if not stock_list: return

    final_results = []
    print(f"[2] 正在进行第一轮筛选 (包含缩量洗盘)...")

    with tqdm(total=len(stock_list), unit="股") as pbar:
        for i in range(0, len(stock_list), BATCH_SIZE):
            batch_codes = stock_list[i: i + BATCH_SIZE]
            batch_res = process_batch(batch_codes, start_date, target_date)
            if batch_res: final_results.extend(batch_res)
            time.sleep(REQUEST_INTERVAL)
            pbar.update(len(batch_codes))

    if not final_results:
        print("\n未发现符合条件的股票。")
        return

    res_df = pd.DataFrame(final_results)

    # 匹配名称
    print("[3] 正在生成中间报表...")
    try:
        basics = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        res_df = pd.merge(res_df, basics, on='ts_code', how='left')
    except:
        res_df['name'] = ''

    # 排序
    sort_map = {"★低位强攻": 0, "☆缩量洗盘": 1, "▲梯量上行": 2}
    res_df['sort_key'] = res_df['strategy'].map(sort_map)
    res_df = res_df.sort_values(by=['sort_key', 'pct_chg'], ascending=[True, False])

    cols = ['ts_code', 'name', 'strategy', 'close', 'pct_chg', 'vol_ratio', 'patterns', 'position_ratio', 'reason']
    res_df = res_df[cols]

    print("\n" + "=" * 80)
    print(f"【第一轮筛选完成】 交易日: {target_date}")
    print(f"入围数量: {len(res_df)} 只")

    # 统计一下各策略数量
    print(res_df['strategy'].value_counts())
    print("=" * 80)

    filename = f'Best_Pick_{target_date}.csv'
    res_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"中间结果已保存至: {filename}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(res_df.head(10))


if __name__ == '__main__':
    try:
        run_screener()
    except KeyboardInterrupt:
        print("\n用户终止")
    except Exception as e:
        print(f"错误: {e}")