# -*- coding: utf-8 -*-
"""
DeepQuant Pro V2.1 - 终极修复版
修复：
1. 整合了高效的数据获取模块 (efficient_fetch_stock_data)
2. 整合了评分重算模块 (calculate_score)
3. 整合了风控双轨制 (保护缩量洗盘)
更新：支持从配置文件读取参数
"""

import tushare as ts
from config import TUSHARE_TOKEN
import pandas as pd
import numpy as np
import time
import datetime
import os
import re
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================


# 参数配置文件
PARAMS_FILE = 'strategy_params.json'

# 默认参数（如果配置文件不存在时使用）
DEFAULT_PARAMS = {
    'SCORE_THRESHOLD_NORMAL': 55,
    'SCORE_THRESHOLD_WASH': 45,
    'TURNOVER_THRESHOLD_NORMAL': 1.5,
    'TURNOVER_THRESHOLD_WASH': 0.6,
    'TOP_N_PER_STRATEGY': 5
}

# 从配置文件加载参数
def load_params():
    """加载参数配置"""
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                second_round = data.get('params', {}).get('second_round', {})
                return {
                    'SCORE_THRESHOLD_NORMAL': second_round.get('SCORE_THRESHOLD_NORMAL', 55),
                    'SCORE_THRESHOLD_WASH': second_round.get('SCORE_THRESHOLD_WASH', 45),
                    'TURNOVER_THRESHOLD_NORMAL': second_round.get('TURNOVER_THRESHOLD_NORMAL', 1.5),
                    'TURNOVER_THRESHOLD_WASH': second_round.get('TURNOVER_THRESHOLD_WASH', 0.6),
                    'TOP_N_PER_STRATEGY': second_round.get('TOP_N_PER_STRATEGY', 5)
                }
        except Exception as e:
            print(f"[警告] 加载参数配置失败，使用默认参数: {e}")
    return DEFAULT_PARAMS

# 加载参数
PARAMS = load_params()
SCORE_THRESHOLD_NORMAL = PARAMS['SCORE_THRESHOLD_NORMAL']
SCORE_THRESHOLD_WASH = PARAMS['SCORE_THRESHOLD_WASH']
TURNOVER_THRESHOLD_NORMAL = PARAMS['TURNOVER_THRESHOLD_NORMAL']
TURNOVER_THRESHOLD_WASH = PARAMS['TURNOVER_THRESHOLD_WASH']
TOP_N_PER_STRATEGY = PARAMS['TOP_N_PER_STRATEGY']

print(f"[系统] 加载参数: SCORE_NORMAL={SCORE_THRESHOLD_NORMAL}, SCORE_WASH={SCORE_THRESHOLD_WASH}")

# 动态生成输入文件名
def get_input_file(target_date):
    """生成输入文件名"""
    return f'Best_Pick_{target_date}.csv'

# ===========================================

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(timeout=30)


def get_trade_context():
    try:
        now_date = datetime.datetime.now().strftime('%Y%m%d')
        cal_df = pro.trade_cal(exchange='', start_date='20200101', end_date=now_date, is_open='1')
        cal_df = cal_df.sort_values('cal_date', ascending=True).reset_index(drop=True)
        last_trade_day = cal_df['cal_date'].values[-1]

        # 计算回溯起始日 (确保有足够数据计算均线)
        start_date_idx = max(0, len(cal_df) - 400)
        start_date = cal_df['cal_date'].values[start_date_idx]

        print(f"[系统] 行情数据截止日: {last_trade_day}")
        return last_trade_day, start_date
    except:
        return None, None


def load_candidate_pool(filename):
    if not os.path.exists(filename):
        print(f"错误：找不到文件 {filename}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(filename, encoding='gbk')
        except:
            return pd.DataFrame()
    return df


def get_daily_data_batch(codes, start_date, end_date):
    """获取日线行情"""
    try:
        df = pro.daily(ts_code=",".join(codes), start_date=start_date, end_date=end_date)
        return df
    except:
        return pd.DataFrame()


# ==============================================================================
# 模块1: 评分重算 (Core)
# ==============================================================================
def calculate_score(strategy, df_daily):
    if len(df_daily) < 60: return 0, 0, 0, 0, 0

    curr = df_daily.iloc[-1]
    prev = df_daily.iloc[-2]

    close = curr['close']
    pct_chg = curr['pct_chg']

    ma20 = df_daily['close'].rolling(20).mean().iloc[-1]
    ma60 = df_daily['close'].rolling(60).mean().iloc[-1]

    vol_ma5 = df_daily['vol'].rolling(5).mean().iloc[-1]
    vol_ratio = curr['vol'] / vol_ma5 if vol_ma5 > 0 else 0
    vol_prev = curr['vol'] / prev['vol'] if prev['vol'] > 0 else 0

    high_250 = df_daily['high'].iloc[-250:].max()
    low_250 = df_daily['low'].iloc[-250:].min()
    pos_ratio = (close - low_250) / (high_250 - low_250) if high_250 != low_250 else 0.5

    # 1. 安全性
    if pos_ratio <= 0.2:
        base_safe = 25
    elif pos_ratio <= 0.4:
        base_safe = 20
    elif pos_ratio <= 0.6:
        base_safe = 15
    else:
        base_safe = 10
    if vol_ratio < 0.8: base_safe += 2
    s_safe = min(25, base_safe)

    # 2. 进攻性 (修正：给洗盘策略补偿)
    s_off = 0
    if "强攻" in str(strategy):
        s_off += 15
    elif "梯量" in str(strategy):
        s_off += 10
    elif "洗盘" in str(strategy):
        s_off += 5

    if vol_ratio > 2.0:
        s_off += 10
    elif vol_ratio > 1.5:
        s_off += 8

    if pct_chg > 5:
        s_off += 10
    elif pct_chg > 2:
        s_off += 5

    # [关键] 缩量洗盘补偿分
    if "洗盘" in str(strategy) and -3 < pct_chg < 3:
        s_off += 10

    s_off = min(35, s_off)

    # 3. 确定性
    s_cert = 10
    if vol_prev > 1.8: s_cert += 5
    if close > ma20 and close > ma60: s_cert += 10
    s_cert = min(25, s_cert)

    # 4. 配合度
    s_match = 10
    if pos_ratio < 0.3 and vol_ratio > 1.5 and pct_chg > 0: s_match += 5
    if pos_ratio < 0.6 and vol_ratio < 0.8 and -3 < pct_chg < 0: s_match += 5
    s_match = min(15, s_match)

    total = s_safe + s_off + s_cert + s_match
    return total, s_safe, s_off, s_cert, s_match


# ==============================================================================
# 模块2: 高效数据获取 (Robust Fetcher)
# ==============================================================================
def fetch_single_stock_basic(ts_code, target_date, max_retries=3):
    """单只股票基本面获取，带智能回溯"""
    # 尝试回溯 5 天，确保拿到数据
    for day_lag in range(5):
        try:
            curr_date = (datetime.datetime.strptime(target_date, '%Y%m%d') - datetime.timedelta(days=day_lag)).strftime(
                '%Y%m%d')

            # 重试机制
            for attempt in range(max_retries):
                try:
                    df = pro.daily_basic(ts_code=ts_code, trade_date=curr_date,
                                         fields='ts_code,pe_ttm,turnover_rate,circ_mv')
                    if not df.empty:
                        return df.iloc[0].to_dict()  # 成功返回
                    break  # 如果日期不对，没必要重试API，直接换日期
                except:
                    time.sleep(0.5)
        except:
            pass
    return None


def efficient_fetch_stock_data(codes, target_date):
    """多线程批量获取"""
    print(f"    启动多线程获取 {len(codes)} 只股票的基本面 (智能回溯)...")
    results = []

    # 线程数不宜过多，防止Tushare封IP
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_code = {executor.submit(fetch_single_stock_basic, code, target_date): code for code in codes}

        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_code), total=len(codes), desc="    获取进度"):
            res = future.result()
            if res:
                results.append(res)

    return pd.DataFrame(results)


# ==============================================================================
# 主流程
# ==============================================================================
def run_system():
    print("=" * 60)
    print("   DeepQuant Pro V2.1 - 终极修复版")
    print("=" * 60)

    # 1. 准备
    target_date, start_date = get_trade_context()
    if not target_date: return

    # 动态生成输入文件名
    input_file = get_input_file(target_date)
    df_pool = load_candidate_pool(input_file)
    if df_pool.empty:
        print(f"[错误] 找不到输入文件: {input_file}")
        return

    stock_list = df_pool['ts_code'].tolist()
    print(f"[1] 候选池: {len(stock_list)} 只")

    strategy_map = df_pool.set_index('ts_code')['strategy'].to_dict()

    # 2. 重新计算评分
    print(f"[2] 重新计算评分...")
    scored_results = []

    batch_size = 50
    # 简单的批处理获取日线
    for i in tqdm(range(0, len(stock_list), batch_size), desc="    日线分析"):
        batch_codes = stock_list[i:i + batch_size]
        df_daily = get_daily_data_batch(batch_codes, start_date, target_date)

        if not df_daily.empty:
            groups = df_daily.groupby('ts_code')
            for code in batch_codes:
                if code in groups.groups:
                    sub_df = groups.get_group(code).sort_values('trade_date')
                    strategy = strategy_map.get(code, "未知")

                    total, s_safe, s_off, s_cert, s_match = calculate_score(strategy, sub_df)
                    curr = sub_df.iloc[-1]

                    scored_results.append({
                        'ts_code': code,
                        'strategy': strategy,
                        'New_Score': total,
                        'S_Safe': s_safe,
                        'close': curr['close'],
                        'pct_chg': curr['pct_chg']
                    })
        time.sleep(0.1)

    df_scored = pd.DataFrame(scored_results)
    print(f"    评分完成，有效: {len(df_scored)} 条")

    # 3. 获取基本面 (使用高效模块)
    print(f"[3] 获取基本面数据...")
    df_basic = efficient_fetch_stock_data(df_scored['ts_code'].tolist(), target_date)

    if df_basic.empty:
        print("    严重错误：无法获取任何基本面数据，请检查网络或Token权限。")
        # 兜底：如果没有基本面，就不卡基本面了，只卡技术分
        print("    [应急模式] 仅使用技术评分进行筛选...")
        df_final = df_scored.copy()
        df_final['pe_ttm'] = -1
        df_final['turnover_rate'] = -1
        df_final['circ_mv'] = -1
    else:
        df_final = pd.merge(df_scored, df_basic, on='ts_code', how='inner')

    # 4. 风控筛选 (双轨制)
    print(f"[4] 执行风控筛选...")

    # 基础条件 (如果有基本面数据才卡)
    cond_pe = (df_final['pe_ttm'] > 0) & (df_final['pe_ttm'] < 100) if 'pe_ttm' in df_final.columns and df_final[
        'pe_ttm'].max() > 0 else True
    # 市值 > 20亿 (200000万元)
    cond_mv = (df_final['circ_mv'] > 200000) if 'circ_mv' in df_final.columns and df_final[
        'circ_mv'].max() > 0 else True

    # 评分与换手率双轨制
    is_wash = df_final['strategy'].str.contains("洗盘")

    # 分数线（使用配置文件中的参数）
    cond_score_normal = (df_final['New_Score'] >= SCORE_THRESHOLD_NORMAL) & (~is_wash)
    cond_score_wash = (df_final['New_Score'] >= SCORE_THRESHOLD_WASH) & (is_wash)
    cond_score = cond_score_normal | cond_score_wash

    # 换手率 (如果有数据，使用配置文件中的参数)
    if 'turnover_rate' in df_final.columns and df_final['turnover_rate'].max() > 0:
        cond_to_normal = (df_final['turnover_rate'] > TURNOVER_THRESHOLD_NORMAL) & (~is_wash)
        cond_to_wash = (df_final['turnover_rate'] > TURNOVER_THRESHOLD_WASH) & (is_wash)
        cond_to = cond_to_normal | cond_to_wash
    else:
        cond_to = True

    df_pass = df_final[cond_pe & cond_mv & cond_to & cond_score].copy()

    print(f"    风控初筛结果: {len(df_pass)} 只")
    
    # ==========================================================================
    # [新增] 终极PK：每种策略只取 Top 3-5
    # ==========================================================================
    print(f"[5] 执行终极PK (优中选优)...")
    
    final_picks = []
    
    # 1. 处理【★低位强攻】
    # 排序逻辑：优先看总分(New_Score)，其次看涨幅(pct_chg)
    # 强攻就要选最强的
    df_attack = df_pass[df_pass['strategy'].str.contains("强攻")].copy()
    if not df_attack.empty:
        df_attack = df_attack.sort_values(by=['New_Score', 'pct_chg'], ascending=[False, False])
        top_attack = df_attack.head(TOP_N_PER_STRATEGY) # 取前N备选
        final_picks.append(top_attack)
        print(f"    ★低位强攻: 入围 {len(df_attack)} -> 精选 {len(top_attack)}")

    # 2. 处理【☆缩量洗盘】
    # 排序逻辑：优先看总分，其次看量比(vol_ratio)越小越好(洗得干净)
    # 我们需要先计算 vol_ratio (如果df_pass里没有，需要从df_scored里拿，或者简单用S_Safe代替)
    # 这里的 New_Score 已经包含了对缩量的加分，直接用 New_Score 排序即可
    df_wash = df_pass[df_pass['strategy'].str.contains("洗盘")].copy()
    if not df_wash.empty:
        # 洗盘股：分数高说明位置好、支撑强；量比低说明洗得干净
        # 这里我们简单按总分排序
        df_wash = df_wash.sort_values(by=['New_Score'], ascending=False)
        top_wash = df_wash.head(TOP_N_PER_STRATEGY)
        final_picks.append(top_wash)
        print(f"    ☆缩量洗盘: 入围 {len(df_wash)} -> 精选 {len(top_wash)}")

    # 3. 处理【▲梯量上行】
    # 排序逻辑：按总分
    df_ladder = df_pass[df_pass['strategy'].str.contains("梯量")].copy()
    if not df_ladder.empty:
        df_ladder = df_ladder.sort_values(by=['New_Score'], ascending=False)
        top_ladder = df_ladder.head(TOP_N_PER_STRATEGY)
        final_picks.append(top_ladder)
        print(f"    ▲梯量上行: 入围 {len(df_ladder)} -> 精选 {len(top_ladder)}")

    # 合并结果
    if not final_picks:
        print("    遗憾，没有股票通过终极PK。")
        return

    df_final_top = pd.concat(final_picks)
    
    # 4. 补充名称和仓位建议 (对精选股进行)
    try:
        names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        df_final_top = pd.merge(df_final_top, names, on='ts_code', how='left')
    except: pass
    
    def get_pos_sugg(row):
        safe = row['S_Safe']
        # 动态仓位：如果是Top1，仓位加成
        base_pos = 10
        if safe >= 20: base_pos += 2
        elif safe < 15: base_pos -= 2
        return f"{base_pos}%"
    
    df_final_top['Pos_Sugg'] = df_final_top.apply(get_pos_sugg, axis=1)
    
    # 转换单位
    if 'circ_mv' in df_final_top.columns:
        df_final_top['MV_Yi'] = round(df_final_top['circ_mv'] / 10000, 2)
    
    # 整理输出
    cols = ['ts_code', 'name', 'industry', 'strategy', 'New_Score', 'Pos_Sugg', 'close', 'pct_chg', 'pe_ttm', 'turnover_rate', 'MV_Yi']
    # 仅保留存在的列
    final_cols = [c for c in cols if c in df_final_top.columns]
    
    df_final_top = df_final_top[final_cols].sort_values(['strategy', 'New_Score'], ascending=[True, False])
    
    outfile = f'DeepQuant_TopPicks_{target_date}.csv'
    df_final_top.to_csv(outfile, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("【👑 皇家精选 Top 15】")
    print("说明：每个策略赛道仅展示前 5 名，建议重点关注前 3 名。")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_final_top)
    print(f"\n结果已保存: {outfile}")

if __name__ == '__main__':
    run_system()