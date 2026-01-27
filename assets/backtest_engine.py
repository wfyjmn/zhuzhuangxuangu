#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepQuant 历史回测引擎
支持参数化策略回测和性能评估
"""

import os
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json


class BacktestEngine:
    """历史回测引擎"""
    
    def __init__(self, params_config: Dict):
        self.params = params_config
        self.backtest_config = params_config['backtest']
        self.scoring_weights = params_config['scoring_weights']
        self.indicators = params_config['indicators']
        self.thresholds = params_config['thresholds']
        
        # 获取Tushare Token
        from dotenv import load_dotenv
        load_dotenv()
        tushare_token = os.getenv("TUSHARE_TOKEN")
        ts.set_token(tushare_token)
        self.pro = ts.pro_api(timeout=30)
        
        print(f"[回测引擎] 初始化完成")
        print(f"[回测引擎] 初始资金: {self.backtest_config['initial_capital']:,}")
    
    def calculate_score_with_params(self, df_daily: pd.DataFrame, strategy: str) -> Tuple[float, Dict]:
        """使用指定参数计算评分"""
        if len(df_daily) < self.indicators['min_data_days']:
            return 0, {'s_safe': 0, 's_off': 0, 's_cert': 0, 's_match': 0}
        
        curr = df_daily.iloc[-1]
        prev = df_daily.iloc[-2]
        close = curr['close']
        pct_chg = curr['pct_chg']
        
        # 计算指标
        ma_short = df_daily['close'].rolling(self.indicators['ma_periods']['short']).mean().iloc[-1]
        ma_medium = df_daily['close'].rolling(self.indicators['ma_periods']['medium']).mean().iloc[-1]
        vol_ma = df_daily['vol'].rolling(self.indicators['vol_ma_period']).mean().iloc[-1]
        vol_ratio = curr['vol'] / vol_ma if vol_ma > 0 else 0
        vol_prev = curr['vol'] / prev['vol'] if prev['vol'] > 0 else 0
        
        high_period = df_daily['high'].iloc[-self.indicators['lookback_period']:].max()
        low_period = df_daily['low'].iloc[-self.indicators['lookback_period']:].min()
        pos_ratio = (close - low_period) / (high_period - low_period) if high_period != low_period else 0.5
        
        # 1. 安全性
        s_safe = self._calculate_safety_score(pos_ratio, vol_ratio)
        # 2. 进攻性
        s_off = self._calculate_offensive_score(strategy, vol_ratio, pct_chg)
        # 3. 确定性
        s_cert = self._calculate_certainty_score(vol_prev, close, ma_short, ma_medium)
        # 4. 配合度
        s_match = self._calculate_match_score(pos_ratio, vol_ratio, pct_chg, strategy)
        
        total = s_safe + s_off + s_cert + s_match
        return total, {'s_safe': s_safe, 's_off': s_off, 's_cert': s_cert, 's_match': s_match}
    
    def _calculate_safety_score(self, pos_ratio: float, vol_ratio: float) -> float:
        """计算安全性评分"""
        weights = self.scoring_weights['safety']
        thresholds = weights['pos_thresholds']
        base_scores = weights['base_scores']
        
        if pos_ratio <= thresholds[0]:
            base_safe = base_scores[0]
        elif pos_ratio <= thresholds[1]:
            base_safe = base_scores[1]
        elif pos_ratio <= thresholds[2]:
            base_safe = base_scores[2]
        else:
            base_safe = base_scores[3]
        
        if vol_ratio < 0.8:
            base_safe += weights['low_vol_bonus']
        
        return min(weights['max_score'], base_safe)
    
    def _calculate_offensive_score(self, strategy: str, vol_ratio: float, pct_chg: float) -> float:
        """计算进攻性评分"""
        weights = self.scoring_weights['offensive']
        s_off = 0
        
        for st_type, base_score in weights['strategy_base'].items():
            if st_type in str(strategy):
                s_off += base_score
                break
        
        for bonus in weights['vol_ratio_bonus']:
            if vol_ratio > bonus['threshold']:
                s_off += bonus['score']
                break
        
        for bonus in weights['pct_chg_bonus']:
            if pct_chg > bonus['threshold']:
                s_off += bonus['score']
                break
        
        wash_config = weights['wash_compensation']
        if wash_config['enabled'] and "洗盘" in str(strategy):
            pct_min, pct_max = wash_config['pct_range']
            if pct_min < pct_chg < pct_max:
                s_off += wash_config['score']
        
        return min(weights['max_score'], s_off)
    
    def _calculate_certainty_score(self, vol_prev: float, close: float, ma_short: float, ma_medium: float) -> float:
        """计算确定性评分"""
        weights = self.scoring_weights['certainty']
        s_cert = weights['base_score']
        
        if vol_prev > weights['vol_threshold']:
            s_cert += weights['vol_bonus']
        
        if close > ma_short and close > ma_medium:
            s_cert += weights['ma_above_bonus']
        
        return min(weights['max_score'], s_cert)
    
    def _calculate_match_score(self, pos_ratio: float, vol_ratio: float, pct_chg: float, strategy: str) -> float:
        """计算配合度评分"""
        weights = self.scoring_weights['match']
        s_match = weights['base_score']
        
        strong_attack = weights['strong_attack_bonus']
        if (pos_ratio < strong_attack['pos_threshold'] and vol_ratio > strong_attack['vol_threshold'] and pct_chg > 0):
            s_match += strong_attack['score']
        
        wash_bonus = weights['wash_bonus']
        if (pos_ratio < wash_bonus['pos_threshold'] and vol_ratio < wash_bonus['vol_threshold'] and 
            wash_bonus['pct_range'][0] < pct_chg < wash_bonus['pct_range'][1]):
            s_match += wash_bonus['score']
        
        return min(weights['max_score'], s_match)


def main():
    with open("strategy_params.json", 'r', encoding='utf-8') as f:
        params = json.load(f)
    engine = BacktestEngine(params)
    print("[回测引擎] 引擎创建成功")


if __name__ == "__main__":
    main()
