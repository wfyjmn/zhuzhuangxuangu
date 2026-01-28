# -*- coding: utf-8 -*-
"""
DeepQuant 天气预报系统（Market Weather Module）
功能：研判大势，根据市场环境调整选股策略

核心功能：
1. 指数趋势判断：上证指数、创业板指的技术分析
2. 市场情绪计算：赚钱效应、跌停家数
3. 策略调整建议：晴天（进攻）、阴天（防守）、暴雨（空仓）
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import os
import time


class MarketWeather:
    """天气预报系统"""
    
    def __init__(self):
        """初始化"""
        from dotenv import load_dotenv
        load_dotenv()
        tushare_token = os.getenv("TUSHARE_TOKEN")
        ts.set_token(tushare_token)
        self.pro = ts.pro_api(timeout=30)
        
        # 指数代码
        self.indices = {
            'sh': '000001.SH',  # 上证指数
            'sz': '399001.SZ'   # 深证成指
        }
        
        # 状态缓存
        self.weather_data = None
        self.market_sentiment = None
        
    def get_index_data(self, index_code: str, days: int = 120) -> pd.DataFrame:
        """
        获取指数K线数据
        
        Args:
            index_code: 指数代码
            days: 获取天数
            
        Returns:
            指数DataFrame
        """
        try:
            time.sleep(0.5)  # 添加延时，避免触发Tushare限流
            
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            
            df = self.pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(df) == 0:
                return pd.DataFrame()
            
            df = df.sort_values('trade_date').tail(days).reset_index(drop=True)
            
            # 计算技术指标
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma60'] = df['close'].rolling(60).mean()
            
            # 计算MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['dif'] = df['ema12'] - df['ema26']
            df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
            df['macd'] = (df['dif'] - df['dea']) * 2
            
            return df
        except Exception as e:
            print(f"[错误] 获取指数数据失败: {e}")
            return pd.DataFrame()
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        分析指数趋势
        
        Args:
            df: 指数数据
            
        Returns:
            趋势分析结果
        """
        if len(df) < 20:
            return {'trend': 'unknown', 'signal': '数据不足'}
        
        latest = df.iloc[-1]
        
        # 判断均线排列
        ma_bullish = (latest['ma5'] > latest['ma10'] > latest['ma20'] > latest['ma60'])
        ma_bearish = (latest['ma5'] < latest['ma10'] < latest['ma20'] < latest['ma60'])
        
        # 判断MACD
        macd_golden = (latest['dif'] > latest['dea']) and (df.iloc[-2]['dif'] <= df.iloc[-2]['dea'])
        macd_death = (latest['dif'] < latest['dea']) and (df.iloc[-2]['dif'] >= df.iloc[-2]['dea'])
        
        # 综合判断
        if ma_bullish and macd_golden:
            trend = 'bullish'
            signal = '多头排列+MACD金叉'
        elif ma_bullish:
            trend = 'bullish_weak'
            signal = '多头排列'
        elif ma_bearish and macd_death:
            trend = 'bearish'
            signal = '空头排列+MACD死叉'
        elif ma_bearish:
            trend = 'bearish_weak'
            signal = '空头排列'
        else:
            trend = 'neutral'
            signal = '震荡整理'
        
        # 计算涨幅
        pct_5d = (latest['close'] / df.iloc[-6]['close'] - 1) * 100 if len(df) > 5 else 0
        pct_20d = (latest['close'] / df.iloc[-21]['close'] - 1) * 100 if len(df) > 20 else 0
        
        return {
            'trend': trend,
            'signal': signal,
            'close': latest['close'],
            'pct_5d': round(pct_5d, 2),
            'pct_20d': round(pct_20d, 2),
            'ma5': latest['ma5'],
            'ma20': latest['ma20'],
            'macd_dif': latest['dif'],
            'macd_dea': latest['dea']
        }
    
    def calculate_market_sentiment(self, trade_date: str = None) -> Dict:
        """
        计算市场情绪指标
        
        Args:
            trade_date: 交易日
            
        Returns:
            市场情绪数据
        """
        if not trade_date:
            trade_date = datetime.now().strftime('%Y%m%d')
        
        try:
            time.sleep(0.5)  # 添加延时，避免触发Tushare限流
            
            # 获取涨跌停数据
            end_date = datetime.strptime(trade_date, '%Y%m%d').strftime('%Y%m%d')
            start_date = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
            
            # 获取涨停跌停数据
            limit_up = self.pro.limit_list_d(
                trade_date=trade_date,
                limit_type='U'
            )
            
            time.sleep(0.5)  # 添加延时
            
            limit_down = self.pro.limit_list_d(
                trade_date=trade_date,
                limit_type='D'
            )
            
            # 计算涨停股今日表现（赚钱效应）
            if len(limit_up) > 0:
                limit_up_stocks = limit_up['ts_code'].tolist()
                # 获取今日数据（简化处理）
                # 赚钱效应暂无法实时计算，使用跌停家数代替
                money_effect = 0
            else:
                limit_up_stocks = []
                money_effect = 0
            
            # 跌停家数
            limit_down_count = len(limit_down)
            limit_up_count = len(limit_up)
            
            # 获取全市场涨跌统计
            time.sleep(0.5)  # 添加延时
            daily = self.pro.daily(
                trade_date=trade_date,
                fields='trade_date,pct_chg'
            )
            
            if len(daily) > 0:
                up_count = (daily['pct_chg'] > 0).sum()
                down_count = (daily['pct_chg'] < 0).sum()
                total_count = len(daily)
                up_ratio = up_count / total_count * 100
            else:
                up_count = 0
                down_count = 0
                up_ratio = 0
                total_count = 0
            
            return {
                'trade_date': trade_date,
                'limit_up_count': limit_up_count,
                'limit_down_count': limit_down_count,
                'up_count': up_count,
                'down_count': down_count,
                'total_count': total_count,
                'up_ratio': round(up_ratio, 2),
                'money_effect': money_effect,
                'high_risk': limit_down_count > 30
            }
        except Exception as e:
            print(f"[错误] 计算市场情绪失败: {e}")
            # 返回默认值，确保返回完整的字典结构
            return {
                'trade_date': trade_date,
                'limit_up_count': 0,
                'limit_down_count': 0,
                'up_count': 0,
                'down_count': 0,
                'total_count': 0,
                'up_ratio': 50.0,
                'money_effect': 0,
                'high_risk': False
            }
    
    def get_weather_forecast(self) -> Dict:
        """
        获取天气预报（综合研判）
        
        Returns:
            天气预报数据
        """
        print("\n" + "="*80)
        print("【🌤️ 天气预报】市场环境研判")
        print("="*80)
        
        # 1. 分析指数趋势
        print("\n[1] 指数趋势分析")
        
        index_analysis = {}
        for name, code in self.indices.items():
            df = self.get_index_data(code)
            if len(df) > 0:
                analysis = self.analyze_trend(df)
                index_analysis[name] = analysis
                
                trend_emoji = {
                    'bullish': '🌞 晴天',
                    'bullish_weak': '⛅ 多云',
                    'neutral': '☁️ 阴天',
                    'bearish_weak': '🌧️ 小雨',
                    'bearish': '⛈️ 暴雨'
                }.get(analysis['trend'], '❓ 未知')
                
                trend_name = {
                    'bullish': '强势多头',
                    'bullish_weak': '偏多',
                    'neutral': '震荡',
                    'bearish_weak': '偏空',
                    'bearish': '弱势空头'
                }.get(analysis['trend'], '未知')
                
                index_name = {'sh': '上证指数', 'sz': '深证成指'}.get(name, code)
                print(f"  {index_name}: {trend_emoji} {trend_name}")
                print(f"    信号: {analysis['signal']}")
                print(f"    收盘: {analysis['close']:.2f} (近5日: {analysis['pct_5d']:+.2f}%)")
        
        # 2. 计算市场情绪
        print("\n[2] 市场情绪指标")
        sentiment = self.calculate_market_sentiment()
        self.market_sentiment = sentiment
        
        print(f"  涨停家数: {sentiment['limit_up_count']}家")
        print(f"  跌停家数: {sentiment['limit_down_count']}家")
        print(f"  上涨家数: {sentiment['up_count']}家")
        print(f"  下跌家数: {sentiment['down_count']}家")
        print(f"  上涨占比: {sentiment['up_ratio']:.1f}%")
        
        # 风险等级
        if sentiment['limit_down_count'] > 50:
            risk_level = "🔴 极高风险"
            risk_score = 5
        elif sentiment['limit_down_count'] > 30:
            risk_level = "🟠 高风险"
            risk_score = 4
        elif sentiment['limit_down_count'] > 10:
            risk_level = "🟡 中等风险"
            risk_score = 3
        elif sentiment['up_ratio'] < 30:
            risk_level = "🟡 中等风险"
            risk_score = 2
        else:
            risk_level = "🟢 低风险"
            risk_score = 1
        
        print(f"  风险等级: {risk_level}")
        
        # 3. 综合研判
        print("\n[3] 综合研判")
        
        # 判断主要指数趋势
        sh_trend = index_analysis.get('sh', {}).get('trend', 'neutral')
        sz_trend = index_analysis.get('sz', {}).get('trend', 'neutral')
        
        # 综合趋势
        if sh_trend == 'bearish' or sz_trend == 'bearish':
            overall_trend = 'bearish'
        elif sh_trend == 'bullish' and sz_trend == 'bullish':
            overall_trend = 'bullish'
        elif sh_trend == 'bearish_weak' or sz_trend == 'bearish_weak':
            overall_trend = 'bearish_weak'
        else:
            overall_trend = 'neutral'
        
        # 天气评级
        if overall_trend == 'bearish' or risk_score >= 4:
            weather = '⛈️ 暴雨'
            action = '空仓休息'
            strategy_adj = '关闭所有策略'
        elif overall_trend == 'bearish_weak' or risk_score >= 3:
            weather = '🌧️ 小雨'
            action = '谨慎防守'
            strategy_adj = '关闭强攻策略，仅保留洗盘/梯量'
        elif overall_trend == 'bullish' and risk_score <= 2:
            weather = '🌞 晴天'
            action = '积极进攻'
            strategy_adj = '正常选股，重点关注强攻策略'
        else:
            weather = '☁️ 阴天'
            action = '适度参与'
            strategy_adj = '正常选股，降低仓位'
        
        print(f"  天气: {weather}")
        print(f"  建议: {action}")
        print(f"  策略调整: {strategy_adj}")
        
        # 4. 参数调整建议
        print("\n[4] 参数调整建议")
        
        threshold_adj = 0
        if overall_trend == 'bearish' or risk_score >= 4:
            threshold_adj = 15  # 暴雨：阈值+15分
            print(f"  评分阈值: +15分 (高风险，大幅提高门槛)")
        elif overall_trend == 'bearish_weak' or risk_score >= 3:
            threshold_adj = 10  # 小雨：阈值+10分
            print(f"  评分阈值: +10分 (中高风险，提高门槛)")
        elif overall_trend == 'bullish' and risk_score <= 2:
            threshold_adj = -5  # 晴天：阈值-5分
            print(f"  评分阈值: -5分 (低风险，适度放松)")
        else:
            threshold_adj = 0
            print(f"  评分阈值: +0分 (正常)")
        
        print("="*80 + "\n")
        
        weather_data = {
            'weather': weather,
            'action': action,
            'strategy_adj': strategy_adj,
            'threshold_adj': threshold_adj,
            'trend': overall_trend,
            'risk_score': risk_score,
            'index_analysis': index_analysis,
            'sentiment': sentiment,
            'close_strong_attack': overall_trend in ['bearish', 'bearish_weak'],
            'allow_trading': overall_trend != 'bearish' and risk_score < 5
        }
        
        self.weather_data = weather_data
        return weather_data
    
    def get_strategy_config(self, original_config: Dict) -> Dict:
        """
        根据天气调整策略配置
        
        Args:
            original_config: 原始配置
            
        Returns:
            调整后的配置
        """
        if not self.weather_data:
            self.get_weather_forecast()
        
        weather = self.weather_data
        
        adjusted_config = original_config.copy()
        
        # 1. 调整评分阈值
        if 'thresholds' in adjusted_config:
            original_normal = adjusted_config['thresholds']['SCORE_THRESHOLD_NORMAL']
            original_wash = adjusted_config['thresholds']['SCORE_THRESHOLD_WASH']
            
            adjusted_config['thresholds']['SCORE_THRESHOLD_NORMAL'] = original_normal + weather['threshold_adj']
            adjusted_config['thresholds']['SCORE_THRESHOLD_WASH'] = original_wash + weather['threshold_adj']
            
            print(f"[天气预报] 评分阈值调整:")
            print(f"  正常策略: {original_normal} → {adjusted_config['thresholds']['SCORE_THRESHOLD_NORMAL']}")
            print(f"  洗盘策略: {original_wash} → {adjusted_config['thresholds']['SCORE_THRESHOLD_WASH']}")
        
        # 2. 调整选股数量
        if weather['close_strong_attack']:
            # 关闭强攻策略
            if 'TOP_N_PER_STRATEGY' in adjusted_config['thresholds']:
                print(f"[天气预报] 关闭强攻策略，仅保留洗盘/梯量")
                # 在实际应用中，这里可以标记禁用某些策略
        elif not weather['allow_trading']:
            # 空仓
            if 'TOP_N_PER_STRATEGY' in adjusted_config['thresholds']:
                adjusted_config['thresholds']['TOP_N_PER_STRATEGY'] = 0
                print(f"[天气预报] 空仓模式，暂停选股")
        
        adjusted_config['weather_info'] = weather
        
        return adjusted_config


def main():
    """测试天气预报系统"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant 天气预报系统")
    print(" " * 30 + "测试运行")
    print("="*80)
    
    weather = MarketWeather()
    
    # 获取天气预报
    forecast = weather.get_weather_forecast()
    
    print("\n[测试结果]")
    print(f"  天气: {forecast['weather']}")
    print(f"  建议: {forecast['action']}")
    print(f"  阈值调整: {forecast['threshold_adj']:+}分")
    print(f"  是否交易: {'是' if forecast['allow_trading'] else '否'}")
    
    print("\n[完成] 天气预报系统测试完成\n")


if __name__ == "__main__":
    main()
