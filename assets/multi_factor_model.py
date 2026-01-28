# -*- coding: utf-8 -*-
"""
DeepQuant 多因子选股模型 (Multi-Factor Selection Model)
版本：v2.1 优化版
修复：
1. 单个板块数据获取失败问题
2. 添加缓存机制避免重复计算
3. 修复 df_basic 作用域问题
优化：
1. 采用全市场数据一次性计算板块效应，大幅减少API请求次数
2. 添加错误处理和降级方案
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class MultiFactorModel:
    """多因子选股模型"""

    def __init__(self):
        """初始化"""
        from dotenv import load_dotenv
        load_dotenv()
        tushare_token = os.getenv("TUSHARE_TOKEN")

        # 如果环境变量没取到，请在此处填入您的 Token
        if not tushare_token:
            # tushare_token = "YOUR_TOKEN_HERE"
            pass

        ts.set_token(tushare_token)
        self.pro = ts.pro_api(timeout=30)

        # 因子权重配置
        self.factor_weights = {
            'moneyflow': 0.3,          # 资金流因子
            'sector_resonance': 0.2,   # 板块共振因子
            'technical': 0.5           # 技术形态因子
        }

        # 缓存交易日历
        self.trade_cal = self._get_trade_cal()

        # 缓存板块统计数据（避免重复计算）
        self._sector_stats_cache = None

    def _get_trade_cal(self):
        """获取最近的交易日历"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            df = self.pro.trade_cal(exchange='SSE', is_open='1', start_date=start_date, end_date=end_date)
            return df['cal_date'].tolist()
        except:
            return []

    def get_latest_trade_date(self):
        """获取最近的一个交易日"""
        if self.trade_cal:
            return self.trade_cal[-1]
        return datetime.now().strftime('%Y%m%d')

    def get_stock_moneyflow(self, ts_code: str, days: int = 20) -> Dict:
        """获取个股资金流向数据"""
        try:
            end_date = self.get_latest_trade_date()
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')

            df = self.pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)

            if len(df) == 0:
                return self._empty_moneyflow_data()

            df = df.sort_values('trade_date').tail(days)

            # 计算指标
            total_net_inflow = df['buy_elg_vol'].sum() - df['sell_elg_vol'].sum()
            total_net_amount = df['buy_elg_amount'].sum() - df['sell_elg_amount'].sum()

            latest = df.iloc[-1]
            latest_net_inflow = latest['buy_elg_vol'] - latest['sell_elg_vol']

            return {
                'total_net_inflow_vol': total_net_inflow,
                'total_net_inflow_amount': total_net_amount,
                'latest_net_inflow_vol': latest_net_inflow,
                'latest_buy_vol': latest['buy_elg_vol'],
                'latest_sell_vol': latest['sell_elg_vol']
            }
        except Exception as e:
            print(f"[错误] 获取资金流向失败 {ts_code}: {e}")
            return self._empty_moneyflow_data()

    def _empty_moneyflow_data(self):
        return {
            'total_net_inflow_vol': 0, 'total_net_inflow_amount': 0,
            'latest_net_inflow_vol': 0, 'latest_buy_vol': 0, 'latest_sell_vol': 0
        }

    def get_batch_moneyflow(self, ts_codes: List[str], days: int = 20, batch_size: int = 50) -> Dict[str, Dict]:
        """批量获取资金流向数据"""
        end_date = self.get_latest_trade_date()
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
        results = {}

        # 如果列表为空直接返回
        if not ts_codes:
            return {}

        def process_batch(batch_codes):
            batch_results = {}
            try:
                df = self.pro.moneyflow(ts_code=",".join(batch_codes), start_date=start_date, end_date=end_date)
                if len(df) > 0:
                    groups = df.groupby('ts_code')
                    for code in batch_codes:
                        if code in groups.groups:
                            sub_df = groups.get_group(code).sort_values('trade_date').tail(days)
                            # 计算累计和最新
                            total_net_inflow = sub_df['buy_elg_vol'].sum() - sub_df['sell_elg_vol'].sum()
                            total_net_amount = sub_df['buy_elg_amount'].sum() - sub_df['sell_elg_amount'].sum()
                            latest = sub_df.iloc[-1]

                            batch_results[code] = {
                                'total_net_inflow_vol': total_net_inflow,
                                'total_net_inflow_amount': total_net_amount,
                                'latest_net_inflow_vol': latest['buy_elg_vol'] - latest['sell_elg_vol'],
                                'latest_buy_vol': latest['buy_elg_vol'],
                                'latest_sell_vol': latest['sell_elg_vol']
                            }
                        else:
                            batch_results[code] = self._empty_moneyflow_data()
                else:
                    for code in batch_codes: batch_results[code] = self._empty_moneyflow_data()
            except Exception as e:
                print(f"    [Batch Error] {e}")
                for code in batch_codes: batch_results[code] = self._empty_moneyflow_data()
            return batch_results

        # 简单的单线程处理小批量，多线程处理大批量
        batches = [ts_codes[i:i + batch_size] for i in range(0, len(ts_codes), batch_size)]

        if len(batches) == 1:
            results.update(process_batch(batches[0]))
        else:
            print(f"    [批量资金流] 正处理 {len(ts_codes)} 只股票，共 {len(batches)} 批次...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_batch, batch): batch for batch in batches}
                for future in as_completed(futures):
                    results.update(future.result())

        return results

    def get_all_sector_performance_snapshot(self) -> Dict[str, Dict]:
        """
        【核心优化】一次性获取全市场所有板块的当日表现
        逻辑：
        1. 获取所有股票的基础信息（所属行业）
        2. 获取最近一个交易日的所有股票行情
        3. 在本地合并并按行业聚合计算

        Returns:
            板块统计数据字典 {industry: {avg_pct_chg, up_ratio, stock_count, is_hot}}
        """
        # 如果有缓存，直接返回
        if self._sector_stats_cache is not None:
            return self._sector_stats_cache

        try:
            trade_date = self.get_latest_trade_date()
            print(f"[板块共振] 正在计算全市场板块热度 (交易日: {trade_date})...")

            # 1. 获取全市场股票及行业
            df_basic = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')

            # 2. 获取全市场当日行情 (一次请求约5000条数据，Tushare允许)
            df_daily = self.pro.daily(trade_date=trade_date, fields='ts_code,pct_chg')

            if len(df_daily) == 0:
                print("[警告] 未获取到今日行情数据，板块共振因子可能无效")
                return {}

            # 3. 合并数据
            merged = pd.merge(df_daily, df_basic, on='ts_code', how='inner')

            # 4. 按行业分组计算统计值
            sector_stats = {}
            grouped = merged.groupby('industry')

            for industry, group in grouped:
                if not industry: continue

                avg_pct_chg = group['pct_chg'].mean()
                up_count = (group['pct_chg'] > 0).sum()
                total_count = len(group)
                up_ratio = (up_count / total_count * 100) if total_count > 0 else 0

                # 热门板块定义：均涨 > 2% 且 上涨率 > 60%
                is_hot = (avg_pct_chg > 2.0) and (up_ratio > 60)

                sector_stats[industry] = {
                    'sector': industry,
                    'avg_pct_chg': round(avg_pct_chg, 2),
                    'up_ratio': round(up_ratio, 2),
                    'stock_count': total_count,
                    'is_hot': is_hot
                }

            # 缓存结果
            self._sector_stats_cache = sector_stats

            print(f"    成功计算 {len(sector_stats)} 个板块的共振数据")
            return sector_stats

        except Exception as e:
            print(f"[错误] 全市场板块计算失败: {e}")
            return {}

    def calculate_moneyflow_score_internal(self, mf_data: Dict) -> float:
        """根据数据计算资金流得分 (纯计算逻辑)"""
        score = 0
        total_inflow = mf_data.get('total_net_inflow_amount', 0)
        latest_inflow = mf_data.get('latest_net_inflow_vol', 0)

        # 1. 累计净流入 (40分)
        if total_inflow > 50000000: score += 40
        elif total_inflow > 20000000: score += 30
        elif total_inflow > 0: score += 20
        elif total_inflow > -20000000: score += 10

        # 2. 最新单日 (30分)
        if latest_inflow > 100000: score += 30
        elif latest_inflow > 0: score += 20
        elif latest_inflow > -50000: score += 10

        # 3. 预留北向资金 (30分) - 此处简化，默认给个基础分
        score += 10

        return min(score, 100)

    def calculate_sector_score_internal(self, sector_perf: Dict) -> float:
        """根据数据计算板块得分 (纯计算逻辑)"""
        score = 0
        if not sector_perf: return 50  # 无数据给中位数

        # 1. 热门板块 (50分)
        if sector_perf.get('is_hot', False): score += 50

        # 2. 涨幅 (30分)
        avg_pct = sector_perf.get('avg_pct_chg', 0)
        if avg_pct > 5: score += 30
        elif avg_pct > 3: score += 25
        elif avg_pct > 1: score += 15
        elif avg_pct > 0: score += 10
        elif avg_pct > -1: score += 5

        # 3. 普涨率 (20分)
        up_ratio = sector_perf.get('up_ratio', 0)
        if up_ratio > 80: score += 20
        elif up_ratio > 70: score += 15
        elif up_ratio > 50: score += 10

        return min(score, 100)

    def batch_calculate_scores(self, stock_list: List[str], technical_scores: Dict[str, float]) -> pd.DataFrame:
        """
        批量计算多因子得分 (高度优化版)

        Args:
            stock_list: 股票代码列表
            technical_scores: 技术评分字典 {ts_code: score}

        Returns:
            综合评分DataFrame，包含以下列：
            - ts_code: 股票代码
            - name: 股票名称
            - industry: 所属行业
            - composite_score: 综合得分
            - tech_score: 技术得分
            - moneyflow_score: 资金流得分（兼容原版本）
            - sector_score: 板块得分
            - sector_name: 板块名称（兼容原版本）
            - sector_hot: 是否热门板块
            - sector_avg_chg: 板块平均涨幅
        """
        results = []
        print(f"\n[多因子模型] 开始计算 {len(stock_list)} 只股票的综合得分...")

        # 1. 批量获取 股票-行业 映射（修复作用域问题）
        print(f"[步骤1] 获取股票基础信息...")
        try:
            df_basic = self.pro.stock_basic(ts_code=",".join(stock_list), fields='ts_code,name,industry')
            # ✅ 提前转换为字典，避免循环中重复查询
            name_map = df_basic.set_index('ts_code')['name'].to_dict()
            industry_map = df_basic.set_index('ts_code')['industry'].to_dict()
        except Exception as e:
            print(f"  [警告] 获取股票基础信息失败: {e}")
            name_map = {}
            industry_map = {}

        # 2. 批量获取 资金流数据
        print(f"[步骤2] 批量获取资金流数据...")
        mf_map = self.get_batch_moneyflow(stock_list, days=20, batch_size=50)

        # 3. 一次性获取 全市场板块表现 (解决单个获取失败的问题)
        print(f"[步骤3] 计算板块共振数据...")
        sector_stats_map = self.get_all_sector_performance_snapshot()

        # 4. 组合计算
        print(f"[步骤4] 合成因子得分...")
        for ts_code in stock_list:
            # 技术分
            tech_score = technical_scores.get(ts_code, 60)

            # 资金分
            mf_data = mf_map.get(ts_code, self._empty_moneyflow_data())
            mf_score = self.calculate_moneyflow_score_internal(mf_data)

            # 板块分
            industry = industry_map.get(ts_code, '未知')
            sector_data = sector_stats_map.get(industry, {})
            sector_score = self.calculate_sector_score_internal(sector_data)

            # 综合分
            comp_score = (
                mf_score * self.factor_weights['moneyflow'] +
                sector_score * self.factor_weights['sector_resonance'] +
                tech_score * self.factor_weights['technical']
            )

            # 记录结果（兼容原版本列名）
            results.append({
                'ts_code': ts_code,
                'name': name_map.get(ts_code, ts_code),
                'industry': industry,
                'composite_score': round(comp_score),
                'tech_score': round(tech_score),
                'moneyflow_score': mf_score,  # ✅ 兼容原版本列名
                'sector_score': sector_score,
                'sector_name': industry,  # ✅ 兼容原版本列名
                'sector_hot': 'YES' if sector_data.get('is_hot') else 'NO',
                'sector_avg_chg': sector_data.get('avg_pct_chg', 0)
            })

        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values('composite_score', ascending=False).reset_index(drop=True)

        print(f"[完成] 计算结束，前3名预览：")
        print(df_result[['ts_code', 'name', 'industry', 'composite_score']].head(3).to_string(index=False))
        return df_result

def main():
    """测试函数"""
    model = MultiFactorModel()

    # 模拟输入
    test_stocks = ['600519.SH', '000001.SZ', '601318.SH', '002594.SZ']
    tech_scores = {'600519.SH': 85, '000001.SZ': 70, '601318.SH': 90, '002594.SZ': 95}

    df = model.batch_calculate_scores(test_stocks, tech_scores)

    # 打印详细结果
    print("\n详细结果:")
    print(df.to_string())

if __name__ == "__main__":
    main()
