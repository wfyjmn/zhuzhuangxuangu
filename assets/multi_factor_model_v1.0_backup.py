# -*- coding: utf-8 -*-
"""
DeepQuant 多因子选股模型 (Multi-Factor Selection Model)
功能：引入资金流因子和板块共振因子，提升选股准确度

核心因子：
1. 资金流因子：北向资金持仓变化、主力净流入额
2. 板块共振因子：所属板块整体涨幅、热门板块加分

设计理念：
- 形态好 + 主力大举买入 = 真突破
- 龙生龙，凤生凤：热门板块内的股票更容易起飞
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class MultiFactorModel:
    """多因子选股模型"""
    
    def __init__(self):
        """初始化"""
        from dotenv import load_dotenv
        load_dotenv()
        tushare_token = os.getenv("TUSHARE_TOKEN")
        ts.set_token(tushare_token)
        self.pro = ts.pro_api(timeout=30)
        
        # 因子权重配置（优化版：提高技术形态权重）
        self.factor_weights = {
            'moneyflow': 0.3,          # 资金流因子权重
            'sector_resonance': 0.2,   # 板块共振因子权重（降低，因为数据有限）
            'technical': 0.5           # 技术因子权重（提高，因为这是核心）
        }
        
        # 缓存数据
        self.moneyflow_cache = {}
        self.sector_data_cache = {}
        
    def get_stock_moneyflow(self, ts_code: str, days: int = 20) -> Dict:
        """
        获取个股资金流向数据
        
        Args:
            ts_code: 股票代码
            days: 获取天数
            
        Returns:
            资金流向数据
        """
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            
            # 获取个股资金流向
            df = self.pro.moneyflow(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(df) == 0:
                return {
                    'net_mf_vol': 0,
                    'net_mf_amount': 0,
                    'buy_elg_vol': 0,
                    'sell_elg_vol': 0
                }
            
            df = df.sort_values('trade_date').tail(days)
            
            # 计算累计主力净流入
            total_net_inflow = df['buy_elg_vol'].sum() - df['sell_elg_vol'].sum()
            total_net_amount = df['buy_elg_amount'].sum() - df['sell_elg_amount'].sum()
            
            # 计算最新单日主力净流入
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
            return {
                'total_net_inflow_vol': 0,
                'total_net_inflow_amount': 0,
                'latest_net_inflow_vol': 0,
                'latest_buy_vol': 0,
                'latest_sell_vol': 0
            }
    
    def get_batch_moneyflow(self, ts_codes: List[str], days: int = 20, batch_size: int = 50) -> Dict[str, Dict]:
        """
        批量获取资金流向数据（优化版：批量获取+多线程）
        
        Args:
            ts_codes: 股票代码列表
            days: 获取天数
            batch_size: 每批股票数量
            
        Returns:
            资金流向数据字典 {ts_code: moneyflow_data}
        """
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
        
        results = {}
        
        # 股票数量少于10，使用单线程
        if len(ts_codes) < 10:
            print(f"    [批量资金流] 股票数 {len(ts_codes)} < 10，使用单线程模式")
            for ts_code in ts_codes:
                results[ts_code] = self.get_stock_moneyflow(ts_code, days)
                time.sleep(0.03)  # 请求间隔
            return results
        
        # 使用批量获取+多线程
        print(f"    [批量资金流] 股票数 {len(ts_codes)}，使用批量多线程模式（batch_size={batch_size}）")
        
        def process_batch(batch_codes):
            """处理一个批次"""
            batch_results = {}
            try:
                # 批量获取资金流数据
                df = self.pro.moneyflow(
                    ts_code=",".join(batch_codes),
                    start_date=start_date,
                    end_date=end_date
                )
                
                if len(df) > 0:
                    # 按股票代码分组
                    groups = df.groupby('ts_code')
                    for code in batch_codes:
                        if code in groups.groups:
                            sub_df = groups.get_group(code).sort_values('trade_date').tail(days)
                            
                            # 计算累计主力净流入
                            total_net_inflow = sub_df['buy_elg_vol'].sum() - sub_df['sell_elg_vol'].sum()
                            total_net_amount = sub_df['buy_elg_amount'].sum() - sub_df['sell_elg_amount'].sum()
                            
                            # 计算最新单日主力净流入
                            latest = sub_df.iloc[-1]
                            latest_net_inflow = latest['buy_elg_vol'] - latest['sell_elg_vol']
                            
                            batch_results[code] = {
                                'total_net_inflow_vol': total_net_inflow,
                                'total_net_inflow_amount': total_net_amount,
                                'latest_net_inflow_vol': latest_net_inflow,
                                'latest_buy_vol': latest['buy_elg_vol'],
                                'latest_sell_vol': latest['sell_elg_vol']
                            }
                        else:
                            batch_results[code] = {
                                'total_net_inflow_vol': 0,
                                'total_net_inflow_amount': 0,
                                'latest_net_inflow_vol': 0,
                                'latest_buy_vol': 0,
                                'latest_sell_vol': 0
                            }
                else:
                    # 批量获取失败，逐个获取
                    for code in batch_codes:
                        batch_results[code] = self.get_stock_moneyflow(code, days)
                        time.sleep(0.03)
                
            except Exception as e:
                print(f"    [错误] 批量获取资金流失败: {e}")
                # 降级：逐个获取
                for code in batch_codes:
                    batch_results[code] = self.get_stock_moneyflow(code, days)
                    time.sleep(0.03)
            
            time.sleep(0.03)  # 请求间隔
            return batch_results
        
        # 分批处理
        batches = [ts_codes[i:i + batch_size] for i in range(0, len(ts_codes), batch_size)]
        
        # 使用多线程处理批次
        with ThreadPoolExecutor(max_workers=min(5, len(batches))) as executor:
            futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_results = future.result()
                    results.update(batch_results)
                    
                    # 显示进度
                    progress = (batch_idx + 1) / len(batches) * 100
                    if (batch_idx + 1) % 5 == 0:
                        print(f"    [批量资金流] 进度: {batch_idx + 1}/{len(batches)} ({progress:.1f}%)")
                except Exception as e:
                    print(f"    [错误] 批次 {batch_idx} 处理失败: {e}")
        
        # 批次间等待
        time.sleep(2)
        
        print(f"    [批量资金流] 完成，成功获取 {len(results)}/{len(ts_codes)} 只股票的资金流数据")
        return results
    
    def get_northbound_holdings(self, ts_code: str, days: int = 20) -> Dict:
        """
        获取北向资金持仓变化
        
        Args:
            ts_code: 股票代码
            days: 获取天数
            
        Returns:
            北向资金持仓数据
        """
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
            
            # 获取沪深港通持股明细
            df = self.pro.hk_hold(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(df) == 0:
                return {
                    'northbound_ratio': 0,
                    'northbound_change': 0,
                    'has_northbound': False
                }
            
            df = df.sort_values('trade_date').tail(days)
            
            # 最新持股比例
            latest = df.iloc[-1]
            latest_ratio = latest['ratio'] / 10000 if 'ratio' in latest else 0
            
            # 持股比例变化
            if len(df) > 1:
                prev = df.iloc[-2]
                prev_ratio = prev['ratio'] / 10000 if 'ratio' in prev else 0
                ratio_change = latest_ratio - prev_ratio
            else:
                ratio_change = 0
            
            return {
                'northbound_ratio': round(latest_ratio, 2),
                'northbound_change': round(ratio_change, 2),
                'has_northbound': latest_ratio > 0
            }
        except Exception as e:
            # 北向资金数据可能不存在，返回默认值
            return {
                'northbound_ratio': 0,
                'northbound_change': 0,
                'has_northbound': False
            }
    
    def calculate_moneyflow_score(self, ts_code: str) -> float:
        """
        计算资金流因子得分（0-100）
        
        评分逻辑：
        - 主力连续净流入：得分高
        - 最新单日大额净流入：得分高
        - 北向资金增持：加分
        
        Returns:
            资金流得分
        """
        # 获取资金流向数据
        mf_data = self.get_stock_moneyflow(ts_code, days=20)
        
        # 获取北向资金数据
        nb_data = self.get_northbound_holdings(ts_code, days=20)
        
        score = 0
        
        # 1. 累计主力净流入得分（40分）
        total_net_inflow = mf_data['total_net_inflow_amount']
        if total_net_inflow > 50000000:  # 净流入 > 5000万
            score += 40
        elif total_net_inflow > 20000000:  # 净流入 > 2000万
            score += 30
        elif total_net_inflow > 0:  # 净流入 > 0
            score += 20
        elif total_net_inflow > -20000000:  # 净流出 < 2000万
            score += 10
        # 净流出 > 2000万：不加分
        
        # 2. 最新单日主力净流入得分（30分）
        latest_net_inflow = mf_data['latest_net_inflow_vol']
        if latest_net_inflow > 100000:  # 单日净流入 > 10万手
            score += 30
        elif latest_net_inflow > 0:
            score += 20
        elif latest_net_inflow > -50000:  # 单日净流出 < 5万手
            score += 10
        
        # 3. 北向资金得分（30分）
        if nb_data['has_northbound']:
            score += 10  # 有北向资金持股：+10分
            
            if nb_data['northbound_change'] > 0.5:  # 增持超过0.5%
                score += 20  # 大幅增持：+20分
            elif nb_data['northbound_change'] > 0:
                score += 10  # 增持：+10分
        
        return min(score, 100)
    
    def get_stock_sector(self, ts_code: str) -> str:
        """
        获取股票所属板块（申万一级分类）
        
        Args:
            ts_code: 股票代码
            
        Returns:
            板块名称
        """
        try:
            # 获取股票基本信息
            df = self.pro.daily_basic(
                ts_code=ts_code,
                trade_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,industry'
            )
            
            if len(df) > 0:
                return df.iloc[0]['industry']
            else:
                return '未知'
        except Exception as e:
            return '未知'
    
    def get_sector_performance(self, sector: str, days: int = 20) -> Dict:
        """
        获取板块整体表现（简化版：使用默认值）
        
        Args:
            sector: 板块名称
            days: 统计天数
            
        Returns:
            板块表现数据
        """
        try:
            # 简化处理：返回默认值
            # 由于API限制，暂时使用默认板块得分
            return {
                'sector': sector,
                'avg_pct_chg': 0,
                'up_ratio': 50,
                'stock_count': 10,
                'is_hot': False
            }
        except Exception as e:
            return {
                'sector': sector,
                'avg_pct_chg': 0,
                'up_ratio': 0,
                'stock_count': 0,
                'is_hot': False
            }
            sector_df = df[df['industry'] == sector]
            
            if len(sector_df) == 0:
                return {
                    'sector': sector,
                    'avg_pct_chg': 0,
                    'up_ratio': 0,
                    'stock_count': 0,
                    'is_hot': False
                }
            
            # 获取最新数据
            latest_date = sector_df['trade_date'].max()
            latest_df = sector_df[sector_df['trade_date'] == latest_date]
            
            # 计算板块表现
            avg_pct_chg = latest_df['pct_chg'].mean()
            up_count = (latest_df['pct_chg'] > 0).sum()
            total_count = len(latest_df)
            up_ratio = up_count / total_count * 100 if total_count > 0 else 0
            
            # 判断是否为热门板块
            # 热门板块标准：平均涨幅 > 3% 且 上涨占比 > 70%
            is_hot = (avg_pct_chg > 3) and (up_ratio > 70)
            
            return {
                'sector': sector,
                'avg_pct_chg': round(avg_pct_chg, 2),
                'up_ratio': round(up_ratio, 2),
                'stock_count': total_count,
                'is_hot': is_hot
            }
        except Exception as e:
            print(f"[错误] 获取板块表现失败 {sector}: {e}")
            return {
                'sector': sector,
                'avg_pct_chg': 0,
                'up_ratio': 0,
                'stock_count': 0,
                'is_hot': False
            }
    
    def calculate_sector_score(self, ts_code: str) -> Tuple[float, str]:
        """
        计算板块共振因子得分（0-100）
        
        评分逻辑：
        - 热门板块内股票：得分高
        - 板块平均涨幅：得分正相关
        - 板块上涨占比：得分正相关
        
        Returns:
            (板块得分, 板块名称)
        """
        # 获取股票所属板块
        sector = self.get_stock_sector(ts_code)
        
        if sector == '未知':
            return 50, sector  # 未知板块给中等分
        
        # 获取板块表现
        sector_perf = self.get_sector_performance(sector, days=20)
        
        score = 0
        
        # 1. 热门板块加分（50分）
        if sector_perf['is_hot']:
            score += 50  # 热门板块：+50分
        
        # 2. 板块平均涨幅得分（30分）
        avg_pct = sector_perf['avg_pct_chg']
        if avg_pct > 5:
            score += 30
        elif avg_pct > 3:
            score += 25
        elif avg_pct > 1:
            score += 15
        elif avg_pct > 0:
            score += 10
        elif avg_pct > -1:
            score += 5
        
        # 3. 板块上涨占比得分（20分）
        up_ratio = sector_perf['up_ratio']
        if up_ratio > 80:
            score += 20
        elif up_ratio > 70:
            score += 15
        elif up_ratio > 60:
            score += 10
        elif up_ratio > 50:
            score += 5
        
        return min(score, 100), sector
    
    def calculate_composite_score(
        self,
        ts_code: str,
        technical_score: float
    ) -> Dict:
        """
        计算综合得分（多因子模型）
        
        综合得分 = 资金流因子 × 40% + 板块共振因子 × 40% + 技术因子 × 20%
        
        Args:
            ts_code: 股票代码
            technical_score: 技术评分（原有形态评分）
            
        Returns:
            综合评分结果
        """
        # 计算各因子得分
        moneyflow_score = self.calculate_moneyflow_score(ts_code)
        sector_score, sector_name = self.calculate_sector_score(ts_code)
        
        # 加权计算综合得分
        composite_score = (
            moneyflow_score * self.factor_weights['moneyflow'] +
            sector_score * self.factor_weights['sector_resonance'] +
            technical_score * self.factor_weights['technical']
        )
        
        # 四舍五入到整数
        composite_score = round(composite_score)
        
        return {
            'ts_code': ts_code,
            'technical_score': round(technical_score),
            'moneyflow_score': moneyflow_score,
            'sector_score': sector_score,
            'sector_name': sector_name,
            'composite_score': composite_score,
            'score_breakdown': {
                '资金流因子': f"{moneyflow_score}分 × {self.factor_weights['moneyflow']*100}%",
                '板块共振': f"{sector_score}分 × {self.factor_weights['sector_resonance']*100}%",
                '技术形态': f"{technical_score}分 × {self.factor_weights['technical']*100}%"
            }
        }
    def batch_calculate_scores(
        self,
        stock_list: List[str],
        technical_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        批量计算多因子得分（优化版：批量获取API数据）
        
        Args:
            stock_list: 股票代码列表
            technical_scores: 技术评分字典 {ts_code: score}
            
        Returns:
            综合评分DataFrame
        """
        results = []
        
        print(f"\n[多因子模型] 开始计算 {len(stock_list)} 只股票的综合得分...")
        
        # 步骤1：批量获取资金流数据
        print(f"[步骤1] 批量获取资金流数据...")
        moneyflow_data = self.get_batch_moneyflow(stock_list, days=20, batch_size=50)
        print(f"  成功获取 {len(moneyflow_data)} 只股票的资金流数据")
        
        # 步骤2：批量获取板块数据（优化为单次查询）
        print(f"[步骤2] 批量获取板块数据...")
        try:
            # 使用stock_basic获取行业信息（更准确）
            df_stock_basic = self.pro.stock_basic(
                ts_code=",".join(stock_list),
                fields='ts_code,name,industry'
            )
            
            if len(df_stock_basic) > 0:
                sector_map = df_stock_basic.set_index('ts_code')['industry'].to_dict()
            else:
                sector_map = {}
            
            print(f"  成功获取 {len(sector_map)} 只股票的板块信息")
            
        except Exception as e:
            print(f"  [警告] 批量获取板块数据失败: {e}")
            sector_map = {}
        
        # 步骤3：批量获取板块表现数据
        print(f"[步骤3] 批量获取板块表现...")
        unique_sectors = list(set([v for v in sector_map.values() if v != '未知']))
        print(f"  共 {len(unique_sectors)} 个板块")
        
        # 获取所有板块的整体表现
        sector_performance_map = {}
        for sector in unique_sectors:
            try:
                perf = self.get_sector_performance(sector, days=20)
                sector_performance_map[sector] = perf
                time.sleep(0.05)  # 请求间隔
            except Exception as e:
                print(f"  [错误] 获取板块 {sector} 表现失败: {e}")
                sector_performance_map[sector] = {
                    'avg_pct_chg': 0,
                    'up_ratio': 0,
                    'stock_count': 0,
                    'is_hot': False
                }
        
        print(f"  成功获取 {len(sector_performance_map)} 个板块的表现数据")
        
        # 步骤4：计算综合得分
        print(f"[步骤4] 计算综合得分...")
        for i, ts_code in enumerate(stock_list):
            try:
                # 获取技术评分
                tech_score = technical_scores.get(ts_code, 60)  # 默认60分
                
                # 获取资金流数据（从缓存）
                mf_data = moneyflow_data.get(ts_code, {
                    'total_net_inflow_vol': 0,
                    'total_net_inflow_amount': 0,
                    'latest_net_inflow_vol': 0,
                    'latest_buy_vol': 0,
                    'latest_sell_vol': 0
                })
                
                # 获取板块数据（从缓存）
                sector_name = sector_map.get(ts_code, '未知')
                sector_perf = sector_performance_map.get(sector_name, {
                    'avg_pct_chg': 0,
                    'up_ratio': 0,
                    'stock_count': 0,
                    'is_hot': False
                })
                
                # 计算资金流得分
                moneyflow_score = self._calculate_moneyflow_score_from_data(mf_data)
                
                # 计算板块得分
                sector_score = self._calculate_sector_score_from_perf(sector_perf)
                
                # 计算综合得分
                composite_score = (
                    moneyflow_score * self.factor_weights['moneyflow'] +
                    sector_score * self.factor_weights['sector_resonance'] +
                    tech_score * self.factor_weights['technical']
                )
                
                results.append({
                    'ts_code': ts_code,
                    'technical_score': round(tech_score),
                    'moneyflow_score': moneyflow_score,
                    'sector_score': sector_score,
                    'sector_name': sector_name,
                    'composite_score': round(composite_score),
                    'score_breakdown': {
                        '资金流因子': f"{moneyflow_score}分 × {self.factor_weights['moneyflow']*100}%",
                        '板块共振': f"{sector_score}分 × {self.factor_weights['sector_resonance']*100}%",
                        '技术形态': f"{tech_score}分 × {self.factor_weights['technical']*100}%"
                    }
                })
                
                # 显示进度
                if (i + 1) % 100 == 0:
                    print(f"  进度: {i + 1}/{len(stock_list)} ({(i+1)/len(stock_list)*100:.1f}%)")
                    
            except Exception as e:
                print(f"[错误] 计算 {ts_code} 失败: {e}")
                continue
        
        print(f"[完成] 综合得分计算完成，共处理 {len(results)} 只股票\n")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 按综合得分降序排列
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        
        return df
    
    def _calculate_moneyflow_score_from_data(self, mf_data: Dict) -> float:
        """从资金流数据计算得分"""
        score = 0
        
        # 1. 累计主力净流入得分（40分）
        total_net_inflow = mf_data.get('total_net_inflow_amount', 0)
        if total_net_inflow > 50000000:  # 净流入 > 5000万
            score += 40
        elif total_net_inflow > 20000000:  # 净流入 > 2000万
            score += 30
        elif total_net_inflow > 0:  # 净流入 > 0
            score += 20
        elif total_net_inflow > -20000000:  # 净流出 < 2000万
            score += 10
        
        # 2. 最新单日主力净流入得分（30分）
        latest_net_inflow = mf_data.get('latest_net_inflow_vol', 0)
        if latest_net_inflow > 100000:  # 单日净流入 > 10万手
            score += 30
        elif latest_net_inflow > 0:
            score += 20
        elif latest_net_inflow > -50000:  # 单日净流出 < 5万手
            score += 10
        
        # 3. 北向资金得分（30分）
        # 注：暂时不计算北向资金，因为API调用次数限制
        # 如果需要，可以单独批量获取
        
        return min(score, 100)
    
    def _calculate_sector_score_from_perf(self, sector_perf: Dict) -> float:
        """从板块表现数据计算得分"""
        score = 0
        
        # 1. 热门板块加分（50分）
        if sector_perf.get('is_hot', False):
            score += 50
        
        # 2. 板块平均涨幅得分（30分）
        avg_pct = sector_perf.get('avg_pct_chg', 0)
        if avg_pct > 5:
            score += 30
        elif avg_pct > 3:
            score += 25
        elif avg_pct > 1:
            score += 15
        elif avg_pct > 0:
            score += 10
        elif avg_pct > -1:
            score += 5
        
        # 3. 板块上涨占比得分（20分）
        up_ratio = sector_perf.get('up_ratio', 0)
        if up_ratio > 80:
            score += 20
        elif up_ratio > 70:
            score += 15
        elif up_ratio > 60:
            score += 10
        elif up_ratio > 50:
            score += 5
        
        return min(score, 100)


def main():
    """测试多因子模型"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant 多因子选股模型")
    print(" " * 30 + "测试运行")
    print("="*80)
    
    model = MultiFactorModel()
    
    # 测试股票列表
    test_stocks = [
        '600997.SH',  # 开滦股份
        '601618.SH',  # 中国中冶
        '605337.SH',  # 李子园
        '600508.SH'   # 上海能源
    ]
    
    # 模拟技术评分
    tech_scores = {
        '600997.SH': 93,
        '601618.SH': 90,
        '605337.SH': 80,
        '600508.SH': 83
    }
    
    # 批量计算综合得分
    df = model.batch_calculate_scores(test_stocks, tech_scores)
    
    print("\n[综合得分结果]")
    print(df[['ts_code', 'technical_score', 'moneyflow_score', 'sector_score', 'sector_name', 'composite_score']].to_string(index=False))
    
    print("\n[详细得分分解]")
    for _, row in df.iterrows():
        print(f"\n【{row['ts_code']}】综合得分: {row['composite_score']}")
        for key, value in row['score_breakdown'].items():
            print(f"  {key}: {value}")
    
    print("\n[完成] 多因子模型测试完成\n")


if __name__ == "__main__":
    main()
