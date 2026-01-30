# -*- coding: utf-8 -*-
"""
DeepQuant AI回测生成器 (AI Backtest Generator) - V5.0
核心升级：
1. 引入【相对收益】标签：在熊市中，跑赢大盘即为赢
2. 添加 V5.0 参数：bear_threshold, alpha_threshold
3. 优化候选股票筛选条件（只训练高流动性股票）
4. 防止数据穿越：严格分离历史（特征）与未来（标签）
5. 集成 Turbo 仓库：自动检测并利用内存加速
"""

import pandas as pd
import numpy as np
import os
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 尝试导入 FeatureExtractor
try:
    from feature_extractor import FeatureExtractor
except ImportError:
    # 简单的 Mock，防止导入失败
    class FeatureExtractor:
        def extract_features(self, df): return df

# 尝试导入 DataWarehouse (优先使用 Turbo)
try:
    from data_warehouse_turbo import DataWarehouse
    IS_TURBO = True
except ImportError:
    from data_warehouse import DataWarehouse
    IS_TURBO = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIBacktestGenerator:
    """AI回测生成器类（V5.0 优化版）"""

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化生成器
        """
        # 数据仓库实例
        self.warehouse = DataWarehouse(data_dir)
        # 特征提取器实例
        self.extractor = FeatureExtractor()

        # V5.0 策略参数
        self.holding_period = 5     # 持仓周期（天）
        self.target_return = 0.03   # 牛市目标收益：3%
        self.bear_threshold = -0.01 # 熊市定义：大盘跌幅超过 1%
        self.alpha_threshold = 0.02 # 熊市Alpha要求：跑赢大盘 2%
        self.max_drawdown_limit = -0.05 # 任何情况下的止损底线

        # 选股过滤参数
        self.amount_threshold = 10000  # 成交额门槛（千元），即1000万
        self.max_candidates = 50       # 每日最大采样数（防止样本过多）
        
        # 缓存大盘指数数据
        self._market_index_cache = None

    def _get_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        [手术三] 修复"相对收益"标签：获取大盘指数数据（上证指数 000001.SH）

        增强功能：
        1. 优先从本地仓库获取
        2. 如果本地没有，尝试临时下载
        3. 确保数据格式正确
        """
        index_code = '000001.SH'  # 上证指数

        # 1. 尝试从本地仓库获取
        df = self.warehouse.get_stock_data(index_code, end_date, days=3650)  # 多取一点，覆盖 10 年

        # 2. 如果仓库里没有，尝试临时下载（如果环境允许）
        if df is None or df.empty:
            logger.warning(f"[警告] 本地未找到指数数据 {index_code}，尝试临时下载...")
            try:
                import tushare as ts
                # 从环境变量获取 Token
                pro = ts.pro_api()

                # 下载指数日线
                df_index = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)

                if not df_index.empty:
                    df_index = df_index.sort_values('trade_date')

                    # 标准化列名（与股票数据一致）
                    df_index.rename(columns={
                        'ts_code': 'ts_code',
                        'trade_date': 'trade_date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'vol',
                        'amount': 'amount'
                    }, inplace=True)

                    # 添加 trade_date_dt 列（用于时间索引）
                    df_index['trade_date_dt'] = pd.to_datetime(df_index['trade_date'])

                    logger.info(f"[成功] 临时下载指数数据 {index_code}，共 {len(df_index)} 条记录")

                    return df_index
            except Exception as e:
                logger.error(f"[错误] 指数下载失败: {e}")

        # 3. 如果还是获取不到，返回空 DataFrame
        if df is None or df.empty:
            logger.error("[严重] 无法获取大盘指数数据，相对收益标签将失效！")
            logger.error(f"  建议操作：手动下载指数数据到 data/daily/{index_code}.csv")
            logger.error(f"  或者检查 Tushare Token 权限")
            return pd.DataFrame(columns=['trade_date', 'close', 'open', 'trade_date_dt'])

        # 4. 确保有 trade_date_dt 列
        if 'trade_date_dt' not in df.columns:
            df['trade_date_dt'] = pd.to_datetime(df['trade_date'])

        return df

    def _calculate_label_v5(self, stock_future: pd.DataFrame, market_future: pd.DataFrame) -> int:
        """
        [核心 V5.0] 动态标签计算逻辑
        
        Args:
            stock_future: 个股未来 N 天数据
            market_future: 大盘未来 N 天数据
            
        Returns:
            1 (正样本/买入), 0 (负样本/观望)
        """
        if stock_future.empty: return 0
        
        # 1. 计算个股收益
        p_start = stock_future['open'].iloc[0] # 以次日开盘价买入
        p_end = stock_future['close'].iloc[-1]
        p_min = stock_future['low'].min()
        
        stock_pct = (p_end / p_start) - 1
        stock_max_loss = (p_min / p_start) - 1

        # 硬性止损检查：如果未来 N 天内触及止损线，直接判负
        if stock_max_loss < self.max_drawdown_limit:
            return 0

        # 2. 计算大盘收益
        market_pct = 0.0
        if not market_future.empty and len(market_future) == len(stock_future):
            m_start = market_future['open'].iloc[0]
            m_end = market_future['close'].iloc[-1]
            market_pct = (m_end / m_start) - 1

        # 3. 动态判定
        is_win = False
        
        if market_pct < self.bear_threshold:
            # 【熊市场景】
            # 条件A: 跑赢大盘一定幅度 (Alpha)
            # 条件B: 自身没有大幅亏损 (例如微跌 1% 但大盘跌 5%，算赢)
            condition_a = stock_pct > (market_pct + self.alpha_threshold)
            condition_b = stock_pct > -0.03 # 允许小幅亏损，但不能深套
            
            if condition_a and condition_b:
                is_win = True
        else:
            # 【牛市/震荡市场景】
            # 纯绝对收益目标
            if stock_pct > self.target_return:
                is_win = True

        return 1 if is_win else 0

    def select_candidates(self, trade_date: str) -> List[str]:
        """
        筛选当日符合条件的候选股票
        模拟真实的选股环境：只看当下热门、流动性好的票
        """
        # 加载当日全市场数据
        df_daily = self.warehouse.load_daily_data(trade_date)
        
        if df_daily is None or df_daily.empty:
            return []

        # 过滤 ST 股 (假设 name 包含 ST)
        # 注意：load_daily_data 通常不包含 name，需要 basic_info
        # 这里简化：只通过流动性和价格筛选
        
        # 1. 过滤成交额过小的（流动性陷阱）
        # amount 单位通常是千元
        mask_liquid = df_daily['amount'] > self.amount_threshold
        
        # 2. 过滤停牌（vol = 0）
        mask_active = df_daily['vol'] > 0
        
        # 3. 过滤高价股和低价股（可选）
        mask_price = (df_daily['close'] > 3) & (df_daily['close'] < 200)

        candidates = df_daily[mask_liquid & mask_active & mask_price]
        
        # 4. 按成交额降序排列，优先选取头部股票（模拟资金关注度）
        candidates = candidates.sort_values('amount', ascending=False)
        
        # 截取前 N 只，防止生成数据太慢
        selected_codes = candidates['ts_code'].head(self.max_candidates).tolist()
        
        return selected_codes

    def generate_dataset(self, start_date: str, end_date: str, max_samples: int = None) -> pd.DataFrame:
        """
        生成训练数据集（主入口）
        """
        logger.info(f"启动 AI 回测生成器 V5.0")
        logger.info(f"范围: {start_date} ~ {end_date}")
        
        # 获取交易日历
        calendar = self.warehouse.get_trade_days(start_date, end_date)
        # 移除最后几天，因为它们没有足够的未来数据来打标签
        calendar = calendar[:-self.holding_period]
        
        logger.info(f"有效交易日: {len(calendar)} 天")
        
        all_samples = []
        total_samples_count = 0
        
        # 预加载大盘数据 (用于计算相对收益)
        # 注意：我们需要比 end_date 更远一点的数据来计算最后一天的 label
        extended_end_date = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=20)).strftime("%Y%m%d")
        market_df = self._get_market_data(start_date, extended_end_date)
        if not market_df.empty:
             market_df['trade_date_dt'] = pd.to_datetime(market_df['trade_date'])
             market_df = market_df.set_index('trade_date_dt').sort_index()

        for i, trade_date in enumerate(calendar):
            # 1. 筛选当日股票
            candidates = self.select_candidates(trade_date)
            if not candidates: continue
            
            # 随机采样（如果候选太多）
            # if len(candidates) > 20:
            #     candidates = random.sample(candidates, 20)
                
            daily_samples = []
            
            for ts_code in candidates:
                # 2. 获取特征数据 (历史 + 当天)
                # 使用 Turbo 仓库的 get_stock_data 极速获取
                # 假设我们需要 60 天历史来计算 MACD 等指标
                hist_data = self.warehouse.get_stock_data(ts_code, trade_date, days=100)
                
                if hist_data is None or len(hist_data) < 60:
                    continue
                
                # 3. 特征提取
                # 注意：FeatureExtractor 必须只使用 hist_data 计算，不能有未来数据
                features = self.extractor.extract_features(hist_data)
                
                if features.empty:
                    continue
                    
                # 取最后一行（也就是 trade_date 当天的特征）
                current_feature = features.iloc[[-1]].copy()
                
                # 4. 获取未来数据 (用于打标签)
                future_data = self.warehouse.get_future_data(ts_code, trade_date, days=self.holding_period)
                
                if future_data is None or len(future_data) < self.holding_period:
                    continue

                # 5. 获取同期大盘数据
                market_future = pd.DataFrame()
                if not market_df.empty:
                    try:
                        start_dt = pd.to_datetime(future_data['trade_date'].iloc[0])
                        end_dt = pd.to_datetime(future_data['trade_date'].iloc[-1])
                        market_future = market_df.loc[start_dt:end_dt]
                    except Exception:
                        pass

                # 6. 计算标签 (V5.0 逻辑)
                label = self._calculate_label_v5(future_data, market_future)
                
                # 7. 组装样本
                # 将元数据保留，方便后续分析，但在训练前需剔除
                current_feature['label'] = label
                current_feature['trade_date'] = trade_date
                current_feature['ts_code'] = ts_code
                
                daily_samples.append(current_feature)
            
            # 只有当采集到样本时才合并
            if daily_samples:
                all_samples.extend(daily_samples)
                total_samples_count += len(daily_samples)
            
            if (i + 1) % 5 == 0:
                logger.info(f"进度: {i+1}/{len(calendar)} | 累计样本: {total_samples_count}")
                
            # 限制总样本数（可选，防止内存溢出）
            if max_samples and total_samples_count >= max_samples:
                logger.info(f"达到最大样本数限制 ({max_samples})，提前停止")
                break

        if not all_samples:
            logger.warning("未生成任何有效样本")
            return pd.DataFrame()

        # 合并所有样本
        final_dataset = pd.concat(all_samples, ignore_index=True)
        
        # 内存优化
        for col in final_dataset.select_dtypes(include=['float64']).columns:
            final_dataset[col] = final_dataset[col].astype('float32')
            
        return final_dataset


if __name__ == '__main__':
    # 简单测试
    gen = AIBacktestGenerator()
    # 假设我们只跑最近几天
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
    
    # 仅演示逻辑，不实际跑（除非有数据）
    print("AIBacktestGenerator V5.0 初始化成功")
    print(f"参数: 目标收益={gen.target_return}, 熊市阈值={gen.bear_threshold}")
