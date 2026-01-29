# -*- coding: utf-8 -*-
"""
DeepQuant 特征提取器 (Feature Extractor) - V5.0 终极版
功能：将原始行情数据转换为 AI 可识别的数值特征向量

优化点：
1. 向量化计算：速度极快
2. 鲁棒性：自动处理 NaN、Inf、除零错误
3. 特征一致性：确保输出特征与定义完全对齐
4. 复权适配：优先使用前复权价格计算技术指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """特征提取器类"""

    def __init__(self):
        """
        初始化特征提取器
        定义特征列表，必须与 extract_features 返回的 key 严格一致
        """
        self.feature_names = [
            # --- 基础量价 (3) ---
            'vol_ratio',          # 量比 (需要数据源包含该列)
            'turnover_rate',      # 换手率 (需要数据源包含该列)
            'pe_ttm',             # 市盈率 (需要数据源包含该列)

            # --- 趋势特征 (5) ---
            'pct_chg_1d',         # 1日涨跌幅
            'pct_chg_5d',         # 5日涨跌幅
            'pct_chg_20d',        # 20日涨跌幅
            'ma5_slope',          # 5日均线斜率(%)
            'ma20_slope',         # 20日均线斜率(%)

            # --- 偏离特征 (2) ---
            'bias_5',             # 5日乖离率
            'bias_20',            # 20日乖离率

            # --- 震荡特征 (2) ---
            'rsi_14',             # RSI指标
            'std_20_ratio',       # 波动率 (20日标准差/均价)

            # --- 相对位置 (2) ---
            'position_20d',       # 近20天位置(0-1)
            'position_250d',      # 年线位置(0-1)

            # --- MACD (3) ---
            'macd_dif',
            'macd_dea',
            'macd_hist',

            # --- 环境特征 (2) ---
            'index_pct_chg',      # 大盘涨跌幅
            'sector_pct_chg',     # 板块涨跌幅

            # --- 评分系统 (3) ---
            'moneyflow_score',    # 资金流得分
            'tech_score',         # 技术形态得分
            'new_score',          # 综合评分
        ]
        
        # 总特征数检查
        # print(f"FeatureExtractor 初始化: 共 {len(self.feature_names)} 个特征")

    def _get_price_col(self, df: pd.DataFrame) -> str:
        """优先使用前复权价格(close_qfq)计算指标"""
        if 'close_qfq' in df.columns:
            return 'close_qfq'
        return 'close'

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一次性计算所有技术指标（向量化加速）
        """
        # 避免修改原始数据
        df = df.copy()
        
        # 确定使用的价格列
        close_col = self._get_price_col(df)
        price = df[close_col]

        # ---------------------------------------------------------
        # 1. 均线 (MA)
        # ---------------------------------------------------------
        df['ma5'] = price.rolling(window=5).mean()
        df['ma20'] = price.rolling(window=20).mean()

        # ---------------------------------------------------------
        # 2. 乖离率 (BIAS)
        # ---------------------------------------------------------
        # 价格偏离均线的百分比
        df['bias_5'] = (price - df['ma5']) / (df['ma5'] + 1e-9) * 100
        df['bias_20'] = (price - df['ma20']) / (df['ma20'] + 1e-9) * 100

        # ---------------------------------------------------------
        # 3. 均线斜率 (Slope %)
        # ---------------------------------------------------------
        # 使用 pct_change 计算斜率，消除高低价股差异
        df['ma5_slope'] = df['ma5'].pct_change() * 100
        df['ma20_slope'] = df['ma20'].pct_change() * 100

        # ---------------------------------------------------------
        # 4. RSI (相对强弱)
        # ---------------------------------------------------------
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # ---------------------------------------------------------
        # 5. MACD
        # ---------------------------------------------------------
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2

        # ---------------------------------------------------------
        # 6. 波动率 (Volatility)
        # ---------------------------------------------------------
        # 20日标准差 / 20日均价
        df['std_20_ratio'] = price.rolling(20).std() / (df['ma20'] + 1e-9) * 100

        # ---------------------------------------------------------
        # 7. 相对位置 (Position)
        # ---------------------------------------------------------
        # (当前价 - 最低价) / (最高价 - 最低价)
        low_20 = price.rolling(20).min()
        high_20 = price.rolling(20).max()
        df['position_20d'] = (price - low_20) / (high_20 - low_20 + 1e-9)

        low_250 = price.rolling(250).min()
        high_250 = price.rolling(250).max()
        df['position_250d'] = (price - low_250) / (high_250 - low_250 + 1e-9)

        # ---------------------------------------------------------
        # 清洗数据
        # ---------------------------------------------------------
        # 前几行计算结果为NaN，使用向后填充或0填充
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        return df

    def extract_features(self, df: pd.DataFrame, index_data: pd.DataFrame = None,
                       sector_data: pd.DataFrame = None, tech_score: float = None,
                       moneyflow_score: float = None, new_score: float = None) -> pd.DataFrame:
        """
        提取单只股票的特征向量（返回 DataFrame 格式，单行）

        Args:
            df: 股票历史数据（至少30天），必须截止到 feature_date
            index_data: 大盘数据
            sector_data: 板块数据
            ...
        """
        # [健壮性] 数据长度检查
        if df is None or len(df) < 30:
            # 返回全0特征的 DataFrame
            return pd.DataFrame(columns=self.feature_names)

        # 1. 计算指标
        df_ind = self.calculate_indicators(df)
        
        # 取最后一行（即 feature_date 当天的数据）
        latest = df_ind.iloc[-1]
        close_col = self._get_price_col(df)

        # 2. 构建特征字典
        features = {}

        # --- 直接从数据源获取的特征 ---
        # 如果数据源里没有这些列，使用默认值
        features['vol_ratio'] = latest.get('vol_ratio', 1.0)
        features['turnover_rate'] = latest.get('turnover_rate', 0.0)
        features['pe_ttm'] = latest.get('pe_ttm', 0.0)

        # --- 计算好的技术指标 ---
        tech_cols = ['bias_5', 'bias_20', 'ma5_slope', 'ma20_slope',
                     'rsi_14', 'std_20_ratio', 'position_20d', 'position_250d',
                     'macd_dif', 'macd_dea', 'macd_hist']
        for col in tech_cols:
            features[col] = latest.get(col, 0.0)

        # --- 涨跌幅特征 ---
        # 重新计算基于 close_col 的涨跌幅，确保准确
        features['pct_chg_1d'] = latest.get('pct_chg', 0.0)
        
        def safe_pct(n):
            if len(df_ind) > n:
                prev = df_ind[close_col].iloc[-(n+1)]
                curr = latest[close_col]
                return (curr - prev) / (prev + 1e-9) * 100
            return 0.0
            
        features['pct_chg_5d'] = safe_pct(5)
        features['pct_chg_20d'] = safe_pct(20)

        # --- 环境特征 ---
        features['index_pct_chg'] = index_data.iloc[-1]['pct_chg'] if (index_data is not None and len(index_data)>0) else 0.0
        features['sector_pct_chg'] = sector_data.iloc[-1]['pct_chg'] if (sector_data is not None and len(sector_data)>0) else 0.0

        # --- 评分特征 ---
        features['moneyflow_score'] = moneyflow_score if moneyflow_score is not None else 0.0
        features['tech_score'] = tech_score if tech_score is not None else 0.0
        features['new_score'] = new_score if new_score is not None else 0.0

        # 3. 转换为 DataFrame 并确保列顺序一致
        # 使用列表封装 dict，创建单行 DataFrame
        feature_df = pd.DataFrame([features])
        
        # [关键] 强制对齐列名，缺失补0，多余丢弃
        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        # 按定义顺序排序
        feature_df = feature_df[self.feature_names]
        
        return feature_df

    def extract_batch_features(self, stock_list: Dict, **kwargs) -> pd.DataFrame:
        """
        批量提取（辅助方法）
        """
        results = []
        for ts_code, data in stock_list.items():
            try:
                # 假设 data 结构为 {'df': ..., 'score': ...}
                df = data.get('df')
                feat_df = self.extract_features(df, **kwargs)
                if not feat_df.empty:
                    # 添加 ID 列用于标识
                    feat_df['ts_code'] = ts_code
                    results.append(feat_df)
            except Exception as e:
                logger.warning(f"Feature extraction failed for {ts_code}: {e}")
                
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()


# 测试代码
if __name__ == '__main__':
    # 模拟数据
    dates = pd.date_range('20240101', periods=60)
    df = pd.DataFrame({
        'trade_date': dates,
        'close': np.random.randn(60).cumsum() + 10,
        'open': np.random.randn(60).cumsum() + 10,
        'high': np.random.randn(60).cumsum() + 12,
        'low': np.random.randn(60).cumsum() + 8,
        'vol': np.random.rand(60) * 1000,
        'amount': np.random.rand(60) * 10000,
        'pct_chg': np.random.randn(60),
        'vol_ratio': 1.5,
        'turnover_rate': 2.0,
        'pe_ttm': 15.0
    })
    # 模拟复权列
    df['close_qfq'] = df['close'] 
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(df, new_score=88)
    
    print("提取结果:")
    print(features.T)
    print(f"\n特征维度: {features.shape}")
