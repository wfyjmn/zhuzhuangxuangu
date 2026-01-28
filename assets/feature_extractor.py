# -*- coding: utf-8 -*-
"""
DeepQuant 特征提取器 (Feature Extractor) - 增强版
优化：
1. 修正斜率计算为百分比，消除高低价股差异
2. 增加除零保护
3. 增加波动率和位置特征
4. 移除硬编码归一化，保留原始特征供 ML 模型处理
5. 增强健壮性（自动路由复权价、处理NaN）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureExtractor:
    """特征提取器类（增强版）"""

    def __init__(self):
        """初始化特征提取器"""
        self.feature_names = [
            # --- 基础量价 ---
            'vol_ratio',          # 量比
            'turnover_rate',      # 换手率
            'pe_ttm',             # 市盈率（TTM）

            # --- 趋势特征 ---
            'pct_chg_1d',         # 1日涨跌幅 (动量)
            'pct_chg_5d',         # 5日涨跌幅
            'pct_chg_20d',        # 20日涨跌幅
            'ma5_slope',          # 5日均线斜率(%) [关键修复：百分比斜率]
            'ma20_slope',         # 20日均线斜率(%)

            # --- 偏离特征 ---
            'bias_5',             # 5日乖离率
            'bias_20',            # 20日乖离率

            # --- 震荡特征 ---
            'rsi_14',             # RSI指标
            'std_20_ratio',       # 20日标准差/均价（波动率）

            # --- 相对位置 ---
            'position_20d',       # 当前价在近20天的位置(0-1)
            'position_250d',      # 当前价在年线的位置(0-1)

            # --- MACD ---
            'macd_dif',           # MACD DIF
            'macd_dea',           # MACD DEA
            'macd_hist',          # MACD 红绿柱

            # --- 环境特征 ---
            'index_pct_chg',      # 大盘涨跌幅
            'sector_pct_chg',     # 板块涨跌幅

            # --- 评分系统 ---
            'moneyflow_score',    # 资金流得分
            'tech_score',         # 技术形态得分
            'new_score',          # 综合评分
        ]

    def _get_price_col(self, df: pd.DataFrame) -> str:
        """
        [健壮性] 自动判断使用复权价还是收盘价
        优先级：复权价 > 收盘价
        """
        if 'close_qfq' in df.columns:
            return 'close_qfq'
        return 'close'

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一次性计算所有技术指标（向量化加速）
        """
        df = df.copy()
        close_col = self._get_price_col(df)
        price = df[close_col]

        # 1. 均线 (MA)
        df['ma5'] = price.rolling(window=5).mean()
        df['ma10'] = price.rolling(window=10).mean()
        df['ma20'] = price.rolling(window=20).mean()

        # 2. 乖离率 (BIAS)
        df['bias_5'] = (price - df['ma5']) / df['ma5'] * 100
        df['bias_20'] = (price - df['ma20']) / df['ma20'] * 100

        # 3. [关键修复] 均线斜率 (Slope %)
        # 计算公式：(今日MA - 昨日MA) / 昨日MA * 100
        # 这样可以消除高价股和低价股的差异
        df['ma5_slope'] = df['ma5'].pct_change() * 100
        df['ma20_slope'] = df['ma20'].pct_change() * 100

        # 4. [关键修复] RSI (相对强弱) - 增加除零保护
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)  # 避免除零
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 5. MACD
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df['macd_dif'] = ema12 - ema26
        df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2

        # 6. [新增] 波动率 (20日标准差 / 均价)
        # 衡量股价的波动程度，对判断洗盘、强攻很有帮助
        df['std_20_ratio'] = price.rolling(20).std() / df['ma20'] * 100

        # 7. [新增] 相对位置 (Position)
        # 计算当前价格在过去一段时间内的相对位置 (0-1)
        # 0 = 最低点, 1 = 最高点
        # 对判断股票所处阶段（吸筹、拉升、出货）非常有帮助
        low_20 = price.rolling(20).min()
        high_20 = price.rolling(20).max()
        df['position_20d'] = (price - low_20) / (high_20 - low_20 + 1e-9)

        low_250 = price.rolling(250).min()
        high_250 = price.rolling(250).max()
        df['position_250d'] = (price - low_250) / (high_250 - low_250 + 1e-9)

        return df

    def extract_features(self, df: pd.DataFrame, index_data: pd.DataFrame = None,
                       sector_data: pd.DataFrame = None, tech_score: float = None,
                       moneyflow_score: float = None, new_score: float = None) -> Dict:
        """
        提取单只股票的特征向量

        Args:
            df: 股票历史数据（至少30天）
            index_data: 大盘指数数据
            sector_data: 板块数据
            tech_score: 技术形态评分
            moneyflow_score: 资金流得分
            new_score: 综合评分

        Returns:
            特征字典
        """
        # [健壮性] 确保数据长度足够
        if len(df) < 30:
            # 返回全0特征，避免报错中断流程
            return {k: 0 for k in self.feature_names}

        # 1. 计算指标（一次性向量化计算所有指标）
        df_ind = self.calculate_indicators(df)
        latest = df_ind.iloc[-1]
        close_col = self._get_price_col(df)

        # 2. 组装特征
        features = {}

        # --- 基础特征 ---
        features['vol_ratio'] = latest.get('vol_ratio', 1.0)  # 默认为1
        features['turnover_rate'] = latest.get('turnover_rate', 0.0)
        features['pe_ttm'] = latest.get('pe_ttm', 0.0)

        # --- 技术特征 (直接从计算好的列取值) ---
        tech_cols = ['bias_5', 'bias_20', 'ma5_slope', 'ma20_slope',
                     'rsi_14', 'std_20_ratio', 'position_20d', 'position_250d',
                     'macd_dif', 'macd_dea', 'macd_hist']

        for col in tech_cols:
            val = latest.get(col, 0)
            # [健壮性] 处理 NaN 和 inf（刚上市或停牌可能导致计算出NaN）
            features[col] = 0 if (pd.isna(val) or np.isinf(val)) else val

        # --- 涨跌幅特征 ---
        # 必须重新计算，确保是基于复权价
        def calc_pct(days: int) -> float:
            """计算N日涨跌幅"""
            if len(df_ind) <= days:
                return 0.0
            prev = df_ind.iloc[-(days + 1)][close_col]
            curr = latest[close_col]
            if prev == 0:
                return 0.0
            return (curr - prev) / prev * 100

        features['pct_chg_1d'] = latest.get('pct_chg', 0)  # 当日涨跌幅
        features['pct_chg_5d'] = calc_pct(5)
        features['pct_chg_20d'] = calc_pct(20)

        # --- 环境特征 ---
        # 取大盘和板块的最后一天涨幅
        if index_data is not None and len(index_data) > 0:
            features['index_pct_chg'] = index_data.iloc[-1]['pct_chg']
        else:
            features['index_pct_chg'] = 0

        if sector_data is not None and len(sector_data) > 0:
            features['sector_pct_chg'] = sector_data.iloc[-1]['pct_chg']
        else:
            features['sector_pct_chg'] = 0

        # --- 评分特征 ---
        features['moneyflow_score'] = moneyflow_score if moneyflow_score else 0
        features['tech_score'] = tech_score if tech_score else 0
        features['new_score'] = new_score if new_score else 0

        return features

    def extract_batch_features(self, stock_list: Dict[str, Dict], index_data: pd.DataFrame = None,
                             sector_data: Dict[str, pd.DataFrame] = None) -> List[Dict]:
        """
        批量提取特征

        Args:
            stock_list: 股票列表 {ts_code: {'df': DataFrame, 'tech_score': float, ...}}
            index_data: 大盘指数数据
            sector_data: 板块数据 {sector_name: DataFrame}

        Returns:
            特征列表
        """
        features_list = []

        for ts_code, stock_data in stock_list.items():
            df = stock_data['df']
            tech_score = stock_data.get('tech_score')
            moneyflow_score = stock_data.get('moneyflow_score')
            new_score = stock_data.get('new_score')

            # 获取板块数据
            industry = stock_data.get('industry')
            sector_df = sector_data.get(industry) if sector_data and industry else None

            try:
                features = self.extract_features(
                    df=df,
                    index_data=index_data,
                    sector_data=sector_df,
                    tech_score=tech_score,
                    moneyflow_score=moneyflow_score,
                    new_score=new_score
                )

                features_list.append({
                    'ts_code': ts_code,
                    **features
                })

            except Exception as e:
                print(f"  [警告] {ts_code} 特征提取失败: {e}")
                continue

        return features_list

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names


def main():
    """测试函数"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant 特征提取器（增强版）")
    print(" " * 30 + "测试运行")
    print("="*80 + "\n")

    # 创建模拟数据（包含复权价）
    dates = pd.date_range('2024-01-01', periods=100)
    df = pd.DataFrame({
        'trade_date': [d.strftime('%Y%m%d') for d in dates],
        'close': np.linspace(10, 15, 100) + np.random.randn(100),  # 模拟上涨
        'adj_factor': [1.0] * 100,
        'vol_ratio': np.random.uniform(0.5, 2.5, 100),
        'turnover_rate': np.random.uniform(1, 5, 100),
        'pct_chg': np.random.randn(100)
    })

    # 构造复权价
    df['close_qfq'] = df['close'] * df['adj_factor']

    # 初始化特征提取器
    extractor = FeatureExtractor()

    # 测试特征提取
    print("[测试] 提取特征")
    feats = extractor.extract_features(df, new_score=85)

    print(f"\n[结果] 提取到 {len(feats)} 个特征:")
    for k, v in feats.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")

    # 测试特征名称
    print(f"\n[特征列表] 共 {len(extractor.feature_names)} 个特征:")
    print(extractor.feature_names)

    print("\n[完成] 特征提取器测试完成\n")


if __name__ == "__main__":
    main()
