# -*- coding: utf-8 -*-
"""
DeepQuant 特征提取器 (Feature Extractor)
功能：
1. 提取技术指标特征
2. 提取基本面特征
3. 提取市场环境特征
4. 特征归一化处理

核心特征：
- 量比
- 换手率
- 市盈率
- 乖离率（BIAS）
- 大盘涨跌幅
- 板块涨跌幅
- 技术形态评分
- 资金流向特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureExtractor:
    """特征提取器类"""

    def __init__(self):
        """初始化特征提取器"""
        self.feature_names = [
            'vol_ratio',          # 量比
            'turnover_rate',      # 换手率
            'pe_ttm',             # 市盈率（TTM）
            'bias_5',             # 5日乖离率
            'bias_10',            # 10日乖离率
            'bias_20',            # 20日乖离率
            'pct_chg_5d',         # 5日涨跌幅
            'pct_chg_10d',        # 10日涨跌幅
            'pct_chg_20d',        # 20日涨跌幅
            'ma5_slope',          # 5日均线斜率
            'ma10_slope',         # 10日均线斜率
            'ma20_slope',         # 20日均线斜率
            'rsi',                # RSI指标
            'macd_dif',           # MACD DIF
            'macd_dea',           # MACD DEA
            'index_pct_chg',      # 大盘涨跌幅
            'sector_pct_chg',     # 板块涨跌幅
            'moneyflow_score',    # 资金流得分
            'tech_score',         # 技术形态得分
            'new_score',          # 综合评分
        ]

    def calculate_ma(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算移动平均线

        Args:
            df: 行情数据
            periods: 周期列表

        Returns:
            添加了MA列的DataFrame
        """
        df = df.copy()
        for period in periods:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
        return df

    def calculate_bias(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算乖离率（BIAS）

        Args:
            df: 行情数据（必须包含ma列）
            periods: 周期列表

        Returns:
            添加了BIAS列的DataFrame
        """
        df = df.copy()
        for period in periods:
            df[f'bias_{period}'] = (df['close'] - df[f'ma{period}']) / df[f'ma{period}'] * 100
        return df

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算RSI指标

        Args:
            df: 行情数据
            period: RSI周期

        Returns:
            添加了RSI列的DataFrame
        """
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标

        Args:
            df: 行情数据

        Returns:
            添加了MACD列的DataFrame
        """
        df = df.copy()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = df['ema12'] - df['ema26']
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2
        return df

    def extract_features(self, df: pd.DataFrame, index_data: pd.DataFrame = None,
                       sector_data: pd.DataFrame = None, tech_score: float = None,
                       moneyflow_score: float = None, new_score: float = None) -> Dict:
        """
        提取特征向量

        Args:
            df: 单只股票的历史行情数据（至少20天）
            index_data: 大盘指数数据
            sector_data: 板块数据
            tech_score: 技术形态评分
            moneyflow_score: 资金流得分
            new_score: 综合评分

        Returns:
            特征字典
        """
        if len(df) < 20:
            raise ValueError("数据长度不足20天")

        # 获取最新一天的收盘价
        latest = df.iloc[-1]

        # 1. 基础特征
        features = {
            'vol_ratio': latest.get('vol_ratio', 0),
            'turnover_rate': latest.get('turnover_rate', 0),
            'pe_ttm': latest.get('pe_ttm', 0),
        }

        # 2. 计算技术指标
        df = self.calculate_ma(df)
        df = self.calculate_bias(df)
        df = self.calculate_rsi(df)
        df = self.calculate_macd(df)

        latest = df.iloc[-1]

        # 3. 乖离率特征
        features['bias_5'] = latest['bias_5']
        features['bias_10'] = latest['bias_10']
        features['bias_20'] = latest['bias_20']

        # 4. 涨跌幅特征
        features['pct_chg_5d'] = (latest['close'] / df.iloc[-5]['close'] - 1) * 100 if len(df) >= 5 else 0
        features['pct_chg_10d'] = (latest['close'] / df.iloc[-10]['close'] - 1) * 100 if len(df) >= 10 else 0
        features['pct_chg_20d'] = (latest['close'] / df.iloc[-20]['close'] - 1) * 100 if len(df) >= 20 else 0

        # 5. 均线斜率（趋势强度）
        features['ma5_slope'] = (latest['ma5'] - df.iloc[-2]['ma5']) if len(df) >= 2 else 0
        features['ma10_slope'] = (latest['ma10'] - df.iloc[-2]['ma10']) if len(df) >= 2 else 0
        features['ma20_slope'] = (latest['ma20'] - df.iloc[-2]['ma20']) if len(df) >= 2 else 0

        # 6. RSI特征
        features['rsi'] = latest['rsi']

        # 7. MACD特征
        features['macd_dif'] = latest['dif']
        features['macd_dea'] = latest['dea']

        # 8. 大盘涨跌幅（如果有）
        if index_data is not None and len(index_data) > 0:
            index_latest = index_data.iloc[-1]
            features['index_pct_chg'] = index_latest.get('pct_chg', 0)
        else:
            features['index_pct_chg'] = 0

        # 9. 板块涨跌幅（如果有）
        if sector_data is not None and len(sector_data) > 0:
            sector_latest = sector_data.iloc[-1]
            features['sector_pct_chg'] = sector_latest.get('pct_chg', 0)
        else:
            features['sector_pct_chg'] = 0

        # 10. 评分特征
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

    def normalize_features(self, features: Dict, method: str = 'minmax') -> Dict:
        """
        特征归一化

        Args:
            features: 特征字典
            method: 归一化方法（minmax/standard）

        Returns:
            归一化后的特征字典
        """
        normalized = features.copy()

        if method == 'minmax':
            # Min-Max归一化（需要提前知道最大最小值）
            # 这里使用简单的逻辑归一化
            for key, value in features.items():
                if key == 'ts_code':
                    continue

                if isinstance(value, (int, float)):
                    # 简单归一化：限制在 -10 到 10 之间
                    normalized[key] = max(-10, min(10, value / 10))

        elif method == 'standard':
            # Z-Score标准化（需要提前知道均值和方差）
            # 这里简化处理
            for key, value in features.items():
                if key == 'ts_code':
                    continue

                if isinstance(value, (int, float)):
                    # 简单标准化
                    normalized[key] = value / 100

        return normalized

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names


def main():
    """测试函数"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant 特征提取器")
    print(" " * 30 + "测试运行")
    print("="*80 + "\n")

    # 创建模拟数据
    dates = pd.date_range('2024-01-01', periods=30)
    data = {
        'trade_date': [d.strftime('%Y%m%d') for d in dates],
        'close': np.random.randn(30).cumsum() + 100,
        'vol_ratio': np.random.randn(30).cumsum() + 1.5,
        'turnover_rate': np.random.rand(30) * 10,
        'pe_ttm': np.random.rand(30) * 50 + 10,
    }
    df = pd.DataFrame(data)

    # 初始化特征提取器
    extractor = FeatureExtractor()

    # 测试特征提取
    print("[测试] 提取特征")
    features = extractor.extract_features(df)

    print(f"  特征数量: {len(features)}")
    print(f"  特征名称: {list(features.keys())}")

    # 打印部分特征
    print("\n  部分特征值:")
    for key, value in list(features.items())[:10]:
        print(f"    {key}: {value:.4f}")

    print("\n[完成] 特征提取器测试完成\n")


if __name__ == "__main__":
    main()
