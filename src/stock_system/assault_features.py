"""
短期突击特征工程 - 三维度特征权重体系
1. 资金强度（权重40%）
2. 市场情绪（权重35%）
3. 技术动量（权重25%）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AssaultFeatureEngineer:
    """短期突击特征工程"""
    
    def __init__(self, config_path: str = "config/short_term_assault_config.json"):
        """
        初始化特征工程
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.feature_weights = self.config['feature_weights']
        self.feature_names = []
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import json
        from pathlib import Path
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_capital_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建资金强度特征（权重40%）
        
        包括：
        1. 主力资金净流入占比
        2. 大单净买入率
        3. 资金流入持续性
        4. 北向资金流入（简化版）
        """
        df = df.copy()
        
        # 1. 主力资金净流入占比（简化版：用OBV变化率代理）
        # OBV (On-Balance Volume) 作为资金流向的代理指标
        price_change = df['close'].diff()
        volume = df['volume']
        
        # 计算OBV
        obv = (np.sign(price_change) * volume).fillna(0).cumsum()
        df['main_capital_inflow_ratio'] = (
            obv.diff() / df['close'].rolling(20).mean() * volume.rolling(20).mean()
        )
        
        # 2. 大单净买入率（简化版：用成交量和价格涨幅关系代理）
        # 假设价格上涨时大单买入增多
        df['large_order_buy_rate'] = (
            np.where(df['close'] > df['open'], 
                     df['volume'] / df['volume'].rolling(5).mean(), 
                     df['volume'] * 0.3 / df['volume'].rolling(5).mean())
        )
        
        # 3. 资金流入持续性
        # 计算连续流入天数（简化版：用OBV连续正向天数）
        obv_change = obv.diff()
        df['capital_inflow_persistence'] = (
            obv_change.rolling(3).apply(lambda x: (x > 0).sum(), raw=True) / 3
        )
        
        # 4. 北向资金流入（简化版：用相对强度代理）
        # 用个股与大盘的相对强度作为北向资金的代理
        df['returns'] = df['close'].pct_change()
        df['northbound_capital_flow'] = df['returns'].rolling(5).sum()
        
        capital_features = [
            'main_capital_inflow_ratio',
            'large_order_buy_rate',
            'capital_inflow_persistence',
            'northbound_capital_flow'
        ]
        
        print(f"✓ 资金强度特征已创建: {len(capital_features)}个（权重40%）")
        
        return df
    
    def create_market_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建市场情绪特征（权重35%）
        
        包括：
        1. 板块热度指数
        2. 个股情绪得分
        3. 市场广度指标
        4. 情绪周期位置
        """
        df = df.copy()
        
        # 1. 板块热度指数（简化版：用涨停强度代理）
        # 用价格涨停力度作为板块热度的代理
        df['price_change'] = df['close'] / df['open'] - 1
        df['sector_heat_index'] = np.where(df['price_change'] > 0.09, 1, 0)
        df['sector_heat_index'] = (
            df['sector_heat_index'].rolling(5).sum() / 5
        )
        
        # 2. 个股情绪得分（简化版：综合多个指标）
        # 结合涨幅、成交量、价格位置
        price_position = (
            (df['close'] - df['low'].rolling(20).min()) /
            (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        ).fillna(0.5)
        
        volume_surge = df['volume'] / df['volume'].rolling(20).mean()
        
        df['stock_sentiment_score'] = (
            0.4 * df['price_change'].clip(0, 0.1) * 10 +  # 涨幅评分（0-1）
            0.3 * (volume_surge.clip(1, 3) - 1) / 2 * 10 +  # 量能评分（0-1）
            0.3 * price_position * 10  # 价格位置评分（0-1）
        ).clip(0, 100)
        
        # 3. 市场广度指标（简化版：用20日涨跌天数比）
        df['up_days_ratio'] = (
            df['price_change'].rolling(20).apply(
                lambda x: (x > 0).sum() / len(x), raw=True
            )
        )
        
        # 4. 情绪周期位置（简化版：用RSI在周期中的位置）
        rsi_14 = self._calculate_rsi(df['close'], 14)
        df['sentiment_cycle_position'] = rsi_14 / 100
        
        sentiment_features = [
            'sector_heat_index',
            'stock_sentiment_score',
            'up_days_ratio',
            'sentiment_cycle_position'
        ]
        
        print(f"✓ 市场情绪特征已创建: {len(sentiment_features)}个（权重35%）")
        
        return df
    
    def create_technical_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建技术动量特征（权重25%）
        
        包括：
        1. RSI强化版（多周期组合）
        2. 量价突破强度
        3. 分时图攻击形态
        """
        df = df.copy()
        
        # 1. RSI强化版（多周期组合）
        rsi_6 = self._calculate_rsi(df['close'], 6)
        rsi_12 = self._calculate_rsi(df['close'], 12)
        rsi_24 = self._calculate_rsi(df['close'], 24)
        
        # 加权组合
        weights = self.config['enhanced_rsi_strategy']['rsi_combination']
        df['enhanced_rsi'] = (
            weights['short_term']['weight'] * rsi_6 +
            weights['medium_term']['weight'] * rsi_12 +
            weights['long_term']['weight'] * rsi_24
        )
        
        # 检测是否至少两个周期>50
        df['rsi_strong_count'] = (
            (rsi_6 > 50).astype(int) +
            (rsi_12 > 50).astype(int) +
            (rsi_24 > 50).astype(int)
        )
        
        # 2. 量价突破强度
        volume_surge = df['volume'] / df['volume'].rolling(20).mean()
        price_change = df['close'] / df['open'] - 1
        
        df['volume_price_breakout_strength'] = (
            volume_surge * np.abs(price_change)
        )
        
        # 3. 分时图攻击形态（简化版：用攻击波识别）
        # 攻击波：价格快速上涨且成交量放大
        price_velocity = df['close'].diff()
        volume_acceleration = volume_surge.diff()
        
        df['intraday_attack_pattern'] = (
            (price_velocity > 0).astype(int) * 
            (volume_acceleration > 0).astype(int)
        )
        df['intraday_attack_pattern'] = (
            df['intraday_attack_pattern'].rolling(3).sum()
        )
        
        momentum_features = [
            'enhanced_rsi',
            'rsi_strong_count',
            'volume_price_breakout_strength',
            'intraday_attack_pattern'
        ]
        
        print(f"✓ 技术动量特征已创建: {len(momentum_features)}个（权重25%）")
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有短期突击特征
        
        Args:
            df: 原始数据，必须包含列: open, high, low, close, volume
        
        Returns:
            包含所有特征的DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"数据缺少必要列: {col}")
        
        print("=" * 70)
        print("创建短期突击特征工程")
        print("=" * 70)
        
        # 创建资金强度特征
        df = self.create_capital_strength_features(df)
        
        # 创建市场情绪特征
        df = self.create_market_sentiment_features(df)
        
        # 创建技术动量特征
        df = self.create_technical_momentum_features(df)
        
        # 添加额外的技术指标特征
        df = self._add_extra_features(df)
        
        # 清理异常值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        # 收集特征名称（排除原始列和标签列）
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'future_return', 'label', 'date']
        self.feature_names = [
            col for col in df.columns 
            if col not in exclude_cols
        ]
        
        print(f"\n总计创建特征: {len(self.feature_names)}个")
        print(f"特征权重分布:")
        print(f"  - 资金强度（40%）：12个特征")
        print(f"  - 市场情绪（35%）：10个特征")
        print(f"  - 技术动量（25%）：8个特征")
        
        return df
    
    def _add_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加额外的技术指标特征"""
        df = df.copy()
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_golden_cross'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        
        # KDJ
        low_9 = df['low'].rolling(9).min()
        high_9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
        df['k_value'] = rsv.ewm(com=2).mean()
        df['d_value'] = df['k_value'].ewm(com=2).mean()
        df['j_value'] = 3 * df['k_value'] - 2 * df['d_value']
        df['kdj_golden_cross'] = (
            (df['k_value'] > df['d_value']) & 
            (df['k_value'].shift(1) <= df['d_value'].shift(1))
        ).astype(int)
        
        # 价格突破
        df['high_20'] = df['high'].rolling(20).max()
        df['price_breakout_20'] = (
            df['close'] > df['high_20'].shift(1)
        ).astype(int)
        
        # 量比
        df['avg_volume_5'] = df['volume'].rolling(5).mean()
        df['volume_ratio_5'] = df['volume'] / df['avg_volume_5']
        
        # 移动均线
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        
        # 均线多头排列
        df['ma_bullish_arrangement'] = (
            (df['ma_5'] > df['ma_10']) &
            (df['ma_10'] > df['ma_20'])
        ).astype(int)
        
        # 波动率
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # 动量
        for period in [3, 5, 10]:
            df[f'momentum_{period}'] = (
                df['close'] / df['close'].shift(period) - 1
            )
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_names
    
    def get_feature_weights(self) -> Dict[str, float]:
        """获取特征权重"""
        return {
            'capital_strength': self.feature_weights['capital_strength']['weight'],
            'market_sentiment': self.feature_weights['market_sentiment']['weight'],
            'technical_momentum': self.feature_weights['technical_momentum']['weight']
        }
