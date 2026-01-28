# -*- coding: utf-8 -*-
"""
DeepQuant AI回测生成器 (AI Backtest Generator)
功能：
1. 事件驱动回测
2. 生成训练数据（特征X + 标签Y）
3. 严格避免未来函数和幸存者偏差
4. 支持多种策略模拟

核心原则：
- 避免未来函数：只能使用T时刻及之前的数据
- 避免幸存者偏差：使用当时在市的股票列表
- 事件驱动：在买入点提取特征，在卖出点计算标签
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from data_warehouse import DataWarehouse
from feature_extractor import FeatureExtractor

class AIBacktestGenerator:
    """AI回测生成器类"""

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化回测生成器

        Args:
            data_dir: 数据目录
        """
        self.warehouse = DataWarehouse(data_dir=data_dir)
        self.extractor = FeatureExtractor()

        # 配置参数
        self.hold_days = 5  # 持有天数
        self.target_return = 3.0  # 目标收益率（%）
        self.stop_loss = -5.0  # 止损（%）

    def select_stocks(self, date: str) -> List[str]:
        """
        模拟选股逻辑（简化版）

        Args:
            date: 选股日期

        Returns:
            选中的股票代码列表
        """
        df = self.warehouse.load_daily_data(date)
        if df is None or len(df) == 0:
            return []

        # 简单筛选条件：
        # 1. 涨跌幅 > 5%（活跃）
        # 2. 成交额 > 1亿（流动性）
        # 3. 换手率 > 2%（活跃度）
        selected = df[
            (df['pct_chg'] > 5) &
            (df['amount'] > 100000000) &
            (df['vol_ratio'] > 2)
        ]

        return selected['ts_code'].tolist()

    def calculate_label(self, df: pd.DataFrame, buy_date: str, hold_days: int = 5,
                       target_return: float = 3.0, stop_loss: float = -5.0) -> int:
        """
        计算标签（5天后是否盈利）

        Args:
            df: 股票行情数据
            buy_date: 买入日期
            hold_days: 持有天数
            target_return: 目标收益率（%）
            stop_loss: 止损（%）

        Returns:
            标签（1=盈利，0=亏损）
        """
        buy_idx = df[df['trade_date'] == buy_date].index

        if len(buy_idx) == 0:
            return 0

        buy_idx = buy_idx[0]
        buy_price = df.loc[buy_idx, 'close']

        # 检查未来数据
        for i in range(1, hold_days + 1):
            if buy_idx + i >= len(df):
                break

            future_price = df.loc[buy_idx + i, 'close']
            pct_return = (future_price - buy_price) / buy_price * 100

            # 止损检查
            if pct_return <= stop_loss:
                return 0

            # 目标收益检查
            if pct_return >= target_return:
                return 1

        # 持有到期后计算最终收益
        if buy_idx + hold_days < len(df):
            final_price = df.loc[buy_idx + hold_days, 'close']
            final_return = (final_price - buy_price) / buy_price * 100
            return 1 if final_return > 0 else 0

        return 0

    def generate_training_data(self, start_date: str, end_date: str,
                             max_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成训练数据（特征X + 标签Y）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_samples: 最大样本数量（None=不限制）

        Returns:
            (X, Y) 特征DataFrame和标签Series
        """
        print(f"\n[回测生成器] 开始生成训练数据")
        print(f"  回测区间: {start_date} ~ {end_date}")
        print(f"  持有天数: {self.hold_days}")
        print(f"  目标收益: {self.target_return}%")
        print(f"  止损: {self.stop_loss}%")

        trade_days = self.warehouse.get_trade_days(start_date, end_date)

        # 排除最后5天（无法计算未来收益）
        trade_days = trade_days[:-self.hold_days]

        features_list = []
        labels_list = []

        for i, date in enumerate(trade_days, 1):
            print(f"\n  [进度] {i}/{len(trade_days)} - {date}")

            # 1. 选股
            selected_stocks = self.select_stocks(date)
            print(f"    选股: {len(selected_stocks)} 只")

            if len(selected_stocks) == 0:
                continue

            # 2. 对每只股票提取特征并计算标签
            for ts_code in selected_stocks:
                try:
                    # 获取股票历史数据（用于提取特征）
                    df = self.warehouse.get_stock_data(ts_code, date, days=30)
                    if df is None or len(df) < 20:
                        continue

                    # 提取特征
                    features = self.extractor.extract_features(df)
                    features['ts_code'] = ts_code
                    features['trade_date'] = date

                    # 计算标签（使用未来的数据）
                    # 注意：这里需要获取包含未来的数据
                    df_future = self.warehouse.get_stock_data(ts_code, date, days=30 + self.hold_days)
                    if df_future is None or len(df_future) < 20 + self.hold_days:
                        continue

                    label = self.calculate_label(
                        df_future,
                        date,
                        self.hold_days,
                        self.target_return,
                        self.stop_loss
                    )

                    features_list.append(features)
                    labels_list.append(label)

                except Exception as e:
                    continue

            # 限制样本数量
            if max_samples and len(features_list) >= max_samples:
                print(f"    [达到最大样本数] {max_samples}")
                break

            # 进度提示
            if i % 10 == 0:
                print(f"    [累计样本] {len(features_list)}")

        # 转换为DataFrame
        X = pd.DataFrame(features_list)
        Y = pd.Series(labels_list, name='label')

        print(f"\n[完成] 生成训练数据")
        print(f"  总样本数: {len(X)}")
        print(f"  正样本（盈利）: {Y.sum()} ({Y.sum()/len(Y)*100:.1f}%)")
        print(f"  负样本（亏损）: {len(Y) - Y.sum()} ({(1-Y.sum()/len(Y))*100:.1f}%)")

        return X, Y

    def generate_validation_data(self, start_date: str, end_date: str,
                                max_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成验证数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_samples: 最大样本数量

        Returns:
            (X, Y) 特征DataFrame和标签Series
        """
        return self.generate_training_data(start_date, end_date, max_samples)

    def save_training_data(self, X: pd.DataFrame, Y: pd.Series, output_dir: str = "data/training"):
        """
        保存训练数据

        Args:
            X: 特征DataFrame
            Y: 标签Series
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存特征
        X_file = os.path.join(output_dir, f"features_{timestamp}.csv")
        X.to_csv(X_file, index=False)
        print(f"\n[保存] 特征数据: {X_file}")

        # 保存标签
        Y_file = os.path.join(output_dir, f"labels_{timestamp}.csv")
        Y.to_csv(Y_file, index=False, header=['label'])
        print(f"[保存] 标签数据: {Y_file}")

        # 保存统计信息
        stats_file = os.path.join(output_dir, f"stats_{timestamp}.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"训练数据统计信息\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"总样本数: {len(X)}\n")
            f.write(f"正样本（盈利）: {Y.sum()} ({Y.sum()/len(Y)*100:.2f}%)\n")
            f.write(f"负样本（亏损）: {len(Y) - Y.sum()} ({(1-Y.sum()/len(Y))*100:.2f}%)\n\n")
            f.write(f"特征列表:\n")
            for col in X.columns:
                if col not in ['ts_code', 'trade_date']:
                    f.write(f"  - {col}\n")
        print(f"[保存] 统计信息: {stats_file}")

    def load_training_data(self, features_file: str, labels_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载训练数据

        Args:
            features_file: 特征文件路径
            labels_file: 标签文件路径

        Returns:
            (X, Y) 特征DataFrame和标签Series
        """
        X = pd.read_csv(features_file)
        Y = pd.read_csv(labels_file, header=None, names=['label'])

        print(f"\n[加载] 训练数据")
        print(f"  特征文件: {features_file}")
        print(f"  标签文件: {labels_file}")
        print(f"  样本数: {len(X)}")

        return X, Y


def main():
    """测试函数"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant AI回测生成器")
    print(" " * 30 + "测试运行")
    print("="*80 + "\n")

    # 初始化回测生成器
    generator = AIBacktestGenerator()

    # 测试：生成少量训练数据
    print("[测试] 生成训练数据（2025年1月）")

    # 为了测试，使用最近的数据
    X, Y = generator.generate_training_data(
        start_date='20250101',
        end_date='20250120',
        max_samples=100  # 限制样本数，快速测试
    )

    if len(X) > 0:
        print(f"\n  特征维度: {X.shape}")
        print(f"  标签分布: 正{Y.sum()}/{len(Y)}负")

        # 保存训练数据
        generator.save_training_data(X, Y)

        print("\n  特征数据预览:")
        print(X.head(3))

        print("\n  标签数据预览:")
        print(Y.head(10))

    print("\n[完成] 回测生成器测试完成\n")


if __name__ == "__main__":
    main()
