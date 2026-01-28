# -*- coding: utf-8 -*-
"""
DeepQuant AI回测生成器 (AI Backtest Generator) - 性能优化版
修复：
1. 修正未来数据获取逻辑，确保标签计算正确（避免数据穿越）
2. 优化数据读取，减少 IO 开销
3. 集成真实的选股策略逻辑，保证样本分布一致性
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from data_warehouse import DataWarehouse
from feature_extractor import FeatureExtractor


class AIBacktestGenerator:
    """AI回测生成器类（优化版）"""

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

    def _get_future_data(self, ts_code: str, start_date: str, days: int) -> Optional[pd.DataFrame]:
        """
        [关键修复1] 获取指定日期之后的未来数据（用于计算标签）

        Args:
            ts_code: 股票代码
            start_date: 起始日期（格式：YYYYMMDD）
            days: 需要的天数

        Returns:
            未来数据的DataFrame
        """
        # 计算大致的结束日期（多取几天以防停牌）
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = start_dt + timedelta(days=days * 2 + 10)
        end_date_str = end_dt.strftime('%Y%m%d')

        # 获取包含起始日期在内的所有数据
        full_df = self.warehouse.get_stock_data(ts_code, end_date_str, days=days + 30)

        if full_df is None or full_df.empty:
            return None

        # [关键修复1] 确保 trade_date 为字符串类型
        full_df['trade_date'] = full_df['trade_date'].astype(str)

        # [关键修复2] 截取 start_date 之后的数据（不包含当天）
        future_df = full_df[full_df['trade_date'] > start_date].sort_values('trade_date')

        # 只取前 N 天
        return future_df.head(days)
    def select_candidates_robust(self, daily_df: pd.DataFrame) -> List[str]:
        """
        [关键修复3] 模拟真实的策略初筛（宽进）
        目的是选出【形态还可以】的股票，让 AI 进一步区分【真龙】还是【杂毛】

        Args:
            daily_df: 当日全市场数据

        Returns:
            候选股票代码列表
        """
        if daily_df.empty:
            return []

        # 计算量比（如果没有的话）
        if 'vol_ratio' not in daily_df.columns:
            # 量比 = 当日成交量 / 前5日平均成交量
            # 由于这里只有当天的数据，暂时设置默认值
            daily_df['vol_ratio'] = 1.0

        # 基础过滤：非ST（通过ts_code过滤），有成交量
        mask = (
            (~daily_df['ts_code'].str.contains('ST|退', na=False)) &  # 过滤ST和退市股票
            (daily_df['amount'] > 1000) &  # 成交额 > 1000万（单位：万元）
            (daily_df['pct_chg'] > -9.8) &    # 非跌停
            (daily_df['pct_chg'] < 9.8)       # 非涨停
        )

        pool = daily_df[mask].copy()

        # [关键修复] 策略逻辑复刻（简化版）
        # 场景A: 放量进攻（涨幅>1%）
        cond_attack = (pool['pct_chg'] > 1.0)

        # 场景B: 缩量洗盘（涨幅 -1% ~ 2%）
        cond_wash = (pool['pct_chg'] > -1.0) & (pool['pct_chg'] < 2.0)

        # 场景C: 梯量上涨（涨幅0-2%）
        cond_ramp = (pool['pct_chg'] > 0) & (pool['pct_chg'] < 2.0)

        # 综合候选池
        candidates = pool[cond_attack | cond_wash | cond_ramp]['ts_code'].tolist()

        return candidates

        # 综合候选池
        candidates = pool[cond_attack | cond_wash | cond_ramp]['ts_code'].tolist()

        return candidates

    def calculate_label(self, future_df: pd.DataFrame, buy_price: float,
                       index_start_price: float = None, index_future_df: pd.DataFrame = None) -> int:
        """
        [关键修复 V5.0] 计算标签（1=盈利，0=亏损）
        使用"相对收益"逻辑，避免 AI 在熊市变成"死空头"

        标签逻辑升级：
        1. 牛市/震荡市：绝对收益 > 3%
        2. 熊市（大盘跌幅 > 2%）：超额收益 > 5%（即便个股跌了，但比大盘少跌很多，也是强势股）

        Args:
            future_df: 未来数据
            buy_price: 买入价格
            index_start_price: 大盘买入时价格（用于计算超额收益）
            index_future_df: 大盘未来数据

        Returns:
            标签（1=盈利，0=亏损）
        """
        if future_df is None or len(future_df) == 0:
            return 0

        price_col = 'close_qfq' if 'close_qfq' in future_df.columns else 'close'

        # 判断市场环境（是否为熊市）
        is_bear_market = False
        index_excess_return = 0

        # 计算大盘收益（判断是否为熊市）
        if index_start_price is not None and index_future_df is not None and len(index_future_df) > 0:
            index_col = 'close_qfq' if 'close_qfq' in index_future_df.columns else 'close'
            index_end_price = index_future_df.iloc[-1][index_col]
            index_return = (index_end_price - index_start_price) / index_start_price * 100

            # 如果大盘跌幅 > 2%，定义为熊市
            if index_return < -2.0:
                is_bear_market = True

        # [动态止盈止损]
        for i, row in future_df.iterrows():
            price = row[price_col]
            pct_return = (price - buy_price) / buy_price * 100

            # 同步计算大盘收益（用于超额收益判断）
            if is_bear_market and index_future_df is not None and len(index_future_df) > 0:
                index_col = 'close_qfq' if 'close_qfq' in index_future_df.columns else 'close'
                idx_price = index_future_df.loc[row.name, index_col] if row.name in index_future_df.index else index_start_price
                index_excess_return = pct_return - ((idx_price - index_start_price) / index_start_price * 100)

            # 止损检查
            if pct_return <= self.stop_loss:
                return 0

            # [熊市] 使用超额收益止盈（跑赢大盘 5% 就算赢）
            if is_bear_market and index_excess_return >= 5.0:
                return 1

            # [牛市/震荡市] 使用绝对收益止盈（3%）
            if not is_bear_market and pct_return >= self.target_return:
                return 1

        # 持有到期后计算最终收益
        final_price = future_df.iloc[-1][price_col]
        final_return = (final_price - buy_price) / buy_price * 100

        # 熊市：超额收益 > 3%
        if is_bear_market:
            index_final = index_future_df.iloc[-1]['close_qfq' if 'close_qfq' in index_future_df.columns else 'close']
            index_final_return = (index_final - index_start_price) / index_start_price * 100
            excess_return = final_return - index_final_return
            return 1 if excess_return > 3.0 else 0

        # 牛市/震荡市：绝对收益 > 0
        return 1 if final_return > 0 else 0

    def generate_training_data(self, start_date: str, end_date: str,
                             max_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        [关键修复2] 生成训练数据（特征X + 标签Y）
        优化数据读取逻辑，减少 IO 开销

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

        # [关键修复1] 排除最后N天（无法计算未来收益）
        valid_days = trade_days[:-(self.hold_days + 2)]

        features_list = []
        labels_list = []

        # [性能优化] 统计信息
        total_days = len(valid_days)
        processed_days = 0
        success_samples = 0

        print(f"\n  [信息] 有效交易日: {total_days} 天")

        for i, date in enumerate(valid_days, 1):
            processed_days += 1

            # 进度提示（每10天显示一次）
            if i % 10 == 0:
                print(f"  [进度] {i}/{total_days} ({i/total_days*100:.1f}%) | 样本: {success_samples}", end="\r")

            # 1. 获取当日全市场数据（内存中操作，快）
            daily_df = self.warehouse.load_daily_data(date)
            if daily_df is None or len(daily_df) == 0:
                continue

            # 2. [关键修复3] 使用真实的策略逻辑筛选候选股
            candidates = self.select_candidates_robust(daily_df)
            if not candidates:
                continue

            # [新增] 获取大盘未来数据（用于相对收益计算）
            index_future_df = self._get_future_data('000001.SH', date, self.hold_days)
            index_df = self.warehouse.get_stock_data('000001.SH', date, days=5)
            index_start_price = index_df.iloc[-1]['close_qfq' if 'close_qfq' in index_df.columns else 'close'] if index_df is not None and len(index_df) > 0 else None

            # 3. 逐个提取特征 + 计算标签
            for ts_code in candidates:
                try:
                    # A. 提取特征（需要 T 及 T 之前的历史数据）
                    hist_df = self.warehouse.get_stock_data(ts_code, date, days=60)
                    if hist_df is None or len(hist_df) < 30:
                        continue

                    # 计算当前买入价（使用复权价格）
                    buy_price_col = 'close_qfq' if 'close_qfq' in hist_df.columns else 'close'
                    buy_price = hist_df.iloc[-1][buy_price_col]

                    # 提取特征 X
                    features = self.extractor.extract_features(hist_df)
                    features['ts_code'] = ts_code
                    features['trade_date'] = date

                    # B. [关键修复 V5.0] 计算标签 Y（使用相对收益，避免死空头）
                    future_df = self._get_future_data(ts_code, date, self.hold_days)
                    label = self.calculate_label(
                        future_df,
                        buy_price,
                        index_start_price=index_start_price,
                        index_future_df=index_future_df
                    )

                    features_list.append(features)
                    labels_list.append(label)
                    success_samples += 1

                except Exception as e:
                    # 静默处理单个股票的错误，避免中断整个流程
                    continue

            # 限制样本数量
            if max_samples and len(features_list) >= max_samples:
                print(f"\n  [达到最大样本数] {max_samples}")
                break

        # 转换为DataFrame
        X = pd.DataFrame(features_list)
        Y = pd.Series(labels_list, name='label')

        print(f"\n\n[完成] 生成训练数据")
        print(f"  处理交易日: {processed_days}/{total_days}")
        print(f"  总样本数: {len(X)}")
        print(f"  正样本（盈利）: {Y.sum()} ({Y.sum()/len(Y)*100:.1f}%)")
        print(f"  负样本（亏损）: {len(Y) - Y.sum()} ({(1-Y.sum()/len(Y))*100:.1f}%)")
        print(f"  胜率: {Y.sum()/len(Y)*100:.2f}%")

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
    print(" " * 20 + "DeepQuant AI回测生成器（优化版）")
    print(" " * 30 + "测试运行")
    print("="*80 + "\n")

    # 初始化回测生成器
    generator = AIBacktestGenerator()

    # 测试：生成少量训练数据（1个月）
    print("[测试] 生成训练数据（2023年1月，最多100个样本）")

    X, Y = generator.generate_training_data(
        start_date='20230101',
        end_date='20230131',
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

    else:
        print("\n  [提示] 未生成样本，可能原因：")
        print("    1. 数据未下载（请先运行 data_warehouse.py 下载数据）")
        print("    2. 选股条件过于严格（没有符合条件的股票）")

    print("\n[完成] 回测生成器测试完成\n")


if __name__ == "__main__":
    main()
