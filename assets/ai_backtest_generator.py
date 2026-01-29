# -*- coding: utf-8 -*-
"""
DeepQuant AI回测生成器 (AI Backtest Generator) - V5.0
核心升级：
1. 引入【相对收益】标签：在熊市中，跑赢大盘即为赢
2. 添加 V5.0 参数：bear_threshold, alpha_threshold
3. 优化候选股票筛选条件（参考代码建议）
4. 添加 generate_dataset 便捷方法
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
    """AI回测生成器类（V5.0 优化版）"""

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化回测生成器

        Args:
            data_dir: 数据目录
        """
        self.warehouse = DataWarehouse(data_dir=data_dir)
        self.extractor = FeatureExtractor()

        # === 基础参数 ===
        self.hold_days = 5  # 持有天数
        self.target_return = 3.0  # 目标收益率（%）
        self.stop_loss = -5.0  # 止损（%）

        # === V5.0 新增参数 ===
        self.bear_threshold = -2.0  # 熊市判定阈值（大盘跌幅 < -2%）
        self.alpha_threshold = 3.0  # 超额收益目标（%）
        self.amount_threshold = 1000  # 成交额阈值（千元，即 100万元）

        # === 筛选参数（参考代码建议）===
        self.vol_ratio_attack_min = 1.2  # 放量进攻最小量比
        self.vol_ratio_wash_max = 1.0  # 缩量洗盘最大量比

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
        V5.0 优化版：模拟真实的策略初筛（宽进）
        目的是选出【形态还可以】的股票，让 AI 进一步区分【真龙】还是【杂毛】

        筛选逻辑（参考代码优化）：
        1. 进攻形态：量比放大（>1.2），涨幅适中（>1%）
        2. 洗盘形态：缩量（<1.0），微跌或微涨（-3% ~ 3%）
        3. [V5.0 放宽] 当没有量比数据时，使用成交量作为替代

        Args:
            daily_df: 当日全市场数据

        Returns:
            候选股票代码列表
        """
        if daily_df.empty:
            return []

        # 基础过滤：非ST（通过ts_code过滤），有成交量
        mask = (
            (~daily_df['ts_code'].str.contains('ST|退', na=False)) &  # 过滤ST和退市股票
            (daily_df['amount'] > self.amount_threshold) &  # 成交额 > 阈值（千元）
            (daily_df['pct_chg'] > -9.8) &    # 非跌停
            (daily_df['pct_chg'] < 9.8)       # 非涨停
        )

        pool = daily_df[mask].copy()

        # 计算量比（如果没有的话）
        if 'vol_ratio' not in pool.columns or pool['vol_ratio'].isna().all():
            # 使用成交量作为替代（成交量 > 75%分位视为"放量"）
            vol_75 = pool['vol'].quantile(0.75)
            pool['vol_ratio_calc'] = pool['vol'] / vol_75
            vol_ratio_col = 'vol_ratio_calc'
        else:
            vol_ratio_col = 'vol_ratio'

        # === V5.0 优化筛选逻辑 ===

        # 场景 A: 进攻形态（放量，涨幅适中）
        cond_attack = (
            (pool[vol_ratio_col] > self.vol_ratio_attack_min) &  # 量比 > 1.2
            (pool['pct_chg'] > 1.0)  # 涨幅 > 1%
        )

        # 场景 B: 洗盘形态（缩量，微跌或微涨）
        cond_wash = (
            (pool[vol_ratio_col] < self.vol_ratio_wash_max) &  # 量比 < 1.0
            (pool['pct_chg'] > -3.0) &   # 涨幅 > -3%
            (pool['pct_chg'] < 3.0)      # 涨幅 < 3%
        )

        # 综合候选池
        candidates = pool[cond_attack | cond_wash]['ts_code'].tolist()

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
        # 需要至少 hold_days + 2 天的数据用于计算未来收益
        min_required_days = self.hold_days + 2

        if len(trade_days) <= min_required_days:
            print(f"\n[警告] 交易日数量不足：{len(trade_days)} 天（需要至少 {min_required_days + 1} 天才能生成样本）")
            return pd.DataFrame(), pd.Series()

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

        if len(X) > 0:
            print(f"  正样本（盈利）: {Y.sum()} ({Y.sum()/len(Y)*100:.1f}%)")
            print(f"  负样本（亏损）: {len(Y) - Y.sum()} ({(1-Y.sum()/len(Y))*100:.1f}%)")
            print(f"  胜率: {Y.sum()/len(Y)*100:.2f}%")
        else:
            print(f"  [警告] 没有生成任何样本，请检查数据或筛选条件")

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

    def generate_dataset(self, start_date: str, end_date: str,
                        max_samples: int = None) -> pd.DataFrame:
        """
        [V5.0 新增] 便捷方法：一步生成完整数据集（特征+标签）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_samples: 最大样本数量（None=不限制）

        Returns:
            完整的DataFrame（包含特征列 + label列 + ts_code + trade_date）
        """
        print(f"\n[{'='*80}]")
        print(f"[V5.0] 生成完整数据集")
        print(f"[{'='*80}]")
        print(f"  回测区间: {start_date} ~ {end_date}")
        print(f"  持有天数: {self.hold_days}")
        print(f"  目标收益: {self.target_return}%")
        print(f"  止损: {self.stop_loss}%")
        print(f"  熊市阈值: {self.bear_threshold}%")
        print(f"  超额收益目标: {self.alpha_threshold}%")

        # 生成数据
        X, Y = self.generate_training_data(start_date, end_date, max_samples)

        if len(X) == 0:
            return pd.DataFrame()

        # 合并特征和标签
        df = X.copy()
        df['label'] = Y.values

        print(f"\n[完成] 生成完整数据集")
        print(f"  总样本数: {len(df)}")
        print(f"  正样本: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.1f}%)")
        print(f"  负样本: {len(df) - df['label'].sum()} ({(1-df['label'].sum()/len(df))*100:.1f}%)")

        # 计算建议权重
        pos_count = df['label'].sum()
        neg_count = len(df) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"  建议 scale_pos_weight: {scale_pos_weight:.2f}")

        return df

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
