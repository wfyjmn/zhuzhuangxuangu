# -*- coding: utf-8 -*-
"""
DeepQuant 数据仓库模块 (Data Warehouse)
功能：
1. 下载历史行情数据到本地
2. 管理本地数据（按日期存储）
3. 提供数据查询接口
4. 防止幸存者偏差（获取当时在市的股票列表）

设计原则：
- 数据本地化（避免频繁API调用）
- 按日期分层存储
- 支持增量更新
"""

import tushare as ts
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

class DataWarehouse:
    """数据仓库类（优化版）"""

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化数据仓库

        Args:
            data_dir: 数据存储目录
        """
        from dotenv import load_dotenv
        load_dotenv()

        tushare_token = os.getenv("TUSHARE_TOKEN")
        if not tushare_token:
            raise ValueError("请配置 TUSHASH_TOKEN 环境变量")

        ts.set_token(tushare_token)
        self.pro = ts.pro_api(timeout=30)

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)  # 确保数据根目录存在

        # 缓存交易日历
        self.trade_cal = self._load_trade_calendar()

        # [优化3] 缓存股票基础信息，避免重复调用 API
        self.basic_info_cache = self._load_basic_info()

        # [新增] 缓存历史数据，避免重复加载
        self._history_data_cache = {}
        self._history_cache_key = None

    def _load_trade_calendar(self) -> List[str]:
        """加载交易日历"""
        cache_file = "data/trade_calendar.csv"
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            # 确保日期为字符串类型
            return df['cal_date'].astype(str).tolist()

        # 从API获取
        df = self.pro.trade_cal(exchange='SSE', start_date='20200101', end_date='20251231')
        df = df[df['is_open'] == 1]
        df.to_csv(cache_file, index=False)
        return df['cal_date'].astype(str).tolist()

    def _load_basic_info(self) -> pd.DataFrame:
        """
        [优化3] 加载或更新股票基础信息缓存
        避免每次下载行情都调用 stock_basic API
        """
        cache_file = "data/stock_basic_cache.csv"

        # 每天只更新一次基础信息
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_time.date() == datetime.now().date():
                return pd.read_csv(cache_file)

        print("[数据仓库] 更新股票基础列表缓存...")
        try:
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name,list_date,delist_date'
            )
            df.to_csv(cache_file, index=False)
            print(f"  [完成] 缓存 {len(df)} 只股票的基础信息")
            return df
        except Exception as e:
            print(f"  [警告] 更新基础信息失败: {e}")
            # 如果失败且有旧缓存，使用旧缓存
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            return pd.DataFrame()

    def get_trade_days(self, start_date: str, end_date: str) -> List[str]:
        """获取指定时间段的交易日列表"""
        cal = [d for d in self.trade_cal if start_date <= d <= end_date]
        return sorted(cal)

    def download_daily_data(self, date: str, force: bool = False) -> Optional[pd.DataFrame]:
        """
        [优化2] 下载指定日期的全市场行情数据（包含复权因子和每日指标）

        [手术二] 注入灵魂：补全缺失的特征（turnover_rate, pe_ttm 等）

        Args:
            date: 日期（格式：YYYYMMDD）
            force: 是否强制重新下载

        Returns:
            行情DataFrame（包含 adj_factor, turnover_rate, pe_ttm 等列）
        """
        filename = os.path.join(self.data_dir, f"{date}.csv")

        # 检查是否已存在
        if os.path.exists(filename) and not force:
            df = pd.read_csv(filename)
            # 检查是否包含必需的特征（turnover_rate, pe_ttm）
            if 'turnover_rate' in df.columns and 'pe_ttm' in df.columns:
                return df
            else:
                print(f"  [重新下载] {date} 的数据缺少关键特征（turnover_rate, pe_ttm）")

        try:
            print(f"[数据仓库] 下载 {date} 的行情数据...")
            time.sleep(0.3)  # 避免触发限流

            # 1. 获取日线行情 (OHLC, vol, amount)
            df_daily = self.pro.daily(trade_date=date)

            if df_daily.empty:
                print(f"  [警告] {date} 没有行情数据")
                return None

            # 2. [手术二] 获取每日指标（turnover_rate, pe_ttm, pb, total_share 等）
            df_basic = self.pro.daily_basic(
                trade_date=date,
                fields='ts_code,turnover_rate,turnover_rate_f,pe_ttm,pb,total_mv,circ_mv'
            )

            # 3. 获取复权因子
            df_adj = self.pro.adj_factor(trade_date=date)

            # --- 关键合并步骤 ---
            # 合并每日指标
            if not df_basic.empty:
                df_daily = pd.merge(df_daily, df_basic, on='ts_code', how='left')
                # 填充缺失值
                df_daily['turnover_rate'] = df_daily['turnover_rate'].fillna(0)
                df_daily['pe_ttm'] = df_daily['pe_ttm'].fillna(0)
                df_daily['pb'] = df_daily['pb'].fillna(0)
                df_daily['total_mv'] = df_daily['total_mv'].fillna(0)
                df_daily['circ_mv'] = df_daily['circ_mv'].fillna(0)

            # 合并复权因子
            if not df_adj.empty:
                df_daily = pd.merge(df_daily, df_adj[['ts_code', 'adj_factor']], on='ts_code', how='left')
                df_daily['adj_factor'] = df_daily['adj_factor'].fillna(1.0)

            # 4. [优化3] 使用缓存的基础信息进行过滤（防止幸存者偏差）
            if not self.basic_info_cache.empty:
                # 过滤：上市日期 <= 当前日期（确保类型一致）
                list_date_str = self.basic_info_cache['list_date'].astype(str)
                valid_codes = self.basic_info_cache[list_date_str <= date]['ts_code']
                df_daily = df_daily[df_daily['ts_code'].isin(valid_codes)]

            # 保存到本地
            df_daily.to_csv(filename, index=False)
            print(f"  [完成] {date} 保存 {len(df_daily)} 只股票的数据（含复权因子、换手率、市盈率）")

            return df_daily

        except Exception as e:
            print(f"  [错误] 下载 {date} 失败: {e}")
            return None

    def download_range_data(self, start_date: str, end_date: str):
        """
        下载指定时间范围的数据

        Args:
            start_date: 开始日期（格式：YYYYMMDD）
            end_date: 结束日期（格式：YYYYMMDD）
        """
        trade_days = self.get_trade_days(start_date, end_date)

        print(f"\n[数据仓库] 开始下载 {start_date} 到 {end_date} 的数据")
        print(f"  交易日数量: {len(trade_days)}")
        print(f"  存储目录: {self.data_dir}\n")

        success_count = 0
        for i, date in enumerate(trade_days, 1):
            df = self.download_daily_data(date)
            if df is not None:
                success_count += 1

            # 进度提示
            if i % 20 == 0:
                print(f"  [进度] {i}/{len(trade_days)} ({i/len(trade_days)*100:.1f}%)")

        print(f"\n[数据仓库] 下载完成！成功: {success_count}/{len(trade_days)}")

    def load_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """
        从本地加载指定日期的行情数据

        Args:
            date: 日期（格式：YYYYMMDD）

        Returns:
            行情DataFrame
        """
        filename = os.path.join(self.data_dir, f"{date}.csv")

        if not os.path.exists(filename):
            print(f"[警告] {date} 的数据不存在，尝试下载...")
            return self.download_daily_data(date)

        return pd.read_csv(filename)

    def load_history_data(self, end_date: str, days: int = 120) -> Dict[str, pd.DataFrame]:
        """
        [优化1] 极速加载历史数据（向量化处理）

        Args:
            end_date: 结束日期
            days: 回溯天数

        Returns:
            {股票代码: DataFrame} 的字典（包含复权价格列）
        """
        # 1. 获取最近 N 个交易日的文件列表
        trade_days = self.get_trade_days("20000101", end_date)[-days:]

        all_dfs = []
        for date in trade_days:
            filepath = os.path.join(self.data_dir, f"{date}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                # 确保有日期列，防止合并后乱序
                if 'trade_date' not in df.columns:
                    df['trade_date'] = date
                all_dfs.append(df)

        if not all_dfs:
            return {}

        # 2. [关键优化] 一次性合并，而不是循环 append
        # 这一步比原来的 iterrows 快 100 倍以上
        big_df = pd.concat(all_dfs, ignore_index=True)

        # 3. [关键修复] 计算前复权价格 (Pre-Adjusted Price)
        # 复权价 = 现价 * 复权因子
        # 这样可以保证所有历史数据的比例关系是正确的
        # 在计算 MA20, MA60, MACD 等指标时，必须使用复权价格
        if 'adj_factor' in big_df.columns:
            big_df['close_qfq'] = big_df['close'] * big_df['adj_factor']
            big_df['high_qfq']  = big_df['high']  * big_df['adj_factor']
            big_df['low_qfq']   = big_df['low']   * big_df['adj_factor']
            big_df['open_qfq']  = big_df['open']  * big_df['adj_factor']
        else:
            # 如果没有复权因子，给出警告并使用原始价格
            print("  [警告] 数据缺少复权因子，将使用原始价格（可能导致回测失真）")
            big_df['close_qfq'] = big_df['close']
            big_df['high_qfq']  = big_df['high']
            big_df['low_qfq']   = big_df['low']
            big_df['open_qfq']  = big_df['open']

        # 4. [关键优化] 使用 groupby 拆分为字典
        # 利用 Pandas 底层的 C 语言优化，能够瞬间处理几十万行数据
        history_data = {
            code: data.sort_values('trade_date').reset_index(drop=True)
            for code, data in big_df.groupby('ts_code')
        }

        return history_data

    def get_stock_data(self, ts_code: str, end_date: str, days: int = 120) -> Optional[pd.DataFrame]:
        """
        获取单只股票的历史数据

        Args:
            ts_code: 股票代码
            end_date: 结束日期
            days: 回溯天数

        Returns:
            股票DataFrame
        """
        history = self.load_history_data(end_date, days)
        return history.get(ts_code)

    def clear_data(self, year: str):
        """
        清理指定年份的数据

        Args:
            year: 年份（如 "2023"）
        """
        import shutil

        pattern = f"{year}*"
        count = 0

        for filename in os.listdir(self.data_dir):
            if filename.startswith(pattern):
                filepath = os.path.join(self.data_dir, filename)
                os.remove(filepath)
                count += 1

        print(f"[数据仓库] 已清理 {year} 年的 {count} 个文件")


def main():
    """测试函数"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant 数据仓库")
    print(" " * 30 + "测试运行")
    print("="*80 + "\n")

    # 初始化数据仓库
    warehouse = DataWarehouse()

    # 测试1：下载一天的数据
    print("[测试1] 下载一天的数据")
    df = warehouse.download_daily_data("20250120", force=True)
    if df is not None:
        print(f"  成功下载 {len(df)} 只股票的数据")
        print(f"  示例数据:\n{df.head(3)}")

    # 测试2：获取交易日历
    print("\n[测试2] 获取交易日历")
    trade_days = warehouse.get_trade_days("20250101", "20250131")
    print(f"  1月交易日数量: {len(trade_days)}")
    print(f"  前5个交易日: {trade_days[:5]}")

    # 测试3：加载历史数据
    print("\n[测试3] 加载历史数据")
    df = warehouse.load_daily_data("20250120")
    if df is not None:
        print(f"  成功加载 {len(df)} 只股票的数据")

    print("\n[完成] 数据仓库测试完成\n")


if __name__ == "__main__":
    main()
