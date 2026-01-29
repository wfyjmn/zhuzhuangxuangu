# -*- coding: utf-8 -*-
"""
DeepQuant 数据仓库模块（Turbo 高性能版）
核心策略：全量预加载 + 内存切片，以空间换时间

优化点：
1. 预加载所有数据到内存，避免重复 IO
2. 使用复合索引 (ts_code, trade_date_dt) 实现极速查询
3. 内存压缩（float64 -> float32），减少内存占用
4. 完全在内存中进行数据切片，无磁盘 IO

适用场景：
- 需要频繁查询历史数据的场景
- 数据生成、回测等需要大量随机访问的场景
- 内存充足的服务器（建议 8GB+）
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 导入原版作为基类
try:
    from data_warehouse import DataWarehouse as OriginalDataWarehouse
except ImportError:
    # 简单的 mock，防止单独运行时报错
    class OriginalDataWarehouse:
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.pro = None
            self.trade_cal = []
            self.basic_info_cache = pd.DataFrame()

        def get_trade_days(self, s, e): return []
        def _load_trade_calendar(self): return []
        def _load_basic_info(self): return pd.DataFrame()

logger = logging.getLogger(__name__)


class DataWarehouseTurbo(OriginalDataWarehouse):
    """
    数据仓库（高性能内存版）
    核心策略：一次性将指定年份的所有数据加载到内存，后续查询纯内存操作
    """

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化 Turbo 数据仓库

        Args:
            data_dir: 数据存储目录
        """
        super().__init__(data_dir)

        # 全局大表：Index=[ts_code, trade_date_dt]
        self.memory_db: Optional[pd.DataFrame] = None
        # 记录当前内存中数据的覆盖范围
        self.loaded_start_date = None
        self.loaded_end_date = None

        logger.info("DataWarehouseTurbo 已初始化 - 准备加速")

    def preload_data(self, start_date: str, end_date: str, lookback_days: int = 120):
        """
        [关键优化] 预加载指定范围内的数据到内存

        Args:
            start_date: 任务开始日期（YYYYMMDD）
            end_date: 任务结束日期（YYYYMMDD）
            lookback_days: 预留的历史回溯缓冲期（如计算60日均线，需要提前加载至少60天）
        """
        # 计算实际需要加载的起始日期（往前推 lookback_days）
        # 使用交易日历更准确，但这里用简单方式估算
        dt_start = datetime.strptime(start_date, '%Y%m%d') - timedelta(days=lookback_days * 1.5)
        real_start_date = dt_start.strftime('%Y%m%d')

        logger.info("=" * 80)
        logger.info("【预加载】开始将数据加载到内存")
        logger.info("=" * 80)
        logger.info(f"  时间范围：{real_start_date} ~ {end_date}")
        logger.info(f"  回溯缓冲：{lookback_days} 天")

        # 1. 扫描目录下所有日线文件（按日期存储：data/daily/20230101.csv）
        data_path = Path(self.data_dir)
        all_files = sorted(list(data_path.glob("*.csv")))

        # 筛选需要加载的文件
        files_to_load = []
        for f in all_files:
            date_str = f.stem  # 文件名就是日期（20230101）
            if real_start_date <= date_str <= end_date:
                files_to_load.append(f)

        if not files_to_load:
            logger.warning(f"[警告] 未找到时间段 {real_start_date} ~ {end_date} 的数据文件")
            return

        logger.info(f"  文件数量：{len(files_to_load)} 个")
        logger.info(f"  预计内存占用：{len(files_to_load) * 0.5:.1f} MB（压缩后）")

        # 2. 批量读取（使用 list comprehension + concat）
        logger.info("\n  [读取] 正在读取数据文件...")
        dfs = []
        total_size = 0

        for i, f in enumerate(files_to_load):
            try:
                # 只读取必要的列，节省内存
                df = pd.read_csv(
                    f,
                    dtype={
                        'ts_code': 'str',
                        'trade_date': 'str',
                        'open': 'float32',
                        'high': 'float32',
                        'low': 'float32',
                        'close': 'float32',
                        'vol': 'float32',
                        'amount': 'float32',
                        'adj_factor': 'float32'
                    }
                )
                dfs.append(df)
                total_size += f.stat().st_size

                # 进度提示
                if (i + 1) % 20 == 0 or (i + 1) == len(files_to_load):
                    progress = (i + 1) / len(files_to_load) * 100
                    logger.info(f"    进度：{progress:.1f}% ({i+1}/{len(files_to_load)})")

            except Exception as e:
                logger.warning(f"  [警告] 文件读取失败 {f.name}: {e}")

        if not dfs:
            logger.error("[错误] 没有成功读取任何数据文件")
            return

        logger.info(f"\n  [合并] 正在合并数据...")

        # 3. 合并为一个巨大的 DataFrame
        self.memory_db = pd.concat(dfs, ignore_index=True)

        # 释放临时列表
        del dfs
        gc.collect()

        # 4. [内存优化] 压缩数据类型
        logger.info(f"  [压缩] 优化内存占用...")
        self._optimize_memory()

        # 5. [极速优化] 设置复合索引并排序
        logger.info(f"  [索引] 创建复合索引（ts_code, trade_date_dt）...")
        self.memory_db['trade_date_dt'] = pd.to_datetime(self.memory_db['trade_date'])
        self.memory_db.sort_values(['ts_code', 'trade_date_dt'], inplace=True)
        self.memory_db.set_index(['ts_code', 'trade_date_dt'], inplace=True)

        self.loaded_start_date = real_start_date
        self.loaded_end_date = end_date

        # 统计信息
        logger.info("\n" + "=" * 80)
        logger.info("【预加载完成】")
        logger.info("=" * 80)
        logger.info(f"  内存表行数：{len(self.memory_db):,}")
        logger.info(f"  覆盖日期：{real_start_date} ~ {end_date}")
        logger.info(f"  唯一股票：{self.memory_db.index.get_level_values(0).nunique():,} 只")
        logger.info(f"  内存占用：{self.memory_db.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # 强制垃圾回收
        gc.collect()

    def _optimize_memory(self):
        """
        内存优化：将 float64 转为 float32，int64 转为 int32
        可减少约 50% 内存占用
        """
        # 压缩浮点数
        for col in self.memory_db.select_dtypes(include=['float64']).columns:
            self.memory_db[col] = self.memory_db[col].astype('float32')

        # 压缩整数
        for col in self.memory_db.select_dtypes(include=['int64']).columns:
            self.memory_db[col] = self.memory_db[col].astype('int32')

    def get_stock_data(self, ts_code: str, end_date: str, days: int = 120) -> Optional[pd.DataFrame]:
        """
        [极速查询] 获取某只股票截止到 end_date 的历史数据

        Args:
            ts_code: 股票代码
            end_date: 结束日期（YYYYMMDD）
            days: 需要的数据天数

        Returns:
            股票DataFrame（索引为日期）
        """
        if self.memory_db is None:
            logger.warning("[警告] 内存数据库未初始化，请先调用 preload_data()")
            return None

        try:
            # 转换日期
            end_dt = pd.Timestamp(end_date)

            # 使用索引访问
            if ts_code not in self.memory_db.index:
                return None

            stock_data = self.memory_db.loc[ts_code]

            # 截取截止日期之前的数据
            slice_data = stock_data.loc[:end_dt]

            if len(slice_data) < days:
                return None  # 数据不足

            # 取最后 N 行
            result = slice_data.iloc[-days:].copy()
            # 添加 trade_date 列（便于后续处理）
            result['trade_date'] = result.index.strftime('%Y%m%d')

            return result

        except KeyError:
            return None
        except Exception as e:
            logger.error(f"查询出错 {ts_code} {end_date}: {e}")
            return None

    def get_future_data(self, ts_code: str, current_date: str, days: int = 5) -> Optional[pd.DataFrame]:
        """
        [极速查询] 获取未来数据（用于打标签）

        Args:
            ts_code: 股票代码
            current_date: 当前日期（YYYYMMDD）
            days: 需要的未来天数

        Returns:
            未来数据DataFrame（索引为日期）
        """
        if self.memory_db is None:
            return None

        try:
            curr_dt = pd.Timestamp(current_date)
            stock_data = self.memory_db.loc[ts_code]

            # 截取当前日期之后的数据（包含当前）
            future_slice = stock_data.loc[curr_dt:]

            # 排除掉当天（如果存在）
            if not future_slice.empty and future_slice.index[0] == curr_dt:
                future_slice = future_slice.iloc[1:]

            if len(future_slice) < days:
                return None  # 未来数据不足（比如到了最新的日期）

            result = future_slice.iloc[:days].copy()
            result['trade_date'] = result.index.strftime('%Y%m%d')

            return result

        except Exception:
            return None

    def load_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """
        [极速查询] 获取当日全市场数据

        Args:
            date: 日期（YYYYMMDD）

        Returns:
            当日全市场数据DataFrame
        """
        if self.memory_db is None:
            return None

        try:
            dt = pd.Timestamp(date)

            # 使用 xs 切片获取某一天的所有股票数据
            daily_data = self.memory_db.xs(dt, level=1)

            # 重置索引（方便后续处理）
            result = daily_data.reset_index()
            result['trade_date'] = date

            return result

        except KeyError:
            return None
        except Exception as e:
            logger.error(f"获取日数据失败 {date}: {e}")
            return None

    def is_loaded(self, start_date: str = None, end_date: str = None) -> bool:
        """
        检查数据是否已加载，以及是否覆盖指定范围

        Args:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            是否已加载并覆盖指定范围
        """
        if self.memory_db is None:
            return False

        if start_date and start_date < self.loaded_start_date:
            return False

        if end_date and end_date > self.loaded_end_date:
            return False

        return True

    def clear_memory(self):
        """
        清除内存中的数据，释放内存
        """
        self.memory_db = None
        self.loaded_start_date = None
        self.loaded_end_date = None
        gc.collect()
        logger.info("[清理] 内存数据已清除")


# 导出 DataWarehouseTurbo 为 DataWarehouse，方便使用
DataWarehouse = DataWarehouseTurbo


# 测试代码
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("DataWarehouseTurbo 性能测试")
    print("=" * 80)

    # 初始化
    dw = DataWarehouseTurbo(data_dir="data/daily")

    # 预加载 2024 年 1 月的数据
    print("\n[1] 预加载数据...")
    dw.preload_data(start_date='20240101', end_date='20240131', lookback_days=120)

    # 测试查询
    print("\n[2] 测试查询...")
    ts_code = '600519.SH'  # 贵州茅台
    end_date = '20240115'

    import time

    # 查询历史数据
    start = time.time()
    hist_data = dw.get_stock_data(ts_code, end_date, days=60)
    elapsed = time.time() - start

    if hist_data is not None:
        print(f"  查询历史数据：{elapsed*1000:.2f} ms")
        print(f"  数据形状：{hist_data.shape}")

    # 查询未来数据
    start = time.time()
    future_data = dw.get_future_data(ts_code, end_date, days=5)
    elapsed = time.time() - start

    if future_data is not None:
        print(f"  查询未来数据：{elapsed*1000:.2f} ms")
        print(f"  数据形状：{future_data.shape}")

    # 查询日数据
    start = time.time()
    daily_data = dw.load_daily_data('20240115')
    elapsed = time.time() - start

    if daily_data is not None:
        print(f"  查询日数据：{elapsed*1000:.2f} ms")
        print(f"  数据形状：{daily_data.shape}")

    print("\n✅ 测试完成")
