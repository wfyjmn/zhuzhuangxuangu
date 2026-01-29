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
        def get_trade_days(self, s, e): return []

logger = logging.getLogger(__name__)


class DataWarehouseTurbo(OriginalDataWarehouse):
    """
    数据仓库（高性能内存版）
    核心策略：一次性将指定年份的所有数据加载到内存，后续查询纯内存操作
    """

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化 Turbo 数据仓库
        """
        super().__init__(data_dir)

        # 全局大表：Index=[ts_code, trade_date_dt]
        self.memory_db: Optional[pd.DataFrame] = None
        # 记录当前内存中数据的覆盖范围
        self.loaded_start_date = None
        self.loaded_end_date = None
        
        # 缓存交易日历
        self._cached_trade_days = []

        logger.info("DataWarehouseTurbo 已初始化 - 准备加速")

    def preload_data(self, start_date: str, end_date: str, lookback_days: int = 120):
        """
        [关键优化] 预加载指定范围内的数据到内存
        """
        # 计算实际需要加载的起始日期（往前推 lookback_days）
        dt_start = datetime.strptime(start_date, '%Y%m%d') - timedelta(days=lookback_days * 1.5)
        real_start_date = dt_start.strftime('%Y%m%d')

        logger.info("=" * 80)
        logger.info("【Turbo 预加载】开始加载数据到内存")
        logger.info("=" * 80)
        logger.info(f"  请求范围：{start_date} ~ {end_date}")
        logger.info(f"  实际加载：{real_start_date} ~ {end_date} (含 {lookback_days} 天回溯)")

        # 1. 扫描目录下所有日线文件
        data_path = Path(self.data_dir)
        all_files = sorted(list(data_path.glob("*.csv")))
        
        if not all_files:
            logger.error(f"[错误] 目录 {self.data_dir} 下未找到任何 CSV 文件")
            return

        # 2. 智能筛选文件
        # 假设文件名是日期格式 (20230101.csv)
        files_to_load = []
        is_date_file = False
        
        # 简单检测文件名格式
        first_file = all_files[0].stem
        if first_file.isdigit() and len(first_file) == 8:
            is_date_file = True
            logger.info("[识别] 检测到文件名格式为 'YYYYMMDD.csv' (按日期存储)")
            for f in all_files:
                date_str = f.stem
                if real_start_date <= date_str <= end_date:
                    files_to_load.append(f)
        else:
            # 可能是按股票代码存储 (000001.SZ.csv)，这种情况下需要读取所有文件并内部过滤日期
            # 注意：这会非常慢且消耗内存，通常 Turbo 模式建议数据按日期存储
            logger.warning("[警告] 文件名似乎不是日期格式，假定为按股票存储。")
            logger.warning("[警告] 这将读取所有文件并在内存中过滤，速度较慢。")
            files_to_load = all_files # 读取所有，后面再 filter

        if not files_to_load:
            logger.warning(f"[警告] 未找到时间段内的任何数据文件")
            return

        logger.info(f"  待读取文件数：{len(files_to_load)} 个")

        # 3. 批量读取
        dfs = []
        
        # 定义读取的列：必须包含 pct_chg 和 pre_close 否则后续计算会崩
        use_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                   'vol', 'amount', 'pct_chg', 'pre_close']
        
        # 并不是所有 CSV 都有 adj_factor，尝试读取时不强制
        
        for i, f in enumerate(files_to_load):
            try:
                # 读取头部检查列名
                header = pd.read_csv(f, nrows=0)
                available_cols = [c for c in use_cols if c in header.columns]
                
                df = pd.read_csv(
                    f,
                    usecols=available_cols,
                    dtype={
                        'ts_code': 'str',
                        'trade_date': 'str',
                        'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32',
                        'vol': 'float32', 'amount': 'float32', 
                        'pct_chg': 'float32', 'pre_close': 'float32'
                    }
                )
                
                # 如果是按股票文件存储，需要在这里过滤日期
                if not is_date_file:
                    df = df[(df['trade_date'] >= real_start_date) & (df['trade_date'] <= end_date)]
                    if df.empty: continue

                dfs.append(df)

                # 进度提示
                if len(files_to_load) > 10 and ((i + 1) % 50 == 0 or (i + 1) == len(files_to_load)):
                    progress = (i + 1) / len(files_to_load) * 100
                    logger.info(f"    读取进度：{progress:.1f}%")

            except Exception as e:
                pass # 忽略单个文件错误

        if not dfs:
            logger.error("[错误] 内存中无数据")
            return

        # 4. 合并与处理
        logger.info("  [合并] 正在构建内存索引...")
        self.memory_db = pd.concat(dfs, ignore_index=True)
        
        del dfs
        gc.collect()

        # 5. [内存优化]
        self._optimize_memory()

        # 6. [极速索引]
        self.memory_db['trade_date_dt'] = pd.to_datetime(self.memory_db['trade_date'])
        
        # 去重（防止文件重叠）
        self.memory_db.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=True)
        
        # 排序并索引
        self.memory_db.sort_values(['ts_code', 'trade_date_dt'], inplace=True)
        self.memory_db.set_index(['ts_code', 'trade_date_dt'], inplace=True)

        self.loaded_start_date = real_start_date
        self.loaded_end_date = end_date
        
        # 7. 更新缓存的交易日历 (直接从数据中提取)
        unique_dates = self.memory_db['trade_date'].unique()
        self._cached_trade_days = sorted(unique_dates)

        logger.info(f"  [完成] 内存表行数：{len(self.memory_db):,}")
        logger.info(f"  [完成] 内存占用：{self.memory_db.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        gc.collect()

    def _optimize_memory(self):
        """内存优化"""
        # 再次确认 float32
        for col in self.memory_db.select_dtypes(include=['float64']).columns:
            self.memory_db[col] = self.memory_db[col].astype('float32')

    def get_trade_days(self, start_date: str, end_date: str) -> List[str]:
        """
        [覆盖父类] 直接从内存数据中获取交易日历
        """
        if self.memory_db is not None and self._cached_trade_days:
            # 使用内存中的缓存
            return [d for d in self._cached_trade_days if start_date <= d <= end_date]
        else:
            # 回退到父类方法（查本地文件或API）
            return super().get_trade_days(start_date, end_date)

    def get_stock_data(self, ts_code: str, end_date: str, days: int = 120) -> Optional[pd.DataFrame]:
        """[极速查询] 获取历史窗口数据"""
        if self.memory_db is None: return None

        try:
            end_dt = pd.Timestamp(end_date)
            
            if ts_code not in self.memory_db.index:
                return None

            # loc[ts_code] 返回该股票的所有数据 (index=date)
            stock_data = self.memory_db.loc[ts_code]
            
            # 利用切片获取数据，非常快
            slice_data = stock_data.loc[:end_dt]
            
            if len(slice_data) == 0:
                return None

            # 取最后 N 行
            result = slice_data.iloc[-days:].copy()
            
            # 还原 columns 里的 trade_date (因为在 index 里)
            result['trade_date'] = result.index.strftime('%Y%m%d')
            
            return result

        except Exception:
            return None

    def get_future_data(self, ts_code: str, current_date: str, days: int = 5) -> Optional[pd.DataFrame]:
        """[极速查询] 获取未来数据"""
        if self.memory_db is None: return None
        
        try:
            curr_dt = pd.Timestamp(current_date)
            
            if ts_code not in self.memory_db.index:
                return None
                
            stock_data = self.memory_db.loc[ts_code]
            
            # 取当前日期之后的数据
            future_slice = stock_data.loc[curr_dt:]
            
            # 如果第一行就是当前日期，去除它（我们要的是未来的）
            if not future_slice.empty and future_slice.index[0] == curr_dt:
                future_slice = future_slice.iloc[1:]
                
            if len(future_slice) < days:
                return None
                
            result = future_slice.iloc[:days].copy()
            result['trade_date'] = result.index.strftime('%Y%m%d')
            return result
            
        except Exception:
            return None

    def load_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """[极速查询] 获取某日全市场数据"""
        if self.memory_db is None: return None
        
        try:
            dt = pd.Timestamp(date)
            # xs 切片获取 Level 1 (Date)
            # 注意：内存表索引是 (ts_code, trade_date_dt)
            # 所以我们要切 Level 1
            daily_data = self.memory_db.xs(dt, level=1)
            
            result = daily_data.reset_index() # ts_code 变回列
            result['trade_date'] = date
            return result
        except KeyError:
            return None
        except Exception:
            return None


# 导出 DataWarehouseTurbo 为 DataWarehouse，方便使用
DataWarehouse = DataWarehouseTurbo
