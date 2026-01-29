"""
DeepQuant 数据仓库模块 (Data Warehouse) - 缓存优化版
功能：
1. 下载历史行情数据到本地
2. 管理本地数据（按日期存储）
3. 提供数据查询接口
4. 防止幸存者偏差（获取当时在市的股票列表）
5. [新增] 缓存历史数据，避免重复加载

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

# 导入原版 DataWarehouse
from data_warehouse import DataWarehouse as OriginalDataWarehouse


class DataWarehouse(OriginalDataWarehouse):
    """数据仓库类（缓存优化版）"""

    def __init__(self, data_dir: str = "data/daily"):
        """
        初始化数据仓库（带缓存）

        Args:
            data_dir: 数据存储目录
        """
        super().__init__(data_dir)

        # [新增] 缓存历史数据，避免重复加载
        self._history_data_cache = {}
        self._history_cache_key = None

    def load_history_data(self, end_date: str, days: int = 120) -> Dict[str, pd.DataFrame]:
        """
        [优化] 极速加载历史数据（向量化处理）+ 缓存机制

        Args:
            end_date: 结束日期
            days: 回溯天数

        Returns:
            {股票代码: DataFrame} 的字典（包含复权价格列）
        """
        # [缓存优化] 检查缓存
        cache_key = f"{end_date}_{days}"
        if self._history_cache_key == cache_key:
            return self._history_data_cache

        # 调用父类方法加载数据
        history_data = super().load_history_data(end_date, days)

        # [缓存优化] 更新缓存
        self._history_data_cache = history_data
        self._history_cache_key = cache_key

        return history_data

    def clear_cache(self):
        """清除历史数据缓存"""
        self._history_data_cache = {}
        self._history_cache_key = None
