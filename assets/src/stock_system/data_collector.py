"""
行情数据采集模块（优化版）
功能：从tushare获取A股实盘行情数据
优化：
1. Token安全：使用环境变量存储token
2. 批量多线程：采用线程池批量获取数据
3. 树形筛选：分层级筛选股票，减少无效调用
4. 智能缓存：完善缓存机制，最大限度减少API调用
"""
import os
import json
import logging
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pandas as pd
import numpy as np
import tushare as ts

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketDataCollector:
    """行情数据采集器（优化版）"""
    
    def __init__(self, config_path: str = None):
        """
        初始化数据采集器
        
        Args:
            config_path: tushare配置文件路径
        """
        if config_path is None:
            workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
            config_path = os.path.join(workspace_path, "config/tushare_config.json")
        
        self.config = self._load_config(config_path)
        
        # API请求限制配置（必须在初始化tushare之前设置）
        self.max_workers = self.config.get('max_workers', 5)  # 线程池大小
        self.request_timeout = self.config.get('timeout', 30)  # 请求超时
        self.retry_count = self.config.get('retry_count', 3)  # 重试次数
        self.rate_limit_delay = self.config.get('rate_limit_delay', 0.1)  # 限流延迟（秒）
        
        # 缓存配置
        self.cache_expiry_hours = self.config.get('cache_expiry_hours', 24)  # 缓存过期时间（小时）
        
        # 初始化tushare和缓存目录
        self.pro = self._init_tushare()
        self.cache_dir = self._init_cache_dir()
        
    def _load_config(self, config_path: str) -> Dict:
        """
        加载配置
        优化：Token优先从环境变量读取，配置文件仅作fallback
        """
        try:
            # 优先从环境变量读取token
            env_token = os.getenv('TUSHARE_TOKEN')
            
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 如果环境变量有token，优先使用环境变量
            if env_token:
                config['token'] = env_token
                logger.info(f"使用环境变量中的token")
            
            logger.info(f"加载配置文件成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _init_tushare(self):
        """初始化tushare连接"""
        try:
            token = self.config.get('token', '')
            if not token:
                logger.warning("未配置tushare token，请先在环境变量TUSHARE_TOKEN中配置")
                return None
            
            ts.set_token(token)
            pro = ts.pro_api(timeout=self.request_timeout)
            logger.info("tushare连接初始化成功")
            return pro
        except Exception as e:
            logger.error(f"初始化tushare失败: {e}")
            raise
    
    def _init_cache_dir(self) -> str:
        """初始化缓存目录"""
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
        cache_dir = os.path.join(workspace_path, "assets/data/market_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"缓存目录初始化: {cache_dir}")
        return cache_dir
    
    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """生成缓存key"""
        key_str = f"{prefix}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """检查缓存是否有效"""
        if not os.path.exists(cache_file):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        return file_time > expiry_time
    
    def _save_pickle_cache(self, data, cache_file: str) -> bool:
        """保存pickle格式的缓存"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"保存缓存成功: {cache_file}")
            return True
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            return False
    
    def _load_pickle_cache(self, cache_file: str):
        """加载pickle格式的缓存"""
        try:
            if not self._is_cache_valid(cache_file):
                logger.debug(f"缓存已过期: {cache_file}")
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"加载缓存成功: {cache_file}")
            return data
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return None
    
    @lru_cache(maxsize=100)
    def get_stock_list(self, market: str = None, status: str = 'L', 
                       use_cache: bool = True) -> pd.DataFrame:
        """
        获取股票列表（带缓存）
        
        Args:
            market: 市场，SSE=上海，SZSE=深圳，None=所有
            status: 状态，L=上市，D=退市，P=暂停上市
            use_cache: 是否使用缓存
            
        Returns:
            股票列表DataFrame
        """
        # 检查缓存
        if use_cache:
            cache_key = self._get_cache_key('stock_list', market=market, status=status)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cached_data = self._load_pickle_cache(cache_file)
            if cached_data is not None:
                logger.info(f"从缓存加载股票列表，共 {len(cached_data)} 只股票")
                return cached_data
        
        try:
            if not self.pro:
                logger.error("tushare未初始化")
                return pd.DataFrame()
            
            df = self.pro.stock_basic(exchange='', list_status=status,
                                      fields='ts_code,symbol,name,area,industry,list_date')
            
            # 树形筛选第一层：市场筛选
            if market:
                if market == 'SSE':
                    df = df[df['ts_code'].str.endswith('.SH')]
                elif market == 'SZSE':
                    df = df[df['ts_code'].str.endswith('.SZ')]
            
            # 树形筛选第二层：排除ST、退市、暂停上市股票
            df = df[~df['name'].str.contains('ST|退|暂停', na=False)]
            
            # 树形筛选第三层：排除新上市股票（不足30天）
            if not df.empty and 'list_date' in df.columns:
                df['list_date'] = pd.to_datetime(df['list_date'])
                min_list_date = datetime.now() - timedelta(days=30)
                df = df[df['list_date'] < min_list_date]
            
            logger.info(f"获取股票列表成功，共 {len(df)} 只股票")
            
            # 保存缓存
            if use_cache:
                self._save_pickle_cache(df, cache_file)
            
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_daily_data(self, ts_code: str, start_date: str, end_date: str = None,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        获取单只股票日线数据（带缓存）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'，默认为今天
            use_cache: 是否使用缓存
            
        Returns:
            日线数据DataFrame
        """
        # 检查缓存
        if use_cache:
            cache_key = self._get_cache_key('daily_data', ts_code=ts_code, 
                                           start_date=start_date, end_date=end_date)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            cached_data = self._load_pickle_cache(cache_file)
            if cached_data is not None:
                logger.debug(f"从缓存加载股票 {ts_code} 的日线数据")
                return cached_data
        
        try:
            if not self.pro:
                logger.error("tushare未初始化")
                return pd.DataFrame()
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            # 重试机制
            for retry in range(self.retry_count):
                try:
                    df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    
                    if df is None or df.empty:
                        logger.warning(f"获取股票 {ts_code} 的日线数据为空")
                        return pd.DataFrame()
                    
                    # 计算涨跌幅
                    df['pct_chg'] = df['pct_chg'].round(2)
                    
                    # 保存缓存
                    if use_cache:
                        self._save_pickle_cache(df, cache_file)
                    
                    logger.debug(f"获取股票 {ts_code} 的日线数据成功，共 {len(df)} 条")
                    return df
                    
                except Exception as e:
                    if retry < self.retry_count - 1:
                        logger.warning(f"获取股票 {ts_code} 数据失败，重试 {retry + 1}/{self.retry_count}: {e}")
                        time.sleep(1)
                    else:
                        raise
                    
        except Exception as e:
            logger.error(f"获取股票 {ts_code} 的日线数据失败: {e}")
            return pd.DataFrame()
    
    def get_batch_daily_data(self, ts_codes: List[str], start_date: str, 
                            end_date: str = None, use_cache: bool = True,
                            use_thread: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的日线数据（多线程+缓存）
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            use_thread: 是否使用多线程
            
        Returns:
            股票代码到日线数据的映射字典
        """
        result = {}
        failed_codes = []
        
        if use_thread and len(ts_codes) > 1:
            # 多线程批量获取
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_code = {
                    executor.submit(
                        self.get_daily_data, ts_code, start_date, end_date, use_cache
                    ): ts_code
                    for ts_code in ts_codes
                }
                
                for future in as_completed(future_to_code):
                    ts_code = future_to_code[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            result[ts_code] = df
                        else:
                            failed_codes.append(ts_code)
                    except Exception as e:
                        logger.error(f"获取股票 {ts_code} 的数据失败: {e}")
                        failed_codes.append(ts_code)
        else:
            # 单线程获取
            for ts_code in ts_codes:
                try:
                    df = self.get_daily_data(ts_code, start_date, end_date, use_cache)
                    if df is not None and not df.empty:
                        result[ts_code] = df
                    else:
                        failed_codes.append(ts_code)
                except Exception as e:
                    logger.error(f"获取股票 {ts_code} 的数据失败: {e}")
                    failed_codes.append(ts_code)
                finally:
                    # 避免请求过快
                    time.sleep(self.rate_limit_delay)
        
        success_count = len(result)
        total_count = len(ts_codes)
        logger.info(f"批量获取日线数据完成，成功 {success_count}/{total_count} 只股票")
        
        if failed_codes:
            logger.warning(f"失败的股票代码: {failed_codes[:10]}...")  # 只显示前10个
        
        return result
    
    def get_stock_pool(self, pool_size: int = 100, market: str = None) -> List[str]:
        """
        获取股票池（向后兼容方法）
        该方法已弃用，建议使用 get_stock_pool_tree
        
        Args:
            pool_size: 池子大小
            market: 市场，None=所有
            
        Returns:
            股票代码列表
        """
        logger.warning("get_stock_pool 方法已弃用，建议使用 get_stock_pool_tree")
        return self.get_stock_pool_tree(
            pool_size=pool_size,
            market=market,
            exclude_st=True,
            min_days_listed=30
        )
    
    def get_stock_pool_tree(self, pool_size: int = 100, market: str = None,
                           industries: List[str] = None, exclude_st: bool = True,
                           min_days_listed: int = 30, use_cache: bool = True) -> List[str]:
        """
        获取股票池（树形筛选）
        
        筛选层级：
        1. 市场筛选（上海/深圳/全部）
        2. 行业筛选（指定行业或全行业）
        3. 质量筛选（排除ST、退市、暂停上市）
        4. 时间筛选（排除新上市股票）
        5. 行业均匀采样（从每个行业均匀选取）
        
        Args:
            pool_size: 池子大小
            market: 市场，None=所有
            industries: 指定行业列表，None=所有行业
            exclude_st: 是否排除ST股票
            min_days_listed: 最小上市天数
            use_cache: 是否使用缓存
            
        Returns:
            股票代码列表
        """
        try:
            df = self.get_stock_list(market=market, use_cache=use_cache)
            
            if df.empty:
                logger.warning("获取股票列表为空，返回空池")
                return []
            
            # 树形筛选第3层：排除ST股票
            if exclude_st:
                df = df[~df['name'].str.contains('ST|退|暂停', na=False)]
                logger.info(f"排除ST后剩余: {len(df)} 只股票")
            
            # 树形筛选第4层：排除新上市股票
            if min_days_listed > 0 and 'list_date' in df.columns:
                df['list_date'] = pd.to_datetime(df['list_date'])
                min_list_date = datetime.now() - timedelta(days=min_days_listed)
                df = df[df['list_date'] < min_list_date]
                logger.info(f"排除新股后剩余: {len(df)} 只股票")
            
            # 树形筛选第2层：行业筛选
            if industries:
                df = df[df['industry'].isin(industries)]
                logger.info(f"行业筛选后剩余: {len(df)} 只股票")
            
            # 树形筛选第5层：按行业均匀采样
            if 'industry' in df.columns:
                industry_groups = df.groupby('industry')
                selected_stocks = []
                
                # 计算每个行业应该选取的数量
                per_industry = max(1, int(pool_size / len(industry_groups)))
                
                for industry, group in industry_groups:
                    # 按上市日期排序，选择更成熟的股票
                    group_sorted = group.sort_values('list_date', ascending=False)
                    selected = group_sorted.head(per_industry)
                    selected_stocks.extend(selected['ts_code'].tolist())
                
                # 如果不足，从其他行业补充
                if len(selected_stocks) < pool_size:
                    remaining = pool_size - len(selected_stocks)
                    available = df[~df['ts_code'].isin(selected_stocks)]
                    extra = available.head(remaining)
                    selected_stocks.extend(extra['ts_code'].tolist())
            else:
                selected_stocks = df['ts_code'].tolist()
            
            # 截断到指定数量
            selected_stocks = selected_stocks[:pool_size]
            
            logger.info(f"获取股票池成功，共 {len(selected_stocks)} 只股票")
            return selected_stocks
            
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return []
    
    def get_realtime_quotes_batch(self, ts_codes: List[str], 
                                  batch_size: int = 100) -> pd.DataFrame:
        """
        批量获取实时行情（分批获取）
        
        Args:
            ts_codes: 股票代码列表
            batch_size: 每批获取的数量
            
        Returns:
            实时行情DataFrame
        """
        try:
            if not self.pro:
                logger.error("tushare未初始化")
                return pd.DataFrame()
            
            all_data = []
            
            # 分批获取，避免API限制
            for i in range(0, len(ts_codes), batch_size):
                batch_codes = ts_codes[i:i + batch_size]
                ts_code_str = ','.join(batch_codes)
                
                try:
                    df = self.pro.daily(ts_code=ts_code_str)
                    if df is not None and not df.empty:
                        all_data.append(df)
                    
                    # 避免请求过快
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"获取批量 {i}-{i+batch_size} 实时行情失败: {e}")
                    continue
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                logger.info(f"获取实时行情成功，共 {len(result)} 只股票")
                return result
            else:
                logger.warning("获取实时行情为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()
    
    def clear_cache(self, older_than_hours: int = None) -> int:
        """
        清理缓存
        
        Args:
            older_than_hours: 清理多少小时前的缓存，None=清理全部
            
        Returns:
            清理的文件数量
        """
        try:
            cleared_count = 0
            current_time = datetime.now()
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    should_delete = False
                    if older_than_hours is None:
                        should_delete = True
                    else:
                        expiry_time = current_time - timedelta(hours=older_than_hours)
                        if file_time < expiry_time:
                            should_delete = True
                    
                    if should_delete:
                        os.remove(file_path)
                        cleared_count += 1
                        logger.debug(f"删除缓存文件: {filename}")
            
            logger.info(f"清理缓存完成，删除 {cleared_count} 个文件")
            return cleared_count
            
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return 0
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        检查数据质量
        
        Args:
            df: 行情数据DataFrame
            
        Returns:
            质量检查结果
        """
        result = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'abnormal_values': {},
            'data_range': {}
        }
        
        # 检查异常值
        numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                # 检查负值
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    result['abnormal_values'][f'{col}_negative'] = negative_count
                
                # 数据范围
                if not df[col].empty:
                    result['data_range'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean())
                    }
        
        logger.info(f"数据质量检查完成: {result}")
        return result
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in cache_files
            )
            
            # 按类型分类统计
            cache_stats = {
                'total_files': len(cache_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': self.cache_dir
            }
            
            # 统计各类型缓存数量
            type_counts = {}
            for filename in cache_files:
                if 'stock_list' in filename:
                    type_counts['stock_list'] = type_counts.get('stock_list', 0) + 1
                elif 'daily_data' in filename:
                    type_counts['daily_data'] = type_counts.get('daily_data', 0) + 1
            
            cache_stats['type_distribution'] = type_counts
            
            return cache_stats
            
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}
