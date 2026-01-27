"""
性能对比测试脚本
演示优化前后的性能差异
"""
import os
import time
import json
from src.stock_system.data_collector import MarketDataCollector

# 设置环境变量
os.environ['TUSHARE_TOKEN'] = 'your_tushare_token_here'

def test_cache_performance():
    """测试缓存性能"""
    print("=" * 60)
    print("测试1：缓存性能对比")
    print("=" * 60)
    
    collector = MarketDataCollector()
    
    # 清理旧缓存
    collector.clear_cache()
    print("\n[清理缓存完成]\n")
    
    # 第一次获取（无缓存）
    print("[第一次获取 - 无缓存]")
    start_time = time.time()
    stock_list_1 = collector.get_stock_list(use_cache=True)
    time_1 = time.time() - start_time
    print(f"耗时: {time_1:.2f} 秒")
    print(f"股票数量: {len(stock_list_1)}")
    
    # 第二次获取（有缓存）
    print("\n[第二次获取 - 有缓存]")
    start_time = time.time()
    stock_list_2 = collector.get_stock_list(use_cache=True)
    time_2 = time.time() - start_time
    print(f"耗时: {time_2:.2f} 秒")
    print(f"股票数量: {len(stock_list_2)}")
    
    # 性能提升
    speedup = time_1 / time_2 if time_2 > 0 else float('inf')
    print(f"\n缓存性能提升: {speedup:.1f} 倍")

def test_thread_pool_performance():
    """测试线程池性能"""
    print("\n" + "=" * 60)
    print("测试2：多线程 vs 单线程性能对比")
    print("=" * 60)
    
    collector = MarketDataCollector()
    
    # 获取股票池
    stock_pool = collector.get_stock_pool_tree(pool_size=20)
    print(f"\n测试股票池: {len(stock_pool)} 只股票")
    
    start_date = '20240101'
    
    # 测试多线程
    print("\n[多线程批量获取]")
    collector.max_workers = 5
    start_time = time.time()
    data_threaded = collector.get_batch_daily_data(
        stock_pool, start_date, use_cache=False, use_thread=True
    )
    time_threaded = time.time() - start_time
    print(f"耗时: {time_threaded:.2f} 秒")
    print(f"成功获取: {len(data_threaded)} 只股票")
    
    # 测试单线程
    print("\n[单线程批量获取]")
    collector.max_workers = 1
    start_time = time.time()
    data_single = collector.get_batch_daily_data(
        stock_pool, start_date, use_cache=False, use_thread=False
    )
    time_single = time.time() - start_time
    print(f"耗时: {time_single:.2f} 秒")
    print(f"成功获取: {len(data_single)} 只股票")
    
    # 性能提升
    speedup = time_single / time_threaded if time_threaded > 0 else 1
    print(f"\n多线程性能提升: {speedup:.1f} 倍")

def test_tree_filtering_performance():
    """测试树形筛选性能"""
    print("\n" + "=" * 60)
    print("测试3：树形筛选效果")
    print("=" * 60)
    
    collector = MarketDataCollector()
    
    # 获取原始股票列表
    print("\n[原始股票列表]")
    start_time = time.time()
    all_stocks = collector.get_stock_list(use_cache=True)
    print(f"股票总数: {len(all_stocks)}")
    
    # 应用树形筛选
    print("\n[树形筛选后]")
    start_time = time.time()
    filtered_pool = collector.get_stock_pool_tree(
        pool_size=100,
        market='SSE',
        exclude_st=True,
        min_days_listed=30,
        use_cache=True
    )
    print(f"筛选后数量: {len(filtered_pool)}")
    print(f"筛选耗时: {time.time() - start_time:.2f} 秒")
    
    # 分析筛选效果
    print("\n[筛选效果分析]")
    if not all_stocks.empty and 'name' in all_stocks.columns:
        st_count = all_stocks['name'].str.contains('ST', na=False).sum()
        print(f"原始ST股票数: {st_count}")
        print(f"已自动排除ST股票")
    
    print(f"\n过滤比例: {(1 - len(filtered_pool) / len(all_stocks)) * 100:.1f}%")

def test_cache_statistics():
    """测试缓存统计"""
    print("\n" + "=" * 60)
    print("测试4：缓存统计信息")
    print("=" * 60)
    
    collector = MarketDataCollector()
    
    stats = collector.get_cache_stats()
    
    print("\n缓存统计:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    print("\n[缓存分布分析]")
    type_dist = stats.get('type_distribution', {})
    for cache_type, count in type_dist.items():
        print(f"  {cache_type}: {count} 个文件")

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("A股模型实盘对比系统 - 性能对比测试")
    print("=" * 60)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 运行各项测试
    try:
        test_cache_performance()
        test_thread_pool_performance()
        test_tree_filtering_performance()
        test_cache_statistics()
        
        print("\n" + "=" * 60)
        print("性能测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
