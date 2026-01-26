#!/usr/bin/env python3
"""
A股模型实盘对比系统 - 主运行脚本
"""
import os
import sys
import argparse
from pathlib import Path

# 添加src到Python路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
src_path = os.path.join(workspace_path, "src")
sys.path.insert(0, src_path)

from stock_system.closed_loop import ClosedLoopSystem


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='A股模型实盘对比系统 - 在线学习闭环迭代平台',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 运行一次完整迭代
  python run_system.py
  
  # 运行多次连续迭代
  python run_system.py --iterations 5
  
  # 指定配置文件
  python run_system.py --config /path/to/config.json
  
  # 运行10次迭代，间隔5天
  python run_system.py --iterations 10 --interval 5
        '''
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=1,
        help='迭代次数 (默认: 1)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径 (默认: config/model_config.json)'
    )
    
    parser.add_argument(
        '--interval', '-I',
        type=int,
        default=5,
        help='迭代间隔天数 (默认: 5)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='开始日期 (格式: YYYYMMDD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='结束日期 (格式: YYYYMMDD)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='测试模式（使用模拟数据）'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("=" * 80)
    print("A股模型实盘对比系统")
    print("在线学习闭环迭代平台")
    print("=" * 80)
    print(f"迭代次数: {args.iterations}")
    print(f"配置文件: {args.config or 'config/model_config.json'}")
    if args.iterations > 1:
        print(f"迭代间隔: {args.interval} 天")
    print("=" * 80)
    print()
    
    # 创建闭环系统
    try:
        system = ClosedLoopSystem(config_path=args.config)
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        sys.exit(1)
    
    # 运行迭代
    try:
        if args.iterations == 1:
            # 单次迭代
            print("开始运行单次迭代...")
            print()
            
            result = system.run_one_iteration(
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            print("\n" + "=" * 80)
            print("迭代完成")
            print("=" * 80)
            
            if result.get('status') == 'completed':
                print("✅ 状态: 成功")
                metrics = result.get('metrics', {})
                print(f"核心指标:")
                print(f"  - Accuracy:  {metrics.get('accuracy', 0):.4f}")
                print(f"  - Precision: {metrics.get('precision', 0):.4f}")
                print(f"  - Recall:    {metrics.get('recall', 0):.4f}")
                print(f"  - F1 Score:  {metrics.get('f1', 0):.4f}")
                print(f"  - AUC:       {metrics.get('auc', 0):.4f}")
                
                if result.get('should_adjust'):
                    print(f"\n⚠️  触发参数调整: {result.get('adjust_reason')}")
                    adjustment = result.get('adjustment', {})
                    if adjustment:
                        print(f"   - 新阈值: {adjustment.get('new_threshold', 0):.4f}")
                
                if result.get('model_updated'):
                    print(f"\n✨ 模型已更新")
                else:
                    print(f"\nℹ️  模型未更新")
                
            else:
                print("❌ 状态: 失败")
                print(f"错误: {result.get('error', '未知错误')}")
            
            print("=" * 80)
            
        else:
            # 多次连续迭代
            print(f"开始运行连续迭代 ({args.iterations} 次)...")
            print()
            
            results = system.run_continuous_iterations(
                max_iterations=args.iterations,
                interval_days=args.interval
            )
            
            print("\n" + "=" * 80)
            print("连续迭代完成")
            print("=" * 80)
            
            success_count = sum(1 for r in results if r.get('status') == 'completed')
            print(f"✅ 成功: {success_count}/{len(results)}")
            print(f"❌ 失败: {len(results) - success_count}/{len(results)}")
            
            # 显示最后几次的性能趋势
            print("\n最近5次性能趋势:")
            print("| 迭代 | Precision | Recall | F1 |")
            print("|------|-----------|--------|-----|")
            
            for i, result in enumerate(results[-5:]):
                iter_num = len(results) - 5 + i + 1
                metrics = result.get('metrics', {})
                print(f"| {iter_num} | {metrics.get('precision', 0):.4f} | "
                      f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |")
            
            print("=" * 80)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，系统退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
