#!/usr/bin/env python3
"""
A股模型实盘对比系统 - 演示脚本
"""
import os
import sys
import logging

# 添加src到Python路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
src_path = os.path.join(workspace_path, "src")
sys.path.insert(0, src_path)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo():
    """演示系统功能"""
    from stock_system.closed_loop import ClosedLoopSystem
    
    print("=" * 80)
    print("A股模型实盘对比系统 - 演示")
    print("=" * 80)
    print()
    
    # 创建闭环系统
    print("正在初始化系统...")
    system = ClosedLoopSystem()
    print("✅ 系统初始化完成\n")
    
    # 运行一次迭代
    print("开始运行一次完整迭代...")
    print("-" * 80)
    
    result = system.run_one_iteration()
    
    print("-" * 80)
    print()
    
    # 显示结果
    print("=" * 80)
    print("迭代结果")
    print("=" * 80)
    
    if result.get('status') == 'completed':
        print("✅ 状态: 成功\n")
        
        metrics = result.get('metrics', {})
        print("核心性能指标:")
        print(f"  📊 Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  🎯 Precision: {metrics.get('precision', 0):.4f}")
        print(f"  🔄 Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  ⚖️  F1 Score:  {metrics.get('f1', 0):.4f}")
        print(f"  📈 AUC:       {metrics.get('auc', 0):.4f}")
        print()
        
        error_analysis = result.get('error_analysis', {})
        print("误差分析:")
        print(f"  ❌ 误差率: {error_analysis.get('error_rate', 0)*100:.2f}%")
        print(f"  ⚠️  假正例: {error_analysis.get('false_positive_count', 0)}")
        print(f"  ⚠️  假负例: {error_analysis.get('false_negative_count', 0)}")
        print()
        
        if result.get('should_adjust'):
            print("⚠️  触发参数调整")
            print(f"   原因: {result.get('adjust_reason', '')}")
            adjustment = result.get('adjustment', {})
            if adjustment:
                print(f"   新阈值: {adjustment.get('new_threshold', 0):.4f}")
        else:
            print("✅ 指标良好，无需调整")
        print()
        
        if result.get('model_updated'):
            print("✨ 模型已更新并保存")
        else:
            print("ℹ️  模型未更新")
        
    else:
        print("❌ 状态: 失败")
        print(f"错误: {result.get('error', '未知错误')}")
    
    print("=" * 80)
    print()
    print("✨ 演示完成！")
    print()
    print("生成的文件:")
    print("  - 预测结果: assets/data/predictions/")
    print("  - 性能报告: assets/logs/performance_report_*.md")
    print("  - 误差报告: assets/logs/error_report_*.md")
    print("  - 模型文件: assets/models/")
    print()


if __name__ == '__main__':
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
