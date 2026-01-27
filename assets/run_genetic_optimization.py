#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepQuant 遗传算法快速启动脚本
一键完成：生成测试数据 -> 运行遗传算法 -> 查看结果
"""

import os
import subprocess
import json
import sys


def print_banner():
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant 遗传算法参数优化系统")
    print(" " * 25 + "一键启动脚本")
    print("="*80 + "\n")


def step1_generate_data():
    """步骤1：生成测试数据"""
    print("[步骤 1/3] 生成测试数据...")
    print("-" * 80)
    result = subprocess.run([sys.executable, "gen_test_data.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[错误] 数据生成失败: {result.stderr}")
        return False
    return True


def step2_run_optimization():
    """步骤2：运行遗传算法优化"""
    print("\n[步骤 2/3] 运行遗传算法优化...")
    print("-" * 80)
    result = subprocess.run([sys.executable, "genetic_optimizer.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[错误] 优化失败: {result.stderr}")
        return False
    return True


def step3_show_results():
    """步骤3：显示优化结果"""
    print("\n[步骤 3/3] 查看优化结果...")
    print("-" * 80)
    
    try:
        with open("strategy_params.json", 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        print("\n✅ 优化完成！")
        print(f"\n📊 优化统计:")
        print(f"  - 版本: {params['version']}")
        print(f"  - 优化状态: {'已优化' if params['optimized'] else '未优化'}")
        print(f"  - 最后更新: {params['last_updated']}")
        print(f"  - 适应度 (Sharpe Ratio): {params['optimization_stats']['sharpe_ratio']:.4f}")
        
        print(f"\n🎯 关键参数变化:")
        print(f"  - 评分阈值(正常): {params['thresholds']['SCORE_THRESHOLD_NORMAL']}")
        print(f"  - 评分阈值(洗盘): {params['thresholds']['SCORE_THRESHOLD_WASH']}")
        
        print(f"\n⚙️  评分权重:")
        print(f"  - 安全分上限: {params['scoring_weights']['safety']['max_score']}")
        print(f"  - 进攻分上限: {params['scoring_weights']['offensive']['max_score']}")
        print(f"  - 确定分上限: {params['scoring_weights']['certainty']['max_score']}")
        print(f"  - 配合分上限: {params['scoring_weights']['match']['max_score']}")
        
        print("\n" + "="*80)
        print("💡 提示:")
        print("  - 优化后的参数已保存到 strategy_params.json")
        print("  - 可以运行选股程序使用新参数: python main_controller.py select")
        print("  - 查看详细文档: cat GENETIC_OPTIMIZATION_README.md")
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        print(f"[错误] 无法读取优化结果: {e}")
        return False


def main():
    print_banner()
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 执行流程
    if not step1_generate_data():
        print("\n❌ 流程中断：数据生成失败")
        return
    
    if not step2_run_optimization():
        print("\n❌ 流程中断：优化失败")
        return
    
    if not step3_show_results():
        print("\n⚠️  警告：无法显示优化结果")
    
    print("\n🎉 完成！遗传算法优化系统运行成功！\n")


if __name__ == "__main__":
    main()
