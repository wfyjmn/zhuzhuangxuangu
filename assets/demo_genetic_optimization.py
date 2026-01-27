#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法优化演示脚本
展示完整的优化流程：生成测试数据 → 运行优化 → 应用参数
"""

import os
import sys
import json
import shutil
from datetime import datetime


def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n{'='*80}")
    print(f"【步骤 {step_num}】{title}")
    print(f"{'='*80}\n")


def main():
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant 遗传算法优化演示")
    print("="*80)
    
    # 步骤1：生成测试数据
    print_step(1, "生成测试验证数据")
    
    print("[提示] 使用模拟数据演示遗传算法优化流程")
    print("[提示] 实际使用时应使用真实的验证跟踪记录\n")
    
    os.system("python scripts/generate_test_validation_data.py")
    
    # 步骤2：显示原始参数
    print_step(2, "查看原始策略参数")
    
    try:
        with open("strategy_params.json", 'r', encoding='utf-8') as f:
            original_params = json.load(f)
        
        print("[原始参数] 关键配置:")
        print(f"  - 安全分上限: {original_params['scoring_weights']['safety']['max_score']}")
        print(f"  - 进攻分上限: {original_params['scoring_weights']['offensive']['max_score']}")
        print(f"  - 确定分上限: {original_params['scoring_weights']['certainty']['max_score']}")
        print(f"  - 配合分上限: {original_params['scoring_weights']['match']['max_score']}")
        print(f"  - 评分阈值(正常): {original_params['thresholds']['SCORE_THRESHOLD_NORMAL']}")
        print(f"  - 评分阈值(洗盘): {original_params['thresholds']['SCORE_THRESHOLD_WASH']}")
    except Exception as e:
        print(f"[错误] 无法读取参数配置: {e}")
        return
    
    # 步骤3：运行遗传算法优化
    print_step(3, "运行遗传算法优化")
    
    print("[提示] 开始优化参数...")
    print("[提示] 这可能需要几分钟时间，请耐心等待\n")
    
    os.system("python genetic_optimizer.py")
    
    # 步骤4：显示优化结果
    print_step(4, "查看优化结果")
    
    try:
        with open("strategy_params_optimized.json", 'r', encoding='utf-8') as f:
            optimized_params = json.load(f)
        
        print("[优化后参数] 关键配置:")
        print(f"  - 安全分上限: {optimized_params['scoring_weights']['safety']['max_score']}")
        print(f"  - 进攻分上限: {optimized_params['scoring_weights']['offensive']['max_score']}")
        print(f"  - 确定分上限: {optimized_params['scoring_weights']['certainty']['max_score']}")
        print(f"  - 配合分上限: {optimized_params['scoring_weights']['match']['max_score']}")
        print(f"  - 评分阈值(正常): {optimized_params['thresholds']['SCORE_THRESHOLD_NORMAL']}")
        print(f"  - 评分阈值(洗盘): {optimized_params['thresholds']['SCORE_THRESHOLD_WASH']}")
        
        # 显示优化统计
        stats = optimized_params['optimization_stats']
        print(f"\n[优化统计]:")
        print(f"  - 优化代数: {stats['generation']}")
        print(f"  - 适应度: {stats['sharpe_ratio']:.4f}")
        print(f"  - 优化时间: {optimized_params['last_updated']}")
    except Exception as e:
        print(f"[错误] 无法读取优化结果: {e}")
        return
    
    # 步骤5：显示优化历史
    print_step(5, "查看优化历史")
    
    try:
        import pandas as pd
        history_df = pd.read_csv("optimization_history.csv")
        
        print("[优化过程] 前10代:")
        print(history_df.head(10).to_string(index=False))
        
        print(f"\n[优化结果] 适应度提升:")
        initial_fitness = history_df.iloc[0]['best_fitness']
        final_fitness = history_df.iloc[-1]['best_fitness']
        improvement = (final_fitness - initial_fitness) / initial_fitness * 100
        
        print(f"  - 初始适应度: {initial_fitness:.4f}")
        print(f"  - 最终适应度: {final_fitness:.4f}")
        print(f"  - 提升幅度: {improvement:.2f}%")
    except Exception as e:
        print(f"[警告] 无法显示优化历史: {e}")
    
    # 步骤6：应用优化参数
    print_step(6, "应用优化后的参数")
    
    print("\n[提示] 优化后的参数已保存到 strategy_params_optimized.json")
    print("[提示] 要应用新参数，请选择以下方式之一：\n")
    
    print("方式1：自动替换（推荐用于演示）")
    print("  执行: python main_controller.py apply-optimized\n")
    
    print("方式2：手动替换")
    print("  1. 备份原配置: cp strategy_params.json strategy_params_backup.json")
    print("  2. 应用新配置: cp strategy_params_optimized.json strategy_params.json\n")
    
    print("方式3：对比调整")
    print("  手动对比两个文件，选择性地应用优化参数\n")
    
    # 询问是否应用
    print("="*80)
    response = input("是否现在应用优化后的参数？(y/n): ").strip().lower()
    
    if response == 'y':
        print("\n[应用] 备份原配置...")
        shutil.copy("strategy_params.json", "strategy_params_backup.json")
        
        print("[应用] 应用优化后的参数...")
        shutil.copy("strategy_params_optimized.json", "strategy_params.json")
        
        print("\n[完成] 优化参数已应用！")
        print("[提示] 下次选股将使用新的参数配置")
    else:
        print("\n[跳过] 参数未被应用")
        print("[提示] 您可以稍后手动应用优化参数")
    
    # 总结
    print("\n" + "="*80)
    print("【演示完成】")
    print("="*80)
    print("\n生成的文件:")
    print("  - validation_records.csv        # 验证数据")
    print("  - strategy_params_backup.json   # 原参数备份（如已应用）")
    print("  - strategy_params_optimized.json # 优化后的参数")
    print("  - optimization_history.csv      # 优化历史记录")
    print("\n相关文档:")
    print("  - GENETIC_OPTIMIZATION_README.md # 详细使用说明")
    print("="*80 + "\n")


if __name__ == "__main__":
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
