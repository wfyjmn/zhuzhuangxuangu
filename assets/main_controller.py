# -*- coding: utf-8 -*-
"""
DeepQuant 主控程序 (Main Controller)
功能：协调各模块运行，实现完整的闭环系统

工作流程：
1. 运行选股筛选（第1轮 + 第2轮）
2. 创建验证跟踪记录
3. 更新验证数据（获取后续表现）
4. 生成验证报告
5. 运行参数优化（可选）
"""

import os
import sys
import subprocess
import json
from datetime import datetime


def print_banner():
    """打印程序横幅"""
    print("\n" + "="*80)
    print(" " * 20 + "DeepQuant Pro V3.0")
    print(" " * 15 + "智能选股 · 验证跟踪 · 参数优化")
    print("="*80)


def load_params():
    """加载参数配置"""
    try:
        with open('strategy_params.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None


def check_dependencies():
    """检查必要的文件是否存在"""
    required_files = [
        'strategy_params.json',
        'validation_records.csv',
        'paper_trading_records.csv',
        'params_history.csv'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("\n[警告] 缺少以下配置文件：")
        for file in missing_files:
            print(f"  - {file}")
        print("\n提示：这些文件将在系统首次运行时自动创建")
        return False

    return True


def run_stock_selection():
    """运行选股流程"""
    print("\n" + "="*80)
    print("【阶段 1】运行选股筛选")
    print("="*80)

    print("\n[步骤 1.1] 运行第1轮筛选...")
    try:
        # 运行第1轮筛选
        result = subprocess.run(
            [sys.executable, '柱形选股-筛选.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[错误] 第1轮筛选失败")
            print(result.stderr)
            return False

        print("[完成] 第1轮筛选成功")

    except Exception as e:
        print(f"[错误] 执行第1轮筛选失败: {e}")
        return False

    print("\n[步骤 1.2] 运行第2轮筛选...")
    try:
        # 运行第2轮筛选
        result = subprocess.run(
            [sys.executable, '柱形选股-第2轮.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[错误] 第2轮筛选失败")
            print(result.stderr)
            return False

        print("[完成] 第2轮筛选成功")

    except Exception as e:
        print(f"[错误] 执行第2轮筛选失败: {e}")
        return False

    print("\n[✅ 完成] 选股筛选流程已完成")
    return True


def run_validation_tracking(mode='update'):
    """运行验证跟踪"""
    print("\n" + "="*80)
    print("【阶段 2】验证跟踪")
    print("="*80)

    print(f"\n[步骤 2.1] 运行验证跟踪（模式: {mode}）...")
    try:
        result = subprocess.run(
            [sys.executable, 'validation_track.py', mode],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[错误] 验证跟踪失败")
            print(result.stderr)
            return False

        print("[完成] 验证跟踪成功")
        return True

    except Exception as e:
        print(f"[错误] 执行验证跟踪失败: {e}")
        return False


def run_parameter_optimization():
    """运行参数优化"""
    print("\n" + "="*80)
    print("【阶段 3】参数优化")
    print("="*80)

    # 检查优化是否启用
    params = load_params()
    if not params:
        print("[警告] 无法加载参数配置")
        return False

    if not params['params'].get('optimization', {}).get('enabled', False):
        print("[信息] 参数优化功能未启用，跳过此阶段")
        return True

    print("\n[步骤 3.1] 运行参数优化...")
    try:
        result = subprocess.run(
            [sys.executable, 'parameter_optimizer.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[错误] 参数优化失败")
            print(result.stderr)
            return False

        print("[完成] 参数优化成功")
        return True

    except Exception as e:
        print(f"[错误] 执行参数优化失败: {e}")
        return False


def run_full_pipeline():
    """运行完整流程：选股 → 验证 → 优化"""
    print_banner()

    # 检查依赖
    check_dependencies()

    # 阶段 1：选股
    if not run_stock_selection():
        print("\n[❌ 失败] 选股阶段失败，流程终止")
        return False

    # 阶段 2：验证跟踪（scan + update + report）
    if not run_validation_tracking(mode='all'):
        print("\n[❌ 失败] 验证跟踪阶段失败")
        return False

    # 阶段 3：参数优化
    if not run_parameter_optimization():
        print("\n[⚠️ 警告] 参数优化阶段失败")
        # 参数优化失败不影响整体流程

    print("\n" + "="*80)
    print("【✅ 完成】完整流程执行完毕")
    print("="*80)
    print("\n数据文件：")
    print("  - 选股结果: DeepQuant_TopPicks_YYYYMMDD.csv")
    print("  - 验证记录: validation_records.csv")
    print("  - 模拟交易: paper_trading_records.csv")
    print("  - 参数配置: strategy_params.json")
    print("  - 参数历史: params_history.csv")
    print("="*80)

    return True


def run_validation_only():
    """仅运行验证跟踪（日常更新）"""
    print_banner()
    print("\n[模式] 仅运行验证跟踪更新")

    # 检查依赖
    check_dependencies()

    # 运行验证更新
    if run_validation_tracking(mode='update'):
        print("\n[✅ 完成] 验证更新完成")
        return True
    else:
        print("\n[❌ 失败] 验证更新失败")
        return False


def run_optimization_only():
    """仅运行参数优化（周期性）"""
    print_banner()
    print("\n[模式] 仅运行参数优化")

    # 检查依赖
    check_dependencies()

    # 运行参数优化
    if run_parameter_optimization():
        print("\n[✅ 完成] 参数优化完成")
        return True
    else:
        print("\n[❌ 失败] 参数优化失败")
        return False


def show_usage():
    """显示使用说明"""
    print_banner()
    print("\n使用说明：")
    print("\n  python main_controller.py [mode]")
    print("\n  模式说明：")
    print("    full    - 运行完整流程（选股 + 验证 + 优化）")
    print("    select  - 仅运行选股筛选")
    print("    validate- 仅运行验证跟踪更新")
    print("    optimize- 仅运行参数优化")
    print("\n  默认模式：full")
    print("\n示例：")
    print("  python main_controller.py           # 运行完整流程")
    print("  python main_controller.py validate  # 仅更新验证数据")
    print("  python main_controller.py optimize  # 仅优化参数")
    print("="*80)


def main():
    """主函数"""
    # 解析命令行参数
    mode = 'full'  # 默认模式

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    # 根据模式执行
    if mode == 'full':
        run_full_pipeline()
    elif mode == 'select':
        print_banner()
        run_stock_selection()
    elif mode == 'validate':
        run_validation_only()
    elif mode == 'optimize':
        run_optimization_only()
    elif mode in ['help', '-h', '--help']:
        show_usage()
    else:
        print(f"[错误] 未知模式: {mode}")
        show_usage()


if __name__ == '__main__':
    main()
