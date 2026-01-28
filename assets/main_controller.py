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
    print("【阶段 0】天气预报（市场环境研判）")
    print("="*80)

    # 🌤️ 运行天气预报系统
    try:
        from market_weather import MarketWeather
        weather = MarketWeather()
        forecast = weather.get_weather_forecast()

        # 如果建议空仓，则跳过选股
        if not forecast['allow_trading']:
            print("\n" + "⚠️"*40)
            print(f"\n[系统提醒] 当前市场天气: {forecast['weather']}")
            print(f"[系统提醒] 系统建议: {forecast['action']}")
            print(f"[系统提醒] 策略调整: {forecast['strategy_adj']}")
            print("\n[决定] 暂停选股，空仓休息")
            print("[提示] '雨天不出门'，保护资金安全比赚钱更重要")
            print("⚠️"*40 + "\n")

            # 记录到日志
            with open('weather_decision.log', 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"天气: {forecast['weather']}\n")
                f.write(f"建议: {forecast['action']}\n")
                f.write(f"决定: 暂停选股（空仓休息）\n")
                f.write(f"{'='*80}\n")

            return True  # 返回True但不执行选股

        # 如果允许交易，根据天气调整参数
        print(f"\n[系统] 当前市场天气: {forecast['weather']}")
        print(f"[系统] 系统建议: {forecast['action']}")
        print(f"[系统] 阈值调整: {forecast['threshold_adj']:+}分")
        print("[系统] 继续执行选股流程\n")

        # 记录决策
        with open('weather_decision.log', 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"天气: {forecast['weather']}\n")
            f.write(f"建议: {forecast['action']}\n")
            f.write(f"阈值调整: {forecast['threshold_adj']:+}分\n")
            f.write(f"决定: 执行选股\n")
            f.write(f"{'='*80}\n")

    except Exception as e:
        print(f"[警告] 天气预报系统运行失败: {e}")
        print("[信息] 继续执行选股流程（使用默认参数）\n")

    print("\n" + "="*80)
    print("【阶段 1】运行选股筛选")
    print("="*80)

    print("\n[步骤 1.1] 运行第1轮筛选...")
    try:
        # 运行第1轮筛选
        result = subprocess.run(
            [sys.executable, '柱形选股-筛选.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'  # 遇到编码错误时替换字符，避免崩溃
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
            text=True,
            encoding='utf-8',
            errors='replace'
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
            text=True,
            encoding='utf-8',
            errors='replace'
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

    if not params.get('params', {}).get('optimization', {}).get('enabled', False):
        print("[信息] 参数优化功能未启用，跳过此阶段")
        return True

    print("\n[步骤 3.1] 运行参数优化...")
    try:
        result = subprocess.run(
            [sys.executable, 'parameter_optimizer.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
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


def run_genetic_optimization():
    """运行遗传算法参数优化"""
    print("\n" + "="*80)
    print("【阶段 4】遗传算法优化")
    print("="*80)

    params = load_params()
    if not params:
        print("[警告] 无法加载参数配置")
        return False

    # 检查是否有足够的历史数据
    try:
        import pandas as pd
        validation_records = pd.read_csv('validation_records.csv', encoding='utf-8-sig')
        if len(validation_records) < 50:
            print(f"[警告] 验证数据不足（{len(validation_records)}条），建议至少50条数据后再运行遗传算法")
            print("[信息] 跳过遗传算法优化")
            return True
    except:
        print("[警告] 无法读取验证记录")
        return True

    print(f"\n[步骤 4.1] 开始遗传算法优化（种群大小: {params['genetic_algorithm']['population_size']}）...")
    try:
        result = subprocess.run(
            [sys.executable, 'genetic_optimizer.py'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            print(f"[错误] 遗传算法优化失败")
            print(result.stderr)
            return False

        print("[完成] 遗传算法优化成功")

        # 提示用户应用优化后的参数
        print("\n[提示] 优化后的参数已保存到 strategy_params_optimized.json")
        print("[提示] 如需应用新参数，请将文件重命名为 strategy_params.json 或手动更新参数")

        return True

    except Exception as e:
        print(f"[错误] 执行遗传算法优化失败: {e}")
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

    # 阶段 4：遗传算法优化（新增）
    if not run_genetic_optimization():
        print("\n[⚠️ 警告] 遗传算法优化阶段失败")
        # 遗传算法失败不影响整体流程

    print("\n" + "="*80)
    print("【✅ 完成】完整流程执行完毕")
    print("="*80)
    print("\n数据文件：")
    print("  - 选股结果: DeepQuant_TopPicks_YYYYMMDD.csv")
    print("  - 验证记录: validation_records.csv")
    print("  - 模拟交易: paper_trading_records.csv")
    print("  - 参数配置: strategy_params.json")
    print("  - 参数历史: params_history.csv")
    print("  - 优化后参数: strategy_params_optimized.json (如有)")
    print("  - 优化历史: optimization_history.csv (如有)")
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
    print("    full    - 运行完整流程（选股 + 验证 + 优化 + 遗传算法）")
    print("    select  - 仅运行选股筛选")
    print("    validate- 仅运行验证跟踪更新")
    print("    optimize- 仅运行参数优化")
    print("    genetic - 仅运行遗传算法优化")
    print("\n  默认模式：full")
    print("\n示例：")
    print("  python main_controller.py           # 运行完整流程")
    print("  python main_controller.py validate  # 仅更新验证数据")
    print("  python main_controller.py optimize  # 仅优化参数")
    print("  python main_controller.py genetic   # 仅运行遗传算法优化")
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
    elif mode == 'genetic':
        print_banner()
        run_genetic_optimization()
    elif mode in ['help', '-h', '--help']:
        show_usage()
    else:
        print(f"[错误] 未知模式: {mode}")
        show_usage()


if __name__ == '__main__':
    main()
