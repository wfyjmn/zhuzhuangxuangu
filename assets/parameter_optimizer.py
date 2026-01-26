# -*- coding: utf-8 -*-
"""
参数优化模块 (Parameter Optimizer)
功能：
1. 分析验证数据，评估策略表现
2. 生成参数优化建议
3. 更新参数配置文件
4. 记录参数变更历史
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ================= 配置区域 =================
PARAMS_FILE = 'strategy_params.json'
VALIDATION_RECORDS_FILE = 'validation_records.csv'
PARAMS_HISTORY_FILE = 'params_history.csv'
# ===========================================


def load_params():
    """加载当前参数配置"""
    try:
        with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[错误] 加载参数配置失败: {e}")
        return None


def save_params(data):
    """保存参数配置"""
    try:
        with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[系统] 参数配置已更新")
        return True
    except Exception as e:
        print(f"[错误] 保存参数配置失败: {e}")
        return False


def load_validation_records():
    """加载验证记录"""
    if not os.path.exists(VALIDATION_RECORDS_FILE):
        print("[错误] 验证记录文件不存在")
        return pd.DataFrame()

    try:
        df = pd.read_csv(VALIDATION_RECORDS_FILE, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"[错误] 加载验证记录失败: {e}")
        return pd.DataFrame()


def record_params_change(old_params, new_params, notes=""):
    """记录参数变更历史"""
    # 创建历史记录
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    effective_date = datetime.now().strftime('%Y-%m-%d')

    # 读取现有历史
    if os.path.exists(PARAMS_HISTORY_FILE):
        df_history = pd.read_csv(PARAMS_HISTORY_FILE, encoding='utf-8-sig')
    else:
        df_history = pd.DataFrame()

    # 生成新版本号
    if df_history.empty:
        version = "1.0"
    else:
        last_version = df_history['version'].iloc[-1]
        # 简单的版本号递增
        major, minor = last_version.split('.')
        version = f"{major}.{int(minor) + 1}"

    # 记录参数变更
    record = {
        'version': version,
        'effective_date': effective_date,
        'change_type': 'optimization',
        'changed_by': 'auto_optimizer',
        'high_risk_pos': new_params.get('first_round', {}).get('HIGH_RISK_POS', ''),
        'strong_chg_pct': new_params.get('first_round', {}).get('STRONG_CHG_PCT', ''),
        'score_threshold_normal': new_params.get('second_round', {}).get('SCORE_THRESHOLD_NORMAL', ''),
        'score_threshold_wash': new_params.get('second_round', {}).get('SCORE_THRESHOLD_WASH', ''),
        'top_n_per_strategy': new_params.get('second_round', {}).get('TOP_N_PER_STRATEGY', ''),
        'enabled': new_params.get('optimization', {}).get('enabled', False),
        'notes': notes
    }

    df_history = pd.concat([df_history, pd.DataFrame([record])], ignore_index=True)
    df_history.to_csv(PARAMS_HISTORY_FILE, index=False, encoding='utf-8-sig')

    print(f"[系统] 参数变更已记录到历史文件，版本: {version}")


def analyze_strategy_performance(df_records):
    """分析策略表现"""
    if df_records.empty:
        return None

    # 只分析已完成验证的记录
    df_completed = df_records[df_records['status'] == 'completed'].copy()

    if df_completed.empty:
        print("[信息] 没有已完成的验证记录，无法进行性能分析")
        return None

    print("\n" + "="*80)
    print("【📈 策略性能分析】")
    print("="*80)

    # 总体统计
    total_count = len(df_completed)
    print(f"\n[总体统计]")
    print(f"  总样本数: {total_count}")

    # 按策略分组分析
    strategies = df_completed['strategy'].unique()
    strategy_performance = {}

    for strategy in strategies:
        df_strategy = df_completed[df_completed['strategy'] == strategy]

        performance = {
            'count': len(df_strategy),
            'day1_avg_return': df_strategy['day1_return'].mean() if 'day1_return' in df_strategy.columns else 0,
            'day3_avg_return': df_strategy['day3_return'].mean() if 'day3_return' in df_strategy.columns else 0,
            'day5_avg_return': df_strategy['day5_return'].mean() if 'day5_return' in df_strategy.columns else 0,
            'day1_win_rate': (df_strategy['day1_return'] > 0).sum() / len(df_strategy) * 100 if 'day1_return' in df_strategy.columns else 0,
            'day3_win_rate': (df_strategy['day3_return'] > 0).sum() / len(df_strategy) * 100 if 'day3_return' in df_strategy.columns else 0,
            'day5_win_rate': (df_strategy['day5_return'] > 0).sum() / len(df_strategy) * 100 if 'day5_return' in df_strategy.columns else 0,
            'max_drawdown': df_strategy['max_drawdown'].min() if 'max_drawdown' in df_strategy.columns else 0,
        }

        strategy_performance[strategy] = performance

        print(f"\n  策略: {strategy}")
        print(f"    样本数: {performance['count']}")
        print(f"    1天平均收益: {performance['day1_avg_return']:.2f}% | 胜率: {performance['day1_win_rate']:.1f}%")
        print(f"    3天平均收益: {performance['day3_avg_return']:.2f}% | 胜率: {performance['day3_win_rate']:.1f}%")
        print(f"    5天平均收益: {performance['day5_avg_return']:.2f}% | 胜率: {performance['day5_win_rate']:.1f}%")
        print(f"    最大回撤: {performance['max_drawdown']:.2f}%")

    return strategy_performance


def generate_optimization_suggestions(performance_data, current_params):
    """生成参数优化建议"""
    if not performance_data:
        return None

    print("\n" + "="*80)
    print("【💡 参数优化建议】")
    print("="*80)

    suggestions = []

    for strategy, perf in performance_data.items():
        # 检查胜率
        if perf['day5_win_rate'] < 40:
            suggestions.append({
                'target': strategy,
                'issue': '胜率过低',
                'metric': f'5天胜率 {perf["day5_win_rate"]:.1f}% < 40%',
                'suggestion': '建议提高筛选标准，如提高评分阈值或增加换手率要求'
            })

        # 检查平均收益
        if perf['day5_avg_return'] < 0:
            suggestions.append({
                'target': strategy,
                'issue': '收益为负',
                'metric': f'5天平均收益 {perf["day5_avg_return"]:.2f}% < 0%',
                'suggestion': '建议暂停该策略或重新调整选股条件'
            })

        # 检查最大回撤
        if perf['max_drawdown'] < -10:
            suggestions.append({
                'target': strategy,
                'issue': '回撤过大',
                'metric': f'最大回撤 {perf["max_drawdown"]:.2f}% < -10%',
                'suggestion': '建议加强止损设置，提高位置要求或降低仓位'
            })

    if not suggestions:
        print("\n  ✅ 当前策略表现良好，暂无优化建议")
        return None

    print("\n  发现以下问题：")
    for i, sug in enumerate(suggestions, 1):
        print(f"\n  [{i}] 策略: {sug['target']}")
        print(f"      问题: {sug['issue']}")
        print(f"      指标: {sug['metric']}")
        print(f"      建议: {sug['suggestion']}")

    return suggestions


def update_params_based_on_suggestions(suggestions, current_data):
    """根据建议更新参数（简化版）"""
    if not suggestions:
        return None

    print("\n" + "="*80)
    print("【⚙️ 参数更新】")
    print("="*80)

    params = current_data['params'].copy()
    changes_made = []

    # 简化的参数调整逻辑（后续可扩展更复杂的优化算法）
    for sug in suggestions:
        target = sug['target']

        if '洗盘' in target:
            # 洗盘策略：如果表现不好，提高评分阈值
            if '胜率过低' in sug['issue']:
                old_threshold = params['second_round']['SCORE_THRESHOLD_WASH']
                new_threshold = min(70, old_threshold + 5)  # 最高不超过70
                params['second_round']['SCORE_THRESHOLD_WASH'] = new_threshold
                changes_made.append(f"洗盘策略评分阈值: {old_threshold} -> {new_threshold}")

        elif '强攻' in target:
            # 强攻策略：如果收益为负，提高涨幅要求
            if '收益为负' in sug['issue']:
                old_threshold = params['first_round']['STRONG_CHG_PCT']
                new_threshold = min(5.0, old_threshold + 0.5)  # 最高不超过5%
                params['first_round']['STRONG_CHG_PCT'] = new_threshold
                changes_made.append(f"强攻涨幅阈值: {old_threshold}% -> {new_threshold}%")

        elif '梯量' in target:
            # 梯量策略：如果表现不好，减少选股数量
            if '胜率过低' in sug['issue']:
                old_n = params['second_round']['TOP_N_PER_STRATEGY']
                new_n = max(3, old_n - 1)  # 最少保留3只
                params['second_round']['TOP_N_PER_STRATEGY'] = new_n
                changes_made.append(f"策略选股数量: {old_n} -> {new_n}")

    if not changes_made:
        print("\n  ℹ️  未生成有效的参数调整方案")
        return None

    print("\n  拟进行的参数调整：")
    for change in changes_made:
        print(f"    • {change}")

    # 更新版本和时间
    params['version'] = f"{float(params['version']) + 0.1:.1f}"
    params['last_updated'] = datetime.now().strftime('%Y-%m-%d')

    return params


def run_optimizer():
    """运行参数优化流程"""
    print("="*80)
    print("   DeepQuant 参数优化系统")
    print("="*80)

    # 1. 加载当前参数
    print("\n[步骤 1] 加载当前参数配置...")
    current_data = load_params()
    if not current_data:
        print("[错误] 无法加载参数配置")
        return

    # 检查优化是否启用
    if not current_data['params'].get('optimization', {}).get('enabled', False):
        print("[信息] 参数优化功能未启用")
        print("提示：在 strategy_params.json 中设置 \"optimization.enabled\": true 以启用")
        return

    # 2. 加载验证记录
    print("\n[步骤 2] 加载验证记录...")
    df_records = load_validation_records()
    if df_records.empty:
        print("[错误] 没有验证记录，无法进行优化")
        return

    # 检查样本量
    df_completed = df_records[df_records['status'] == 'completed']
    min_records = current_data['params'].get('optimization', {}).get('MIN_RECORDS', 30)

    if len(df_completed) < min_records:
        print(f"[信息] 验证记录不足（{len(df_completed)} < {min_records}），建议继续积累数据")
        return

    # 3. 分析策略表现
    print("\n[步骤 3] 分析策略表现...")
    performance_data = analyze_strategy_performance(df_records)

    if not performance_data:
        return

    # 4. 生成优化建议
    print("\n[步骤 4] 生成优化建议...")
    suggestions = generate_optimization_suggestions(performance_data, current_data['params'])

    if not suggestions:
        return

    # 5. 更新参数
    print("\n[步骤 5] 更新参数配置...")
    new_params = update_params_based_on_suggestions(suggestions, current_data)

    if not new_params:
        return

    # 保存参数
    if save_params(new_params):
        # 记录变更历史
        notes = "基于验证数据的自动优化"
        for sug in suggestions:
            notes += f"\n- {sug['target']}: {sug['issue']}"

        record_params_change(current_data['params'], new_params['params'], notes)

        print("\n[✅ 完成] 参数优化流程已完成")
        print("\n  请在下次选股时使用新参数运行筛选程序")
    else:
        print("\n[❌ 失败] 参数保存失败")


if __name__ == '__main__':
    run_optimizer()
