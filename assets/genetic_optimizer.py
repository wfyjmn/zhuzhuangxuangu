#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepQuant 遗传算法参数优化器
用于自动优化选股策略的权重和阈值参数
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import copy

# ============================================================================
# 遗传算法核心类
# ============================================================================

class GeneticOptimizer:
    """遗传算法参数优化器"""
    
    def __init__(self, config_path="strategy_params.json"):
        """
        初始化优化器
        
        Args:
            config_path: 策略参数配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.ga_config = self.config['genetic_algorithm']
        self.population = []
        self.best_individual = None
        self.fitness_history = []
        
        print(f"[遗传算法] 初始化完成")
        print(f"[遗传算法] 种群大小: {self.ga_config['population_size']}")
        print(f"[遗传算法] 迭代次数: {self.ga_config['generations']}")
        print(f"[遗传算法] 变异率: {self.ga_config['mutation_rate']}")
        print(f"[遗传算法] 交叉率: {self.ga_config['crossover_rate']}")
    
    def initialize_population(self):
        """初始化种群"""
        print(f"\n[遗传算法] 开始初始化种群...")
        self.population = []
        
        for i in range(self.ga_config['population_size']):
            individual = self._create_random_individual()
            self.population.append(individual)
        
        print(f"[遗传算法] 种群初始化完成，共 {len(self.population)} 个个体")
        return self.population
    
    def _create_random_individual(self) -> Dict:
        """创建随机个体（随机参数组合）"""
        individual = copy.deepcopy(self.config)
        
        # 1. 随机化评分权重
        individual['scoring_weights'] = self._mutate_scoring_weights(
            individual['scoring_weights']
        )
        
        # 2. 随机化指标参数
        individual['indicators'] = self._mutate_indicators(
            individual['indicators']
        )
        
        # 3. 随机化阈值
        individual['thresholds'] = self._mutate_thresholds(
            individual['thresholds']
        )
        
        return individual
    
    def _mutate_scoring_weights(self, weights: Dict) -> Dict:
        """变异评分权重"""
        mutated = copy.deepcopy(weights)
        
        # 变异安全性权重
        mutated['safety']['base_scores'] = [
            max(15, min(30, s + random.randint(-5, 5)))
            for s in mutated['safety']['base_scores']
        ]
        mutated['safety']['low_vol_bonus'] = max(0, min(5, 
            mutated['safety']['low_vol_bonus'] + random.randint(-1, 1)))
        mutated['safety']['max_score'] = min(40, max(20, 
            mutated['safety']['max_score'] + random.randint(-5, 5)))
        
        # 变异进攻性权重
        for strategy in mutated['offensive']['strategy_base']:
            mutated['offensive']['strategy_base'][strategy] = max(5, min(20,
                mutated['offensive']['strategy_base'][strategy] + random.randint(-3, 3)))
        
        for bonus in mutated['offensive']['vol_ratio_bonus']:
            bonus['score'] = max(3, min(15, 
                bonus['score'] + random.randint(-2, 2)))
        
        for bonus in mutated['offensive']['pct_chg_bonus']:
            bonus['score'] = max(2, min(15, 
                bonus['score'] + random.randint(-2, 2)))
        
        if mutated['offensive']['wash_compensation']['enabled']:
            mutated['offensive']['wash_compensation']['score'] = max(3, min(15,
                mutated['offensive']['wash_compensation']['score'] + random.randint(-3, 3)))
        
        mutated['offensive']['max_score'] = min(50, max(20,
            mutated['offensive']['max_score'] + random.randint(-5, 5)))
        
        # 变异确定性权重
        mutated['certainty']['base_score'] = max(5, min(15,
            mutated['certainty']['base_score'] + random.randint(-2, 2)))
        mutated['certainty']['vol_threshold'] = max(1.0, min(2.5,
            mutated['certainty']['vol_threshold'] + random.uniform(-0.2, 0.2)))
        mutated['certainty']['vol_bonus'] = max(2, min(10,
            mutated['certainty']['vol_bonus'] + random.randint(-2, 2)))
        mutated['certainty']['ma_above_bonus'] = max(5, min(15,
            mutated['certainty']['ma_above_bonus'] + random.randint(-2, 2)))
        mutated['certainty']['max_score'] = min(40, max(15,
            mutated['certainty']['max_score'] + random.randint(-5, 5)))
        
        # 变异配合度权重
        mutated['match']['base_score'] = max(5, min(15,
            mutated['match']['base_score'] + random.randint(-2, 2)))
        mutated['match']['strong_attack_bonus']['score'] = max(2, min(10,
            mutated['match']['strong_attack_bonus']['score'] + random.randint(-2, 2)))
        mutated['match']['wash_bonus']['score'] = max(2, min(10,
            mutated['match']['wash_bonus']['score'] + random.randint(-2, 2)))
        mutated['match']['max_score'] = min(25, max(10,
            mutated['match']['max_score'] + random.randint(-3, 3)))
        
        return mutated
    
    def _mutate_indicators(self, indicators: Dict) -> Dict:
        """变异指标参数"""
        mutated = copy.deepcopy(indicators)
        
        # 变异均线周期
        mutated['ma_periods']['short'] = max(5, min(60,
            int(mutated['ma_periods']['short'] + random.randint(-5, 5))))
        mutated['ma_periods']['medium'] = max(20, min(120,
            int(mutated['ma_periods']['medium'] + random.randint(-10, 10))))
        
        # 变异量均线周期
        mutated['vol_ma_period'] = max(3, min(10,
            int(mutated['vol_ma_period'] + random.randint(-1, 1))))
        
        return mutated
    
    def _mutate_thresholds(self, thresholds: Dict) -> Dict:
        """变异阈值"""
        mutated = copy.deepcopy(thresholds)
        
        # 变异评分阈值
        mutated['SCORE_THRESHOLD_NORMAL'] = max(50, min(90,
            mutated['SCORE_THRESHOLD_NORMAL'] + random.randint(-5, 5)))
        mutated['SCORE_THRESHOLD_WASH'] = max(45, min(85,
            mutated['SCORE_THRESHOLD_WASH'] + random.randint(-5, 5)))
        
        # 变异换手率阈值
        mutated['TURNOVER_THRESHOLD_NORMAL'] = max(5, min(20,
            int(mutated['TURNOVER_THRESHOLD_NORMAL'] + random.randint(-2, 2))))
        mutated['TURNOVER_THRESHOLD_WASH'] = max(2, min(10,
            int(mutated['TURNOVER_THRESHOLD_WASH'] + random.randint(-1, 1))))
        
        # 变异选股数量
        mutated['TOP_N_PER_STRATEGY'] = max(5, min(20,
            int(mutated['TOP_N_PER_STRATEGY'] + random.randint(-2, 2))))
        
        return mutated
    
    def evaluate_fitness(self, individual: Dict, validation_records: pd.DataFrame) -> float:
        """
        计算个体适应度（基于夏普比率和胜率）
        
        Args:
            individual: 参数个体
            validation_records: 验证记录数据
            
        Returns:
            适应度值
        """
        if len(validation_records) == 0:
            return 0.0
        
        # 1. 使用新的参数重新计算每只股票的评分
        threshold_normal = individual['thresholds']['SCORE_THRESHOLD_NORMAL']
        threshold_wash = individual['thresholds']['SCORE_THRESHOLD_WASH']
        
        # 重新计算评分（使用新的权重参数）
        # 注意：这里需要完整的日线数据才能重新计算，简化起见，我们使用原始记录中的分项得分
        # 并用新的权重重新计算总分
        
        def recalc_score(row, weights):
            """使用新权重重新计算总分"""
            # 调整后的分项得分（按比例缩放以匹配新上限）
            s_safe = min(weights['safety']['max_score'], 
                        row['s_safe'] * weights['safety']['max_score'] / 25)
            s_off = min(weights['offensive']['max_score'],
                       row['s_off'] * weights['offensive']['max_score'] / 35)
            s_cert = min(weights['certainty']['max_score'],
                        row['s_cert'] * weights['certainty']['max_score'] / 25)
            s_match = min(weights['match']['max_score'],
                         row['s_match'] * weights['match']['max_score'] / 15)
            return s_safe + s_off + s_cert + s_match
        
        weights = individual['scoring_weights']
        validation_records['recalc_score'] = validation_records.apply(
            lambda row: recalc_score(row, weights), axis=1
        )
        
        # 2. 根据新评分筛选股票
        selected_stocks = []
        for _, row in validation_records.iterrows():
            strategy = row['strategy']
            if "洗盘" in str(strategy):
                threshold = threshold_wash
            else:
                threshold = threshold_normal
            
            if row['recalc_score'] >= threshold:
                selected_stocks.append(row)
        
        if len(selected_stocks) == 0:
            return 0.0
        
        selected_df = pd.DataFrame(selected_stocks)
        
        # 3. 计算筛选后股票的表现
        sharpe = self._calculate_sharpe_ratio(selected_df)
        win_rate = self._calculate_win_rate(selected_df)
        avg_return = selected_df['pct_5d'].mean()
        
        # 4. 综合评分（夏普比率权重最高，同时考虑选股数量）
        # 惩罚选股数量过少或过多的情况
        selection_count = len(selected_df)
        if selection_count < 10:
            count_penalty = (10 - selection_count) / 10 * 10  # 选股太少惩罚
        elif selection_count > 50:
            count_penalty = (selection_count - 50) / 50 * 5   # 选股太多惩罚
        else:
            count_penalty = 0
        
        fitness = sharpe * 0.5 + win_rate * 100 * 0.3 + avg_return * 0.2 - count_penalty
        
        return max(0, fitness)  # 确保非负
    
    def _calculate_sharpe_ratio(self, records: pd.DataFrame) -> float:
        """计算夏普比率"""
        if len(records) < 2:
            return 0.0
        
        returns = records['pct_5d'].values  # 使用5日收益率
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # 年化收益率（假设252个交易日）
        avg_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
        
        sharpe = avg_return / std_return
        return sharpe
    
    def _calculate_win_rate(self, records: pd.DataFrame) -> float:
        """计算胜率"""
        if len(records) == 0:
            return 0.0
        
        # 使用5日收益率计算胜率
        winning_trades = (records['pct_5d'] > 0).sum()
        total_trades = len(records)
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def selection(self, fitness_scores: List[float]) -> List[Dict]:
        """
        锦标赛选择
        
        Args:
            fitness_scores: 适应度列表
            
        Returns:
            选择的个体列表
        """
        selected = []
        tournament_size = self.ga_config['tournament_size']
        
        # 保留精英个体
        elite_indices = np.argsort(fitness_scores)[-self.ga_config['elite_size']:][::-1]
        for idx in elite_indices:
            selected.append(self.population[idx])
        
        # 锦标赛选择剩余个体
        while len(selected) < len(self.population):
            # 随机选择tournament_size个个体
            tournament_indices = random.sample(
                range(len(self.population)), 
                min(tournament_size, len(self.population))
            )
            # 选择适应度最高的
            winner_idx = tournament_indices[
                np.argmax([fitness_scores[i] for i in tournament_indices])
            ]
            selected.append(self.population[winner_idx])
        
        return selected
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        交叉操作（参数交叉）
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            子代1, 子代2
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        if random.random() < self.ga_config['crossover_rate']:
            # 评分权重交叉
            child1['scoring_weights'], child2['scoring_weights'] = self._crossover_dict(
                parent1['scoring_weights'], 
                parent2['scoring_weights']
            )
            
            # 指标参数交叉
            child1['indicators'], child2['indicators'] = self._crossover_dict(
                parent1['indicators'], 
                parent2['indicators']
            )
            
            # 阈值交叉
            child1['thresholds'], child2['thresholds'] = self._crossover_dict(
                parent1['thresholds'], 
                parent2['thresholds']
            )
        
        return child1, child2
    
    def _crossover_dict(self, dict1: Dict, dict2: Dict) -> Tuple[Dict, Dict]:
        """字典交叉（随机交换部分参数）"""
        new_dict1 = copy.deepcopy(dict1)
        new_dict2 = copy.deepcopy(dict2)
        
        for key in dict1.keys():
            if isinstance(dict1[key], dict):
                # 递归交叉
                new_dict1[key], new_dict2[key] = self._crossover_dict(
                    dict1[key], dict2[key]
                )
            elif isinstance(dict1[key], list):
                # 列表交叉：随机选择交换点
                if len(dict1[key]) > 1 and random.random() < 0.5:
                    new_dict1[key] = copy.deepcopy(dict2[key])
                    new_dict2[key] = copy.deepcopy(dict1[key])
            elif isinstance(dict1[key], (int, float)) and random.random() < 0.5:
                # 数值交叉：简单交换
                new_dict1[key], new_dict2[key] = dict2[key], dict1[key]
        
        return new_dict1, new_dict2
    
    def mutate(self, individual: Dict) -> Dict:
        """
        变异操作
        
        Args:
            individual: 待变异个体
            
        Returns:
            变异后个体
        """
        mutated = copy.deepcopy(individual)
        
        if random.random() < self.ga_config['mutation_rate']:
            # 变异评分权重
            mutated['scoring_weights'] = self._mutate_scoring_weights(
                mutated['scoring_weights']
            )
        
        if random.random() < self.ga_config['mutation_rate']:
            # 变异指标参数
            mutated['indicators'] = self._mutate_indicators(
                mutated['indicators']
            )
        
        if random.random() < self.ga_config['mutation_rate']:
            # 变异阈值
            mutated['thresholds'] = self._mutate_thresholds(
                mutated['thresholds']
            )
        
        return mutated
    
    def optimize(self, validation_records: pd.DataFrame) -> Dict:
        """
        执行遗传算法优化
        
        Args:
            validation_records: 历史验证记录（用于计算适应度）
            
        Returns:
            最优参数个体
        """
        print(f"\n{'='*60}")
        print(f"[遗传算法] 开始参数优化...")
        print(f"[遗传算法] 验证数据量: {len(validation_records)} 条")
        print(f"{'='*60}")
        
        # 初始化种群
        if not self.population:
            self.initialize_population()
        
        # 迭代优化
        best_fitness = 0
        stagnation_count = 0
        
        for generation in range(self.ga_config['generations']):
            # 评估适应度
            fitness_scores = []
            for individual in self.population:
                fitness = self.evaluate_fitness(individual, validation_records)
                fitness_scores.append(fitness)
            
            # 记录最优个体
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                self.best_individual = copy.deepcopy(self.population[current_best_idx])
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            self.fitness_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'current_best': current_best_fitness
            })
            
            # 打印进度
            if (generation + 1) % 10 == 0 or generation == 0:
                print(f"\n[遗传算法] 第 {generation + 1}/{self.ga_config['generations']} 代")
                print(f"  - 最优适应度: {best_fitness:.4f}")
                print(f"  - 平均适应度: {np.mean(fitness_scores):.4f}")
                print(f"  - 当前最优: {current_best_fitness:.4f}")
            
            # 检查收敛
            if stagnation_count >= self.ga_config['stagnation_generations']:
                print(f"\n[遗传算法] 连续 {stagnation_count} 代无提升，提前终止优化")
                break
            
            # 选择
            selected = self.selection(fitness_scores)
            
            # 交叉和变异
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群
            self.population = new_population[:self.ga_config['population_size']]
        
        print(f"\n{'='*60}")
        print(f"[遗传算法] 优化完成！")
        print(f"[遗传算法] 最优适应度: {best_fitness:.4f}")
        print(f"[遗传算法] 优化代数: {len(self.fitness_history)}")
        print(f"{'='*60}\n")
        
        return self.best_individual
    
    def save_best_params(self, output_path="strategy_params_optimized.json"):
        """保存最优参数"""
        if self.best_individual is None:
            print("[错误] 没有最优参数可保存")
            return
        
        # 更新元数据
        self.best_individual['optimized'] = True
        self.best_individual['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.best_individual['optimization_stats']['generation'] = len(self.fitness_history)
        self.best_individual['optimization_stats']['sharpe_ratio'] = self.fitness_history[-1]['best_fitness']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.best_individual, f, indent=2, ensure_ascii=False)
        
        print(f"[保存] 最优参数已保存到: {output_path}")
    
    def export_fitness_history(self, output_path="optimization_history.csv"):
        """导出优化历史"""
        if not self.fitness_history:
            print("[警告] 没有优化历史可导出")
            return
        
        df = pd.DataFrame(self.fitness_history)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[保存] 优化历史已保存到: {output_path}")


# ============================================================================
# 工具函数
# ============================================================================

def load_validation_records(filepath="validation_records.csv") -> pd.DataFrame:
    """加载验证记录"""
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        # 过滤有效数据
        df = df[df['pct_5d'].notna()]
        print(f"[数据] 加载验证记录: {len(df)} 条")
        return df
    except FileNotFoundError:
        print(f"[错误] 验证记录文件不存在: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[错误] 加载验证记录失败: {e}")
        return pd.DataFrame()


def main():
    """主函数 - 演示使用"""
    # 1. 加载历史验证记录
    validation_records = load_validation_records("validation_records.csv")
    
    if len(validation_records) == 0:
        print("[提示] 没有历史验证数据，生成模拟数据用于演示...")
        # 生成模拟数据
        np.random.seed(42)
        validation_records = pd.DataFrame({
            'ts_code': [f'00000{i}.SZ' for i in range(100)],
            'strategy': ['强攻'] * 50 + ['洗盘'] * 50,
            'score': np.random.uniform(60, 90, 100),
            'pct_1d': np.random.uniform(-5, 8, 100),
            'pct_3d': np.random.uniform(-8, 12, 100),
            'pct_5d': np.random.uniform(-10, 15, 100)
        })
    
    # 2. 创建优化器
    optimizer = GeneticOptimizer("strategy_params.json")
    
    # 3. 执行优化
    best_params = optimizer.optimize(validation_records)
    
    # 4. 保存结果
    optimizer.save_best_params("strategy_params_optimized.json")
    optimizer.export_fitness_history("optimization_history.csv")
    
    # 5. 打印最优参数
    print("\n[最优参数] 关键权重配置:")
    print(f"  - 安全分上限: {best_params['scoring_weights']['safety']['max_score']}")
    print(f"  - 进攻分上限: {best_params['scoring_weights']['offensive']['max_score']}")
    print(f"  - 确定分上限: {best_params['scoring_weights']['certainty']['max_score']}")
    print(f"  - 配合分上限: {best_params['scoring_weights']['match']['max_score']}")
    print(f"  - 评分阈值(正常): {best_params['thresholds']['SCORE_THRESHOLD_NORMAL']}")
    print(f"  - 评分阈值(洗盘): {best_params['thresholds']['SCORE_THRESHOLD_WASH']}")


if __name__ == "__main__":
    main()
