# -*- coding: utf-8 -*-
"""
DeepQuant 遗传算法参数优化器 (Evolution Engine)
功能：读取验证记录 -> 模拟不同参数下的选股表现 -> 进化出最优参数
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import copy
import os

class GeneticOptimizer:
    def __init__(self, config_path="strategy_params.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}，请确保已创建Version 2.0配置文件")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 兼容性检查
        if 'genetic_algorithm' not in self.config:
            raise ValueError("配置文件缺少 'genetic_algorithm' 字段，请更新 strategy_params.json")

        self.ga_config = self.config['genetic_algorithm']
        self.population = []
        self.best_individual = None
        self.fitness_history = []
        
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.ga_config['population_size']):
            individual = self._create_random_individual()
            self.population.append(individual)
        return self.population
    
    def _create_random_individual(self) -> Dict:
        """创建随机变异的个体"""
        individual = copy.deepcopy(self.config)
        # 对三个核心维度进行随机扰动
        individual['scoring_weights'] = self._mutate_scoring_weights(individual['scoring_weights'])
        individual['indicators'] = self._mutate_indicators(individual['indicators'])
        individual['thresholds'] = self._mutate_thresholds(individual['thresholds'])
        return individual
    
    def _mutate_scoring_weights(self, weights: Dict) -> Dict:
        mutated = copy.deepcopy(weights)
        
        # 变异安全性 (Safety)
        # 随机调整基础分列表，例如 [25, 20...] -> [26, 19...]
        mutated['safety']['base_scores'] = [
            max(5, min(35, s + random.randint(-5, 5))) for s in mutated['safety']['base_scores']
        ]
        # 变异上限
        mutated['safety']['max_score'] = max(15, min(40, mutated['safety']['max_score'] + random.randint(-5, 5)))

        # 变异进攻性 (Offensive)
        for strategy in mutated['offensive']['strategy_base']:
            val = mutated['offensive']['strategy_base'][strategy]
            mutated['offensive']['strategy_base'][strategy] = max(0, min(25, val + random.randint(-3, 3)))
        
        # 变异上限
        mutated['offensive']['max_score'] = max(20, min(50, mutated['offensive']['max_score'] + random.randint(-5, 5)))
        
        # 变异其他项略... (保持结构完整即可)
        return mutated
    
    def _mutate_indicators(self, indicators: Dict) -> Dict:
        mutated = copy.deepcopy(indicators)
        # 均线周期变异
        mutated['ma_periods']['short'] = max(5, min(40, int(mutated['ma_periods']['short'] + random.randint(-5, 5))))
        mutated['ma_periods']['medium'] = max(30, min(120, int(mutated['ma_periods']['medium'] + random.randint(-10, 10))))
        return mutated
    
    def _mutate_thresholds(self, thresholds: Dict) -> Dict:
        mutated = copy.deepcopy(thresholds)
        # 评分门槛变异 (这是最重要的参数)
        mutated['SCORE_THRESHOLD_NORMAL'] = max(40, min(80, mutated['SCORE_THRESHOLD_NORMAL'] + random.randint(-5, 5)))
        mutated['SCORE_THRESHOLD_WASH'] = max(30, min(70, mutated['SCORE_THRESHOLD_WASH'] + random.randint(-5, 5)))
        return mutated

    def evaluate_fitness(self, individual: Dict, validation_records: pd.DataFrame) -> float:
        """计算适应度：使用新参数重跑历史数据"""
        if len(validation_records) == 0: return 0.0
        
        threshold_normal = individual['thresholds']['SCORE_THRESHOLD_NORMAL']
        threshold_wash = individual['thresholds']['SCORE_THRESHOLD_WASH']
        weights = individual['scoring_weights']

        # --- 核心逻辑：基于分项分重算总分 ---
        # 假设历史记录里保存了原始的 s_safe, s_off 等数值
        # 我们根据新旧权重的比例来调整分数 (模拟)
        
        # 获取旧配置中的最大值 (用于归一化)
        # 实际应用中建议将原始数据标准化，这里简化处理：
        # 新得分 = 原始分项分 * (新权重上限 / 旧权重上限)
        
        # 为防止除零错误，这里简化为直接叠加扰动
        def fast_recalc(row):
            # 这里是一个简化的估算模型
            # 实际上应该完全复现 calculate_score 函数的逻辑
            # 但为了速度，我们假设 s_safe 等字段是线性的
            
            # 安全分调整系数
            adj_safe = weights['safety']['max_score'] / 25.0 
            adj_off = weights['offensive']['max_score'] / 35.0
            
            new_score = (row['s_safe'] * adj_safe) + \
                        (row['s_off'] * adj_off) + \
                        row['s_cert'] + row['s_match'] # 假设这两项变动不大
            return new_score

        # 计算新分数
        recalc_scores = validation_records.apply(fast_recalc, axis=1)
        
        # 筛选：哪些股票在新的标准下会被选中？
        # 如果是洗盘策略用洗盘阈值，否则用正常阈值
        is_wash = validation_records['strategy'].astype(str).str.contains("洗盘")
        passed_mask = (is_wash & (recalc_scores >= threshold_wash)) | \
                      (~is_wash & (recalc_scores >= threshold_normal))
        
        selected_df = validation_records[passed_mask]
        
        if len(selected_df) < 5: 
            return 0.0 # 选不出股票，适应度为0
            
        # 计算指标
        win_rate = (selected_df['pct_5d'] > 0).mean()
        avg_return = selected_df['pct_5d'].mean()
        sharpe = avg_return / (selected_df['pct_5d'].std() + 1e-6)
        
        # 适应度公式 = 夏普比率 * 0.4 + 胜率 * 0.4 + 收益率 * 0.1
        # 并对选股数量过少进行惩罚
        fitness = (sharpe * 0.4) + (win_rate * 2.0) + (avg_return * 0.1)
        
        return max(0.0, fitness)

    def optimize(self, validation_records: pd.DataFrame):
        print(f"\n{'='*60}")
        print(f"[遗传算法] 启动进化引擎...")
        print(f"[遗传算法] 训练数据: {len(validation_records)} 条")
        
        self.initialize_population()
        
        generations = self.ga_config['generations']
        
        for gen in range(generations):
            fitness_scores = [self.evaluate_fitness(ind, validation_records) for ind in self.population]
            
            best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[best_idx]
            
            # 保存历史最佳
            if self.best_individual is None or current_best_fitness > self.fitness_history[-1]['best_fitness'] if self.fitness_history else 0:
                self.best_individual = copy.deepcopy(self.population[best_idx])
            
            avg_fit = np.mean(fitness_scores)
            self.fitness_history.append({'gen': gen, 'best_fitness': current_best_fitness, 'avg': avg_fit})
            
            print(f"[Gen {gen+1}/{generations}] Best Fitness: {current_best_fitness:.4f} | Avg: {avg_fit:.4f}")
            
            # 简单的选择和变异逻辑 (截断选择)
            # 保留前 30% 优秀的个体直接进入下一代
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_n = int(len(self.population) * 0.3)
            parents = [self.population[i] for i in sorted_indices[:top_n]]
            
            # 繁殖下一代
            new_pop = parents[:] # 精英保留
            while len(new_pop) < len(self.population):
                # 随机选一个父母进行变异
                parent = random.choice(parents)
                child = self._create_random_individual() # 这里简化为直接随机生成+父母特征混合
                # 简单混合: 取父母的阈值，但变异指标
                child['thresholds'] = parent['thresholds'] # 继承阈值
                child['thresholds'] = self._mutate_thresholds(child['thresholds']) # 再变异
                new_pop.append(child)
            
            self.population = new_pop
            
        return self.best_individual

    def save_best_params(self, output_path="strategy_params.json"):
        if not self.best_individual: return
        
        # 标记为已优化
        self.best_individual['optimized'] = True
        self.best_individual['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        self.best_individual['optimization_stats']['sharpe_ratio'] = self.fitness_history[-1]['best_fitness']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.best_individual, f, indent=2, ensure_ascii=False)
        print(f"[保存] 进化后的基因已写入: {output_path}")

def main():
    # 1. 加载数据
    try:
        df = pd.read_csv("validation_records.csv", encoding='utf-8-sig')
    except:
        print("未找到数据，请先运行 gen_test_data.py")
        return

    # 2. 运行优化
    optimizer = GeneticOptimizer()
    best_params = optimizer.optimize(df)
    
    # 3. 结果展示
    print("\n[进化完成] 最优参数对比:")
    print(f"原始阈值: {optimizer.config['thresholds']['SCORE_THRESHOLD_NORMAL']}")
    print(f"进化阈值: {best_params['thresholds']['SCORE_THRESHOLD_NORMAL']}")
    
    # 4. 保存
    optimizer.save_best_params()

if __name__ == "__main__":
    main()
