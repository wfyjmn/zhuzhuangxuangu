# DeepQuant 遗传算法参数优化系统

## 概述

DeepQuant 遗传算法参数优化系统是一个基于进化算法的智能参数调优工具，能够自动优化选股策略的各项权重和阈值参数，提升策略的盈利能力和稳定性。

## 核心功能

### 1. 参数化配置
- 将评分系统中的固定权重转化为可配置参数
- 支持安全性、进攻性、确定性、配合度四大评分维度的权重优化
- 支持均线周期、量比阈值、涨幅阈值等技术指标的优化

### 2. 遗传算法优化
- **种群初始化**：生成随机的参数组合作为初始种群
- **适应度评估**：基于夏普比率、胜率、平均收益率综合评估参数表现
- **选择操作**：采用锦标赛选择，保留优秀个体
- **交叉操作**：参数交叉，产生新的参数组合
- **变异操作**：随机变异参数，探索参数空间

### 3. 性能评估
- **夏普比率**：衡量风险调整后的收益
- **胜率**：衡量选股的准确性
- **平均收益率**：衡量选股的盈利能力

## 文件结构

```
assets/
├── genetic_optimizer.py              # 遗传算法核心引擎
├── backtest_engine.py               # 历史回测引擎
├── strategy_params.json             # 当前策略参数
├── strategy_params_optimized.json   # 优化后的策略参数
├── validation_records.csv           # 验证跟踪记录
├── optimization_history.csv         # 优化历史记录
├── main_controller.py               # 主控程序（已集成遗传算法）
└── scripts/
    └── generate_test_validation_data.py  # 测试数据生成脚本
```

## 使用方法

### 1. 准备验证数据

在使用遗传算法前，需要先积累足够的验证跟踪数据。建议至少50条历史选股记录。

```bash
# 运行完整的选股和验证流程
python main_controller.py full
```

### 2. 生成测试数据（用于测试）

如果只是测试遗传算法功能，可以生成模拟数据：

```bash
python scripts/generate_test_validation_data.py
```

### 3. 运行遗传算法优化

#### 方式一：直接运行优化器

```bash
python genetic_optimizer.py
```

#### 方式二：通过主控程序运行

```bash
# 运行完整流程（包含遗传算法优化）
python main_controller.py full

# 仅运行遗传算法优化
python main_controller.py genetic
```

### 4. 查看优化结果

优化完成后，会生成以下文件：

- **strategy_params_optimized.json**：优化后的参数配置
- **optimization_history.csv**：优化过程记录

## 参数配置说明

### 遗传算法参数

在 `strategy_params.json` 中配置：

```json
{
  "genetic_algorithm": {
    "population_size": 50,      # 种群大小
    "generations": 100,         # 最大迭代次数
    "mutation_rate": 0.1,       # 变异率
    "crossover_rate": 0.7,      # 交叉率
    "elite_size": 5,            # 精英个体数量
    "tournament_size": 3,       # 锦标赛选择大小
    "convergence_threshold": 0.001,    # 收敛阈值
    "stagnation_generations": 10       # 连续无提升的代数
  }
}
```

### 可优化参数

#### 评分权重

1. **安全性评分**
   - `base_scores`: 位置区间基础分 [25, 20, 15, 10]
   - `pos_thresholds`: 位置阈值 [0.2, 0.4, 0.6]
   - `low_vol_bonus`: 缩量加分
   - `max_score`: 安全分上限

2. **进攻性评分**
   - `strategy_base`: 策略基础分（强攻、梯量、洗盘）
   - `vol_ratio_bonus`: 量比加分
   - `pct_chg_bonus`: 涨幅加分
   - `wash_compensation`: 洗盘补偿分
   - `max_score`: 进攻分上限

3. **确定性评分**
   - `base_score`: 基础分
   - `vol_threshold`: 量比阈值
   - `vol_bonus`: 量比加分
   - `ma_above_bonus`: 均线加分
   - `max_score`: 确定分上限

4. **配合度评分**
   - `base_score`: 基础分
   - `strong_attack_bonus`: 强攻加分
   - `wash_bonus`: 洗盘加分
   - `max_score`: 配合分上限

#### 指标参数

- `ma_periods`: 均线周期（短期、中期）
- `vol_ma_period`: 量均线周期
- `lookback_period`: 回溯周期
- `min_data_days`: 最少数据天数

#### 阈值参数

- `SCORE_THRESHOLD_NORMAL`: 正常策略评分阈值
- `SCORE_THRESHOLD_WASH`: 洗盘策略评分阈值
- `TURNOVER_THRESHOLD_NORMAL`: 正常换手率阈值
- `TURNOVER_THRESHOLD_WASH`: 洗盘换手率阈值
- `TOP_N_PER_STRATEGY`: 每个策略选股数量

## 优化流程

```
1. 初始化种群
   ↓
2. 评估适应度
   ↓
3. 选择（锦标赛）
   ↓
4. 交叉（参数组合）
   ↓
5. 变异（随机调整）
   ↓
6. 更新种群
   ↓
7. 判断收敛
   ├─ 否 → 返回步骤2
   └─ 是 → 输出最优参数
```

## 适应度计算

适应度 = 夏普比率 × 0.5 + 胜率 × 100 × 0.3 + 平均收益率 × 0.2 - 选股数量惩罚

其中：
- **夏普比率**：衡量风险调整后的收益，越高越好
- **胜率**：衡量选股的准确性，越高越好
- **平均收益率**：衡量选股的盈利能力，越高越好
- **选股数量惩罚**：防止选股过多或过少

## 优化结果示例

优化前：
- 安全分上限：25
- 进攻分上限：35
- 确定分上限：25
- 配合分上限：15
- 评分阈值（正常）：70
- 评分阈值（洗盘）：65

优化后：
- 安全分上限：20
- 进攻分上限：29
- 确定分上限：22
- 配合分上限：19
- 评分阈值（正常）：74
- 评分阈值（洗盘）：70

适应度提升：31.09 → 38.85（+24.9%）

## 应用优化后的参数

### 方式一：替换配置文件

```bash
# 备份原配置
cp strategy_params.json strategy_params_backup.json

# 应用优化后的参数
cp strategy_params_optimized.json strategy_params.json
```

### 方式二：手动调整

打开 `strategy_params.json`，对比 `strategy_params_optimized.json`，手动调整参数。

## 注意事项

1. **数据量要求**：建议至少50条历史选股记录后再运行遗传算法
2. **优化频率**：建议每周或每月运行一次，避免过度优化
3. **参数范围**：算法会自动约束参数在合理范围内，避免极端值
4. **过拟合风险**：过度优化可能导致过拟合，建议使用交叉验证

## 常见问题

### Q: 适应度一直不变怎么办？

A: 检查以下几点：
1. 验证数据量是否足够（建议至少50条）
2. 验证数据是否有足够的多样性
3. 尝试增加变异率或调整种群大小

### Q: 优化速度很慢怎么办？

A: 可以通过以下方式提速：
1. 减少种群大小（population_size）
2. 减少迭代次数（generations）
3. 减少验证数据量

### Q: 优化后的参数表现不如预期？

A: 可能原因：
1. 历史数据不足，优化结果不具代表性
2. 市场环境变化，历史规律失效
3. 过度拟合，建议使用交叉验证

## 扩展开发

### 添加新的可优化参数

1. 在 `strategy_params.json` 中添加新参数
2. 在 `genetic_optimizer.py` 的 `_create_random_individual()` 中添加参数初始化逻辑
3. 在 `genetic_optimizer.py` 的 `_mutate_*()` 方法中添加变异逻辑
4. 在 `backtest_engine.py` 中应用新参数

### 自定义适应度函数

修改 `genetic_optimizer.py` 中的 `evaluate_fitness()` 方法，自定义适应度计算逻辑。

## 版本历史

- **v2.0** (2026-01-27)
  - 新增遗传算法参数优化系统
  - 实现参数化配置
  - 集成到主控程序

## 联系与反馈

如有问题或建议，请在 GitHub 仓库提交 Issue。
