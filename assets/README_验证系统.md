# DeepQuant 验证与优化系统使用指南

## 系统概述

本系统为柱形选股策略提供了完整的**验证跟踪**和**参数优化**闭环功能，实现了：

1. **前瞻性测试**：跟踪选股后1天、3天、5天的实际表现
2. **模拟交易**：记录每次选股的虚拟交易
3. **验证分析**：自动计算收益率、胜率、最大回撤等指标
4. **参数优化**：根据验证结果自动调整策略参数
5. **闭环系统**：选股 → 验证 → 优化 → 选股的持续改进

---

## 文件结构

```
assets/
├── strategy_params.json          # 策略参数配置文件
├── validation_records.csv        # 验证跟踪记录
├── paper_trading_records.csv     # 模拟交易记录
├── params_history.csv            # 参数变更历史
├── validation_track.py           # 验证跟踪程序
├── parameter_optimizer.py        # 参数优化程序
├── main_controller.py            # 主控程序
├── 柱形选股-筛选.py              # 第1轮筛选（已修改）
├── 柱形选股-第2轮.py            # 第2轮筛选（已修改）
└── DeepQuant_TopPicks_YYYYMMDD.csv  # 选股结果文件
```

---

## 快速开始

### 1. 首次使用

```bash
# 安装依赖
pip install tushare

# 运行完整流程（选股 + 验证）
python main_controller.py full
```

### 2. 日常更新验证数据

```bash
# 仅更新验证跟踪数据（每日运行）
python main_controller.py validate

# 或直接使用验证程序
python validation_track.py update
```

### 3. 周期性参数优化

```bash
# 运行参数优化（每周运行）
python main_controller.py optimize

# 或直接使用优化程序
python parameter_optimizer.py
```

---

## 核心程序说明

### 1. 主控程序（main_controller.py）

协调各模块运行，提供统一的入口：

```bash
python main_controller.py [mode]

模式说明：
  full     - 运行完整流程（选股 + 验证 + 优化）
  select   - 仅运行选股筛选
  validate - 仅运行验证跟踪更新
  optimize - 仅运行参数优化
```

### 2. 验证跟踪程序（validation_track.py）

功能：
- 扫描选股结果文件，创建验证记录
- 获取后续交易日数据，计算收益率
- 生成验证报告

```bash
python validation_track.py [mode]

模式说明：
  scan    - 扫描新的选股结果文件
  update  - 更新现有验证记录
  report  - 生成验证报告
  all     - 执行全部流程（默认）
```

### 3. 参数优化程序（parameter_optimizer.py）

功能：
- 分析验证数据，评估策略表现
- 生成参数优化建议
- 更新参数配置文件
- 记录参数变更历史

```bash
python parameter_optimizer.py
```

---

## 数据文件说明

### 1. 策略参数配置（strategy_params.json）

```json
{
  "version": "1.0",
  "last_updated": "2026-01-23",
  "params": {
    "first_round": {
      "HIGH_RISK_POS": 0.8,
      "STRONG_CHG_PCT": 2.5,
      ...
    },
    "second_round": {
      "SCORE_THRESHOLD_NORMAL": 55,
      "SCORE_THRESHOLD_WASH": 45,
      ...
    },
    "validation": {
      "TRACK_DAYS": [1, 3, 5],
      "MAX_POSITION_PER_STOCK": 10,
      ...
    },
    "optimization": {
      "MIN_RECORDS": 30,
      "enabled": false
    }
  }
}
```

**重要**：启用参数优化需要设置 `"optimization.enabled": true`

### 2. 验证跟踪记录（validation_records.csv）

字段说明：
- `record_id`: 记录ID（股票代码_选股日期）
- `ts_code`: 股票代码
- `pick_date`: 选股日期
- `strategy`: 选股策略
- `buy_price`: 买入价格
- `day1_return`: 1天收益率
- `day3_return`: 3天收益率
- `day5_return`: 5天收益率
- `max_drawdown`: 最大回撤
- `status`: 验证状态（validating/completed）

### 3. 模拟交易记录（paper_trading_records.csv）

字段说明：
- `trade_date`: 交易日期
- `ts_code`: 股票代码
- `strategy`: 选股策略
- `action`: 交易动作（BUY/SELL）
- `price`: 交易价格
- `quantity`: 交易数量
- `stop_loss`: 止损价
- `status`: 交易状态（open/closed）

### 4. 参数历史记录（params_history.csv）

记录所有参数变更历史，方便追踪和回溯。

---

## 使用流程

### 完整闭环流程

```
1. 运行选股
   ↓
   生成 DeepQuant_TopPicks_YYYYMMDD.csv

2. 验证跟踪（自动）
   ↓
   创建验证记录 + 模拟交易记录

3. 每日更新验证
   ↓
   计算实际收益率

4. 周期性优化（可选）
   ↓
   分析策略表现 → 调整参数

5. 下一轮选股
   ↓
   使用优化后的参数
```

### 推荐使用节奏

| 操作 | 频率 | 命令 |
|------|------|------|
| 选股 | 每日 | `python main_controller.py select` |
| 验证更新 | 每日 | `python main_controller.py validate` |
| 参数优化 | 每周 | `python main_controller.py optimize` |
| 完整流程 | 每日 | `python main_controller.py full` |

---

## 验证报告示例

```
================================================================================
【📊 验证报告】
================================================================================

[总体概况]
  总记录数: 150
  已完成验证: 120
  验证中: 30

[策略表现（5天收益率）]
  策略: ★低位强攻
    样本数: 45
    1天平均收益: 2.35% | 胜率: 68.9%
    3天平均收益: 3.12% | 胜率: 71.1%
    5天平均收益: 4.56% | 胜率: 73.3%
    最大回撤: -8.20%

  策略: ☆缩量洗盘
    样本数: 40
    1天平均收益: 0.85% | 胜率: 62.5%
    3天平均收益: 1.52% | 胜率: 65.0%
    5天平均收益: 2.85% | 胜率: 67.5%
    最大回撤: -5.60%
```

---

## 参数优化机制

### 优化触发条件

1. 验证记录数量达到阈值（默认30条）
2. 参数优化功能启用（`optimization.enabled = true`）

### 优化逻辑

系统会自动检测以下问题并调整参数：

- **胜率过低**（< 40%）：提高筛选标准，如提高评分阈值
- **收益为负**：暂停策略或调整选股条件
- **回撤过大**（< -10%）：加强止损设置或降低仓位

### 优化记录

所有参数变更都会记录到 `params_history.csv`，方便追溯：

```
version, effective_date, change_type, changed_by, notes
1.1, 2026-01-27, optimization, auto_optimizer, 洗盘策略胜率过低
```

---

## 注意事项

### 1. 时间格式统一

所有日期使用 `YYYYMMDD` 格式（字符串），例如：
- 选股日期：`20260123`
- 文件名：`DeepQuant_TopPicks_20260123.csv`

### 2. CSV 编码

所有 CSV 文件使用 `utf-8-sig` 编码，确保 Excel 正确打开。

### 3. 数据连续性

- 确保选股后持续运行验证更新
- 建议设置定时任务每日运行验证程序

### 4. 参数调优

- 初期建议关闭参数优化（`optimization.enabled = false`）
- 积累足够数据（至少30条记录）后再启用优化
- 定期查看 `params_history.csv` 追踪参数变化

---

## 常见问题

### Q1: 验证记录显示为空？

**A**: 检查是否有选股结果文件（`DeepQuant_TopPicks_*.csv`），并运行 `python validation_track.py scan`

### Q2: 参数优化没有生效？

**A**:
1. 检查 `optimization.enabled` 是否设置为 `true`
2. 确认验证记录数量是否达到阈值（默认30条）
3. 查看是否有优化建议生成

### Q3: 如何手动调整参数？

**A**: 直接编辑 `strategy_params.json`，修改对应参数值。下次选股时会自动生效。

### Q4: 如何查看历史参数？

**A**: 查看 `params_history.csv` 文件，包含所有参数变更记录。

---

## 扩展功能

### 1. 添加自定义指标

在 `validation_track.py` 的 `update_validation_records` 函数中添加新的计算逻辑。

### 2. 自定义优化策略

在 `parameter_optimizer.py` 的 `update_params_based_on_suggestions` 函数中添加自定义调整逻辑。

### 3. 集成实时提醒

在验证报告中添加异常情况的告警机制。

---

## 技术支持

如有问题，请检查：
1. 日志输出
2. 数据文件格式
3. 参数配置是否正确

---

## 版本历史

- **V3.0** (2026-01-27)
  - 新增验证跟踪系统
  - 新增参数优化模块
  - 新增主控程序
  - 实现完整闭环系统

- **V2.1** (原有系统)
  - 两轮筛选机制
  - 评分系统
  - 风控双轨制

---

## 许可证

本系统仅供学习和研究使用。
