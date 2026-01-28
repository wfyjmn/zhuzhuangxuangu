# DeepQuant 智能选股系统 - 文件说明文档

**文档版本**：v3.0
**更新日期**：2026-01-29
**系统版本**：V3.0

---

## 📋 目录

1. [核心选股程序](#核心选股程序)
2. [多因子模型](#多因子模型)
3. [天气预报系统](#天气预报系统)
4. [遗传算法优化](#遗传算法优化)
5. [配置文件](#配置文件)
6. [文档文件](#文档文件)
7. [结果文件](#结果文件)
8. [辅助模块](#辅助模块)
9. [测试与验证](#测试与验证)

---

## 核心选股程序

### 主程序

| 文件名 | 版本 | 状态 | 说明 |
|--------|------|------|------|
| `柱形选股-筛选.py` | v2.0 | ✅ 生产 | 第1轮筛选程序，从全市场扫描候选池 |
| `柱形选股-第2轮.py` | v2.1 | ✅ 生产 | 第2轮筛选程序，多因子评分和最终选股 |
| `main_controller.py` | v3.0 | ✅ 生产 | 主控程序，协调各模块运行 |

**功能说明**：
- **第1轮筛选**：从3000+股票中扫描候选池（约400-500只）
- **第2轮筛选**：候选池中进行多因子评分，最终选出15只精选股
- **主控程序**：集成天气预报系统，根据市场环境调整策略

---

## 多因子模型

### 核心文件

| 文件名 | 版本 | 状态 | 说明 |
|--------|------|------|------|
| `multi_factor_model.py` | v2.1 | ✅ 生产 | 多因子选股模型（当前使用） |
| `multi_factor_model_v1.0_backup.py` | v1.0 | 🔒 备份 | 原版本备份 |
| `multi_factor_model_v2.1.py` | v2.1 | 🔒 源文件 | 新版本源文件 |

**功能说明**：
- **资金流因子**（30%）：主力净流入、北向资金
- **板块共振因子**（20%）：板块涨幅、热门板块
- **技术形态因子**（50%）：K线形态、均线排列

**性能优化**：
- API调用次数：↓ 98.3%（884次 → 15次）
- 运行时间：↓ 67%（5分钟 → 1分40秒）
- 板块识别正确率：100%

---

## 天气预报系统

### 核心文件

| 文件名 | 版本 | 状态 | 说明 |
|--------|------|------|------|
| `market_weather.py` | v1.0 | ✅ 生产 | 天气预报系统（大势择时） |

**功能说明**：
- **指数趋势分析**：上证指数、深证成指的技术分析
- **市场情绪计算**：赚钱效应、跌停家数
- **策略调整建议**：
  - 🌞 晴天：积极进攻
  - ☁️ 阴天：适度参与
  - 🌧️ 小雨：谨慎防守
  - ⛈️ 暴雨：空仓休息

**调整范围**：
- 评分阈值（+15分、+10分、-5分）
- 选股策略（关闭强攻策略）
- 选股数量

---

## 遗传算法优化

### 核心文件

| 文件名 | 版本 | 状态 | 说明 |
|--------|------|------|------|
| `genetic_optimizer.py` | v1.0 | ✅ 生产 | 遗传算法优化器 |
| `run_genetic_optimization.py` | v1.0 | ✅ 生产 | 遗传算法运行脚本 |
| `demo_genetic_optimization.py` | v1.0 | 📝 示例 | 遗传算法演示示例 |

**功能说明**：
- **参数变异**：随机调整评分阈值、换手率阈值等参数
- **优胜劣汰**：基于夏普比率评估策略表现
- **持续进化**：自动优化选股参数

**优化参数**：
- 评分阈值（正常策略、洗盘策略）
- 换手率阈值
- 选股数量

---

## 配置文件

### 参数配置

| 文件名 | 版本 | 状态 | 说明 |
|--------|------|------|------|
| `strategy_params.json` | v3.0 | ✅ 生产 | 策略参数配置（多因子模型适配） |
| `strategy_params_multi_factor.json` | v3.0 | ✅ 生产 | 多因子模型参数配置 |
| `strategy_params_optimized.json` | v2.0 | 🔒 备份 | 优化后的参数配置 |
| `strategy_params_v1_backup.json` | v1.0 | 🔒 备份 | 原始参数配置备份 |

**配置项说明**：
```json
{
  "params": {
    "first_round": {
      "MIN_PCT_CHG": 2.0,           // 最小涨幅
      "MIN_VOL_RATIO": 1.2,          // 最小量比
      "MIN_POS_RATIO": 0.15          // 最小持仓比
    },
    "second_round": {
      "SCORE_THRESHOLD_NORMAL": 40,  // 正常策略评分阈值
      "SCORE_THRESHOLD_WASH": 30,    // 洗盘策略评分阈值
      "TURNOVER_THRESHOLD_NORMAL": 1.5,  // 正常策略换手率阈值
      "TURNOVER_THRESHOLD_WASH": 0.6,    // 洗盘策略换手率阈值
      "TOP_N_PER_STRATEGY": 5         // 每种策略选股数量
    }
  }
}
```

### 模型配置

| 文件名 | 版本 | 状态 | 说明 |
|--------|------|------|------|
| `config/model_config.json` | v1.0 | ✅ 生产 | 模型基础配置 |
| `config/aggressive_config.json` | v1.0 | ✅ 生产 | 激进策略配置 |
| `config/auto_threshold_config.json` | v1.0 | ✅ 生产 | 自动阈值配置 |
| `config/short_term_assault_config.json` | v1.0 | ✅ 生产 | 短期强攻配置 |

---

## 文档文件

### 系统文档

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `README.md` | 📖 系统说明 | 项目主文档 |
| `系统交付总结.md` | 📖 交付说明 | 系统交付总结 |
| `PROJECT_STRUCTURE.md` | 📖 结构说明 | 项目结构说明 |
| `INSTALLATION_GUIDE.md` | 📖 安装指南 | 安装配置指南 |
| `安全配置指南.md` | 📖 安全说明 | 安全配置说明 |

### 功能文档

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `WEATHER_SYSTEM_README.md` | 📖 使用文档 | 天气预报系统使用文档 |
| `MULTI_FACTOR_MODEL_README.md` | 📖 使用文档 | 多因子模型使用文档 |
| `GENETIC_OPTIMIZATION_README.md` | 📖 使用文档 | 遗传算法使用文档 |

### 测试报告

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `WEATHER_SYSTEM_TEST_REPORT.md` | 📊 测试报告 | 天气预报系统测试报告 |
| `MULTI_FACTOR_MODEL_TEST_REPORT.md` | 📊 测试报告 | 多因子模型测试报告 |
| `MULTI_FACTOR_VALIDATION_REPORT.md` | 📊 验证报告 | 多因子模型验证报告 |
| `MULTI_FACTOR_MODEL_V2.1_VALIDATION_REPORT.md` | 📊 验证报告 | 多因子模型v2.1验证报告 |
| `API_OPTIMIZATION_REPORT.md` | 📊 优化报告 | API优化验证报告 |

### 升级文档

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `MULTI_FACTOR_MODEL_UPGRADE_GUIDE.md` | 📖 升级指南 | 多因子模型升级指南 |
| `REPOSITORY_UPDATE_LOG_20260129.md` | 📖 更新日志 | 仓库更新日志 |
| `REPOSITORY_UPDATE_SUMMARY.md` | 📖 更新总结 | 仓库更新总结 |

---

## 结果文件

### 选股结果

| 文件名 | 日期 | 数量 | 说明 |
|--------|------|------|------|
| `DeepQuant_TopPicks_20260129.csv` | 2026-01-29 | 12只 | 最新选股结果 |
| `DeepQuant_TopPicks_20260128.csv` | 2026-01-28 | 15只 | 选股结果 |
| `DeepQuant_TopPicks_20260120.csv` | 2026-01-20 | - | 历史选股结果 |

**结果文件包含字段**：
- ts_code：股票代码
- name：股票名称
- industry：所属行业
- strategy：策略类型（★低位强攻、☆缩量洗盘、▲梯量上行）
- New_Score：综合得分
- Pos_Sugg：仓位建议
- close：收盘价
- pct_chg：涨跌幅
- pe_ttm：市盈率
- turnover_rate：换手率
- MV_Yi：市值（亿）

### 候选池

| 文件名 | 日期 | 数量 | 说明 |
|--------|------|------|------|
| `Best_Pick_20260129.csv` | 2026-01-29 | 474只 | 第1轮候选池 |
| `Best_Pick_20260128.csv` | 2026-01-28 | 474只 | 候选池备份 |

---

## 辅助模块

### 数据收集

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `src/stock_system/data_collector.py` | ✅ 生产 | 数据收集模块 |
| `src/stock_system/predictor.py` | ✅ 生产 | 预测器模块 |

### 参数优化

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `parameter_optimizer.py` | ✅ 生产 | 参数优化器 |
| `src/stock_system/parameter_tuner.py` | ✅ 生产 | 参数调优模块 |

### 阈值优化

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `src/stock_system/auto_threshold_optimizer.py` | ✅ 生产 | 自动阈值优化 |
| `src/stock_system/capital_threshold_optimizer.py` | ✅ 生产 | 资金阈值优化 |
| `src/stock_system/rsi_threshold_optimizer.py` | ✅ 生产 | RSI阈值优化 |

### 交易系统

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `src/stock_system/assault_trading.py` | ✅ 生产 | 强攻交易系统 |
| `src/stock_system/closed_loop.py` | ✅ 生产 | 闭环控制系统 |
| `src/stock_system/triple_confirmation.py` | ✅ 生产 | 三重确认机制 |

### 监控与报告

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `src/stock_system/monitor.py` | ✅ 生产 | 监控模块 |
| `src/stock_system/model_reporter.py` | ✅ 生产 | 模型报告器 |
| `src/stock_system/report_generator.py` | ✅ 生产 | 报告生成器 |

### 缓存管理

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `src/stock_system/cache_manager.py` | ✅ 生产 | 缓存管理器 |
| `cache/logs/*.json` | 📁 缓存数据 | 缓存日志文件 |

### 可视化

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `src/stock_system/visualizer.py` | ✅ 生产 | 可视化模块 |

---

## 测试与验证

### 测试脚本

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `test_system.py` | ✅ 可用 | 系统测试脚本 |
| `test_performance.py` | ✅ 可用 | 性能测试脚本 |
| `test_threshold_optimization.py` | ✅ 可用 | 阈值优化测试 |
| `test_visualization.py` | ✅ 可用 | 可视化测试 |
| `demo_system.py` | 📝 示例 | 系统演示示例 |

### 验证脚本

| 文件名 | 状态 | 说明 |
|--------|------|------|
| `run_practical_validation.py` | ✅ 可用 | 实用验证脚本 |
| `validation_track.py` | ✅ 可用 | 验证跟踪脚本 |

---

## 文件状态说明

### 图例

| 图标 | 说明 |
|------|------|
| ✅ | 生产环境使用 |
| 📝 | 文档/示例 |
| 🔒 | 备份/归档 |
| 📁 | 目录 |
| 📊 | 报告 |
| 📖 | 说明文档 |

### 版本管理

**多因子模型版本**：
- v1.0：原始版本（存在板块识别问题）
- v2.1：优化版本（解决板块识别+优化API调用）

**策略参数版本**：
- v1.0：原始参数
- v2.0：优化参数
- v3.0：多因子模型适配参数

---

## 文件更新记录

### 2026-01-29 更新

**新增文件**：
- `multi_factor_model_v2.1.py`：新版本源文件
- `multi_factor_model_v1.0_backup.py`：原版本备份
- `MULTI_FACTOR_MODEL_UPGRADE_GUIDE.md`：升级指南
- `MULTI_FACTOR_MODEL_V2.1_VALIDATION_REPORT.md`：验证报告
- `REPOSITORY_UPDATE_LOG_20260129.md`：更新日志

**更新文件**：
- `multi_factor_model.py`：升级到v2.1
- `DeepQuant_TopPicks_20260129.csv`：最新选股结果
- `Best_Pick_20260129.csv`：候选池数据

---

## 使用建议

### 核心程序使用

1. **运行选股**：
   ```bash
   cd assets
   python3 柱形选股-筛选.py  # 第1轮筛选
   python3 柱形选股-第2轮.py  # 第2轮筛选
   ```

2. **使用主控程序**：
   ```bash
   python3 main_controller.py full  # 完整流程
   ```

3. **查看结果**：
   ```bash
   cat DeepQuant_TopPicks_YYYYMMDD.csv
   ```

### 参数调整

1. **修改评分阈值**：编辑 `strategy_params.json`
2. **调整因子权重**：编辑 `multi_factor_model.py`
3. **优化参数**：运行 `genetic_optimizer.py`

### 系统维护

1. **定期更新**：查看 `REPOSITORY_UPDATE_LOG_YYYYMMDD.md`
2. **问题排查**：查看 `test_system.py` 日志
3. **性能监控**：查看 `test_performance.py` 报告

---

## 注意事项

### 文件保护

- **备份文件**（`*_backup.py`、`*_backup.json`）不要修改
- **源文件**（`*_v2.1.py`）保留用于参考
- **生产文件**（`multi_factor_model.py`）是当前使用的版本

### 数据安全

- 选股结果文件建议定期备份
- 配置文件修改前建议备份
- 测试数据与生产数据分离

### 版本兼容

- 多因子模型v2.1兼容现有配置文件
- 天气预报系统需要最新交易日历数据
- 遗传算法需要历史回测数据

---

## 联系支持

如有问题，请查看：
1. `README.md`：系统概述
2. `INSTALLATION_GUIDE.md`：安装配置
3. `WEATHER_SYSTEM_README.md`：天气预报系统
4. `MULTI_FACTOR_MODEL_README.md`：多因子模型

---

**文档维护**：定期更新以反映系统变化
**最后更新**：2026-01-29
**文档版本**：v3.0
