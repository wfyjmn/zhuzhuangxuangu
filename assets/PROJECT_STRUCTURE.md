# 项目结构说明

```
/workspace/projects/
│
├── config/                          # 配置文件目录
│   ├── model_config.json           # 模型配置
│   └── tushare_config.json         # 数据源配置
│
├── src/                            # 源代码目录
│   └── stock_system/               # 股票系统核心模块
│       ├── __init__.py
│       ├── data_collector.py        # 数据采集模块
│       ├── predictor.py             # 预测生成模块
│       ├── analyzer.py              # 对比分析模块
│       ├── error_tracker.py         # 误差溯源模块
│       ├── parameter_tuner.py       # 参数调整模块
│       ├── model_updater.py         # 模型更新模块
│       ├── closed_loop.py           # 闭环迭代主流程
│       └── cache_manager.py         # 缓存管理模块
│
├── assets/                         # 资源目录
│   ├── models/                     # 模型文件
│   │   ├── best_model.pkl          # 最新模型
│   │   ├── model_metadata.json     # 模型元数据
│   │   └── model_*.pkl            # 历史模型备份
│   ├── data/                       # 数据目录
│   │   ├── predictions/            # 预测结果
│   │   └── market_cache/          # 行情数据缓存
│   ├── logs/                       # 日志目录
│   │   ├── performance_report_*.md # 性能报告
│   │   ├── error_report_*.md       # 误差报告
│   │   ├── summary_report.md       # 总结报告
│   │   └── iteration_results.json # 迭代结果
│   └── cache/                      # 通用缓存
│       ├── predictions/
│       ├── market_data/
│       ├── models/
│       ├── metrics/
│       ├── errors/
│       └── logs/
│
├── docs/                           # 文档目录
│   └── STOCK_SYSTEM_README.md     # 系统使用文档
│
├── scripts/                        # 脚本目录（内置）
│   └── load_env.py
│
├── tests/                          # 测试目录
│
├── run_system.py                   # 主运行脚本
├── test_system.py                  # 系统测试脚本
├── demo_system.py                  # 演示脚本
├── INSTALLATION_GUIDE.md           # 快速开始指南
├── PROJECT_STRUCTURE.md            # 项目结构说明（本文件）
├── requirements.txt                # Python依赖
├── README.md                       # 项目README
└── .coze                          # 配置文件
```

## 核心模块说明

### 1. data_collector.py（数据采集模块）
- 功能：从tushare获取A股行情数据
- 主要方法：
  - `get_stock_list()` - 获取股票列表
  - `get_daily_data()` - 获取日线数据
  - `get_batch_daily_data()` - 批量获取数据
  - `get_stock_pool()` - 获取股票池
  - `save_cache()` / `load_cache()` - 缓存管理

### 2. predictor.py（预测生成模块）
- 功能：基于XGBoost模型生成股票预测
- 主要方法：
  - `predict()` - 单次预测
  - `predict_batch()` - 批量预测
  - `generate_features_from_price()` - 特征生成
  - `save_predictions()` - 保存预测结果

### 3. analyzer.py（对比分析模块）
- 功能：对比预测与实盘，计算指标
- 主要方法：
  - `align_predictions_with_actuals()` - 数据对齐
  - `calculate_metrics()` - 计算指标
  - `generate_confusion_matrix()` - 混淆矩阵
  - `should_trigger_adjustment()` - 判断是否调整
  - `generate_report()` - 生成报告

### 4. error_tracker.py（误差溯源模块）
- 功能：分析误差来源
- 主要方法：
  - `analyze_errors()` - 误差分析
  - `identify_error_stocks()` - 识别误差股票
  - `get_feature_importance()` - 特征重要性
  - `analyze_error_by_threshold()` - 阈值分析

### 5. parameter_tuner.py（参数调整模块）
- 功能：自适应调整模型参数
- 主要方法：
  - `determine_adjustment_strategy()` - 确定调整策略
  - `apply_adjustments()` - 应用调整
  - `rollback_parameters()` - 参数回滚
  - `auto_adjust()` - 自动调整

### 6. model_updater.py（模型更新模块）
- 功能：增量训练和模型管理
- 主要方法：
  - `save_model()` - 保存模型
  - `load_model()` - 加载模型
  - `incremental_train()` - 增量训练
  - `update_model_with_new_data()` - 使用新数据更新
  - `rollback_to_version()` - 回滚到指定版本

### 7. closed_loop.py（闭环迭代主流程）
- 功能：整合所有模块，实现完整闭环
- 主要方法：
  - `run_one_iteration()` - 运行一次迭代
  - `run_continuous_iterations()` - 连续迭代
  - `_generate_summary_report()` - 生成总结报告

### 8. cache_manager.py（缓存管理模块）
- 功能：统一管理各类数据缓存
- 主要方法：
  - `save()` - 保存数据
  - `load()` - 加载数据
  - `clear_expired_cache()` - 清理过期缓存
  - `get_cache_stats()` - 获取缓存统计

## 运行脚本说明

### run_system.py（主运行脚本）
- 功能：系统主入口
- 使用方法：
  ```bash
  python run_system.py                    # 单次迭代
  python run_system.py --iterations 5    # 5次迭代
  python run_system.py --config xxx.json # 自定义配置
  ```

### test_system.py（系统测试脚本）
- 功能：测试所有模块是否正常工作
- 使用方法：
  ```bash
  python test_system.py
  ```

### demo_system.py（演示脚本）
- 功能：快速演示系统功能
- 使用方法：
  ```bash
  python demo_system.py
  ```

## 配置文件说明

### config/model_config.json
模型和性能配置：
- `xgboost.params` - XGBoost模型参数
- `xgboost.threshold` - 预测阈值
- `performance.targets` - 性能目标
- `performance.trigger_thresholds` - 触发调整的阈值
- `adjustment` - 参数调整策略

### config/tushare_config.json
数据源配置：
- `token` - tushare API token（必须配置）
- `timeout` - 超时时间
- `retry_count` - 重试次数

## 输出文件说明

### 预测结果
- 位置：`assets/data/predictions/predictions_*.json`
- 内容：每只股票的预测标签和概率

### 性能报告
- 位置：`assets/logs/performance_report_*.md`
- 内容：Markdown格式的性能分析报告

### 误差报告
- 位置：`assets/logs/error_report_*.md`
- 内容：误差分析和调整建议

### 模型文件
- 位置：`assets/models/model_*.pkl`
- 内容：XGBoost模型文件

### 总结报告
- 位置：`assets/logs/summary_report.md`
- 内容：多次迭代的性能趋势

## 数据流说明

```
1. 数据采集
   tushare → 历史行情数据 → 缓存

2. 特征生成
   历史行情 → 技术指标 → 特征向量

3. 模型预测
   特征向量 → XGBoost模型 → 预测结果

4. 实盘验证
   新行情 → 对齐 → 对比分析

5. 误差分析
   对比结果 → 误差统计 → 问题定位

6. 参数调整
   误差分析 → 调整策略 → 新参数

7. 模型更新
   新数据 + 新参数 → 增量训练 → 新模型

8. 闭环迭代
   回到步骤1，开始新一轮
```
