# A股模型实盘对比系统 - 快速开始指南

## 系统概述

这是一个完整的A股股票预测模型实盘验证闭环系统，实现了从预测生成到模型优化的完整流程。

## ✅ 已完成的功能

### 1. 核心模块（7个）

✅ **数据采集模块** (`data_collector.py`)
- 从tushare获取股票列表和行情数据
- 支持批量数据采集
- 数据缓存和质量检查

✅ **预测生成模块** (`predictor.py`)
- 加载XGBoost模型
- 生成股票涨跌预测
- 自动特征生成

✅ **对比分析模块** (`analyzer.py`)
- 计算核心性能指标（Accuracy, Precision, Recall, F1, AUC）
- 生成混淆矩阵
- 投资组合指标计算（夏普比率、最大回撤等）

✅ **误差溯源模块** (`error_tracker.py`)
- 分析误差分布
- 识别误差股票
- 特征重要性分析

✅ **参数调整模块** (`parameter_tuner.py`)
- 自适应参数调整策略
- 小步迭代机制
- 参数回滚功能

✅ **模型更新模块** (`model_updater.py`)
- 增量训练
- 模型版本管理
- 自动备份

✅ **闭环迭代主流程** (`closed_loop.py`)
- 完整的迭代流程
- 自动化运行
- 报告生成

### 2. 辅助模块（1个）

✅ **缓存管理模块** (`cache_manager.py`)
- 统一的数据缓存管理
- 过期缓存自动清理
- 缓存统计

### 3. 配置文件（2个）

✅ `config/model_config.json` - 模型配置
✅ `config/tushare_config.json` - 数据源配置

### 4. 运行脚本（3个）

✅ `run_system.py` - 主运行脚本
✅ `test_system.py` - 系统测试脚本
✅ `demo_system.py` - 演示脚本

### 5. 文档（2个）

✅ `docs/STOCK_SYSTEM_README.md` - 详细使用文档
✅ `INSTALLATION_GUIDE.md` - 本文档

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤2: 配置tushare token

编辑 `config/tushare_config.json`：

```json
{
  "token": "你的tushare_token",
  "timeout": 30,
  "retry_count": 3
}
```

获取token：访问 https://tushare.pro 注册账号

### 步骤3: 运行测试

```bash
python test_system.py
```

预期输出：所有7个测试都通过 ✅

### 步骤4: 运行演示

```bash
python demo_system.py
```

### 步骤5: 运行完整系统

```bash
# 单次迭代
python run_system.py

# 多次迭代
python run_system.py --iterations 5
```

## 📊 系统输出

运行后会生成以下文件：

```
assets/
├── data/
│   ├── predictions/          # 预测结果
│   └── market_data/          # 行情数据缓存
├── models/                   # 模型文件
│   ├── best_model.pkl
│   └── model_*.pkl
├── logs/
│   ├── performance_report_*.md    # 性能报告
│   ├── error_report_*.md        # 误差报告
│   └── summary_report.md        # 总结报告
└── cache/                    # 数据缓存
```

## 🎯 核心指标

系统跟踪的关键指标：

### 分类指标
- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1 Score
- AUC

### 投资组合指标
- 累计收益率
- 年化收益率
- 夏普比率
- 最大回撤
- 胜率

## 🔧 参数调整策略

系统会自动根据误差调整参数：

1. **假正例过多** → 提高分类阈值
2. **假负例过多** → 降低分类阈值
3. **召回率过低** → 增加scale_pos_weight
4. **精确率过低** → 调整学习率

## 📝 注意事项

1. **必须配置tushare token**才能获取实时行情数据
2. 首次运行会创建示例模型，后续会用实际数据训练
3. 建议在交易日后运行，以确保数据完整
4. 系统会自动保存日志和模型文件

## 🆘 故障排除

### 问题1: 测试失败 - 缺少依赖

**解决方案**：
```bash
pip install tushare xgboost scikit-learn
```

### 问题2: 运行失败 - 未配置token

**解决方案**：
编辑 `config/tushare_config.json`，填入你的tushare token

### 问题3: 数据采集失败

**解决方案**：
- 检查网络连接
- 检查tushare token是否有效
- 查看日志文件了解详细错误

## 📖 文档

详细文档请参考 `docs/STOCK_SYSTEM_README.md`

## ✨ 特性亮点

1. **完全自动化**：一次配置，自动运行
2. **闭环迭代**：预测→验证→优化→重训
3. **智能调参**：根据误差自动调整参数
4. **全面分析**：多维度性能指标和误差分析
5. **版本管理**：模型版本自动管理，支持回滚
6. **缓存机制**：高效的数据缓存和复用

## 🎉 总结

系统已完整实现所有核心功能，可以立即使用。只需配置tushare token后即可开始实盘验证！

---
**祝你使用愉快！** 🚀
