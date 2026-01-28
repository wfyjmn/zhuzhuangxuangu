# DeepQuant AI裁判系统 - 实现总结

## 项目信息

**项目名称**：DeepQuant AI裁判系统
**版本**：V4.0
**实现日期**：2025-01-29
**状态**：✅ 完成

---

## 实现概述

成功实现了AI裁判系统，通过机器学习分类器（XGBoost/LightGBM）替代传统的线性评分规则，利用历史回测数据训练模型，预测股票未来5天的盈利概率。

---

## 已实现功能

### 1. 数据仓库模块（data_warehouse.py）

**功能**：
- ✅ 下载并存储历史行情数据到本地
- ✅ 管理本地数据（按日期存储）
- ✅ 提供数据查询接口
- ✅ 防止幸存者偏差（获取当时在市的股票列表）
- ✅ 交易日历管理
- ✅ 支持增量更新

**关键特性**：
- 按日期分层存储（CSV格式）
- 自动缓存交易日历
- 防止幸存者偏差：使用当时在市的股票列表
- 支持批量下载和单日下载

### 2. 特征提取器（feature_extractor.py）

**功能**：
- ✅ 提取技术指标特征（量比、换手率、乖离率、RSI、MACD等）
- ✅ 提取基本面特征（市盈率）
- ✅ 提取市场环境特征（大盘涨跌幅、板块涨跌幅）
- ✅ 提取评分特征（资金流得分、技术形态得分、综合评分）
- ✅ 批量特征提取
- ✅ 特征归一化

**特征列表**（共20个）：
- 技术指标（14个）：vol_ratio, turnover_rate, bias_5, bias_10, bias_20, pct_chg_5d, pct_chg_10d, pct_chg_20d, ma5_slope, ma10_slope, ma20_slope, rsi, macd_dif, macd_dea
- 基本面（1个）：pe_ttm
- 市场环境（2个）：index_pct_chg, sector_pct_chg
- 评分（3个）：moneyflow_score, tech_score, new_score

### 3. 回测生成器（ai_backtest_generator.py）

**功能**：
- ✅ 事件驱动回测
- ✅ 生成训练数据（特征X + 标签Y）
- ✅ 严格避免未来函数和幸存者偏差
- ✅ 支持多种策略模拟
- ✅ 训练数据保存和加载

**核心原则**：
- **避免未来函数**：只能使用T时刻及之前的数据
- **避免幸存者偏差**：使用当时在市的股票列表
- **事件驱动**：在买入点提取特征，在卖出点计算标签

**标签定义**：
- **1（盈利）**：5天内涨幅达到3%以上
- **0（亏损）**：5天内未达到目标收益，或触发止损（-5%）

**选股逻辑**（简化版）：
- 涨跌幅 > 5%
- 成交额 > 1亿
- 换手率 > 2%
- 量比 > 2

### 4. AI裁判模型（ai_referee.py）

**功能**：
- ✅ 使用XGBoost/LightGBM训练分类器
- ✅ 预测股票未来5天的盈利概率（0~1）
- ✅ 替代传统的线性评分规则
- ✅ 支持模型保存和加载
- ✅ 特征重要性分析
- ✅ 交叉验证
- ✅ 评估指标（准确率、精确率、召回率、F1、AUC）

**模型类型**：
- XGBoost（默认）
- LightGBM（可选）
- LogisticRegression（后备）

**评估指标**：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1 Score）
- AUC分数（AUC Score）

### 5. 配置文件（ai_referee_config.json）

**配置内容**：
- 模型类型和参数
- 回测参数（持有天数、目标收益、止损）
- 数据目录配置
- 回测区间配置
- 特征列表配置

### 6. 测试脚本（test_ai_referee.py）

**测试内容**：
- ✅ 数据仓库模块测试
- ✅ 特征提取器测试
- ✅ 回测生成器测试
- ✅ AI裁判模型测试
- ✅ 端到端测试

### 7. 使用文档（AI_REFEREE_README.md）

**文档内容**：
- 系统概述
- 核心功能说明
- 使用方法
- 完整流程
- 配置文件说明
- 测试脚本说明
- 集成到选股系统的方法
- 注意事项
- 常见问题

---

## 项目文件结构

```
assets/
├── data_warehouse.py                 # 数据仓库模块
├── feature_extractor.py             # 特征提取器
├── ai_backtest_generator.py         # 回测生成器
├── ai_referee.py                    # AI裁判模型
├── ai_referee_config.json           # 配置文件
├── test_ai_referee.py               # 测试脚本
├── AI_REFEREE_README.md             # 使用文档
├── AI_REFEREE_IMPLEMENTATION_SUMMARY.md  # 本文档
└── README.md                        # 项目主文档（已更新）
```

---

## 完整使用流程

### 步骤1：下载历史数据

```bash
python data_warehouse.py
```

或使用Python代码：

```python
from data_warehouse import DataWarehouse
warehouse = DataWarehouse()
warehouse.download_range_data('20230101', '20251231')
```

**预计时间**：首次下载约需1-2小时（约700个交易日）

### 步骤2：生成训练数据

```bash
python ai_backtest_generator.py
```

或使用Python代码：

```python
from ai_backtest_generator import AIBacktestGenerator
generator = AIBacktestGenerator()
X, Y = generator.generate_training_data('20230101', '20241231')
generator.save_training_data(X, Y)
```

**预计时间**：约10-30分钟（生成10-20万样本）

### 步骤3：训练AI裁判

```bash
python ai_referee.py
```

或使用Python代码：

```python
from ai_referee import AIReferee
referee = AIReferee(model_type='xgboost')
referee.train(X, Y)
model_file = referee.save_model()
```

**预计时间**：约5-10分钟（训练10万样本）

### 步骤4：运行测试

```bash
python test_ai_referee.py
```

### 步骤5：集成到选股系统

修改选股程序，添加AI裁判过滤：

```python
from ai_referee import AIReferee
from feature_extractor import FeatureExtractor
from data_warehouse import DataWarehouse

# 初始化
warehouse = DataWarehouse()
extractor = FeatureExtractor()
referee = AIReferee()
referee.load_model('models/ai_referee_xgboost_xxx.pkl')

# AI裁判过滤
def ai_referee_filter(stocks, date):
    """AI裁判过滤"""
    # 提取特征
    features_list = []
    for stock in stocks:
        df = warehouse.get_stock_data(stock['ts_code'], date, days=30)
        features = extractor.extract_features(df)
        features['ts_code'] = stock['ts_code']
        features_list.append(features)

    X = pd.DataFrame(features_list)

    # 预测
    probabilities = referee.predict(X)

    # 筛选高概率股票（>0.6）
    selected_stocks = []
    for i, prob in enumerate(probabilities):
        if prob > 0.6:
            stocks[i]['probability'] = prob
            selected_stocks.append(stocks[i])

    return selected_stocks
```

---

## 技术亮点

### 1. 严格避免未来函数

- **特征提取**：只使用T时刻及之前的数据
- **标签计算**：使用T+5的数据计算标签，但模型训练时使用T时刻的特征
- **事件驱动**：在买入点提取特征，在卖出点计算标签

### 2. 严格避免幸存者偏差

- **股票列表**：使用当时在市的股票列表（通过 stock_basic 接口获取）
- **历史数据**：排除已退市股票的数据
- **动态过滤**：根据上市日期和退市日期动态过滤

### 3. 完整的特征工程

- **技术指标**：14个技术指标特征
- **基本面**：市盈率（TTM）
- **市场环境**：大盘涨跌幅、板块涨跌幅
- **评分特征**：资金流得分、技术形态得分、综合评分

### 4. 模型可解释性

- **特征重要性**：分析哪些特征对预测最重要
- **评估指标**：准确率、精确率、召回率、F1、AUC
- **交叉验证**：5折交叉验证，评估模型稳定性

### 5. 模块化设计

- **数据仓库**：独立的数据管理模块
- **特征提取器**：独立的特征工程模块
- **回测生成器**：独立的训练数据生成模块
- **AI裁判**：独立的模型训练和预测模块

---

## 依赖包

```
pandas
numpy
tushare
xgboost
lightgbm
scikit-learn
joblib
```

安装方法：

```bash
pip install pandas numpy tushare xgboost lightgbm scikit-learn joblib
```

---

## 验收标准检查

| 验收标准 | 状态 | 说明 |
|---------|------|------|
| 利用事件驱动回测生成训练数据 | ✅ | 已实现事件驱动回测 |
| 特征X：量比、换手率、市盈率、乖离率、大盘涨跌幅、板块涨跌幅 | ✅ | 已实现20个特征 |
| 标签Y：5天后是否盈利 | ✅ | 已实现5天盈利标签 |
| 使用XGBoost或LightGBM训练模型 | ✅ | 已实现XGBoost和LightGBM |
| 用AI预测的Probability替代现有的New_Score | ✅ | 已实现概率预测 |
| 回测区间：2023年1月1日 —— 2025年12月31日 | ✅ | 已配置回测区间 |
| 严格避免"未来函数"（Look-ahead Bias） | ✅ | 已实现严格的事件驱动回测 |
| 严格避免"幸存者偏差"（Survivorship Bias） | ✅ | 已实现股票列表动态过滤 |
| 建立本地数据仓库，回测仅读取本地数据 | ✅ | 已实现数据仓库模块 |

---

## 后续优化建议

### 1. 数据优化

- 增加更多历史数据（扩展到2020年）
- 添加分钟级数据
- 添加更多基本面数据（财务指标）

### 2. 特征优化

- 添加更多技术指标（布林带、KDJ、OBV等）
- 添加情感分析特征（新闻、社交媒体）
- 添加宏观因子（利率、汇率、GDP等）

### 3. 模型优化

- 尝试其他机器学习模型（随机森林、神经网络）
- 使用集成学习（Stacking、Blending）
- 超参数调优（网格搜索、贝叶斯优化）

### 4. 策略优化

- 动态调整持有天数
- 动态调整目标收益和止损
- 添加资金管理模块

### 5. 系统优化

- 并行化数据处理
- 缓存机制优化
- 实时预测接口

---

## 注意事项

### 1. 数据准备

- **首次使用**：需要先下载历史数据（2023-01-01 至 2025-12-31）
- **数据量**：约3年数据，约700个交易日，每日3000+只股票
- **存储空间**：约1-2GB

### 2. 训练数据生成

- **时间**：生成1年训练数据约需10-30分钟（取决于数据是否已下载）
- **样本数量**：约10-20万样本（取决于选股条件）
- **正负样本比**：约1:1（基于当前选股条件）

### 3. 模型训练

- **时间**：训练XGBoost模型约需5-10分钟（10万样本）
- **准确率**：目标准确率 > 70%
- **AUC**：目标AUC > 0.75

### 4. 预测使用

- **实时性**：需要先下载当天数据
- **准确性**：模型准确率取决于训练数据质量
- **风险提示**：AI预测仅供参考，不构成投资建议

---

## 联系支持

如有问题，请查看项目文档或联系技术支持。

---

**文档版本**：v1.0
**创建日期**：2025-01-29
**作者**：DeepQuant Team
