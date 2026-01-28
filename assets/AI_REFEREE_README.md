# DeepQuant AI裁判系统使用文档

## 概述

AI裁判系统是DeepQuant智能选股系统的核心升级模块，通过机器学习分类器（XGBoost/LightGBM）替代传统的线性评分规则，利用历史回测数据训练模型，预测股票未来5天的盈利概率。

## 核心功能

### 1. 数据仓库模块（data_warehouse.py）

**功能**：
- 下载并存储历史行情数据到本地
- 管理本地数据（按日期存储）
- 提供数据查询接口
- 防止幸存者偏差（获取当时在市的股票列表）

**使用方法**：
```python
from data_warehouse import DataWarehouse

# 初始化数据仓库
warehouse = DataWarehouse(data_dir="data/daily")

# 下载一天的数据
df = warehouse.download_daily_data("20250120")

# 下载时间范围的数据
warehouse.download_range_data("20230101", "20251231")

# 加载数据
df = warehouse.load_daily_data("20250120")

# 获取交易日历
trade_days = warehouse.get_trade_days("20230101", "20251231")
```

### 2. 特征提取器（feature_extractor.py）

**功能**：
- 提取技术指标特征（量比、换手率、乖离率、RSI、MACD等）
- 提取基本面特征（市盈率）
- 提取市场环境特征（大盘涨跌幅、板块涨跌幅）
- 提取评分特征（资金流得分、技术形态得分、综合评分）

**使用方法**：
```python
from feature_extractor import FeatureExtractor

# 初始化特征提取器
extractor = FeatureExtractor()

# 提取单只股票的特征
features = extractor.extract_features(df)

# 批量提取特征
features_list = extractor.extract_batch_features(stock_list)

# 获取特征名称列表
feature_names = extractor.get_feature_names()
```

**特征列表**（共20个）：
- 技术指标：vol_ratio, turnover_rate, bias_5, bias_10, bias_20, pct_chg_5d, pct_chg_10d, pct_chg_20d, ma5_slope, ma10_slope, ma20_slope, rsi, macd_dif, macd_dea
- 基本面：pe_ttm
- 市场环境：index_pct_chg, sector_pct_chg
- 评分：moneyflow_score, tech_score, new_score

### 3. 回测生成器（ai_backtest_generator.py）

**功能**：
- 事件驱动回测
- 生成训练数据（特征X + 标签Y）
- 严格避免未来函数和幸存者偏差
- 支持多种策略模拟

**核心原则**：
- **避免未来函数**：只能使用T时刻及之前的数据
- **避免幸存者偏差**：使用当时在市的股票列表
- **事件驱动**：在买入点提取特征，在卖出点计算标签

**使用方法**：
```python
from ai_backtest_generator import AIBacktestGenerator

# 初始化回测生成器
generator = AIBacktestGenerator()

# 生成训练数据
X, Y = generator.generate_training_data(
    start_date="20230101",
    end_date="20241231",
    max_samples=10000  # 可选，限制样本数量
)

# 保存训练数据
generator.save_training_data(X, Y)

# 加载训练数据
X, Y = generator.load_training_data(
    features_file="data/training/features_xxx.csv",
    labels_file="data/training/labels_xxx.csv"
)
```

**标签定义**：
- **1（盈利）**：5天内涨幅达到3%以上
- **0（亏损）**：5天内未达到目标收益，或触发止损（-5%）

### 4. AI裁判模型（ai_referee.py）

**功能**：
- 使用XGBoost/LightGBM训练分类器
- 预测股票未来5天的盈利概率（0~1）
- 替代传统的线性评分规则
- 支持模型保存和加载
- 特征重要性分析

**使用方法**：
```python
from ai_referee import AIReferee

# 初始化AI裁判
referee = AIReferee(model_type='xgboost')

# 训练模型
referee.train(X, Y, validation_split=0.2)

# 预测概率
probabilities = referee.predict(X_test)

# 获取特征重要性
importance_df = referee.get_feature_importance()

# 保存模型
model_file = referee.save_model()

# 加载模型
new_referee = AIReferee()
new_referee.load_model(model_file)

# 交叉验证
cv_results = referee.cross_validate(X, Y, cv=5)
```

## 完整流程

### 步骤1：下载历史数据

```bash
# 测试数据仓库
python data_warehouse.py

# 下载2023年1月到2025年12月的数据
python -c "
from data_warehouse import DataWarehouse
warehouse = DataWarehouse()
warehouse.download_range_data('20230101', '20251231')
"
```

### 步骤2：生成训练数据

```bash
# 测试回测生成器
python ai_backtest_generator.py

# 生成训练数据（2023-2024年）
python -c "
from ai_backtest_generator import AIBacktestGenerator
generator = AIBacktestGenerator()
X, Y = generator.generate_training_data('20230101', '20241231')
generator.save_training_data(X, Y)
"
```

### 步骤3：训练AI裁判

```bash
# 测试AI裁判
python ai_referee.py

# 训练并保存模型
python -c "
from ai_referee import AIReferee
from ai_backtest_generator import AIBacktestGenerator

# 生成训练数据
generator = AIBacktestGenerator()
X, Y = generator.generate_training_data('20230101', '20241231')

# 训练模型
referee = AIReferee(model_type='xgboost')
referee.train(X, Y)

# 保存模型
model_file = referee.save_model()
print(f'模型已保存: {model_file}')
"
```

### 步骤4：使用AI裁判预测

```python
from ai_referee import AIReferee
from feature_extractor import FeatureExtractor
from data_warehouse import DataWarehouse

# 初始化
warehouse = DataWarehouse()
extractor = FeatureExtractor()
referee = AIReferee()

# 加载模型
referee.load_model('models/ai_referee_xgboost_20250129_120000.pkl')

# 加载今天的数据
df_today = warehouse.load_daily_data('20250129')

# 提取特征
features_list = []
for ts_code in df_today['ts_code']:
    df = warehouse.get_stock_data(ts_code, '20250129', days=30)
    features = extractor.extract_features(df)
    features['ts_code'] = ts_code
    features_list.append(features)

X_predict = pd.DataFrame(features_list)

# 预测
probabilities = referee.predict(X_predict)

# 合并结果
df_today['probability'] = probabilities.values

# 筛选高概率股票（>0.6）
selected = df_today[df_today['probability'] > 0.6]
print(f"选出的股票: {len(selected)} 只")
print(selected[['ts_code', 'name', 'probability']])
```

## 配置文件

配置文件：`ai_referee_config.json`

```json
{
  "ai_referee": {
    "model_type": "xgboost",
    "model_params": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1
    }
  },
  "backtest": {
    "hold_days": 5,
    "target_return": 3.0,
    "stop_loss": -5.0
  },
  "backtest_range": {
    "train_start": "20230101",
    "train_end": "20241231",
    "validation_start": "20250101",
    "validation_end": "20251231"
  }
}
```

## 测试脚本

运行完整测试：

```bash
python test_ai_referee.py
```

测试包含：
1. 数据仓库模块测试
2. 特征提取器测试
3. 回测生成器测试
4. AI裁判模型测试
5. 端到端测试

## 集成到选股系统

将AI裁判集成到现有的选股流程：

```python
from ai_referee import AIReferee
from feature_extractor import FeatureExtractor
from data_warehouse import DataWarehouse

class EnhancedStockSelector:
    def __init__(self):
        self.warehouse = DataWarehouse()
        self.extractor = FeatureExtractor()
        self.referee = AIReferee()

        # 加载模型
        self.referee.load_model('models/ai_referee_xgboost_xxx.pkl')

    def select_stocks(self, date: str):
        """增强选股流程"""

        # 步骤1：第一轮筛选（基础条件）
        stocks = self.first_round_filter(date)

        # 步骤2：天气预报（大势研判）
        weather = self.check_market_weather(date)
        if weather == 'rain':
            print("市场环境不佳，暂停选股")
            return []

        # 步骤3：多因子筛选
        stocks = self.multi_factor_filter(stocks)

        # 步骤4：技术形态分析
        stocks = self.technical_analysis(stocks)

        # 步骤5：AI裁判预测（新增）
        stocks = self.ai_referee_filter(stocks, date)

        return stocks

    def ai_referee_filter(self, stocks, date):
        """AI裁判过滤"""

        # 提取特征
        features_list = []
        for stock in stocks:
            df = self.warehouse.get_stock_data(stock['ts_code'], date, days=30)
            features = self.extractor.extract_features(df)
            features['ts_code'] = stock['ts_code']
            features_list.append(features)

        X = pd.DataFrame(features_list)

        # 预测
        probabilities = self.referee.predict(X)

        # 筛选高概率股票（>0.6）
        selected_stocks = []
        for i, prob in enumerate(probabilities):
            if prob > 0.6:
                stocks[i]['probability'] = prob
                selected_stocks.append(stocks[i])

        return selected_stocks
```

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

### 4. 避免未来函数
- **特征提取**：只能使用T时刻及之前的数据
- **标签计算**：使用T+5的数据计算标签，但模型训练时使用T时刻的特征

### 5. 避免幸存者偏差
- **股票列表**：使用当时在市的股票列表（通过 stock_basic 接口获取）
- **历史数据**：排除已退市股票的数据

## 依赖包安装

```bash
pip install pandas numpy tushare xgboost lightgbm scikit-learn joblib
```

或者使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 文件说明

| 文件名 | 说明 |
|--------|------|
| data_warehouse.py | 数据仓库模块 |
| feature_extractor.py | 特征提取器 |
| ai_backtest_generator.py | 回测生成器 |
| ai_referee.py | AI裁判模型 |
| ai_referee_config.json | 配置文件 |
| test_ai_referee.py | 测试脚本 |
| AI_REFEREE_README.md | 本文档 |

## 常见问题

### Q1: 为什么没有生成训练数据？
A: 检查是否已经下载历史数据。如果没有，请先运行 `data_warehouse.py` 下载数据。

### Q2: 模型准确率很低？
A: 可能的原因：
1. 训练数据太少（增加训练时间范围）
2. 选股条件不合理（调整选股条件）
3. 特征不够丰富（添加更多特征）
4. 模型参数不合适（调整模型参数）

### Q3: 预测概率都很低？
A: 可能的原因：
1. 市场环境不佳（查看天气预报）
2. 模型训练不充分（增加训练样本）
3. 特征提取有问题（检查特征提取逻辑）

### Q4: 如何提高预测准确率？
A: 可以尝试：
1. 增加训练数据量（扩展回测区间）
2. 优化特征工程（添加更有价值的特征）
3. 调整模型参数（网格搜索优化）
4. 集成学习（使用多个模型集成）

## 联系支持

如有问题，请查看项目文档或联系技术支持。

---

**版本**：v1.0
**更新日期**：2025-01-29
**作者**：DeepQuant Team
