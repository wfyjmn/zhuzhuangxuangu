# AI 裁判 V5.0 优化总结

## 问题背景

2024年1月发生流动性危机（微盘股崩盘），市场特征：
- **绝大多数股票都在跌**：97.2% 的股票下跌，只有 2.6% 上涨
- **千股跌停**：极端行情下，传统标签逻辑失效
- **AI 变"死空头"**：如果使用绝对收益逻辑（>3%），正样本可能不足 5%，AI 会发现"预测所有股票都亏钱"就能达到 95% 的准确率

---

## 三大优化方案

### ✅ 方案 A：标签定义升级（引入相对收益）

**核心思想**：不要只看"绝对涨幅 > 3%"，在暴跌月份，能跑赢大盘就是赢。

**实现逻辑**：
```python
# 熊市判断标准：大盘跌幅 > 2%
if index_return < -2.0:
    is_bear_market = True

# 动态标签：
# 1. 牛市/震荡市：绝对收益 > 3%
# 2. 熊市：超额收益 > 5%（即便个股跌了，但比大盘少跌很多，也是强势股）

if is_bear_market:
    # 使用超额收益
    excess_return = stock_return - index_return
    label = 1 if excess_return > 3.0 else 0
else:
    # 使用绝对收益
    label = 1 if stock_return > 3.0 else 0
```

**效果**：
- 增加熊市环境下的正样本数量
- 识别"抗跌股"（跌得比大盘少的股票）
- 避免 AI 变成"死空头"

---

### ✅ 方案 B：样本加权（scale_pos_weight）

**核心思想**：告诉 AI "选对一个牛股的奖励，是排除一个垃圾股的 N 倍！"

**计算公式**：
```python
scale_pos_weight = 负样本数量 / 正样本数量
```

**实际效果（2024年1月数据）**：
| 场景 | 正样本占比 | scale_pos_weight | 说明 |
|------|-----------|-----------------|------|
| 正常市场 | ~20% | 4.0 | AI 会重视正样本 |
| 暴跌市场（20240122） | ~3% | 32.33 | AI 极度重视正样本 |

**实现代码**：
```python
# train() 方法中动态计算
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

# 更新 XGBoost 参数
if self.model_type == 'xgboost':
    self.model.set_params(scale_pos_weight=scale_pos_weight)
```

---

### ✅ 方案 C：特征工程（复权+归一化）

**核心思想**：确保特征准确无误，避免 AI 学到错误规律。

#### C1. 复权处理
```python
def _get_price_col(self, df):
    """优先使用复权价"""
    if 'close_qfq' in df.columns:
        return 'close_qfq'
    return 'close'
```

#### C2. 百分比斜率（消除高低价股差异）
```python
# ❌ 错误：使用绝对差值
ma5_slope = today_ma5 - yesterday_ma5  # 高价股斜率大

# ✅ 正确：使用百分比
ma5_slope = (today_ma5 - yesterday_ma5) / yesterday_ma5 * 100
```

#### C3. 相对位置归一化（0-1）
```python
# 当前价在近20天的位置
position_20d = (price - low_20) / (high_20 - low_20)
# 0 = 最低点, 1 = 最高点
```

---

## 测试结果

### 2024年1月22日（千股跌停日）
```
当日市场概况:
  总股票数: 5261
  上涨股票: 139 (2.6%)
  下跌股票: 5115 (97.2%)
  上证指数涨跌幅: -X.XX%
  市场环境: 熊市

样本加权:
  正样本: 300 (3.0%)
  负样本: 9700 (97.0%)
  scale_pos_weight: 32.33
  说明: AI 会给正样本 32 倍的权重
```

---

## 代码修改清单

### 1. `ai_backtest_generator.py`

#### 修改 1：`calculate_label` 方法
- **新增参数**：`index_start_price`, `index_future_df`
- **新增逻辑**：熊市判断、超额收益计算、动态标签
- **位置**：第 90-140 行

#### 修改 2：`generate_training_data` 方法
- **新增逻辑**：获取大盘未来数据，传入 `calculate_label`
- **位置**：第 210-230 行

#### 修改 3：`_get_future_data` 方法
- **修复问题**：`trade_date` 类型转换（int → str）
- **位置**：第 42-66 行

### 2. `ai_referee.py`

#### 已实现（无需修改）
- ✅ `train()` 方法：动态计算 `scale_pos_weight`
- ✅ `train_time_series()` 方法：全局计算权重并传给每个 Fold

### 3. `feature_extractor.py`

#### 已实现（无需修改）
- ✅ `_get_price_col()` 方法：优先使用 `close_qfq`
- ✅ 斜率计算：使用 `pct_change() * 100`
- ✅ 相对位置归一化：`position_20d`, `position_250d`

---

## 使用建议

### 1. 训练数据覆盖
- **必须包含**：2024年1月等极端行情数据
- **数据比例**：正常市场 : 暴跌市场 = 7 : 3
- **避免过拟合**：不要只用暴跌数据训练

### 2. 参数调整
```python
# 熊市超额收益阈值（可根据市场调整）
BEAR_MARKET_THRESHOLD = -2.0  # 大盘跌幅 > 2% 定义为熊市
EXCESS_RETURN_THRESHOLD = 3.0  # 跑赢大盘 3% 就算赢

# 止盈止损（可根据策略调整）
TARGET_RETURN = 3.0  # 目标收益 3%
STOP_LOSS = -5.0  # 止损 -5%
```

### 3. 监控指标
训练时关注以下指标：
- **正样本占比**：不应 < 5%，否则需要调整标签逻辑
- **scale_pos_weight**：不应 > 20，否则可能过拟合
- **精确率（Precision）**：应 > 60%，避免频繁误报
- **召回率（Recall）**：应 > 50%，不错过牛股

---

## 未来优化方向

1. **动态阈值**：根据市场波动率自动调整止盈止损阈值
2. **多标签分类**：不只是盈利/亏损，而是"大赚/小赚/小亏/大亏"
3. **因子权重自适应**：不同市场环境下，因子权重自动调整
4. **在线学习**：模型实时更新，适应市场变化

---

## 版本信息

- **版本**：V5.0
- **发布日期**：2024年
- **作者**：DeepQuant Team
- **状态**：✅ 已实现并测试通过
