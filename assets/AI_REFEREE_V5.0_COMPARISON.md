# AI 裁判 V5.0 实现对比分析

## 参考代码 vs 当前实现

### 1. 大盘指数数据获取方式

| 版本 | 实现方式 | 优点 | 缺点 |
|------|---------|------|------|
| **参考代码** | `self.warehouse.pro.index_daily()` | 直接获取，速度快 | 依赖 tushare pro 接口 |
| **当前实现** | `get_stock_data('000001.SH')` | 统一接口，兼容性好 | 数据可能不完整 |

**建议**：保持当前实现，但需要确保指数数据已下载到本地仓库。

---

### 2. 候选股票筛选条件

| 项目 | 参考代码 | 当前实现 | 差异分析 |
|------|---------|---------|---------|
| ST 过滤 | `name.str.contains('ST')` | `ts_code.str.contains('ST\|退')` | **参考代码可能报错**（name列不存在） |
| 成交额阈值 | `> 30000`（3亿） | `> 1000`（1000万） | 参考代码过滤更严 |
| 量比条件 | `> 1.2`（进攻） | 固定 1.0 | 参考代码更精细 |
| 洗盘形态 | `vol_ratio < 1.0` | 无 | 参考代码考虑了缩量回调 |

**关键问题**：参考代码使用了 `name` 列，但实际数据可能没有这个列！

```python
# 参考代码（可能报错）
mask = (daily_df['name'].str.contains('ST') == False)  # KeyError: 'name'

# 当前实现（已修复）
mask = (~daily_df['ts_code'].str.contains('ST|退', na=False))
```

---

### 3. 标签计算逻辑

#### 参考代码逻辑
```python
# 1. 止损检查（优先）
if min_return <= self.stop_loss:
    return 0

# 2. 熊市判断
if index_return < self.bear_threshold:
    # 超额收益 > 3% 就算赢
    alpha = stock_return - index_return
    return 1 if alpha >= self.alpha_threshold else 0
else:
    # 牛市：绝对收益 > 3%
    return 1 if stock_return >= self.target_return else 0
```

#### 当前实现逻辑
```python
# 1. 动态止盈止损（每日检查）
for price in future_df:
    if pct_return <= self.stop_loss:
        return 0  # 立即止损
    if is_bear_market and excess_return >= 5.0:
        return 1  # 熊市超额收益止盈
    if not is_bear_market and pct_return >= 3.0:
        return 1  # 牛市绝对收益止盈

# 2. 到期结算
return 1 if excess_return > 3.0 else 0  # 熊市
return 1 if final_return > 0 else 0  # 牛市
```

**差异**：
- 参考代码：只看最终收益（不提前止盈）
- 当前实现：每日检查止盈止损（更贴近实盘）

---

### 4. 数据生成方法

| 方法 | 返回值 | 用途 |
|------|-------|------|
| 参考代码 `generate_dataset` | `pd.DataFrame`（含特征+标签） | 一步生成，方便 |
| 当前实现 `generate_training_data` | `(X, Y)`（分开） | 符合 sklearn 规范 |

**建议**：添加 `generate_dataset` 方法作为便捷接口。

---

## 优化建议

### 1. 修复参考代码的 Bug

参考代码中的 `name` 列过滤会导致 `KeyError`，需要修复：

```python
# ❌ 错误
mask = (daily_df['name'].str.contains('ST') == False)

# ✅ 正确
mask = (~daily_df['ts_code'].str.contains('ST|退', na=False))
```

### 2. 调整成交额阈值

参考代码使用 `> 30000`（3亿），当前实现使用 `> 1000`（1000万）。

**建议**：
- 如果数据单位是万元，当前实现是正确的（1000万 = 1000万元）
- 如果数据单位是元，需要调整

让我检查一下数据单位...

```python
# 检查 amount 列的单位
daily_df['amount'].describe()
# 如果 mean 在 100000 左右 → 单位是元
# 如果 mean 在 100 左右 → 单位是万元
```

### 3. 添加缺失的 V5.0 参数

参考代码定义了清晰的参数：

```python
self.bear_threshold = -2.0   # 熊市判定阈值
self.alpha_threshold = 3.0   # 超额收益目标
```

**建议**：将这些参数添加到当前实现的 `__init__` 中。

---

## 推荐实现方案

### 方案 A：修复参考代码的 Bug，直接使用

**优点**：
- 代码结构清晰
- 参数定义明确
- 一步生成数据

**缺点**：
- 需要修复 `name` 列问题
- 需要调整成交额阈值
- 需要适配 DataWarehouse 接口

### 方案 B：融合两个版本的优点

**优点**：
- 保留当前实现的稳健性
- 借鉴参考代码的参数定义
- 兼顾两种数据生成方式

**缺点**：
- 需要额外代码

---

## 下一步行动

我建议采用 **方案 B**，具体步骤：

1. ✅ 添加 V5.0 参数到 `__init__`
2. ✅ 优化 `select_candidates_robust` 的成交额阈值
3. ✅ 添加 `generate_dataset` 方法
4. ✅ 保留动态止盈止损逻辑（比参考代码更贴近实盘）
5. ✅ 测试 2024年1月数据生成效果

需要我执行这个优化方案吗？
