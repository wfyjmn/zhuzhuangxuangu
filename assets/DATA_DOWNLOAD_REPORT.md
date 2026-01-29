# 2023-2024 年历史数据下载完成报告

## 下载概况

### 数据范围
- **2023 年**：2023-01-01 ~ 2023-12-31
- **2024 年**：2024-01-01 ~ 2024-12-31
- **总交易日**：484 天

### 下载统计

| 年份 | 交易日数 | 已下载 | 缺失 | 完整度 |
|------|---------|-------|------|--------|
| 2023 | 242 | 242 | 0 | 100.0% |
| 2024 | 242 | 242 | 0 | 100.0% |
| **总计** | **484** | **484** | **0** | **100.0%** |

### 数据规模
- **总文件数**：491 个 CSV 文件
- **总数据大小**：205.3 MB
- **平均每日大小**：428.2 KB
- **平均每日股票数**：约 5,200 只

---

## 数据质量

### 数据内容
每个交易日数据包含以下字段：
- `ts_code`：股票代码
- `trade_date`：交易日期
- `open`：开盘价
- `high`：最高价
- `low`：最低价
- `close`：收盘价
- `pre_close`：昨收价
- `change`：涨跌额
- `pct_chg`：涨跌幅（%）
- `vol`：成交量
- `amount`：成交额（千元）
- `adj_factor`：复权因子

### 数据特点
- ✅ **含复权因子**：所有数据都包含 `adj_factor` 列，可用于计算前复权价格
- ✅ **防幸存者偏差**：只下载上市日期 <= 当前日期的股票
- ✅ **数据完整**：无缺失日期

---

## 使用建议

### 1. 训练数据生成
现在可以使用完整的历史数据生成训练样本：

```python
from ai_backtest_generator import AIBacktestGenerator

generator = AIBacktestGenerator()

# 生成 2023-2024 年的训练数据
df = generator.generate_dataset('20230101', '20241231')

# 或者分别生成
df_2023 = generator.generate_dataset('20230101', '20231231')
df_2024 = generator.generate_dataset('20240101', '20241231')
```

### 2. 数据集划分建议
- **训练集**：2023 年（242 天）
- **验证集**：2024 年上半年（约 120 天）
- **测试集**：2024 年下半年（约 120 天）

### 3. 预期样本量
根据之前的测试，每天筛选候选股票约 2,700 只（51%），预计总样本量：
- 理论最大值：484 × 2,700 = 1,306,800 条
- 实际预估（考虑历史数据不足）：650,000 ~ 900,000 条

---

## 数据质量验证

### 检查脚本
使用以下命令检查数据完整性：

```bash
cd assets
python -c "
from data_warehouse import DataWarehouse
import os

warehouse = DataWarehouse()

# 检查 2023-2024 年
for year in ['2023', '2024']:
    start = f'{year}0101'
    end = f'{year}1231'
    trade_days = warehouse.get_trade_days(start, end)
    missing = [d for d in trade_days if not os.path.exists(f'data/daily/{d}.csv')]

    print(f'{year}: {len(trade_days)} 天, 缺失 {len(missing)} 天')
"
```

### 预期输出
```
2023: 242 天, 缺失 0 天
2024: 242 天, 缺失 0 天
```

---

## 下载脚本清单

本次使用的下载脚本：

1. **`test_download_2023.py`** - 测试下载功能（下载 5 天）
2. **`download_2023_missing.py`** - 下载 2023 年 1-11 月缺失数据（221 天）
3. **`download_2023_nov_missing.py`** - 下载 2023 年 11 月缺失数据（19 天）
4. **`download_2023_dec_missing.py`** - 下载 2023 年 12 月缺失数据（16 天）
5. **`download_2023_2024_data.py`** - 完整下载脚本（2023-2024 年，未使用）

---

## 下一步操作

### 1. 生成训练数据
```bash
cd assets
python generate_training_data_2024_simple.py
```

### 2. 训练 AI 裁判模型
```bash
python train_ai_referee_v4.5.py
```

### 3. 测试模型
```bash
python test_ai_referee_v4.5.py
```

---

## 注意事项

1. **数据时效性**：数据截至 2024 年 12 月 31 日，如需更新至最新日期，请使用 `download_2024_data.py`

2. **API 限流**：Tushare API 有调用频率限制，下载大量数据时需注意限流

3. **存储空间**：当前数据大小约 205 MB，建议定期备份

4. **复权因子**：数据已包含复权因子，计算前复权价格时使用：
   ```python
   close_qfq = close * adj_factor
   ```

---

## 版本信息
- **下载日期**：2025 年 1 月 29 日
- **数据范围**：2023-01-01 ~ 2024-12-31
- **数据版本**：V1.0
- **状态**：✅ 完成
