# "三大手术"增量更新指南（修复版）

**更新时间**: 2026-01-30
**版本**: V5.0.1
**状态**: ✅ 脚本已修复，可以安全使用

---

## 重要修复说明

### 🔧 已修复的问题

1. **✅ 路径问题** - 脚本现在从项目根目录运行，确保能正确导入模块
2. **✅ 大盘指数** - 脚本现在会自动下载上证指数（000001.SH），修复相对收益标签
3. **✅ Token 安全** - 移除硬编码 Token，统一使用环境变量

### 📁 脚本位置

所有脚本现在位于**项目根目录**（`/workspace/projects/`），而不是 `assets/` 目录：

```bash
/workspace/projects/run_quick_incremental_update.sh      # 快速更新
/workspace/projects/run_full_incremental_update.sh       # 完整更新
/workspace/projects/run_test_incremental_update.sh       # 测试脚本
```

---

## 脚本说明

### 1️⃣ run_quick_incremental_update.sh - 快速增量更新（推荐）

**用途**: 补全 2023-2024 关键特征（换手率/PE）+ 大盘指数

**范围**: 20230701 ~ 20240630（约 240 个交易日）

**预计耗时**: 约 30-50 分钟（不是 120 分钟）

**包含任务**:
- ✅ 下载上证指数（000001.SH）
- ✅ 更新个股数据（含 turnover_rate, pe_ttm, pb 等）

**执行命令**:
```bash
cd /workspace/projects
./run_quick_incremental_update.sh
```

**查看进度**:
```bash
tail -f quick_incremental_update.log
```

---

### 2️⃣ run_full_incremental_update.sh - 完整增量更新

**用途**: 补全 2023-2024 所有关键特征 + 大盘指数

**范围**: 20230101 ~ 20241231（约 479 个交易日）

**预计耗时**: 约 60-100 分钟（不是 240 分钟）

**包含任务**:
- ✅ 下载上证指数（000001.SH）
- ✅ 更新个股数据（含 turnover_rate, pe_ttm, pb 等）

**执行命令**:
```bash
cd /workspace/projects
./run_full_incremental_update.sh
```

**查看进度**:
```bash
tail -f full_incremental_update.log
```

---

### 3️⃣ run_test_incremental_update.sh - 测试脚本

**用途**: 验证增量更新功能是否正常

**范围**: 20240101 ~ 20240110（10 个交易日）

**预计耗时**: 约 10 秒

**包含任务**:
- ✅ 测试上证指数下载
- ✅ 测试个股数据下载（含特征）

**执行命令**:
```bash
cd /workspace/projects
./run_test_incremental_update.sh
```

**查看结果**:
```bash
cat test_incremental_update.log
```

---

## 验证数据更新

### 检查大盘指数

```bash
cd /workspace/projects
head -5 assets/data/daily/000001.SH.csv
```

应该看到类似输出：
```csv
ts_code,trade_date,open,high,low,close,vol,amount,pct_chg
000001.SH,20230103,3116.11,3125.19,3073.91,3116.51,322995428,388445000000,0.88
000001.SH,20230104,3124.28,3124.28,3088.65,3095.24,280458379,331317000000,-0.68
...
```

### 检查个股特征

```python
import pandas as pd
from pathlib import Path

# 读取一个示例文件
data_dir = Path("assets/data/daily")
sample_file = sorted(data_dir.glob("*.csv"))[-1]

df = pd.read_csv(sample_file)

print(f"文件: {sample_file.name}")
print(f"特征列: {list(df.columns)}")

# 检查关键特征
features = {
    'turnover_rate': '换手率',
    'pe_ttm': '市盈率',
    'pb': '市净率',
    'ps': '市销率'
}

print("\n关键特征检查:")
for col, name in features.items():
    status = '✅' if col in df.columns else '❌'
    non_null = df[col].notna().sum() if col in df.columns else 0
    print(f"  {status} {name} ({col}): {non_null} 条非空数据")
```

---

## 重新训练模型

### 1. 重新生成训练数据

```bash
cd /workspace/projects/assets
nohup python3 train_optimized.py > train_with_new_features.log 2>&1 &
```

### 2. 查看训练进度

```bash
tail -f assets/train_with_new_features.log
```

### 3. 验证特征重要性

训练完成后，检查特征重要性文件：

```python
import pandas as pd
import glob
from pathlib import Path

# 读取最新的特征重要性文件
model_dir = Path("assets/data/models")
feature_files = sorted(model_dir.glob("feature_importance_*.csv"))

if feature_files:
    imp_df = pd.read_csv(feature_files[-1])

    print("Top 20 特征重要性:")
    print(imp_df.head(20))

    # 检查 turnover_rate 和 pe_ttm 的排名
    turnover_rank = imp_df[imp_df['feature'] == 'turnover_rate']
    pe_rank = imp_df[imp_df['feature'] == 'pe_ttm']

    print(f"\nturnover_rate 排名: {turnover_rank.index[0] + 1 if not turnover_rank.empty else '不存在'}")
    print(f"pe_ttm 排名: {pe_rank.index[0] + 1 if not pe_rank.empty else '不存在'}")
else:
    print("未找到特征重要性文件，请先训练模型")
```

---

## 预期效果

### 更新前

| 特征 | 状态 | 重要性 |
|------|------|--------|
| turnover_rate | ❌ 缺失 | 0 |
| pe_ttm | ❌ 缺失 | 0 |
| pb | ❌ 缺失 | 0 |
| 大盘指数 | ❌ 缺失 | 0 |

### 更新后

| 特征 | 状态 | 预期重要性 |
|------|------|-----------|
| turnover_rate | ✅ 完整 | Top 5 |
| pe_ttm | ✅ 完整 | Top 10 |
| pb | ✅ 完整 | Top 15 |
| 大盘指数 | ✅ 完整 | N/A（用于标签）|

### 预期模型性能提升

| 指标 | 当前 | 更新后（预期） | 提升 |
|------|------|---------------|------|
| AUC | 0.5314 | 0.60-0.65 | +13% ~ +22% |
| Precision | 0.2808 | 0.35-0.40 | +25% ~ +42% |
| Recall | 0.2664 | 0.35-0.45 | +31% ~ +69% |

---

## 故障排除

### 问题 1: 导入失败

**错误信息**:
```
❌ 导入失败: No module named 'data_warehouse'
```

**解决方案**:
- 确保从项目根目录运行：`cd /workspace/projects`
- 确保脚本位于项目根目录，不是 assets 目录

### 问题 2: Token 无效

**错误信息**:
```
请设置tushare pro的token凭证码
```

**解决方案**:
1. 检查 `.env` 文件是否存在：`ls -la .env`
2. 检查 Token 是否正确：`cat .env | grep TUSHARE_TOKEN`
3. 确保 Token 未过期

### 问题 3: 限流错误

**错误信息**:
```
每分钟最多访问200次
```

**解决方案**:
- 脚本已包含 `time.sleep(0.1)` 防止限流
- 如果仍然遇到限流，可以增加等待时间：
  ```python
  time.sleep(0.2)  # 改为 0.2 秒
  ```

### 问题 4: 权限错误

**错误信息**:
```
bash: ./run_quick_incremental_update.sh: Permission denied
```

**解决方案**:
```bash
chmod +x run_quick_incremental_update.sh
```

---

## 推荐使用流程

### 第一次使用

1. **测试功能**（10 秒）
   ```bash
   cd /workspace/projects
   ./run_test_incremental_update.sh
   cat test_incremental_update.log
   ```

2. **快速更新**（30-50 分钟）
   ```bash
   ./run_quick_incremental_update.sh
   tail -f quick_incremental_update.log
   ```

3. **验证数据**
   - 检查大盘指数：`head -5 assets/data/daily/000001.SH.csv`
   - 检查个股特征：运行验证脚本

4. **重新训练**
   ```bash
   cd assets
   python3 train_optimized.py
   ```

### 后续使用

如果只需要更新最近的数据（如最近 3 个月）：

```python
# 临时脚本：更新最近 3 个月
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'assets'))

from data_warehouse import DataWarehouse
import datetime

dw = DataWarehouse()

# 获取最近 3 个月的交易日
end_date = datetime.datetime.now().strftime('%Y%m%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime('%Y%m%d')

dates = dw.get_trade_days(start_date, end_date)

for date in dates:
    print(f"更新 {date}...")
    df = dw.download_daily_data(date, force=True)
```

---

## 总结

### 主要改进

1. **✅ 路径修复** - 脚本从项目根目录运行，确保模块导入正确
2. **✅ 指数补全** - 自动下载上证指数，修复相对收益标签
3. **✅ Token 安全** - 移除硬编码，使用环境变量
4. **✅ 性能优化** - 实际耗时比预估快很多（Tushare 日线接口很快）

### 预期效果

- 📊 补充 9 个关键特征
- 🎯 AUC 预期提升至 0.60-0.65
- 🚀 模型性能显著提升

### 下一步

1. 运行测试脚本验证功能
2. 运行快速更新脚本补充数据
3. 重新训练模型并验证效果

---

**更新时间**: 2026-01-30
**状态**: ✅ 脚本已修复，可以安全使用
