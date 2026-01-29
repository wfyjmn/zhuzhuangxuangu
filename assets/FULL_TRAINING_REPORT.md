# 完整训练流程报告

## 执行概况

本次完成了一个完整的 AI 裁判模型训练流程，包括数据生成、模型训练和评估。

---

## 训练流程

### 步骤 1：生成训练数据

**数据规模**：
- 总样本数：5,000 条
- 正样本：760 条（15.2%）
- 负样本：4,240 条（84.8%）
- 特征数：21 个

**特征列表**：
1. vol_ratio（量比）
2. turnover_rate（换手率）
3. pe_ttm（市盈率）
4. pct_chg_1d（1日涨跌幅）
5. pct_chg_5d（5日涨跌幅）
6. pct_chg_20d（20日涨跌幅）
7. ma5_slope（5日均线斜率）
8. ma20_slope（20日均线斜率）
9. bias_5（5日乖离率）
10. bias_20（20日乖离率）
11. rsi_14（RSI指标）
12. std_20_ratio（波动率）
13. position_20d（20日相对位置）
14. position_250d（250日相对位置）
15. macd_dif（MACD DIF）
16. macd_dea（MACD DEA）
17. macd_hist（MACD 红绿柱）
18. index_pct_chg（大盘涨跌幅）
19. sector_pct_chg（板块涨跌幅）
20. moneyflow_score（资金流得分）
21. tech_score（技术形态得分）

---

### 步骤 2：训练 AI 裁判模型

**模型配置**：
- 模型类型：LogisticRegression（XGBoost 未安装，自动降级）
- 样本权重：scale_pos_weight = 5.58（负样本/正样本）
- 交叉验证：5 折时序交叉验证

**交叉验证结果**：

| Fold | 训练样本 | 验证样本 | Accuracy | Precision | Recall | F1 | AUC |
|------|---------|---------|----------|-----------|--------|-----|-----|
| 1 | 835 | 833 | 0.5186 | 0.1354 | 0.4298 | 0.2059 | 0.5096 |
| 2 | 1,668 | 833 | 0.4814 | 0.1318 | 0.3897 | 0.1970 | 0.4519 |
| 3 | 2,501 | 833 | 0.5258 | 0.1487 | 0.4793 | 0.2270 | 0.4720 |
| 4 | 3,334 | 833 | 0.5126 | 0.1542 | 0.5378 | 0.2397 | 0.5083 |
| 5 | 4,167 | 833 | 0.5162 | 0.1733 | 0.5036 | 0.2578 | 0.4989 |
| **平均** | - | - | **0.5109** | **0.1487** | **0.4680** | **0.2255** | **0.4881** |

**模型选择**：
- 已保存第 5 Fold 的模型（看过的历史数据最多）

---

### 步骤 3：模型评估

**训练集评估指标**：
- 准确率（Accuracy）：52.80%
- 精确率（Precision）：16.64%
- 召回率（Recall）：52.50%
- F1分数：25.27%
- AUC分数：53.79%

**混淆矩阵**：
```
预测负样本: TN=2,241, FP=1,999
预测正样本: FN=361, TP=399
```

**详细分类报告**：
```
              precision    recall  f1-score   support

           0     0.8613    0.5285    0.6551      4240
           1     0.1664    0.5250    0.2527       760

    accuracy                         0.5280      5000
   macro avg     0.5138    0.5268    0.4539      5000
weighted avg     0.7556    0.5280    0.5939      5000
```

**预测概率分布**：
- 平均概率：49.75%
- 正样本概率：50.24%
- 负样本概率：49.67%
- 概率中位数：49.79%

---

## 特征重要性（Top 10）

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | sector_pct_chg（板块涨跌幅） | 0.1143 |
| 2 | index_pct_chg（大盘涨跌幅） | 0.0968 |
| 3 | pct_chg_20d（20日涨跌幅） | 0.0920 |
| 4 | ma20_slope（20日均线斜率） | 0.0845 |
| 5 | pct_chg_1d（1日涨跌幅） | 0.0786 |
| 6 | bias_20（20日乖离率） | 0.0554 |
| 7 | pct_chg_5d（5日涨跌幅） | 0.0464 |
| 8 | vol_ratio（量比） | 0.0368 |
| 9 | turnover_rate（换手率） | 0.0273 |
| 10 | macd_dif（MACD DIF） | 0.0178 |

**分析**：
- 环境特征（大盘、板块涨跌幅）最重要，说明市场环境影响显著
- 趋势特征（20日涨跌幅、均线斜率）次之
- 动量特征（1日涨跌幅）也有一定重要性

---

## 数据文件

**训练数据**：
- 文件：`data/training/mock_training_data.csv`
- 大小：约 500 KB

**模型文件**：
- 文件：`data/training/ai_referee_model.pkl/ai_referee_xgboost_20260129_122228.pkl`
- 格式：Joblib 格式

---

## 使用说明

### 加载模型

```python
from ai_referee import AIReferee

referee = AIReferee()
referee.load_model('data/training/ai_referee_model.pkl')
```

### 使用模型预测

```python
# 准备特征数据
features = {
    'vol_ratio': 1.2,
    'turnover_rate': 3.5,
    'pct_chg_1d': 2.5,
    # ... 其他特征
}

# 转换为 DataFrame
import pandas as pd
X = pd.DataFrame([features])

# 预测
probability = referee.predict_proba(X)[0][1]
print(f"盈利概率: {probability:.2%}")
```

---

## 注意事项

### 数据问题

**真实数据生成速度**：
- 单只股票完整流程：约 23 秒
- 生成 1,000 只股票：约 383 分钟（6.4 小时）
- 生成 65万~90万条数据：约 4~6 天

**原因**：
- 每只股票需要加载 60 天历史数据：13.84 秒
- 每只股票需要获取 5 天未来数据：9.49 秒
- 数据仓库没有缓存机制，每次都从文件读取

### 模型性能

**当前性能**（使用模拟数据）：
- AUC：48.81%（接近随机猜测）
- 精确率：14.87%（较低）
- 召回率：46.80%（中等）

**原因**：
- 使用模拟数据，特征与标签之间没有真实关联
- 使用 LogisticRegression（线性模型），能力有限
- XGBoost 未安装，无法使用更强的树模型

---

## 优化建议

### 1. 数据生成优化

**添加缓存机制**：
```python
class CachedDataWarehouse:
    def __init__(self):
        self.cache = {}

    def get_stock_data(self, ts_code, date, days):
        cache_key = f"{ts_code}_{date}_{days}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self._load_from_file(ts_code, date, days)
        return self.cache[cache_key]
```

**批量读取数据**：
```python
def load_multiple_stocks(self, ts_codes, date, days):
    """一次性加载多只股票的历史数据"""
    # 优化 IO 操作
    pass
```

### 2. 模型优化

**安装 XGBoost**：
```bash
pip install xgboost
```

**调整参数**：
```python
params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5.58  # 动态计算
}
```

### 3. 特征优化

**添加更多特征**：
- KDJ 指标
- BOLL 指标
- ATR（平均真实波幅）
- 量价配合特征

**特征选择**：
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X, y)
```

---

## 总结

本次训练流程成功完成，展示了完整的数据生成、模型训练和评估流程。

**成果**：
- ✅ 完整的训练流程脚本
- ✅ 时序交叉验证实现
- ✅ 模型保存和加载机制
- ✅ 特征重要性分析

**下一步**：
1. 安装 XGBoost：`pip install xgboost`
2. 优化数据仓库缓存机制
3. 使用真实历史数据训练
4. 持续优化模型参数和特征

---

## 版本信息

- **版本**：V1.0
- **训练日期**：2025-01-29
- **模型类型**：LogisticRegression
- **数据类型**：模拟数据（演示用）
