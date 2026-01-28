# DeepQuant 智能选股系统 - 项目打包清单

**打包日期**：2026-01-29
**版本**：V3.0
**文件名**：zhuzhuangxuangu_v3.0_20260129.tar.gz
**文件大小**：7.1M
**文件数量**：454个

---

## 📦 打包信息

### 基本信息

| 项目 | 内容 |
|------|------|
| 打包格式 | tar.gz |
| 压缩算法 | gzip |
| 文件大小 | 7.1M |
| 文件数量 | 454个 |
| 打包时间 | 2026-01-29 00:40 |
| 版本号 | V3.0 |

### 排除内容

- ✅ `.git/` - Git版本控制目录
- ✅ `__pycache__/` - Python缓存目录
- ✅ `*.pyc` - Python字节码文件
- ✅ `*.pyo` - Python优化字节码文件
- ✅ `.vscode/` - VSCode配置
- ✅ `.idea/` - PyCharm配置
- ✅ `.DS_Store` - macOS系统文件
- ✅ `*.log` - 日志文件
- ✅ `*.egg-info` - Python包信息
- ✅ `.env` - 环境变量文件（保护敏感信息）

---

## 📁 项目结构

### 根目录文件

| 文件 | 说明 |
|------|------|
| `PROJECT_STRUCTURE.md` | 项目结构说明 |
| `PRECISION_BALANCED_REPORT.md` | 精度平衡报告 |
| `GITHUB_README.md` | GitHub README |
| `cleanup_tokens.py` | Token清理脚本 |
| `test_system.py` | 系统测试脚本 |

### assets/ 目录

#### 核心程序

| 文件 | 说明 |
|------|------|
| `柱形选股-筛选.py` | 第1轮筛选程序 |
| `柱形选股-第2轮.py` | 第2轮筛选程序 |
| `main_controller.py` | 主控程序 |
| `multi_factor_model.py` | 多因子模型 v2.1 |
| `market_weather.py` | 天气预报系统 |
| `genetic_optimizer.py` | 遗传算法优化器 |

#### 配置文件

| 文件 | 说明 |
|------|------|
| `strategy_params.json` | 策略参数配置 |
| `strategy_params_multi_factor.json` | 多因子模型配置 |
| `strategy_params_optimized.json` | 优化参数配置 |
| `config/*.json` | 模型配置文件 |

#### 结果文件

| 文件 | 说明 |
|------|------|
| `DeepQuant_TopPicks_20260129.csv` | 最新选股结果（12只） |
| `DeepQuant_TopPicks_20260128.csv` | 选股结果（15只） |
| `Best_Pick_20260129.csv` | 候选池（474只） |
| `Best_Pick_20260128.csv` | 候选池备份 |

#### 文档文件

| 文件 | 说明 |
|------|------|
| `README.md` | 项目主文档 |
| `FILES_GUIDE.md` | 文件说明文档 |
| `INSTALLATION_GUIDE.md` | 安装配置指南 |
| `WEATHER_SYSTEM_README.md` | 天气预报系统文档 |
| `MULTI_FACTOR_MODEL_README.md` | 多因子模型文档 |
| `GENETIC_OPTIMIZATION_README.md` | 遗传算法文档 |
| `DOCUMENTATION_UPDATE_SUMMARY.md` | 文档更新总结 |
| `REPOSITORY_UPDATE_LOG_20260129.md` | 仓库更新日志 |
| `系统交付总结.md` | 系统交付总结 |

#### 测试报告

| 文件 | 说明 |
|------|------|
| `MULTI_FACTOR_MODEL_V2.1_VALIDATION_REPORT.md` | 多因子模型验证报告 |
| `API_OPTIMIZATION_REPORT.md` | API优化报告 |
| `WEATHER_SYSTEM_TEST_REPORT.md` | 天气预报系统测试报告 |
| `MULTI_FACTOR_MODEL_TEST_REPORT.md` | 多因子模型测试报告 |

#### 升级文档

| 文件 | 说明 |
|------|------|
| `MULTI_FACTOR_MODEL_UPGRADE_GUIDE.md` | 多因子模型升级指南 |

#### 备份文件

| 文件 | 说明 |
|------|------|
| `multi_factor_model_v1.0_backup.py` | 多因子模型v1.0备份 |
| `multi_factor_model_v2.1.py` | 多因子模型v2.1源文件 |
| `strategy_params_v1_backup.json` | 策略参数v1.0备份 |

#### src/ 目录

| 目录 | 说明 |
|------|------|
| `src/agents/` | Agent代理模块 |
| `src/graphs/` | 图模块 |
| `src/storage/` | 存储模块（数据库、内存、对象存储） |
| `src/tools/` | 工具模块 |
| `src/utils/` | 工具函数（文件、日志、消息） |
| `src/stock_system/` | 股票系统（数据收集、预测、优化） |

#### cache/ 目录

| 目录 | 说明 |
|------|------|
| `cache/logs/` | 缓存日志 |
| `cache/models/` | 缓存模型 |
| `cache/market_data/` | 市场数据缓存 |
| `cache/metrics/` | 指标缓存 |
| `cache/predictions/` | 预测缓存 |
| `cache/errors/` | 错误缓存 |

#### data/ 目录

| 目录 | 说明 |
|------|------|
| `data/market_cache/` | 市场数据缓存 |

#### models/ 目录

| 文件 | 说明 |
|------|------|
| `models/*.json` | 模型配置文件 |
| `models/model_metadata_*.json` | 模型元数据文件 |

#### reports/ 目录

| 文件 | 说明 |
|------|------|
| `reports/strategy_comparison.json` | 策略对比报告 |
| `reports/*.png` | 可视化报告图片 |
| `reports/*.html` | 训练报告HTML |
| `reports/*.flag` | 训练完成标记 |

---

## 📊 文件统计

### 按类型统计

| 类型 | 数量 |
|------|------|
| Python文件（.py） | 约80个 |
| JSON文件（.json） | 约50个 |
| Markdown文件（.md） | 约20个 |
| CSV文件（.csv） | 约10个 |
| 图片文件（.png） | 约30个 |
| HTML文件（.html） | 约10个 |
| 其他文件 | 约250个 |

### 按目录统计

| 目录 | 文件数量 |
|------|----------|
| `assets/` | 约350个 |
| `assets/src/` | 约100个 |
| `assets/cache/` | 约20个 |
| `assets/models/` | 约15个 |
| `assets/reports/` | 约80个 |
| `assets/config/` | 约5个 |
| `根目录` | 约5个 |

---

## 🚀 快速开始

### 解压安装

```bash
# 解压文件
tar -xzf zhuzhuangxuangu_v3.0_20260129.tar.gz

# 进入项目目录
cd zhuzhuangxuangu/projects

# 查看文件
ls -la
```

### 环境配置

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 TUSHARE_TOKEN

# 3. 验证安装
python3 test_system.py
```

### 运行选股

```bash
# 方法1：使用主控程序
cd assets
python3 main_controller.py full

# 方法2：分步运行
python3 柱形选股-筛选.py
python3 柱形选股-第2轮.py

# 查看结果
cat DeepQuant_TopPicks_20260129.csv
```

---

## 📖 文档导航

### 快速入门

1. **README.md** - 项目主文档，快速了解系统
2. **FILES_GUIDE.md** - 文件说明文档，了解项目结构
3. **INSTALLATION_GUIDE.md** - 安装配置指南

### 功能说明

4. **WEATHER_SYSTEM_README.md** - 天气预报系统
5. **MULTI_FACTOR_MODEL_README.md** - 多因子模型
6. **GENETIC_OPTIMIZATION_README.md** - 遗传算法

### 测试报告

7. **MULTI_FACTOR_MODEL_V2.1_VALIDATION_REPORT.md** - 验证报告
8. **API_OPTIMIZATION_REPORT.md** - 优化报告
9. **WEATHER_SYSTEM_TEST_REPORT.md** - 测试报告

### 升级文档

10. **MULTI_FACTOR_MODEL_UPGRADE_GUIDE.md** - 升级指南
11. **REPOSITORY_UPDATE_LOG_20260129.md** - 更新日志
12. **DOCUMENTATION_UPDATE_SUMMARY.md** - 文档更新总结

---

## ⚠️ 注意事项

### 环境要求

- Python 3.8+
- Tushare Pro 账号
- 网络连接（获取行情数据）

### 配置要求

- 必须配置 TUSHARE_TOKEN（环境变量）
- 建议配置数据库（可选）
- 建议配置对象存储（可选）

### 数据安全

- ⚠️ **不要**将 `.env` 文件提交到公开仓库
- ⚠️ **不要**在代码中硬编码 Token
- ✅ 使用环境变量存储敏感信息

---

## 🔧 故障排查

### 常见问题

1. **无法获取数据**
   - 检查网络连接
   - 检查 TUSHARE_TOKEN 是否正确
   - 检查 API 额度是否充足

2. **模块导入失败**
   - 检查依赖是否安装完整
   - 检查 Python 版本是否符合要求
   - 检查环境变量是否正确配置

3. **选股结果为空**
   - 检查评分阈值是否过高
   - 检查市场数据是否正常
   - 查看日志文件排查问题

---

## 📞 技术支持

### 问题反馈

- GitHub仓库：https://github.com/wfyjmn/zhuzhuangxuangu.git
- 提交Issue反馈问题

### 文档查询

- 查看 `FILES_GUIDE.md` 了解文件结构
- 查看 `README.md` 了解系统功能
- 查看对应的 README 文件了解具体模块

---

## 📝 版本信息

| 项目 | 内容 |
|------|------|
| 系统版本 | V3.0 |
| 多因子模型版本 | v2.1 |
| 天气预报系统版本 | v1.0 |
| 遗传算法版本 | v1.0 |
| 打包日期 | 2026-01-29 |
| 文件大小 | 7.1M |
| 文件数量 | 454个 |

---

## 🎯 系统特性

### 核心功能

- ✅ 两轮筛选系统
- ✅ 天气预报系统（大势择时）
- ✅ 多因子选股模型（资金流+板块共振+技术形态）
- ✅ 遗传算法优化
- ✅ 参数动态调整

### 性能优化

- ✅ API调用次数减少98.3%
- ✅ 运行时间减少67%
- ✅ 板块识别正确率100%

### 文档完善

- ✅ 完整的文件说明文档
- ✅ 详细的使用指南
- ✅ 全面的测试报告
- ✅ 清晰的升级指南

---

**打包完成时间**：2026-01-29 00:40
**文档版本**：v3.0
**状态**：✅ 完成
