# DeepQuant 智能选股系统

<div align="center">

**一个基于技术分析的智能选股系统，具备验证跟踪和参数优化功能**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tushare](https://img.shields.io/badge/Data-Tushare-orange.svg)](https://tushare.pro/)

[功能介绍](#-核心功能) • [快速开始](#-快速开始) • [文档](#-文档) • [贡献](#-贡献)

</div>

---

## 📖 项目简介

DeepQuant 是一个基于技术分析的智能选股系统，通过多轮筛选、验证跟踪和参数优化，实现策略的持续改进。

### 主要特性

- 🎯 **智能选股**：两轮筛选机制，优中选优
- 📊 **验证跟踪**：自动跟踪选股后1/3/5天表现
- 🔄 **参数优化**：基于实际数据自动调整策略参数
- 🔒 **安全可靠**：Token 环境变量隔离，防止泄露
- 📝 **完整文档**：详细的使用指南和技术文档

---

## 🌟 核心功能

### 1. 智能选股系统

- **第1轮筛选**：宽进严出，快速扫描全市场
- **第2轮精选**：优中选优，评分排序
- **三种策略**：
  - ★ 低位强攻
  - ☆ 缩量洗盘
  - ▲ 梯量上行

### 2. 验证跟踪系统

- 自动跟踪选股后1天、3天、5天表现
- 计算收益率、胜率、最大回撤
- 生成详细的验证报告
- 模拟交易记录

### 3. 参数优化模块

- 分析策略表现，识别问题
- 智能调整参数阈值
- 记录参数变更历史
- 持续改进策略

### 4. Token 安全保护

- 环境变量隔离存储
- 统一配置管理
- 防止代码泄露
- 完整的安全文档

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Tushare Token（从 [Tushare Pro](https://tushare.pro/) 免费获取）

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/wfyjmn/zhuzhuangxuangu.git
cd zhuzhuangxuangu
```

#### 2. 配置 Token

创建 `.env` 文件（在 `assets/` 目录下）：

```bash
cd assets
cp .env.example .env
```

编辑 `.env` 文件，填入您的 Tushare Token：

```env
TUSHARE_TOKEN=your_tushare_token_here
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 运行系统

```bash
cd assets
python main_controller.py full
```

---

## 📚 使用指南

### 日常使用

| 操作 | 命令 | 频率 |
|------|------|------|
| 完整流程 | `python main_controller.py full` | 每日 |
| 仅选股 | `python main_controller.py select` | 每日 |
| 验证更新 | `python main_controller.py validate` | 每日 |
| 参数优化 | `python main_controller.py optimize` | 每周 |

### 推荐使用节奏

**第1周**：运行完整流程，积累数据

**第2-4周**：每日更新验证，分析表现

**第1个月后**：启用参数优化，持续改进

---

## 📁 项目结构

```
zhuzhuangxuangu/
├── assets/                      # 主程序目录
│   ├── main_controller.py       # 主控程序 ⭐
│   ├── 柱形选股-筛选.py         # 第1轮筛选
│   ├── 柱形选股-第2轮.py        # 第2轮筛选
│   ├── validation_track.py      # 验证跟踪
│   ├── parameter_optimizer.py   # 参数优化
│   ├── config.py                # 配置管理
│   ├── strategy_params.json     # 参数配置
│   ├── .env.example             # 环境变量示例
│   └── *.md                     # 文档文件
├── src/                         # 源代码
├── requirements.txt             # 依赖列表
├── README.md                    # 本文档
└── .gitignore                   # Git 忽略规则
```

---

## 📊 系统架构

```
┌─────────────────────────────────────┐
│      主控程序 (main_controller)     │
│         统一入口，协调运行           │
└──────────────┬──────────────────────┘
               │
       ┌───────┼───────┐
       │       │       │
  ┌────▼──┐ ┌─▼────┐ ┌─▼─────────┐
  │ 第1轮 │ │ 第2轮 │ │  验证跟踪  │
  │ 筛选  │ │ 筛选  │ │           │
  └───┬───┘ └───┬───┘ └─────┬─────┘
      │         │             │
      └─────────┴─────────────┘
                │
        ┌───────▼────────┐
        │  选股结果文件   │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  验证跟踪系统   │
        │ - 计算收益率   │
        │ - 生成报告     │
        │ - 模拟交易     │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  参数优化系统   │
        │ - 分析表现     │
        │ - 调整参数     │
        │ - 记录历史     │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  下一轮选股     │
        └────────────────┘
```

---

## 📖 文档

### 快速开始

- [下载说明.md](assets/下载说明.md) - 详细的安装步骤
- [使用指南](assets/README_验证系统.md) - 验证系统使用说明

### 技术文档

- [项目结构](assets/PROJECT_STRUCTURE.md) - 项目目录结构
- [系统交付总结](assets/系统交付总结.md) - 系统功能和架构
- [打包内容清单](assets/打包内容清单.md) - 文件清单

### 安全文档

- [安全配置指南](assets/安全配置指南.md) - Token 配置步骤
- [Token 安全保护说明](assets/Token安全保护说明.md) - 安全机制说明

---

## 🔒 安全说明

### Token 安全

- ✅ Token 存储在 `.env` 文件中
- ✅ `.env` 文件已添加到 `.gitignore`
- ✅ 不会被提交到版本控制系统
- ✅ 需要手动创建和配置

### 敏感文件

以下文件不会被提交到 Git：
- `.env` - 包含 Token
- `*.csv` - 数据文件
- `*.backup` - 备份文件
- `__pycache__/` - Python 缓存

---

## ⚠️ 免责声明

本系统仅供学习和研究使用，不构成任何投资建议。使用本系统进行交易的所有风险由用户自行承担。

- 本系统不保证选股的准确性
- 股市有风险，投资需谨慎
- 请根据自身情况合理使用
- 建议结合其他分析工具

---

## 🤝 贡献

欢迎贡献代码、报告 Bug 或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 开源协议

本项目采用 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送 Pull Request
- 邮件联系

---

## 🙏 致谢

- [Tushare](https://tushare.pro/) - 提供数据接口
- [Python](https://www.python.org/) - 编程语言
- [Pandas](https://pandas.pydata.org/) - 数据分析
- [Tushare](https://tushare.pro/) - 数据接口

---

<div align="center">

**如果觉得这个项目有用，请给个 ⭐ Star 支持一下！**

Made with ❤️ by wfyjmn

</div>
