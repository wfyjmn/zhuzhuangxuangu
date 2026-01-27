# Windows 编码问题修复说明

## 问题描述

在 Windows 系统上运行 `main_controller.py` 时出现编码错误：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xae in position 63: illegal multibyte sequence
```

### 原因

- Windows 默认使用 GBK 编码
- 程序输出包含中文字符
- `subprocess.run()` 没有指定编码，导致解码失败

## 解决方案

在 `main_controller.py` 中所有的 `subprocess.run()` 调用中添加：

```python
encoding='utf-8',
errors='replace'  # 遇到编码错误时替换字符，避免崩溃
```

### 修改的文件

`assets/main_controller.py` - 4 处 subprocess.run 调用

1. 第1轮筛选调用
2. 第2轮筛选调用
3. 验证跟踪调用
4. 参数优化调用

## 使用说明

### 本地更新代码

如果您已经下载了代码到本地，请：

1. **方法1：从 GitHub 重新下载**
```bash
git pull origin main
```

2. **方法2：手动修改**
打开 `assets/main_controller.py`，找到所有的 `subprocess.run()` 调用，添加编码参数：

```python
# 修改前
result = subprocess.run(
    [sys.executable, '文件名.py'],
    capture_output=True,
    text=True
)

# 修改后
result = subprocess.run(
    [sys.executable, '文件名.py'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)
```

### 测试修复

```bash
cd assets
python main_controller.py full
```

## 技术细节

### encoding='utf-8'

- 指定使用 UTF-8 编码解析输出
- Python 3 默认使用 UTF-8，但 subprocess 在 Windows 上可能使用系统默认编码

### errors='replace'

- 遇到无法解码的字符时，替换为 � 字符
- 防止程序因编码错误而崩溃
- 保证程序可以继续运行

## 其他可能的编码问题

### 如果问题仍然存在

检查 Python 文件的编码格式：

1. 使用支持 UTF-8 的编辑器（VS Code、PyCharm）
2. 确保文件保存为 UTF-8 编码
3. 在文件开头添加编码声明：

```python
# -*- coding: utf-8 -*-
```

### 终端编码问题

如果终端输出仍然乱码：

**Windows CMD**：
```cmd
chcp 65001
```

**PowerShell**：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 总结

这个修复确保了程序在 Windows 系统上能够正确处理中文输出，不会因为编码问题而崩溃。

---

*修复日期: 2026-01-27*
*版本: 1.0*
