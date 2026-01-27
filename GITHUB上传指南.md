# 🚀 GitHub 上传指南

## ✅ 安全检查完成

您的项目已经通过了安全检查，可以安全上传到 GitHub。

### 已完成的清理工作

- ✅ 清理了 6 个文件中的硬编码 Token
- ✅ 所有 Token 已替换为占位符 `your_tushare_token_here`
- ✅ `.env` 文件已在 `.gitignore` 中
- ✅ 备份文件已删除

---

## 📋 上传步骤

### 步骤 1: 初始化 Git 仓库（如果还没有）

```bash
# 进入项目目录
cd /workspace/projects

# 初始化 Git
git init
```

### 步骤 2: 添加远程仓库

```bash
# 添加 GitHub 仓库（替换为您的仓库地址）
git remote add origin https://github.com/wfyjmn/zhuzhuangxuangu.git

# 或使用 SSH（推荐）
git remote add origin git@github.com:wfyjmn/zhuzhuangxuangu.git
```

### 步骤 3: 添加文件到 Git

```bash
# 添加所有文件
git add .

# 检查将要提交的文件
git status
```

### 步骤 4: 提交更改

```bash
# 提交
git commit -m "feat: DeepQuant 智能选股系统 V3.0

- 实现两轮筛选机制（强攻/洗盘/梯量）
- 新增验证跟踪系统（1/3/5天表现）
- 新增参数优化模块
- 实现 Token 安全保护
- 完整的文档体系"
```

### 步骤 5: 推送到 GitHub

```bash
# 推送到主分支
git push -u origin main

# 或者推送到 master 分支
git push -u origin master
```

---

## ⚠️ 重要提示

### 上传前检查清单

- [x] 真实 Token 已清理
- [x] `.env` 文件在 `.gitignore` 中
- [x] 备份文件已删除
- [x] README.md 已准备
- [x] LICENSE 已添加（可选）

### 检查是否包含敏感信息

```bash
# 检查是否有真实的 Token
grep -r "8f5cd68a" . --include="*.py" --include="*.md" --include="*.json"
```

如果输出为空，说明没有敏感信息。

### 检查将要上传的文件

```bash
# 查看将要提交的文件
git status

# 查看文件大小
du -sh * .* 2>/dev/null | sort -h
```

---

## 📝 创建 README.md

将 `GITHUB_README.md` 的内容复制为 `README.md`：

```bash
cp GITHUB_README.md README.md
git add README.md
git commit -m "docs: 添加 GitHub README"
```

---

## 🔒 安全验证

### 验证 .gitignore

```bash
# 查看 .gitignore 内容
cat .gitignore

# 确保包含以下内容
echo "/.env
.env
*.backup
__pycache__/
*.pyc"
```

### 验证敏感文件

```bash
# 检查 .env 是否会被上传
git check-ignore -v .env

# 应该输出：.env	.gitignore:1:/.env
```

---

## 📊 推荐的分支策略

```
main (主分支，稳定版本)
├── feature (功能开发)
├── bugfix (错误修复)
└── docs (文档更新)
```

---

## 🎯 初次推送命令汇总

```bash
# 完整的首次推送命令
cd /workspace/projects
git init
git remote add origin https://github.com/wfyjmn/zhuzhuangxuangu.git
git add .
git commit -m "feat: DeepQuant 智能选股系统 V3.0"
git push -u origin main
```

---

## ⚙️ 配置建议

### 1. 添加 LICENSE

```bash
# 创建 MIT LICENSE
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 wfyjmn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

git add LICENSE
git commit -m "docs: 添加 MIT License"
```

### 2. 创建 .gitignore

如果还没有 `.gitignore`，创建一个：

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# 虚拟环境
venv/
env/
.venv/
ENV/

# 环境变量（包含敏感信息）
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# 操作系统
.DS_Store
Thumbs.db

# 数据文件
*.csv
*.xlsx

# 备份文件
*.backup

# 日志
*.log
logs/

# 临时文件
temp/
tmp/
EOF
```

### 3. 添加 GitHub 标签

在 README.md 顶部添加：

```markdown
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
```

---

## 🎉 上传完成

上传完成后，访问您的 GitHub 仓库：
https://github.com/wfyjmn/zhuzhuangxuangu

---

## 📞 需要帮助？

如果遇到问题：

1. 检查 Git 是否安装：`git --version`
2. 检查网络连接
3. 确认 GitHub 仓库地址正确
4. 查看错误信息

---

**准备就绪，开始上传吧！** 🚀
