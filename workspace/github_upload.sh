#!/bin/bash
# GitHub 上传脚本

echo "=========================================================="
echo "  GitHub 上传脚本"
echo "=========================================================="

# 检查 Git 是否安装
if ! command -v git &> /dev/null; then
    echo "❌ Git 未安装，请先安装 Git"
    exit 1
fi

echo "✅ Git 已安装: $(git --version)"

# 检查是否已初始化
if [ ! -d ".git" ]; then
    echo ""
    echo "[步骤 1] 初始化 Git 仓库..."
    git init
    echo "✅ Git 仓库已初始化"
else
    echo ""
    echo "[步骤 1] Git 仓库已存在"
fi

# 检查远程仓库
if git remote get-url origin &> /dev/null; then
    echo ""
    echo "[步骤 2] 远程仓库已配置: $(git remote get-url origin)"
else
    echo ""
    echo "[步骤 2] 配置远程仓库..."
    read -p "请输入 GitHub 仓库地址: " repo_url
    git remote add origin "$repo_url"
    echo "✅ 远程仓库已配置: $repo_url"
fi

# 添加文件
echo ""
echo "[步骤 3] 添加文件到 Git..."
git add .
echo "✅ 文件已添加"

# 查看状态
echo ""
echo "[步骤 4] 查看将要提交的文件..."
git status --short

echo ""
read -p "是否继续提交？(yes/no): " confirm

if [ "$confirm" != "yes" ] && [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

# 提交
echo ""
echo "[步骤 5] 提交更改..."
git commit -m "feat: DeepQuant 智能选股系统 V3.0

- 实现两轮筛选机制（强攻/洗盘/梯量）
- 新增验证跟踪系统（1/3/5天表现）
- 新增参数优化模块
- 实现 Token 安全保护
- 完整的文档体系"

echo "✅ 提交完成"

# 推送
echo ""
echo "[步骤 6] 推送到 GitHub..."
read -p "请输入分支名称（默认 main）: " branch_name
branch_name=${branch_name:-main}

echo "正在推送到 $branch_name 分支..."
git push -u origin "$branch_name"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "  ✅ 上传成功！"
    echo "=========================================================="
    echo ""
    echo "您的项目已成功上传到 GitHub"
    echo "请访问: $(git remote get-url origin)"
else
    echo ""
    echo "❌ 推送失败，请检查错误信息"
fi
