#!/bin/bash

# 完整安全备份脚本
# 作者: Coze Coding
# 日期: 2026-02-01
# 功能: 备份整个项目，排除敏感信息

set -e  # 遇到错误立即退出

echo "============================================================"
echo "🔒 DeepQuant 智能选股系统 - 完整安全备份"
echo "============================================================"
echo ""

# 配置
PROJECT_DIR="/workspace/projects"
BACKUP_DIR="${PROJECT_DIR}/backup"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="DeepQuant_Complete_Backup_${TIMESTAMP}"
BACKUP_FILE="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"

# 创建备份目录
echo "📁 创建备份目录..."
mkdir -p "${BACKUP_DIR}"

# 显示备份信息
echo "📋 备份信息："
echo "  项目目录: ${PROJECT_DIR}"
echo "  备份目录: ${BACKUP_DIR}"
echo "  备份文件: ${BACKUP_FILE}"
echo "  时间戳: ${TIMESTAMP}"
echo ""

# 进入项目目录
cd "${PROJECT_DIR}"

echo "🔄 开始备份..."
echo ""

# 创建临时排除文件
EXCLUDE_FILE="${BACKUP_DIR}/exclude_patterns.txt"
cat > "${EXCLUDE_FILE}" << 'EOF'
# 敏感信息
.env
*.pem
*.key
*.crt
config/tushare_config.json

# 缓存文件
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# 数据缓存（过大，不包含在备份中）
assets/data/market_cache/*.pkl
assets/daily_prediction_*.csv
assets/atm_prediction_*.csv

# 临时文件
*.log
*.tmp
temp/
tmp/

# 测试文件
test_*.py
*_test.py

# 系统文件
.DS_Store
Thumbs.db

# 备份文件本身
backup/
*.tar.gz
*.zip

# Git 文件
.git/
.gitignore

# IDE 文件
.vscode/
.idea/
*.swp
*.swo
*~

# 大型压缩包
*.tar.gz
压缩包*.tar.gz
EOF

# 执行备份
echo "📦 正在压缩文件..."
tar -czf "${BACKUP_FILE}" \
    --exclude-from="${EXCLUDE_FILE}" \
    --exclude="${BACKUP_FILE}" \
    --exclude="${EXCLUDE_FILE}" \
    .

# 获取备份文件大小
BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)

echo "✅ 备份完成！"
echo ""
echo "📊 备份统计："
echo "  备份文件: ${BACKUP_FILE}"
echo "  文件大小: ${BACKUP_SIZE}"
echo ""

# 列出备份内容
echo "📄 备份内容预览："
tar -tzf "${BACKUP_FILE}" | head -n 30
echo "  ... (共 $(tar -tzf "${BACKUP_FILE}" | wc -l) 个文件)"
echo ""

# 验证备份
echo "🔍 验证备份完整性..."
if tar -tzf "${BACKUP_FILE}" > /dev/null 2>&1; then
    echo "✅ 备份完整性验证通过"
else
    echo "❌ 备份完整性验证失败！"
    exit 1
fi

# 生成备份清单
echo "📋 生成备份清单..."
MANIFEST_FILE="${BACKUP_DIR}/${BACKUP_NAME}_manifest.txt"
cat > "${MANIFEST_FILE}" << EOF
============================================================
DeepQuant 智能选股系统 - 完整备份清单
============================================================

备份时间: $(date +"%Y-%m-%d %H:%M:%S")
备份文件: ${BACKUP_FILE}
文件大小: ${BACKUP_SIZE}
文件数量: $(tar -tzf "${BACKUP_FILE}" | wc -l)

备份范围:
- ✅ 源代码 (src/)
- ✅ 脚本文件 (scripts/)
- ✅ 配置文件 (config/)
- ✅ 训练模型 (assets/models/)
- ✅ 文档资料 (assets/*.md)
- ✅ 依赖配置 (requirements.txt)
- ❌ 敏感信息 (.env, *.key)
- ❌ 数据缓存 (assets/data/market_cache/*.pkl)
- ❌ 临时文件 (*.log, temp/)

============================================================
文件清单:
============================================================
EOF

tar -tzf "${BACKUP_FILE}" | sort >> "${MANIFEST_FILE}"

echo "  清单文件: ${MANIFEST_FILE}"
echo ""

# 清理临时文件
rm -f "${EXCLUDE_FILE}"

echo "============================================================"
echo "✅ 备份完成！"
echo "============================================================"
echo ""
echo "📦 备份文件位置:"
echo "  ${BACKUP_FILE}"
echo ""
echo "📋 备份清单位置:"
echo "  ${MANIFEST_FILE}"
echo ""
echo "🔍 查看备份内容:"
echo "  tar -tzf ${BACKUP_FILE} | less"
echo ""
echo "💾 恢复备份:"
echo "  tar -xzf ${BACKUP_FILE} -C /path/to/restore"
echo ""
echo "📊 备份统计:"
echo "  文件数量: $(tar -tzf "${BACKUP_FILE}" | wc -l)"
echo "  文件大小: ${BACKUP_SIZE}"
echo ""

# 显示备份文件详情
echo "📁 备份目录内容:"
ls -lh "${BACKUP_DIR}" | grep -E "tar.gz|manifest"
echo ""

echo "✨ 备份操作完成！"
