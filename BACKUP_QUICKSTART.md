# 程序和数据保护 - 快速入门

## 🚀 3 分钟快速设置

### 1. 立即备份（推荐）

```bash
# 运行完整备份
bash scripts/backup_all.sh
```

这将备份：
- ✅ 所有模型文件（38 个）
- ✅ 训练数据（6 个）
- ✅ 配置文件
- ✅ 重要日志

### 2. 设置自动备份

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天凌晨 2 点自动备份）
0 2 * * * /workspace/projects/scripts/backup_all.sh >> /workspace/backups/backup.log 2>&1
```

### 3. 推送到 GitHub

```bash
# 添加所有更改
git add .

# 提交
git commit -m "feat: 添加备份保护系统"

# 推送
git push origin main
```

---

## 📋 常用命令

### 备份相关

```bash
# 完整备份
bash scripts/backup_all.sh

# 系统健康检查
bash scripts/system_health_check.sh

# 从备份恢复
bash scripts/restore_from_backup.sh backup_20260131_193249.tar.gz
```

### Git 相关

```bash
# 查看状态
git status

# 提交更改
git add .
git commit -m "your message"

# 推送到远程
git push origin main

# 查看提交历史
git log --oneline -10
```

### 模型备份（Python）

```python
from scripts.model_backup_manager import ModelBackupManager

# 创建备份管理器
backup_manager = ModelBackupManager()

# 备份模型
backup_manager.backup_model(
    model=your_model,
    metadata={
        'metrics': {'auc': 0.78, 'precision': 0.60},
        'config': {'n_estimators': 100}
    },
    model_name="my_model"
)

# 列出所有备份
backups = backup_manager.list_backups()
for backup in backups:
    print(f"{backup['path']} - {backup['metrics']}")

# 恢复最新模型
model = backup_manager.restore_latest_model("my_model")
```

---

## 📊 当前系统状态

### ✅ 已完成的保护措施

1. **备份脚本** - `scripts/backup_all.sh`
   - ✅ 已创建
   - ✅ 已测试
   - ✅ 备份大小: 11M

2. **恢复脚本** - `scripts/restore_from_backup.sh`
   - ✅ 已创建
   - ✅ 可用

3. **健康检查** - `scripts/system_health_check.sh`
   - ✅ 已创建
   - ✅ 当前评分: 70/100
   - ✅ 状态: 良好

4. **模型备份管理器** - `scripts/model_backup_manager.py`
   - ✅ 已创建
   - ✅ 可用

### 📁 备份内容

```
backup_20260131_193249.tar.gz (11M)
├── models/ (38 个文件)
│   ├── *.pkl (模型文件)
│   └── *.json (配置文件)
├── training_data/ (6 个文件)
│   └── *.csv (训练数据)
├── config/
│   └── .env (环境变量)
└── logs/ (2 个文件)
    └── *.log (日志文件)
```

---

## ⚠️ 当前问题和建议

### 问题 1: 配置文件缺失
```
❌ .env 不存在
✓ assets/.env 存在
```

**解决方案**:
```bash
# 从备份创建 .env
cp assets/.env .env

# 或者设置环境变量
export TUSHARE_TOKEN="your_token_here"
```

### 问题 2: 有未提交的更改
```
⚠ 有 6 个文件未提交
```

**解决方案**:
```bash
# 提交更改
git add .
git commit -m "feat: 添加备份保护系统"
git push origin main
```

---

## 🔄 完整工作流程

### 日常开发流程

```bash
# 1. 开始工作前
git pull origin main  # 获取最新代码

# 2. 工作中
# ... 进行开发 ...

# 3. 提交更改
git add .
git commit -m "feat: 新功能"

# 4. 推送到远程
git push origin main
```

### 训练模型流程

```python
# 1. 训练前先备份现有模型
bash scripts/backup_all.sh

# 2. 训练模型
# ... 训练代码 ...

# 3. 保存模型并自动备份
from scripts.model_backup_manager import save_model_with_backup

save_model_with_backup(
    model=your_model,
    metadata={
        'metrics': {'auc': 0.78, 'precision': 0.60},
        'config': training_config
    },
    save_path="assets/models/my_model.pkl"
)

# 4. 推送到 Git
git add assets/models/
git commit -m "feat: 训练新模型"
git push origin main
```

---

## 📝 检查清单

### 每日检查
- [ ] 推送代码到 GitHub
- [ ] 检查健康状态: `bash scripts/system_health_check.sh`
- [ ] 提交未更改的文件

### 每周检查
- [ ] 运行完整备份: `bash scripts/backup_all.sh`
- [ ] 检查备份文件完整性
- [ ] 清理过期备份（系统自动清理 7 天前的）

### 每月检查
- [ ] 验证备份可以正常恢复
- [ ] 更新保护策略
- [ ] 检查存储空间

---

## 🆘 恢复流程

### 从 Git 恢复

```bash
# 查看提交历史
git log --oneline -20

# 恢复到特定版本
git checkout <commit-hash> -- <file>

# 或恢复整个项目
git checkout <commit-hash>
```

### 从备份恢复

```bash
# 列出所有备份
ls -lh /workspace/backups/backup_*.tar.gz

# 恢复最新备份
bash scripts/restore_from_backup.sh backup_20260131_193249.tar.gz
```

---

## 💡 最佳实践

### 1. 频繁提交
- 每完成一个功能就提交
- 每天结束前提交所有更改

### 2. 定期备份
- 训练模型前备份
- 重大变更前备份
- 每天自动备份（crontab）

### 3. 验证备份
- 定期检查备份完整性
- 测试恢复流程

### 4. 文档记录
- 记录模型版本和性能
- 保留训练日志
- 维护变更记录

---

## 📞 故障排除

### 问题: 备份失败

**检查**:
```bash
# 检查磁盘空间
df -h

# 检查权限
ls -la /workspace/backups/

# 查看日志
cat /workspace/backups/backup.log
```

### 问题: Git 推送失败

**检查**:
```bash
# 检查远程状态
git remote -v

# 拉取最新代码
git pull origin main

# 查看冲突
git status
```

### 问题: 模型损坏

**恢复**:
```bash
# 1. 检查备份
bash scripts/system_health_check.sh

# 2. 从备份恢复
bash scripts/restore_from_backup.sh <backup-file>

# 3. 验证恢复
python -c "import pickle; model=pickle.load(open('assets/models/my_model.pkl','rb')); print(model)"
```

---

## ✅ 系统健康评分

### 评分标准

| 分数 | 状态 | 说明 |
|------|------|------|
| 90-100 | 优秀 ✓ | 所有保护措施已就位 |
| 70-89 | 良好 ✓ | 基本保护措施已完成 |
| 50-69 | 一般 ⚠ | 部分保护措施缺失 |
| 0-49 | 较差 ❌ | 需要立即采取行动 |

### 当前状态

```
健康评分: 70/100
状态: 良好 ✓

✅ 模型文件: 23 个，全部正常
✅ 训练数据: 6 个文件
✅ 备份文件: 1 个（11M）
✅ Git 同步: 正常

⚠ 配置文件: .env 缺失
⚠ 未提交更改: 6 个文件
```

---

**更新时间**: 2026-01-31
**状态**: ✅ 保护系统已设置完成
