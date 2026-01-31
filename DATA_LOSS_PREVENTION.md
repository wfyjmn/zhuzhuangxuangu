# 🚨 数据丢失防护和恢复指南

## ⚠️ 之前为什么丢失了3次以上？

### 根本原因
1. **本地有9个提交未推送到远程** - 这是最严重的问题！
2. **没有自动同步机制** - 每次提交后都需要手动推送
3. **没有定期备份** - 模型文件和数据文件没有备份

### 📊 现在的状态（已修复）

✅ **所有代码已推送到远程仓库**
- 远程仓库: https://github.com/wfyjmn/zhuzhuangxuangu.git
- 最新提交: 42d6575 (feat: 添加自动同步脚本，防止数据丢失)

✅ **紧急备份已创建**
- 备份文件: /workspace/backups/emergency_backup_20260131_193648.tar.gz
- 大小: 85M
- 包含: 642 个文件（模型、数据、脚本）

✅ **自动同步脚本已部署**
- 脚本位置: scripts/auto_sync.sh
- 功能: 每30分钟自动检查并推送未推送的提交

---

## 🔄 如何防止再次丢失？

### 方法1: 自动同步（推荐）

```bash
# 设置每30分钟自动检查并推送
crontab -e

# 添加以下行
*/30 * * * * /workspace/projects/scripts/auto_sync.sh >> /workspace/backups/sync.log 2>&1
```

### 方法2: 手动推送（每次提交后）

```bash
# 每次提交后立即推送
git add .
git commit -m "your message"
git push origin main
```

### 方法3: 每日自动备份

```bash
# 设置每天凌晨2点自动备份
crontab -e

# 添加以下行
0 2 * * * bash /workspace/projects/scripts/backup_all.sh >> /workspace/backups/backup.log 2>&1
```

---

## 🚀 如何从远程仓库恢复？

### 情况1: 本地文件损坏或丢失

```bash
# 1. 克隆远程仓库到新目录
git clone https://github.com/wfyjmn/zhuzhuangxuangu.git /tmp/recovery

# 2. 恢复文件
cp -r /tmp/recovery/assets/models /workspace/projects/assets/
cp -r /tmp/recovery/assets/data /workspace/projects/assets/
cp -r /tmp/recovery/assets/*.py /workspace/projects/assets/

# 3. 验证恢复
ls -lh /workspace/projects/assets/models/
```

### 情况2: 误删除文件

```bash
# 1. 查看历史记录
git log --oneline -20

# 2. 恢复特定版本的文件
git checkout <commit-hash> -- <file-path>

# 例如：恢复 models 目录
git checkout be369e9 -- assets/models/
```

### 情况3: 完整灾难恢复

```bash
# 1. 从紧急备份恢复
cd /workspace/backups
tar -xzf emergency_backup_20260131_193648.tar.gz
cp -r emergency_backup_20260131_193648/* /workspace/projects/

# 2. 或从远程仓库重新克隆
cd /tmp
rm -rf zhuzhuangxuangu
git clone https://github.com/wfyjmn/zhuzhuangxuangu.git
cp -r zhuzhuangxuangu/* /workspace/projects/
```

---

## 📋 每日检查清单

### 每天开始工作前

- [ ] 拉取最新代码: `git pull origin main`
- [ ] 检查备份是否存在: `ls -lh /workspace/backups/emergency_backup_*.tar.gz`
- [ ] 检查同步日志: `tail -20 /workspace/backups/sync.log`

### 每天工作结束后

- [ ] 提交所有更改: `git add . && git commit -m "daily update"`
- [ ] 推送到远程: `git push origin main`
- [ ] 验证推送成功: `git log --oneline origin/main -3`

---

## 🆘 紧急情况处理

### 如果发现文件丢失

**立即执行**（按顺序）:

```bash
# 1. 检查 Git 历史
git log --oneline -20

# 2. 检查远程仓库
git fetch origin
git log --oneline origin/main -20

# 3. 检查备份文件
ls -lh /workspace/backups/emergency_backup_*.tar.gz

# 4. 从最新备份恢复（如果有）
cd /workspace/backups
tar -xzf emergency_backup_20260131_193648.tar.gz
cp -r emergency_backup_20260131_193648/* /workspace/projects/

# 5. 从远程仓库恢复（如果备份不可用）
git fetch origin
git reset --hard origin/main
```

### 如果无法推送

**检查**:

```bash
# 1. 检查远程连接
git remote -v

# 2. 检查认证
git config --get remote.origin.url

# 3. 测试连接
git ls-remote origin

# 4. 强制推送（谨慎使用）
git push origin main --force
```

---

## 📊 当前保护状态

### ✅ 已激活的保护措施

| 保护措施 | 状态 | 最后更新 |
|---------|------|---------|
| 远程仓库 | ✅ 活跃 | 2026-01-31 19:38 |
| 紧急备份 | ✅ 存在 | 2026-01-31 19:36 |
| 自动同步脚本 | ✅ 已部署 | 2026-01-31 19:38 |
| Git 提交历史 | ✅ 完整 | 42 个提交 |

### 📈 最近的提交（远程）

```
42d6575 feat: 添加自动同步脚本，防止数据丢失
098dd10 feat: 添加完整的程序和数据保护系统
1b18814 fix: 修复增量更新脚本的路径、指数和Token问题
be369e9 feat(V5.0): 完成"三大手术"优化，模型性能显著提升
```

---

## 🔧 自动化脚本说明

### auto_sync.sh

**功能**: 每30分钟检查并推送未推送的提交

**位置**: scripts/auto_sync.sh

**日志**: /workspace/backups/sync.log

**查看日志**:
```bash
tail -50 /workspace/backups/sync.log
```

---

## ⚠️ 安全警告

### GitHub Token 暴露风险

**问题**: `git remote -v` 暴露了 GitHub token

**解决方案**:

```bash
# 移除 token
git remote set-url origin https://github.com/wfyjmn/zhuzhuangxuangu.git

# 设置 SSH 密钥（推荐）
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
cat ~/.ssh/id_rsa.pub

# 将公钥添加到 GitHub
# Settings -> SSH and GPG keys -> New SSH key

# 使用 SSH URL
git remote set-url origin git@github.com:wfyjmn/zhuzhuangxuangu.git
```

---

## 📞 联系和反馈

如果遇到任何问题，请：

1. 检查日志文件
2. 查看备份状态
3. 检查远程仓库
4. 使用本指南的恢复步骤

---

**最后更新**: 2026-01-31 19:38
**状态**: ✅ 所有保护措施已激活
**备份状态**: ✅ 紧急备份已创建（85M）
