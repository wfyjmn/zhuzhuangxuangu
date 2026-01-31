# 🛡️ 数据保护系统 - 使用指南

## ✅ 当前状态

```
[✓] 守护进程正在运行 (PID: 1967)
[✓] 所有提交已推送到远程
[✓] 紧急备份存在 (85M)
[✓] 找到 23 个模型文件
[✓] 找到 494 个数据文件
```

---

## 🚀 快速命令

### 查看系统状态
```bash
bash scripts/check_status.sh
```

### 管理守护进程
```bash
# 查看状态
bash scripts/manage_daemon.sh status

# 停止守护进程
bash scripts/manage_daemon.sh stop

# 启动守护进程
bash scripts/manage_daemon.sh start

# 重启守护进程
bash scripts/manage_daemon.sh restart

# 查看日志
bash scripts/manage_daemon.sh log
```

### 手动同步
```bash
# 推送到远程
git push origin main

# 或使用自动同步脚本
bash scripts/auto_sync.sh
```

---

## 🔄 自动同步机制

**守护进程** 每30分钟自动检查并推送未推送的提交。

**日志位置**: `/workspace/backups/daemon.log`

**查看日志**:
```bash
tail -30 /workspace/backups/daemon.log
```

---

## 🆘 恢复数据

### 从远程仓库恢复
```bash
git fetch origin
git reset --hard origin/main
```

### 从紧急备份恢复
```bash
cd /workspace/backups
tar -xzf emergency_backup_20260131_193648.tar.gz
cp -r emergency_backup_20260131_193648/* /workspace/projects/
```

---

## 📊 防护措施

| 措施 | 状态 | 说明 |
|------|------|------|
| 守护进程 | ✅ 运行中 | 每30分钟自动同步 |
| 远程仓库 | ✅ 已连接 | https://github.com/wfyjmn/zhuzhuangxuangu.git |
| 紧急备份 | ✅ 存在 | 85M，642个文件 |
| 自动同步脚本 | ✅ 已部署 | 可手动运行 |

---

## ⚠️ 重要提示

1. **每次提交后立即推送**:
   ```bash
   git add .
   git commit -m "your message"
   git push origin main
   ```

2. **定期检查系统状态**:
   ```bash
   bash scripts/check_status.sh
   ```

3. **守护进程会自动运行**，无需手动干预。

---

**更新时间**: 2026-01-31 19:40
**守护进程状态**: ✅ 运行中
**远程仓库**: ✅ 已同步
