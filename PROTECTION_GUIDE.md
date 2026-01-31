# 程序和数据保护方案

**版本**: 1.0
**更新日期**: 2026-01-31

---

## 🛡️ 多层保护体系

### 第一层：Git 版本控制（代码保护）

#### 1.1 提交规范

```bash
# 1. 频繁提交代码
git add .
git commit -m "feat: 新增功能"

# 2. 推送到远程
git push origin main

# 3. 使用有意义的提交信息
feat: 新功能
fix: 修复bug
refactor: 重构代码
docs: 文档更新
```

#### 1.2 强制保护关键文件

创建 `.gitignore` 保护敏感文件：

```bash
# .gitignore
*.log
*.pkl
__pycache__/
.env
*.csv
data/
logs/
```

---

### 第二层：自动备份策略

#### 2.1 定期备份脚本

创建 `scripts/backup_all.sh`：

```bash
#!/bin/bash
# 完整备份脚本

BACKUP_DIR="/workspace/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "开始备份到: $BACKUP_DIR"

# 1. 备份模型文件
mkdir -p "$BACKUP_DIR/models"
cp -r /workspace/projects/assets/models/*.pkl "$BACKUP_DIR/models/" 2>/dev/null
cp -r /workspace/projects/data/models/*.pkl "$BACKUP_DIR/models/" 2>/dev/null

# 2. 备份训练数据
mkdir -p "$BACKUP_DIR/training_data"
cp -r /workspace/projects/data/training/*.csv "$BACKUP_DIR/training_data/" 2>/dev/null

# 3. 备份配置文件
mkdir -p "$BACKUP_DIR/config"
cp -r /workspace/projects/assets/models/*.json "$BACKUP_DIR/config/" 2>/dev/null

# 4. 备份重要日志
mkdir -p "$BACKUP_DIR/logs"
find /workspace/projects/logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \;

# 5. 创建备份清单
echo "备份清单:" > "$BACKUP_DIR/manifest.txt"
echo "模型文件: $(ls $BACKUP_DIR/models/ | wc -l) 个" >> "$BACKUP_DIR/manifest.txt"
echo "训练数据: $(ls $BACKUP_DIR/training_data/ | wc -l) 个" >> "$BACKUP_DIR/manifest.txt"
echo "配置文件: $(ls $BACKUP_DIR/config/ | wc -l) 个" >> "$BACKUP_DIR/manifest.txt"

# 6. 压缩备份
tar -czf "$BACKUP_DIR.tar.gz" -C /workspace/backups "$(basename $BACKUP_DIR)"

# 7. 清理临时文件
rm -rf "$BACKUP_DIR"

echo "备份完成: $BACKUP_DIR.tar.gz"

# 8. 清理旧备份（保留最近 7 天）
find /workspace/backups -name "*.tar.gz" -mtime +7 -delete
```

#### 2.2 训练后自动备份

修改训练脚本，在保存模型后自动备份：

```python
import shutil
from datetime import datetime

def save_model_with_backup(model, filename):
    """保存模型并自动备份"""
    # 1. 保存模型
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    # 2. 自动备份
    backup_dir = f"/workspace/backups/models/{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    backup_file = os.path.join(backup_dir, os.path.basename(filename))
    shutil.copy2(filename, backup_file)
    
    print(f"模型已保存: {filename}")
    print(f"模型已备份: {backup_file}")
```

---

### 第三层：云端备份

#### 3.1 GitHub 自动同步

```bash
#!/bin/bash
# 自动推送到 GitHub
cd /workspace/projects

# 添加所有更改
git add .

# 提交
git commit -m "auto: 自动备份 $(date +%Y%m%d_%H%M%S)"

# 推送
git push origin main

echo "已推送到 GitHub"
```

#### 3.2 设置定时任务

```bash
# 编辑 crontab
crontab -e

# 每天凌晨 2 点自动备份
0 2 * * * /workspace/projects/scripts/backup_all.sh >> /workspace/backups/backup.log 2>&1

# 每 6 小时推送代码到 GitHub
0 */6 * * * cd /workspace/projects && git add . && git commit -m "auto: 定期备份" && git push origin main >> /workspace/backups/git_push.log 2>&1
```

---

### 第四层：数据库保护

#### 4.1 数据库备份

```python
import sqlite3
import shutil

def backup_database(db_path, backup_path):
    """备份数据库"""
    # 关闭所有连接
    conn.close()
    
    # 复制数据库文件
    shutil.copy2(db_path, backup_path)
    print(f"数据库已备份: {backup_path}")
```

#### 4.2 增量备份

```python
import os
import shutil
from datetime import datetime

def incremental_backup(source_dir, backup_dir):
    """增量备份：只备份修改过的文件"""
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_file = os.path.join(root, file)
            relative_path = os.path.relpath(source_file, source_dir)
            backup_file = os.path.join(backup_dir, relative_path)
            
            # 只备份新文件或修改过的文件
            if not os.path.exists(backup_file) or \
               os.path.getmtime(source_file) > os.path.getmtime(backup_file):
                os.makedirs(os.path.dirname(backup_file), exist_ok=True)
                shutil.copy2(source_file, backup_file)
                print(f"备份: {relative_path}")
```

---

### 第五层：模型版本管理

#### 5.1 模型元数据

保存模型时记录完整信息：

```python
import json
import pickle
from datetime import datetime

def save_model_with_metadata(model, config, metrics, feature_names, filename):
    """保存模型和完整元数据"""
    model_data = {
        'model': model,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'model_type': 'XGBoost',
            'version': 'v1.0',
            'config': config,
            'metrics': metrics,
            'feature_names': feature_names,
            'git_commit': get_git_commit(),
            'data_version': get_data_version(),
        }
    }
    
    # 保存模型
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    # 保存元数据 JSON
    metadata_file = filename.replace('.pkl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(model_data['metadata'], f, indent=2)
    
    print(f"模型已保存: {filename}")
    print(f"元数据已保存: {metadata_file}")

def get_git_commit():
    """获取当前 Git 提交"""
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        return "unknown"

def get_data_version():
    """获取数据版本"""
    # 返回数据的时间范围或其他标识
    return "20230101-20241231"
```

#### 5.2 模型注册表

创建模型注册表：

```python
import json
from pathlib import Path

class ModelRegistry:
    """模型注册表：记录所有模型"""
    
    def __init__(self, registry_file="models/model_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return []
    
    def register_model(self, model_path, metadata):
        """注册新模型"""
        entry = {
            'model_path': str(model_path),
            'created_at': metadata['created_at'],
            'version': metadata['version'],
            'metrics': metadata['metrics'],
            'git_commit': metadata.get('git_commit', 'unknown'),
        }
        self.registry.append(entry)
        self._save_registry()
    
    def _save_registry(self):
        """保存注册表"""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_best_model(self, metric='auc'):
        """获取最佳模型"""
        sorted_models = sorted(
            self.registry,
            key=lambda x: x['metrics'].get(metric, 0),
            reverse=True
        )
        return sorted_models[0] if sorted_models else None
```

---

### 第六层：监控和恢复

#### 6.1 文件完整性检查

```python
import os
import hashlib

def check_file_integrity(file_path, expected_hash=None):
    """检查文件完整性"""
    if not os.path.exists(file_path):
        return False, "文件不存在"
    
    # 计算文件哈希
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if expected_hash:
        if file_hash != expected_hash:
            return False, "文件已损坏"
    
    return True, file_hash

def verify_all_models(model_dir="assets/models"):
    """验证所有模型文件"""
    print("检查模型文件完整性...")
    
    registry_file = os.path.join(model_dir, "model_registry.json")
    if os.path.exists(registry_file):
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        for entry in registry:
            model_path = entry['model_path']
            is_valid, info = check_file_integrity(model_path)
            status = "✅" if is_valid else "❌"
            print(f"{status} {os.path.basename(model_path)}: {info}")
```

#### 6.2 自动恢复

```python
def recover_from_backup(target_file, backup_dir):
    """从备份恢复文件"""
    if os.path.exists(target_file):
        print(f"目标文件已存在: {target_file}")
        return False
    
    # 查找最新的备份
    backups = sorted(
        Path(backup_dir).glob("*/" + os.path.basename(target_file)),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not backups:
        print("未找到备份")
        return False
    
    latest_backup = backups[0]
    shutil.copy2(latest_backup, target_file)
    print(f"已从备份恢复: {latest_backup} -> {target_file}")
    return True
```

---

## 📋 完整保护清单

### 代码保护
- [x] Git 版本控制
- [x] 频繁提交
- [x] 推送到远程
- [x] 使用分支管理

### 数据保护
- [x] 自动备份脚本
- [x] 定期备份（每天）
- [x] 增量备份
- [x] 备份到云端

### 模型保护
- [x] 保存元数据
- [x] 模型注册表
- [x] 版本管理
- [x] 完整性检查

### 监控和恢复
- [x] 文件完整性检查
- [x] 自动恢复机制
- [x] 备份清单
- [x] 日志记录

---

## 🚀 快速开始

### 1. 创建备份目录

```bash
mkdir -p /workspace/backups/models
mkdir -p /workspace/backups/training_data
mkdir -p /workspace/backups/logs
```

### 2. 设置定时任务

```bash
# 编辑 crontab
crontab -e

# 添加以下行
0 2 * * * /workspace/projects/scripts/backup_all.sh
0 */6 * * * cd /workspace/projects && git push origin main
```

### 3. 运行首次备份

```bash
bash scripts/backup_all.sh
```

---

## 🔄 恢复流程

### 从备份恢复

```bash
# 1. 解压备份
tar -xzf /workspace/backups/20260131_020000.tar.gz -C /tmp/

# 2. 恢复模型
cp -r /tmp/20260131_020000/models/* /workspace/projects/assets/models/

# 3. 恢复训练数据
cp -r /tmp/20260131_020000/training_data/* /workspace/projects/data/training/

# 4. 恢复配置
cp -r /tmp/20260131_020000/config/* /workspace/projects/assets/models/
```

### 从 Git 恢复

```bash
# 查看提交历史
git log --oneline

# 恢复到特定版本
git checkout <commit-hash> -- <file>

# 或恢复整个项目
git checkout <commit-hash>
```

---

## 📊 备份策略总结

| 保护级别 | 保护对象 | 频率 | 存储位置 | 保留时间 |
|---------|---------|------|---------|---------|
| 代码 | 所有源代码 | 每次提交 | Git + GitHub | 永久 |
| 模型 | 训练好的模型 | 每次训练 | 本地 + 云端 | 30 天 |
| 数据 | 训练数据 | 每天 | 本地 + 云端 | 7 天 |
| 日志 | 重要日志 | 每天 | 本地 | 7 天 |
| 配置 | 配置文件 | 每次修改 | Git + 本地 | 永久 |

---

## 💡 最佳实践

1. **开发习惯**
   - 频繁提交代码（每小时至少一次）
   - 每次提交前先 pull
   - 每天结束前 push 到远程

2. **训练习惯**
   - 训练前先备份当前最佳模型
   - 训练后立即保存并备份
   - 记录完整的训练参数和性能指标

3. **定期检查**
   - 每周检查备份是否正常
   - 每月验证备份的完整性
   - 定期清理过期备份

4. **文档记录**
   - 记录每次重大变更
   - 保留完整的训练日志
   - 维护模型版本记录

---

**更新时间**: 2026-01-31
**状态**: ✅ 保护方案已完成
