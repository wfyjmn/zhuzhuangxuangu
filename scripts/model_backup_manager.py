"""
模型备份管理器
自动备份训练好的模型和相关数据
"""
import os
import shutil
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


class ModelBackupManager:
    """模型备份管理器"""

    def __init__(self, backup_root: str = "/workspace/backups"):
        self.backup_root = Path(backup_root)
        self.models_backup = self.backup_root / "models"
        self.models_backup.mkdir(parents=True, exist_ok=True)

    def backup_model(
        self,
        model: Any,
        metadata: Dict[str, Any],
        model_name: str = "model"
    ) -> str:
        """
        备份模型和元数据

        Args:
            model: 模型对象
            metadata: 模型元数据
            model_name: 模型名称

        Returns:
            备份文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 保存模型
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.models_backup / model_filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # 2. 保存元数据
        metadata_filename = f"{model_name}_{timestamp}_metadata.json"
        metadata_path = self.models_backup / metadata_filename

        # 添加额外元数据
        metadata['backup_timestamp'] = timestamp
        metadata['backup_hash'] = self._calculate_hash(model_path)
        metadata['git_commit'] = self._get_git_commit()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 3. 更新注册表
        self._update_registry({
            'model_name': model_name,
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'timestamp': timestamp,
            'metrics': metadata.get('metrics', {}),
            'git_commit': metadata.get('git_commit', 'unknown'),
        })

        print(f"✓ 模型已备份: {model_path}")
        print(f"✓ 元数据已备份: {metadata_path}")

        return str(model_path)

    def backup_training_data(
        self,
        training_data_file: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        备份训练数据

        Args:
            training_data_file: 训练数据文件路径
            metadata: 元数据

        Returns:
            备份文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.backup_root / "training_data"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # 复制训练数据
        backup_path = backup_dir / f"{Path(training_data_file).stem}_{timestamp}.csv"
        shutil.copy2(training_data_file, backup_path)

        # 保存元数据
        if metadata:
            metadata_path = backup_dir / f"{Path(training_data_file).stem}_{timestamp}_metadata.json"
            metadata['backup_timestamp'] = timestamp
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ 训练数据已备份: {backup_path}")

        return str(backup_path)

    def verify_backup(self, backup_path: str) -> bool:
        """
        验证备份文件完整性

        Args:
            backup_path: 备份文件路径

        Returns:
            是否有效
        """
        if not os.path.exists(backup_path):
            return False

        # 读取元数据中的哈希值
        metadata_path = backup_path.replace('.pkl', '_metadata.json')
        if not os.path.exists(metadata_path):
            return True  # 没有元数据，无法验证

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        expected_hash = metadata.get('backup_hash')
        if not expected_hash:
            return True  # 没有哈希值，无法验证

        # 计算当前文件的哈希值
        current_hash = self._calculate_hash(backup_path)

        return current_hash == expected_hash

    def restore_latest_model(self, model_name: str = "model") -> Optional[Any]:
        """
        从备份恢复最新的模型

        Args:
            model_name: 模型名称

        Returns:
            模型对象，如果不存在则返回 None
        """
        # 查找最新的备份
        backups = sorted(
            self.models_backup.glob(f"{model_name}_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not backups:
            print(f"未找到模型 {model_name} 的备份")
            return None

        latest_backup = backups[0]

        # 验证备份
        if not self.verify_backup(latest_backup):
            print(f"备份文件已损坏: {latest_backup}")
            return None

        # 加载模型
        with open(latest_backup, 'rb') as f:
            model = pickle.load(f)

        print(f"✓ 模型已恢复: {latest_backup}")

        return model

    def list_backups(self, model_name: Optional[str] = None) -> list:
        """
        列出所有备份

        Args:
            model_name: 模型名称（可选）

        Returns:
            备份列表
        """
        if model_name:
            backups = list(self.models_backup.glob(f"{model_name}_*.pkl"))
        else:
            backups = list(self.models_backup.glob("*.pkl"))

        # 按时间排序
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        result = []
        for backup in backups:
            metadata_path = backup.with_suffix('.pkl').with_name(backup.stem + '_metadata.json')
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            result.append({
                'path': str(backup),
                'size': backup.stat().st_size,
                'timestamp': backup.stem.split('_')[-2] + '_' + backup.stem.split('_')[-1],
                'metrics': metadata.get('metrics', {}),
                'valid': self.verify_backup(backup)
            })

        return result

    def _calculate_hash(self, file_path: str) -> str:
        """计算文件 SHA256 哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _get_git_commit(self) -> str:
        """获取当前 Git 提交"""
        import subprocess
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd="/workspace/projects",
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            return "unknown"

    def _update_registry(self, entry: Dict[str, Any]):
        """更新备份注册表"""
        registry_file = self.backup_root / "backup_registry.json"

        registry = []
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)

        registry.append(entry)

        # 只保留最近 100 条记录
        if len(registry) > 100:
            registry = registry[-100:]

        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)


# ============================================
# 便捷函数
# ============================================

def save_model_with_backup(
    model: Any,
    metadata: Dict[str, Any],
    save_path: str,
    backup_manager: Optional[ModelBackupManager] = None
):
    """
    保存模型并自动备份

    Args:
        model: 模型对象
        metadata: 模型元数据
        save_path: 保存路径
        backup_manager: 备份管理器（可选）
    """
    # 保存模型到指定位置
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ 模型已保存: {save_path}")

    # 自动备份
    if backup_manager is None:
        backup_manager = ModelBackupManager()

    model_name = Path(save_path).stem
    backup_manager.backup_model(model, metadata, model_name)


# ============================================
# 示例用法
# ============================================

if __name__ == "__main__":
    # 示例：备份管理器使用
    backup_manager = ModelBackupManager()

    # 示例：列出所有备份
    print("\n所有备份:")
    backups = backup_manager.list_backups()
    for backup in backups:
        status = "✓" if backup['valid'] else "❌"
        print(f"  {status} {backup['path']} ({backup['size']} bytes)")
        if backup['metrics']:
            print(f"     指标: {backup['metrics']}")
