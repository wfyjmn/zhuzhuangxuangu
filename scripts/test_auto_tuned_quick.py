#!/usr/bin/env python3
"""
快速测试自动化调优脚本
使用少量股票和试验快速验证功能
"""
import sys
import os

# 添加路径
workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
sys.path.insert(0, os.path.join(workspace_path, "src"))
sys.path.insert(0, os.path.join(workspace_path, "scripts"))

from train_auto_tuned import AutoTunedTrainer


def test_quick():
    """快速测试"""
    print("=" * 70)
    print("自动化调优快速测试")
    print("=" * 70)
    print()

    # 创建训练器
    trainer = AutoTunedTrainer()

    # 修改配置为快速测试
    trainer.config['data']['n_stocks'] = 10  # 只用10只股票
    trainer.config['data']['start_date'] = "2023-01-01"  # 缩短数据范围
    trainer.config['optuna']['n_trials'] = 5  # 只做5次试验
    trainer.config['optuna']['timeout'] = 600  # 10分钟超时

    print("快速测试配置:")
    print(f"  股票数: {trainer.config['data']['n_stocks']}")
    print(f"  数据起始: {trainer.config['data']['start_date']}")
    print(f"  Optuna 试验: {trainer.config['optuna']['n_trials']} 次")
    print(f"  超时时间: {trainer.config['optuna']['timeout']} 秒")
    print()

    try:
        # 运行训练
        model, threshold = trainer.train_full_pipeline()

        print("\n" + "=" * 70)
        print("✓ 快速测试成功！")
        print("=" * 70)
        print(f"最优阈值: {threshold:.2f}")
        print(f"最优参数: {trainer.best_params}")
        print()

        return True

    except Exception as e:
        print(f"\n❌ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_quick()
    sys.exit(0 if success else 1)
