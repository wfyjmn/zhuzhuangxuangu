import os
import time

def monitor_training(log_file="assets/training_v5.log"):
    """监控训练进度"""
    print("=" * 60)
    print("📊 训练进度监控")
    print("=" * 60)

    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return

    while True:
        # 读取日志文件最后50行
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) > 50:
            lines = lines[-50:]

        print("\n" + "=" * 60)
        print(f"📊 训练进度 (最后50行)")
        print("=" * 60)
        for line in lines:
            print(line.rstrip())

        # 检查是否完成
        if any("训练完成" in line for line in lines):
            print("\n✅ 训练已完成！")
            break

        # 检查是否有错误
        if any("Error" in line or "error" in line for line in lines[-10:]):
            print("\n⚠️ 检测到错误，请查看日志文件")
            break

        # 等待60秒
        print("\n⏳ 60秒后再次检查...")
        time.sleep(60)

if __name__ == "__main__":
    monitor_training()
