#!/bin/bash
# 完整增量更新脚本 - 重新下载 2023-2024 年的所有数据

cd /workspace/projects/assets

# 加载 .env 文件
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "开始完整增量更新..."
echo "时间范围: 20230101 ~ 20241231"
echo "预计耗时: 约 240 分钟（479 个交易日）"

# 后台运行
nohup python3 << 'EOF' > full_incremental_update.log 2>&1 &
import os
import sys
import time
from pathlib import Path

# 设置 Token
os.environ['TUSHARE_TOKEN'] = '8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7'

sys.path.insert(0, str(Path.cwd()))

from data_warehouse import DataWarehouse

print("=" * 80)
print("完整增量更新 - 重新下载 2023-2024 年的所有数据")
print("=" * 80)

dw = DataWarehouse()

# 获取所有交易日
trade_days = dw.get_trade_days('20230101', '20241231')

print(f"\n总交易日数: {len(trade_days)}")
print(f"预计耗时: {len(trade_days) * 0.5 / 60:.1f} 分钟")

success_count = 0
fail_count = 0

for i, date in enumerate(trade_days, 1):
    print(f"\n[{i}/{len(trade_days)}] {date}")
    try:
        df = dw.download_daily_data(date, force=True)
        if df is not None:
            has_turnover = 'turnover_rate' in df.columns
            has_pe = 'pe_ttm' in df.columns
            if has_turnover and has_pe:
                print(f"  ✅ 成功")
                success_count += 1
            else:
                print(f"  ⚠️  部分特征缺失")
                success_count += 1
        else:
            print(f"  ❌ 失败")
            fail_count += 1
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        fail_count += 1

    # 每 50 天输出一次进度
    if i % 50 == 0:
        print(f"\n[进度] 已完成 {i}/{len(trade_days)} 天 ({i/len(trade_days)*100:.1f}%)")

print("\n" + "=" * 80)
print("增量更新完成")
print("=" * 80)
print(f"成功: {success_count} 天")
print(f"失败: {fail_count} 天")
print(f"总计: {len(trade_days)} 天")

if fail_count > 0:
    print(f"\n⚠️ 有 {fail_count} 天下载失败，请检查日志")
else:
    print("\n✅ 所有数据下载完成！")

EOF

echo "后台任务已启动，使用以下命令查看进度："
echo "  tail -f full_incremental_update.log"
