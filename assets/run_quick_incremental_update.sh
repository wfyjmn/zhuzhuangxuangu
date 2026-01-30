#!/bin/bash
# 快速更新策略：只更新 2023 年下半年和 2024 年上半年（关键训练数据）

cd /workspace/projects/assets

# 加载 .env 文件
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "快速增量更新策略..."
echo "更新范围: 20230701 ~ 20240630（12 个月，约 240 个交易日）"
echo "预计耗时: 约 120 分钟"

# 后台运行
nohup python3 << 'EOF' > quick_incremental_update.log 2>&1 &
import os
import sys
from pathlib import Path

# 设置 Token
os.environ['TUSHARE_TOKEN'] = '8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7'

sys.path.insert(0, str(Path.cwd()))

from data_warehouse import DataWarehouse

print("=" * 80)
print("快速增量更新 - 更新 2023-2024 关键训练数据")
print("=" * 80)

dw = DataWarehouse()

# 只更新关键月份：2023年7-12月和2024年1-6月
dates1 = dw.get_trade_days('20230701', '20231231')
dates2 = dw.get_trade_days('20240101', '20240630')
trade_days = dates1 + dates2

print(f"\n更新天数: {len(trade_days)}")
print(f"预计耗时: {len(trade_days) * 0.5 / 60:.1f} 分钟")

success_count = 0

for i, date in enumerate(trade_days, 1):
    print(f"\n[{i}/{len(trade_days)}] {date}")
    try:
        df = dw.download_daily_data(date, force=True)
        if df is not None and 'turnover_rate' in df.columns and 'pe_ttm' in df.columns:
            print(f"  ✅ 成功")
            success_count += 1
        else:
            print(f"  ⚠️  特征不完整")
    except Exception as e:
        print(f"  ❌ 错误: {e}")

    # 每 50 天输出一次进度
    if i % 50 == 0:
        print(f"\n[进度] 已完成 {i}/{len(trade_days)} 天 ({i/len(trade_days)*100:.1f}%)")

print("\n" + "=" * 80)
print(f"完成：成功 {success_count}/{len(trade_days)} 天")
print("=" * 80)

if success_count > len(trade_days) * 0.9:
    print("\n✅ 更新成功！可以重新训练模型")
else:
    print("\n⚠️ 部分数据更新失败")

EOF

echo "后台任务已启动，使用以下命令查看进度："
echo "  tail -f quick_incremental_update.log"
