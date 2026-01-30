#!/bin/bash
# 设置 Token 并运行增量更新

cd /workspace/projects/assets

# 加载 .env 文件
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 检查 Token
if [ -z "$TUSHARE_TOKEN" ]; then
    echo "错误：TUSHARE_TOKEN 未设置"
    exit 1
fi

echo "Token 设置成功: ${TUSHARE_TOKEN:0:20}..."

# 运行增量更新脚本（先测试小范围）
python3 << 'EOF'
import os
import sys
from pathlib import Path

# 设置 Token
os.environ['TUSHARE_TOKEN'] = '8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7'

sys.path.insert(0, str(Path.cwd()))

from data_warehouse import DataWarehouse

print("=" * 80)
print("测试：重新下载最近 10 个交易日的数据（补充 turnover_rate, pe_ttm）")
print("=" * 80)

dw = DataWarehouse()

# 获取最近 10 个交易日
trade_days = dw.get_trade_days('20241201', '20241231')

if not trade_days:
    print("[错误] 无法获取交易日历")
    sys.exit(1)

# 取最后 10 天
test_days = trade_days[-10:]

print(f"\n测试日期范围：{test_days[0]} ~ {test_days[-1]}")
print(f"共 {len(test_days)} 个交易日\n")

# 重新下载数据
success_count = 0
for date in test_days:
    print(f"\n[{date}] 重新下载...")
    try:
        df = dw.download_daily_data(date, force=True)
        if df is not None:
            has_turnover = 'turnover_rate' in df.columns
            has_pe = 'pe_ttm' in df.columns
            print(f"  ✅ 成功 | turnover_rate: {'✅' if has_turnover else '❌'} | pe_ttm: {'✅' if has_pe else '❌'}")
            success_count += 1
        else:
            print(f"  ❌ 失败")
    except Exception as e:
        print(f"  ❌ 错误: {e}")

print(f"\n" + "=" * 80)
print(f"测试完成：成功 {success_count}/{len(test_days)} 天")
print("=" * 80)

if success_count == len(test_days):
    print("\n✅ 测试成功！可以运行完整增量更新")
else:
    print("\n⚠️ 测试部分失败，请检查错误信息")
EOF
