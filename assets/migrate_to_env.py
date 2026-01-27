# -*- coding: utf-8 -*-
"""
Token 安全迁移脚本
自动将硬编码的 Token 替换为从环境变量读取
"""

import os
import re
from pathlib import Path


# 目标文件列表
TARGET_FILES = [
    '柱形选股-筛选.py',
    '柱形选股-第2轮.py',
    'validation_track.py',
]


def migrate_file(filepath):
    """迁移单个文件"""
    print(f"\n处理文件: {filepath}")

    # 读取文件内容
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ 无法读取文件: {e}")
        return False

    original_content = content

    # 1. 删除硬编码的 Token 行
    patterns_to_remove = [
        r"MY_TOKEN\s*=\s*['\"][a-f0-9]+['\"]",  # MY_TOKEN = '...'
        r"#.*Token.*[:].*['\"][a-f0-9]+['\"]",  # 注释中的 Token
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, "", content)

    # 2. 检查是否已导入 config
    if "from config import" not in content:
        # 在导入部分添加 config 导入
        import_section = [
            "import tushare as ts",
            "import pandas as pd",
            "import json",
            "import os",
        ]

        for import_line in import_section:
            if import_line in content:
                # 在找到的第一个导入后添加
                content = content.replace(
                    import_line,
                    import_line + "\nfrom config import TUSHARE_TOKEN",
                    1
                )
                break

    # 3. 替换 MY_TOKEN 为 TUSHARE_TOKEN
    # 找到 ts.set_token(MY_TOKEN) 并替换
    content = re.sub(
        r"ts\.set_token\(MY_TOKEN\)",
        "ts.set_token(TUSHARE_TOKEN)",
        content
    )

    # 检查是否还有未替换的 MY_TOKEN 引用
    if "MY_TOKEN" in content:
        print(f"  ⚠️  警告: 文件中仍存在 MY_TOKEN 引用，请手动检查")

    # 如果内容没有变化，跳过
    if content == original_content:
        print(f"  ℹ️  文件无需修改（可能已经迁移或没有硬编码 Token）")
        return False

    # 备份原文件
    backup_path = filepath + ".backup"
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  ✅ 已创建备份: {backup_path}")
    except Exception as e:
        print(f"  ⚠️  备份失败: {e}")

    # 写入新内容
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 文件已更新")
        return True
    except Exception as e:
        print(f"  ❌ 写入文件失败: {e}")
        return False


def check_env_file():
    """检查 .env 文件是否存在"""
    env_path = Path(".env")

    if not env_path.exists():
        print("\n" + "="*60)
        print("⚠️  .env 文件不存在")
        print("="*60)
        print("\n请执行以下步骤：")
        print("1. 复制 .env.example 为 .env:")
        print("   cp .env.example .env")
        print("\n2. 编辑 .env 文件，填入您的 Token:")
        print("   TUSHARE_TOKEN=your_actual_token")
        print("\n3. 安装依赖:")
        print("   pip install python-dotenv")
        print("="*60)
        return False

    # 检查 Token 是否已配置
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv('TUSHARE_TOKEN')

        if not token or token == 'your_token_here':
            print("\n⚠️  .env 文件中的 Token 未配置")
            print("请在 .env 文件中设置: TUSHARE_TOKEN=your_actual_token")
            return False

        print(f"\n✅ .env 文件已配置 (Token: {token[:10]}...)")
        return True

    except ImportError:
        print("\n❌ 缺少依赖: python-dotenv")
        print("请运行: pip install python-dotenv")
        return False


def main():
    """主函数"""
    print("="*60)
    print("  Token 安全迁移脚本")
    print("="*60)

    # 检查 .env 文件
    if not check_env_file():
        print("\n⚠️  请先配置 .env 文件后再运行迁移")
        return

    # 确认执行
    print("\n即将修改以下文件:")
    for filepath in TARGET_FILES:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (不存在)")

    confirm = input("\n是否继续? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("已取消迁移")
        return

    # 执行迁移
    success_count = 0
    for filepath in TARGET_FILES:
        if os.path.exists(filepath):
            if migrate_file(filepath):
                success_count += 1

    # 总结
    print("\n" + "="*60)
    print(f"迁移完成: {success_count}/{len(TARGET_FILES)} 个文件已更新")
    print("="*60)

    if success_count > 0:
        print("\n后续步骤:")
        print("1. 检查代码是否正常运行: python 柱形选股-筛选.py")
        print("2. 如果一切正常，删除备份文件 (*.backup)")
        print("3. 提交代码时，确保 .env 文件在 .gitignore 中")

    print("\n详细信息请查看: 安全配置指南.md")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已中断迁移")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
