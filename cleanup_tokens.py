#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token 清理脚本 - 上传到 GitHub 前的安全清理
"""

import os
import re
from pathlib import Path

# 真实 Token（实际使用时替换为您的 Token）
REAL_TOKEN = '8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7'
# 占位符
PLACEHOLDER = 'your_tushare_token_here'

# 需要清理的文件模式
PATTERNS = {
    '.py': r'(MY_TOKEN\s*=\s*[\'\"])' + REAL_TOKEN + r'([\'\"])',
    '.py': r'(TUSHARE_TOKEN\s*=\s*[\'\"])' + REAL_TOKEN + r'([\'\"])',
    '.py': r'(os\.environ\[.TUSHARE_TOKEN.\]\s*=\s*[\'\"])' + REAL_TOKEN + r'([\'\"])',
    '.py': r'(Token.*[:]\s*[\'\"])' + REAL_TOKEN + r'([\'\"])',
}

# 跳过的文件（不清理）
SKIP_FILES = ['.env', '.env.example']

def clean_file(filepath):
    """清理单个文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 清理各种格式的 Token
        content = re.sub(
            r"(MY_TOKEN\s*=\s*['\"])8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7(['\"])",
            r"\1your_tushare_token_here\2",
            content
        )

        content = re.sub(
            r"(TUSHARE_TOKEN\s*=\s*['\"])8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7(['\"])",
            r"\1your_tushare_token_here\2",
            content
        )

        content = re.sub(
            r"(os\.environ\[['\"]TUSHARE_TOKEN['\"]\]\s*=\s*['\"])8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7(['\"])",
            r"\1your_tushare_token_here\2",
            content
        )

        # 文档中的示例
        content = re.sub(
            r"(TUSHARE_TOKEN=)8f5cd68a38bb5bd3fe035ff544bc8c71c6c97e70b081d9a58f8d0bd7",
            r"\1your_tushare_token_here",
            content
        )

        # 如果内容有变化，写回文件
        if content != original_content:
            # 创建备份
            backup_path = str(filepath) + '.token_backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # 写入清理后的内容
            with open(str(filepath), 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        return False

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False

def main():
    print("="*60)
    print("  Token 清理脚本 - GitHub 上传前安全检查")
    print("="*60)

    # 搜索包含 Token 的文件
    print("\n[步骤 1] 搜索包含真实 Token 的文件...")

    files_with_token = []

    # 搜索 assets 目录
    for filepath in Path('assets').rglob('*'):
        if filepath.is_file() and filepath.suffix in ['.py', '.md']:
            filename = filepath.name
            if filename in SKIP_FILES:
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if REAL_TOKEN in content:
                        files_with_token.append(filepath)
            except:
                continue

    print(f"  找到 {len(files_with_token)} 个包含真实 Token 的文件")

    if not files_with_token:
        print("\n✅ 没有找到需要清理的文件")
        return

    print("\n文件列表:")
    for i, filepath in enumerate(files_with_token, 1):
        print(f"  {i}. {filepath}")

    # 清理文件
    print("\n[步骤 2] 清理文件...")

    cleaned_count = 0
    for filepath in files_with_token:
        print(f"  清理: {filepath.name}...", end=' ')
        if clean_file(filepath):
            print("✅")
            cleaned_count += 1
        else:
            print("⏭️  (无需清理)")

    print(f"\n✅ 已清理 {cleaned_count} 个文件")

    # 验证
    print("\n[步骤 3] 验证清理结果...")

    remaining_tokens = []
    for filepath in Path('assets').rglob('*'):
        if filepath.is_file() and filepath.suffix in ['.py', '.md']:
            if filepath.name in SKIP_FILES:
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    if REAL_TOKEN in f.read():
                        remaining_tokens.append(filepath)
            except:
                continue

    if remaining_tokens:
        print(f"  ⚠️  警告: 还有 {len(remaining_tokens)} 个文件包含 Token:")
        for filepath in remaining_tokens:
            print(f"    - {filepath}")
    else:
        print("  ✅ 所有 Token 已清理完毕")

    # 检查 .gitignore
    print("\n[步骤 4] 检查 .gitignore...")

    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
            if '.env' in gitignore_content:
                print("  ✅ .env 已在 .gitignore 中")
            else:
                print("  ⚠️  .env 不在 .gitignore 中，建议添加")
    else:
        print("  ⚠️  找不到 .gitignore 文件")

    # 总结
    print("\n" + "="*60)
    print("  清理完成")
    print("="*60)

    if cleaned_count > 0:
        print(f"\n已创建备份文件 (*.token_backup)")
        print("如果确认无误，可以删除备份文件:")
        print("  find assets -name '*.token_backup' -delete")

    if remaining_tokens:
        print("\n⚠️  仍有文件包含真实 Token，请手动检查")
    else:
        print("\n✅ 可以安全上传到 GitHub")

if __name__ == '__main__':
    main()
