# -*- coding: utf-8 -*-
"""
配置文件 - 统一管理敏感信息
所有敏感信息都通过环境变量获取，不硬编码在代码中
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def get_tushare_token():
    """
    获取 Tushare Token

    优先级：
    1. 环境变量 TUSHARE_TOKEN
    2. .env 文件中的配置
    3. 抛出异常（要求必须配置）

    Returns:
        str: Tushare API Token

    Raises:
        ValueError: 如果没有配置 Token
    """
    token = os.getenv('TUSHARE_TOKEN')

    if not token or token == 'your_token_here':
        raise ValueError(
            "❌ 未配置 Tushare Token！\n"
            "请执行以下步骤：\n"
            "1. 复制 .env.example 为 .env\n"
            "2. 在 .env 文件中填入您的 Token:\n"
            "   TUSHARE_TOKEN=your_actual_token\n"
            "3. 安装依赖: pip install python-dotenv"
        )

    return token


# 其他配置项（可以继续扩展）
def get_config(key, default=None):
    """
    获取配置项

    Args:
        key (str): 配置键名
        default: 默认值

    Returns:
        配置值或默认值
    """
    return os.getenv(key, default)


# 便捷导出
TUSHARE_TOKEN = get_tushare_token()
