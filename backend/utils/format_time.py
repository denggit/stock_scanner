#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 12:19 PM
@File       : format_time.py
@Description: 
"""


def format_time(seconds: int) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}小时{minutes}分钟{seconds}秒"
    elif minutes > 0:
        return f"{minutes}分钟{seconds}秒"
    else:
        return f"{seconds}秒"
