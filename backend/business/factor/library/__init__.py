#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : __init__.py
@Description: 因子库模块
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

from . import akshare_factors
from . import channel_factors
from . import fundamental_factors
# 导入所有因子库，自动注册所有因子
from . import technical_factors
from . import worldquant_factors

__all__ = [
    'technical_factors',
    'fundamental_factors',
    'worldquant_factors',
    'channel_factors',
    'akshare_factors'
]
