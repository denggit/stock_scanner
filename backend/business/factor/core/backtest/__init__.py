#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : __init__.py
@Description: 回测模块
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

from .backtest_engine import FactorBacktestEngine

__all__ = [
    'FactorBacktestEngine'
]
