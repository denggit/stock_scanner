#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : __init__.py
@Description: 因子回测框架核心模块
"""

from .base_factor import BaseFactor
from .factor_engine import FactorEngine
from .data_manager import FactorDataManager
from .backtest_engine import FactorBacktestEngine
from .analyzer import FactorAnalyzer
from .report_generator import FactorReportGenerator

__all__ = [
    'BaseFactor',
    'FactorEngine', 
    'FactorDataManager',
    'FactorBacktestEngine',
    'FactorAnalyzer',
    'FactorReportGenerator'
]
