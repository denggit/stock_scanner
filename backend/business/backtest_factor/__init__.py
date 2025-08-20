#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : __init__.py
@Description: 因子回测框架主模块
"""

from .factor_framework import FactorFramework
from .core.base_factor import BaseFactor, CommonFactors
from .core.data_manager import FactorDataManager
from .core.factor_engine import FactorEngine
from .core.backtest_engine import FactorBacktestEngine
from .core.analyzer import FactorAnalyzer
from .core.report_generator import FactorReportGenerator

__all__ = [
    'FactorFramework',
    'BaseFactor',
    'CommonFactors',
    'FactorDataManager',
    'FactorEngine',
    'FactorBacktestEngine',
    'FactorAnalyzer',
    'FactorReportGenerator'
]

__version__ = '1.0.0'
__author__ = 'Zijun Deng'
__description__ = '基于vectorbt的因子回测框架，提供完整的因子研究到策略落地流程'
