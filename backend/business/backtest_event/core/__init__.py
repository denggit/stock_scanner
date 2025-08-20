#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测框架核心模块
"""

from .backtest_engine import BacktestEngine, BacktestFactory
from .base_strategy import BaseStrategy
from .data_manager import DataManager
from .result_analyzer import ResultAnalyzer

__all__ = [
    'BaseStrategy',
    'BacktestEngine',
    'BacktestFactory',
    'DataManager',
    'ResultAnalyzer'
]
