#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通道分析器模块
提供通道分析相关的管理器和工具函数
"""

from .manager import ChannelAnalyzerManager
from .utils import ChannelAnalysisUtils, RegressionUtils, parse_r2_bounds

__all__ = [
    'ChannelAnalyzerManager',
    'ChannelAnalysisUtils',
    'RegressionUtils',
    'parse_r2_bounds'
]
