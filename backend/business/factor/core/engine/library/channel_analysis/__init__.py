#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : AI Assistant
@Date       : 2024-12-19
@File       : __init__.py
@Description: 通道分析因子模块

该模块实现了各种通道分析算法，包括：
1. 上升通道回归分析
2. 下降通道分析
3. 通道状态管理
4. 锚点检测算法

主要功能：
- 自动检测和维护股票价格通道
- 提供通道突破和重锚机制
- 支持多种通道状态监控
"""

from .rising_channel import AscendingChannelRegression
from .channel_state import ChannelState
from .pivot_detector import PivotDetector

__all__ = [
    'AscendingChannelRegression',
    'ChannelState',
    'PivotDetector'
]
