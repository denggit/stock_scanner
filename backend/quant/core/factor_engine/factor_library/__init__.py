#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : AI Assistant
@Date       : 2024-12-19
@File       : __init__.py
@Description: 因子库模块 - 包含各种量化因子的实现

该模块提供了一套完整的因子库架构，支持：
1. 技术指标因子
2. 通道分析因子
3. 统计类因子
4. 基本面因子

所有因子都通过统一的注册机制进行管理，支持动态加载和配置。
"""

from .channel_analysis import *

__version__ = "1.0.0"
__author__ = "AI Assistant" 