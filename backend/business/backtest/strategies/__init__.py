#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略模块
包含各种交易策略实现

新架构组织：
- core/: 核心通用组件（所有策略都使用）
- analyzers/: 分析器模块（按分析类型分组）
- implementations/: 具体策略实现（按策略类型分类）
- factory/: 策略工厂（工厂模式）
"""

# 核心组件
from .base import BaseStrategy

# 具体策略实现
from .implementations.rising_channel_strategy import (
    RisingChannelStrategy,
    run_rising_channel_backtest,
    run_rising_channel_quick_test,
    create_rising_channel_strategy
)

__all__ = [
    # 核心基础
    'BaseStrategy',

    # 策略实现
    'RisingChannelStrategy',
    
    # 上升通道策略相关函数
    'run_rising_channel_backtest',
    'run_rising_channel_quick_test',
    'create_rising_channel_strategy'
]
