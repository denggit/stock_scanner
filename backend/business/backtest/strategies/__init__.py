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
from .core import BaseStrategy
from .core.managers import DataManager, PositionManager, TradeManager, TradeLogger
from .core.utils import SignalUtils, ParameterUtils, PriceUtils, DataUtils

# 策略工厂 (暂时注释掉，factory 模块尚未实现)
# from .factory import StrategyFactory, create_strategy, list_strategies, register_strategy

# 具体策略实现
from .implementations.channel import RisingChannelStrategy

__all__ = [
    # 核心基础
    'BaseStrategy',
    'DataManager',
    'PositionManager', 
    'TradeManager',
    'TradeLogger',
    
    # 通用工具
    'SignalUtils',
    'ParameterUtils',
    'PriceUtils', 
    'DataUtils',
    
    # 策略工厂 (暂时注释掉)
    # 'StrategyFactory',
    # 'create_strategy',
    # 'list_strategies',
    # 'register_strategy',
    
    # 策略实现
    'RisingChannelStrategy'
]
