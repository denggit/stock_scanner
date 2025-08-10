#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略工厂模块
提供统一的策略创建和管理接口
"""

from .strategy_factory import (
    StrategyFactory, StrategyRegistry, 
    create_strategy, list_strategies, register_strategy
)

__all__ = [
    'StrategyFactory',
    'StrategyRegistry',
    'create_strategy',
    'list_strategies', 
    'register_strategy'
]
