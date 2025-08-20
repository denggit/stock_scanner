#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : __init__.py
@Description: 因子模块
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

from .base_factor import BaseFactor
from .factor_registry import FactorRegistry, factor_registry, register_factor, get_factor, list_factors

__all__ = [
    'BaseFactor',
    'FactorRegistry',
    'factor_registry',
    'register_factor',
    'get_factor',
    'list_factors',
    'library'
]
