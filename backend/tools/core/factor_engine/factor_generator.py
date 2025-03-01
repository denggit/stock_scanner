#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:00 PM
@File       : factor_generator.py
@Description: 实现因子注册机制，支持动态加载因子类（如FactorMomentum）


设计要点：
1. 性能优化 - 因子并行计算：使用Dask或Ray实现跨CPU核的因子并行计算

2. 灵活拓展 - 插件式因子：通过装饰器自动注册新因子

使用方法：
mgmt = DatabaseManager()
df = mgmt.get_stock_daily(code="sh.605300", start_date="2024-01-01", end_date="2025-03-01")

# 获取所有注册的因子
factors = get_registered_factors()

# 计算单个因子
momentum_1m = factors['momentum_1m'](df['close'])
print("1个月动量因子:")
print(momentum_1m)
"""

from functools import wraps
from typing import Callable, Dict

import numpy as np
import pandas as pd

# 因子注册表
FACTOR_REGISTRY: Dict[str, Callable] = {}


def register_factor(name: str):
    """
    因子注册装饰器

    Args:
        name: 因子名称

    Returns:
        装饰器函数
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        FACTOR_REGISTRY[name] = wrapper
        return wrapper

    return decorator


# --- 动量类因子 ---
@register_factor(name='momentum_1m')
def momentum_1m(close: pd.Series) -> pd.Series:
    """
    1个月动量因子

    Args:
        close: 收盘价序列

    Returns:
        1个月动量值
    """
    return close.pct_change(21)  # 约21个交易日


@register_factor(name='momentum_12m')
def momentum_12m(close: pd.Series) -> pd.Series:
    """
    12个月动量因子，剔除最近1个月

    Args:
        close: 收盘价序列

    Returns:
        12个月动量值
    """
    return (close / close.shift(252)) / (close / close.shift(21)) - 1


# --- 波动率类因子 ---
@register_factor(name='volatility_1m')
def volatility_1m(returns: pd.Series) -> pd.Series:
    """
    1个月历史波动率

    Args:
        returns: 收益率序列

    Returns:
        1个月历史波动率
    """
    return returns.rolling(21).std() * np.sqrt(252)


# --- 均值回归类因子 ---
@register_factor(name='mean_reversion')
def mean_reversion(close: pd.Series, window: int = 20) -> pd.Series:
    """
    均值回归因子

    Args:
        close: 收盘价序列
        window: 移动平均窗口

    Returns:
        当前价格相对移动平均的偏离度
    """
    ma = close.rolling(window).mean()
    return (close - ma) / ma


# --- 技术指标类因子 ---
@register_factor(name='rsi')
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    相对强弱指标(RSI)

    Args:
        close: 收盘价序列
        window: RSI计算窗口

    Returns:
        RSI值
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --- 成交量类因子 ---
@register_factor(name='volume_price_corr')
def volume_price_corr(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    成交量-价格相关性因子

    Args:
        close: 收盘价序列
        volume: 成交量序列
        window: 相关系数计算窗口

    Returns:
        成交量和价格的滚动相关系数
    """
    return close.rolling(window).corr(volume)


def get_registered_factors() -> Dict[str, Callable]:
    """
    获取所有已注册的因子

    Returns:
        因子注册表
    """
    return FACTOR_REGISTRY
