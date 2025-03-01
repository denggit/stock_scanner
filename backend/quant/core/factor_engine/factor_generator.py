#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:00 PM
@File       : factor_generator.py
@Description: 建立基础因子库，实现因子注册机制，支持动态加载因子

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

# 全局因子注册表
FACTOR_REGISTRY: Dict[str, Callable] = {}


class BaseFactor:
    """因子基类，提供基础功能和通用方法"""

    @staticmethod
    def register_factor(name: str):
        """因子注册装饰器"""

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            FACTOR_REGISTRY[name] = wrapper
            return wrapper

        return decorator


class MomentumFactors(BaseFactor):
    """动量类因子"""

    @BaseFactor.register_factor(name='momentum_1m')
    @staticmethod
    def momentum_1m(close: pd.Series) -> pd.Series:
        """
        1个月动量因子

        Args:
            close: 收盘价序列
        Returns:
            1个月动量值
        """
        return close.pct_change(21)

    @BaseFactor.register_factor(name='momentum_12m')
    @staticmethod
    def momentum_12m(close: pd.Series) -> pd.Series:
        """
        12个月动量因子，剔除最近1个月
        """
        return (close / close.shift(252)) / (close / close.shift(21)) - 1


class VolatilityFactors(BaseFactor):
    """波动率类因子"""

    @BaseFactor.register_factor(name='volatility_1m')
    @staticmethod
    def volatility_1m(returns: pd.Series) -> pd.Series:
        """1个月历史波动率"""
        return returns.rolling(21).std() * np.sqrt(252)

    @BaseFactor.register_factor(name='parkinson_volatility')
    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
        """Parkinson波动率"""
        return (np.log(high / low) ** 2 / (4 * np.log(2))).rolling(window).mean().pow(0.5)


class MeanReversionFactors(BaseFactor):
    """均值回归类因子"""

    @BaseFactor.register_factor(name='mean_reversion')
    @staticmethod
    def mean_reversion(close: pd.Series, window: int = 20) -> pd.Series:
        """均值回归因子"""
        ma = close.rolling(window).mean()
        return (close - ma) / ma

    @BaseFactor.register_factor(name='bollinger_score')
    @staticmethod
    def bollinger_score(close: pd.Series, window: int = 20) -> pd.Series:
        """布林带得分"""
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        return (close - ma) / (2 * std)


class TechnicalFactors(BaseFactor):
    """技术指标类因子"""

    @BaseFactor.register_factor(name='rsi')
    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """相对强弱指标(RSI)"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @BaseFactor.register_factor(name='macd')
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD指标"""
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line


class VolumeFactors(BaseFactor):
    """成交量类因子"""

    @BaseFactor.register_factor(name='volume_price_corr')
    @staticmethod
    def volume_price_corr(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """成交量-价格相关性因子"""
        return close.rolling(window).corr(volume)

    @BaseFactor.register_factor(name='obv')
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """能量潮指标(OBV)"""
        return (np.sign(close.diff()) * volume).cumsum()


def get_registered_factors() -> Dict[str, Callable]:
    """获取所有已注册的因子"""
    return FACTOR_REGISTRY


def get_factor_by_type(factor_type: str) -> Dict[str, Callable]:
    """
    按类型获取因子

    Args:
        factor_type: 因子类型名称（如 'momentum', 'volatility' 等）
    Returns:
        该类型的所有因子字典
    """
    return {name: func for name, func in FACTOR_REGISTRY.items()
            if name.startswith(factor_type.lower())}
