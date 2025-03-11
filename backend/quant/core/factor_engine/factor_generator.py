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
    def volatility_1m(pct_chg: pd.Series) -> pd.Series:
        """1个月历史波动率"""
        return pct_chg.rolling(21).std() * np.sqrt(252)

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


class ShortTermFactors(BaseFactor):
    """短期交易因子 - 适合1-30天的做多策略"""

    @BaseFactor.register_factor(name='momentum_accel')
    @staticmethod
    def momentum_accel(close: pd.Series) -> pd.Series:
        """
        动量加速度因子 - 短期动量相对中期动量的加速度

        Args:
            close: 收盘价序列
        Returns:
            动量加速度因子值
        """
        ret_5 = close.pct_change(5)
        ret_3 = close.pct_change(3)
        return (ret_3 - ret_5) / ret_5.abs().replace(0, 1e-6)

    @BaseFactor.register_factor(name='gap_strength')
    @staticmethod
    def gap_strength(open_price: pd.Series, preclose: pd.Series) -> pd.Series:
        """
        跳空高开强度 - 相对于历史均值的跳空强度

        Args:
            open_price: 开盘价序列
            preclose: 前收盘价序列
        Returns:
            跳空强度值
        """
        gap = (open_price - preclose) / preclose
        gap_strength = gap - gap.rolling(20).mean()
        return gap_strength

    @BaseFactor.register_factor(name='vol_break')
    @staticmethod
    def volatility_breakout(high: pd.Series, low: pd.Series, preclose: pd.Series) -> pd.Series:
        """
        波动突破因子 - 当日波动相对历史波动突破的强度

        Args:
            high: 最高价序列
            low: 最低价序列
            preclose: 前收盘价序列
        Returns:
            波动突破指标（大于1表示突破）
        """
        # 计算真实波幅
        tr1 = (high - low)
        tr2 = (high - preclose).abs()
        tr3 = (low - preclose).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 当日波动与过去5日平均波幅的比值
        daily_range = (high - low) / preclose
        avg_range = true_range.rolling(5).mean() / preclose
        return daily_range / avg_range.replace(0, 1e-6)

    @BaseFactor.register_factor(name='pv_resonance')
    @staticmethod
    def price_volume_resonance(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """
        量价共振因子 - 价格创新高且成交量放大

        Args:
            close: 收盘价序列
            volume: 成交量序列
            window: 回顾窗口
        Returns:
            量价共振指标（越大越强）
        """
        price_ratio = close / close.rolling(window).max().shift()
        volume_ratio = volume / volume.rolling(window).mean().shift()
        return price_ratio * volume_ratio

    @BaseFactor.register_factor(name='block_strength')
    @staticmethod
    def block_trade_strength(amount: pd.Series, turn: pd.Series) -> pd.Series:
        """
        大单强度因子 - 成交额相对于流通市值的异常强度

        Args:
            amount: 成交额序列
            turn: 换手率序列(%)
        Returns:
            大单强度指标
        """
        # 近似流通市值 = 成交额 / (换手率/100)
        circ_mv = amount / (turn / 100 + 1e-6)

        # 成交额异常值
        amount_deviation = amount - amount.rolling(20).mean()

        return amount_deviation / circ_mv

    @BaseFactor.register_factor(name='upper_pressure')
    @staticmethod
    def upper_pressure(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        盘口压力因子 - 收盘价距离当日最高价的反转指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        Returns:
            盘口压力指标（越小越好）
        """
        return (high - close) / (high - low + 1e-6)

    @BaseFactor.register_factor(name='overnight_momentum')
    @staticmethod
    def overnight_momentum(open_price: pd.Series, preclose: pd.Series, window: int = 3) -> pd.Series:
        """
        隔夜动量因子 - 过去几天隔夜收益的平均值

        Args:
            open_price: 开盘价序列
            preclose: 前收盘价序列
            window: 平均窗口
        Returns:
            隔夜动量因子值
        """
        overnight_ret = (open_price - preclose) / preclose
        return overnight_ret.rolling(window).mean()

    @BaseFactor.register_factor(name='intraday_trend')
    @staticmethod
    def intraday_trend(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        日内趋势因子 - 收盘价相对开盘价在日内高低点范围的相对位置

        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        Returns:
            日内趋势指标（1为全天上涨，-1为全天下跌）
        """
        return (close - open_price) / (high - low + 1e-6)

    @BaseFactor.register_factor(name='value_momentum')
    @staticmethod
    def value_momentum(close: pd.Series, pe_ttm: pd.Series, window: int = 5) -> pd.Series:
        """
        估值动量因子 - 短期动量与PE估值的复合因子

        Args:
            close: 收盘价序列
            pe_ttm: PE(TTM)序列
            window: 动量计算窗口
        Returns:
            估值动量因子值
        """
        # 短期动量
        momentum = close.pct_change(window)

        # PE排名（越低越好）
        pe_rank = 1 - pe_ttm.rolling(20).rank(pct=True)

        # 复合因子
        return momentum * pe_rank

    @BaseFactor.register_factor(name='smart_money')
    @staticmethod
    def smart_money(close: pd.Series, open_price: pd.Series, pct_chg: pd.Series) -> pd.Series:
        """
        聪明钱因子 - 尾盘收益占比

        Args:
            close: 收盘价序列
            open_price: 开盘价序列
            pct_chg: 日涨跌幅(%)
        Returns:
            聪明钱指标（越大表示尾盘拉升越强）
        """
        # 模拟最后30分钟收益占比(假定交易时段为4小时)
        last_hour_ret = close / open_price.shift(3) - 1

        # 防止除0
        abs_pct_chg = pct_chg.abs() / 100 + 1e-6

        # 计算尾盘收益占比
        return (pct_chg / 100 - last_hour_ret) / abs_pct_chg


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
