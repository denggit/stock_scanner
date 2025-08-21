#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : technical_factors.py
@Description: 技术因子库
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import numpy as np
import pandas as pd

from ..core.factor.base_factor import register_technical_factor


# ==================== 动量类因子 ====================

@register_technical_factor(name='momentum_5d', description='5日动量因子')
def momentum_5d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """5日动量因子：过去5日收益率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['close'].pct_change(5)


@register_technical_factor(name='momentum_20d', description='20日动量因子')
def momentum_20d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """20日动量因子：过去20日收益率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['close'].pct_change(20)


@register_technical_factor(name='momentum_60d', description='60日动量因子')
def momentum_60d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """60日动量因子：过去60日收益率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['close'].pct_change(60)


# ==================== 波动率类因子 ====================

@register_technical_factor(name='volatility_20d', description='20日波动率因子')
def volatility_20d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """20日波动率因子：过去20日收益率的标准差
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    returns = data['close'].pct_change()
    return returns.rolling(20).std()


@register_technical_factor(name='volatility_60d', description='60日波动率因子')
def volatility_60d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """60日波动率因子：过去60日收益率的标准差
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    returns = data['close'].pct_change()
    return returns.rolling(60).std()


# ==================== 成交量类因子 ====================

@register_technical_factor(name='volume_ratio_5d', description='5日成交量比率因子')
def volume_ratio_5d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """5日成交量比率因子：当前成交量与过去5日平均成交量的比值
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['volume'] / data['volume'].rolling(5).mean()


@register_technical_factor(name='volume_ratio_20d', description='20日成交量比率因子')
def volume_ratio_20d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """20日成交量比率因子：当前成交量与过去20日平均成交量的比值
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['volume'] / data['volume'].rolling(20).mean()


# ==================== 价格位置类因子 ====================

@register_technical_factor(name='price_position_20d', description='20日价格位置因子')
def price_position_20d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """20日价格位置因子：当前价格在20日高低点之间的位置
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high_20d = data['high'].rolling(20).max()
    low_20d = data['low'].rolling(20).min()
    return (data['close'] - low_20d) / (high_20d - low_20d)


@register_technical_factor(name='price_position_60d', description='60日价格位置因子')
def price_position_60d(data: pd.DataFrame, **kwargs) -> pd.Series:
    """60日价格位置因子：当前价格在60日高低点之间的位置
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high_60d = data['high'].rolling(60).max()
    low_60d = data['low'].rolling(60).min()
    return (data['close'] - low_60d) / (high_60d - low_60d)


# ==================== 均线类因子 ====================

@register_technical_factor(name='ma_cross_5_20', description='5日与20日均线交叉因子')
def ma_cross_5_20(data: pd.DataFrame, **kwargs) -> pd.Series:
    """5日与20日均线交叉因子：5日均线相对20日均线的位置
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    ma5 = data['close'].rolling(5).mean()
    ma20 = data['close'].rolling(20).mean()
    return (ma5 - ma20) / ma20


@register_technical_factor(name='ma_cross_10_60', description='10日与60日均线交叉因子')
def ma_cross_10_60(data: pd.DataFrame, **kwargs) -> pd.Series:
    """10日与60日均线交叉因子：10日均线相对60日均线的位置
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    ma10 = data['close'].rolling(10).mean()
    ma60 = data['close'].rolling(60).mean()
    return (ma10 - ma60) / ma60


# ==================== 技术指标类因子 ====================

@register_technical_factor(name='rsi_14', description='14日RSI因子')
def rsi_14(data: pd.DataFrame, **kwargs) -> pd.Series:
    """14日RSI因子：相对强弱指数
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@register_technical_factor(name='rsi_21', description='21日RSI因子')
def rsi_21(data: pd.DataFrame, **kwargs) -> pd.Series:
    """21日RSI因子：相对强弱指数
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(21).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(21).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@register_technical_factor(name='bollinger_position', description='布林带位置因子')
def bollinger_position(data: pd.DataFrame, window: int = 20, num_std: float = 2, **kwargs) -> pd.Series:
    """布林带位置因子：价格在布林带中的相对位置
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        window: 计算窗口，默认 20
        num_std: 标准差倍数，默认 2
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return (close - lower) / (upper - lower)


@register_technical_factor(name='macd_histogram', description='MACD柱状图因子')
def macd_histogram(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.Series:
    """MACD柱状图因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        fast: 快线周期，默认 12
        slow: 慢线周期，默认 26
        signal: 信号线周期，默认 9
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line


@register_technical_factor(name='williams_r', description='威廉指标因子')
def williams_r(data: pd.DataFrame, window: int = 14, **kwargs) -> pd.Series:
    """威廉指标因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 14
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    low = data['low']
    close = data['close']
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    return (highest_high - close) / (highest_high - lowest_low) * -100


@register_technical_factor(name='cci', description='商品通道指数因子')
def cci(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """商品通道指数因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 20
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    low = data['low']
    close = data['close']
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window).mean()
    mad = typical_price.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (typical_price - sma) / (0.015 * mad)


# ==================== 高级技术因子 ====================

@register_technical_factor(name='kama', description='KAMA自适应移动平均因子')
def kama(data: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    """KAMA自适应移动平均因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        window: 计算窗口，默认 10
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']

    def calculate_kama(prices, window):
        if len(prices) < window:
            return prices.iloc[-1]

        change = abs(prices.iloc[-1] - prices.iloc[-window])
        volatility = sum(abs(prices.iloc[i] - prices.iloc[i - 1]) for i in range(1, len(prices)))

        if volatility == 0:
            er = 0
        else:
            er = change / volatility

        sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2

        return prices.iloc[-2] + sc * (prices.iloc[-1] - prices.iloc[-2])

    return close.rolling(window).apply(calculate_kama, raw=False)


@register_technical_factor(name='atr', description='平均真实波幅因子')
def atr(data: pd.DataFrame, window: int = 14, **kwargs) -> pd.Series:
    """平均真实波幅因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 14
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    low = data['low']
    close = data['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


@register_technical_factor(name='adx', description='平均趋向指数因子')
def adx(data: pd.DataFrame, window: int = 14, **kwargs) -> pd.Series:
    """平均趋向指数因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 14
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # 计算+DM和-DM
    high_diff = high - high.shift(1)
    low_diff = low.shift(1) - low

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    # 计算TR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([pd.Series(tr1), pd.Series(tr2), pd.Series(tr3)], axis=1).max(axis=1)

    # 计算平滑值
    plus_di = pd.Series(plus_dm).rolling(window).mean() / tr.rolling(window).mean() * 100
    minus_di = pd.Series(minus_dm).rolling(window).mean() / tr.rolling(window).mean() * 100

    # 计算DX和ADX
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx_result = dx.rolling(window).mean()

    return adx_result


# ==================== 自定义技术因子 ====================

@register_technical_factor(name='volume_price_momentum', description='成交量价格动量因子')
def volume_price_momentum(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    成交量价格动量因子：价格动量与成交量变化的结合
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        window: 计算窗口，默认 20
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
    price_momentum = close.pct_change(window)
    volume_momentum = volume.pct_change(window)

    return price_momentum * volume_momentum


@register_technical_factor(name='gap_strength', description='跳空强度因子')
def gap_strength(data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    跳空强度因子：开盘价相对前收盘价的跳空幅度
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, preclose 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open_price = data['open']
    preclose = data['preclose']
    return (open_price - preclose) / preclose


@register_technical_factor(name='intraday_volatility', description='日内波动率因子')
def intraday_volatility(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    日内波动率因子：基于日内高低点的波动率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 20
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    low = data['low']
    close = data['close']
    intraday_range = (high - low) / close
    return intraday_range.rolling(window).mean()
