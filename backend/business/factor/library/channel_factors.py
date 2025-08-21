#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : channel_factors.py
@Description: 通道分析因子库
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import pandas as pd

from ..core.factor.base_factor import register_technical_factor


# ==================== 通道分析因子 ====================

@register_technical_factor(name='channel_distance', description='通道距离因子')
def channel_distance(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    通道距离因子：价格在通道中的相对位置
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 20
        **kwargs: 其他参数
        
    Returns:
        通道距离因子值 (-1到1之间，-1表示在下沿，1表示在上沿)
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # 计算通道边界
    upper_channel = high.rolling(window).max()
    lower_channel = low.rolling(window).min()

    # 计算价格在通道中的相对位置
    channel_range = upper_channel - lower_channel
    price_position = (close - lower_channel) / channel_range

    # 转换为-1到1的范围
    return 2 * price_position - 1


@register_technical_factor(name='channel_breakout', description='通道突破因子')
def channel_breakout(data: pd.DataFrame, window: int = 20, threshold: float = 0.05, **kwargs) -> pd.Series:
    """
    通道突破因子：检测价格突破通道边界
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 20
        threshold: 突破阈值，默认 0.05
        **kwargs: 其他参数
        
    Returns:
        突破因子值 (1表示向上突破，-1表示向下突破，0表示无突破)
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # 计算通道边界
    upper_channel = high.rolling(window).max()
    lower_channel = low.rolling(window).min()

    # 检测突破
    upper_breakout = (close > upper_channel * (1 + threshold)).astype(int)
    lower_breakout = (close < lower_channel * (1 - threshold)).astype(int)

    return upper_breakout - lower_breakout


@register_technical_factor(name='channel_width', description='通道宽度因子')
def channel_width(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    通道宽度因子：通道的相对宽度
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 20
        **kwargs: 其他参数
        
    Returns:
        通道宽度因子值
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # 计算通道边界
    upper_channel = high.rolling(window).max()
    lower_channel = low.rolling(window).min()

    # 计算通道宽度（相对于价格）
    channel_width = (upper_channel - lower_channel) / close

    return channel_width


@register_technical_factor(name='channel_trend', description='通道趋势因子')
def channel_trend(data: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    通道趋势因子：通道的趋势方向
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, close 等列
        window: 计算窗口，默认 20
        **kwargs: 其他参数
        
    Returns:
        通道趋势因子值 (正值表示上升趋势，负值表示下降趋势)
    """
    high = data['high']
    low = data['low']
    
    # 计算通道中点
    upper_channel = high.rolling(window).max()
    lower_channel = low.rolling(window).min()
    channel_mid = (upper_channel + lower_channel) / 2

    # 计算通道中点的变化率
    channel_trend = channel_mid.pct_change()

    return channel_trend
