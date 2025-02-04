#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 1:22 AM
@File       : indicators.py
@Description: 
"""
from typing import Tuple

import numpy as np
import pandas as pd


class CalIndicators:
    def __init__(self):
        pass

    @staticmethod
    def macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean().round(2)
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean().round(2)
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean().round(2)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    @staticmethod
    def ema(df: pd.DataFrame, period: int, cal_value: str = 'close') -> pd.Series:
        """计算EMA指标"""
        return df[cal_value].ewm(span=period, min_periods=period, adjust=False).mean().round(2)

    @staticmethod
    def sma(df: pd.DataFrame, period: int, cal_value: str = 'close') -> pd.Series:
        """计算SMA指标"""
        return df[cal_value].rolling(window=period, min_periods=period, center=False).mean().round(2)

    @staticmethod
    def amplitude(df: pd.DataFrame, lookback_period: int = 14) -> pd.Series:
        """计算振幅
        振幅 = (当日最高价 - 当日最低价) / 前收盘价 × 100%
        """
        amplitude = ((df['high'] - df['low']) / df['close'].shift(1) * 100).round(2)
        if lookback_period > 1:
            amplitude = amplitude.rolling(window=lookback_period).mean().round(2)
        return amplitude

    @staticmethod
    def volatility(df: pd.DataFrame, lookback_period: int = 14, annualized: bool = True) -> pd.Series:
        """计算波动率
        Args:
            df: 数据框
            lookback_period: 回看周期
            annualized: 是否年化，默认为True
        Returns:
            波动率序列
        """
        # 计算对数收益率
        log_returns = np.log(df['close'] / df['close'].shift(1))
        # 计算滚动标准差
        vol = log_returns.rolling(window=lookback_period).std()

        if annualized:
            # 假设一年250个交易日，年化处理
            vol = vol * np.sqrt(250)

        return vol.round(4) * 100  # 转换为百分比并保留4位小数

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, ma_period: int = 20, bollinger_k: int = 2) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        bb_mid = CalIndicators.ema(df, ma_period)
        std = df['close'].rolling(window=ma_period).std()
        bb_upper = (bb_mid + std * bollinger_k).round(2)
        bb_lower = (bb_mid - std * bollinger_k).round(2)
        return bb_mid, bb_upper, bb_lower

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - 100 / (1 + rs)).round(2)

    @staticmethod
    def support(df: pd.DataFrame, lookback_period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """计算支撑"""
        return df['low'].rolling(window=lookback_period).min()

    @staticmethod
    def resistance(df: pd.DataFrame, lookback_period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """计算阻力"""
        return df['high'].rolling(window=lookback_period).max()
