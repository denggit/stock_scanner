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

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
        """计算ROC (Rate of Change) 指标
        ROC = (当前收盘价 - n日前收盘价) / n日前收盘价 × 100
        """
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100).round(2)
        return roc

    @staticmethod
    def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算KDJ指标
        Args:
            df: 数据框
            n: RSV计算周期
            m1: K值平滑因子
            m2: D值平滑因子
        Returns:
            K, D, J值序列
        """
        low_list = df['low'].rolling(window=n).min()
        high_list = df['high'].rolling(window=n).max()
        
        rsv = ((df['close'] - low_list) / (high_list - low_list) * 100).round(2)
        
        k = pd.Series(index=df.index, dtype=float)
        d = pd.Series(index=df.index, dtype=float)
        
        k[0] = 50
        d[0] = 50
        
        for i in range(1, len(df)):
            k[i] = (2/3 * k[i-1] + 1/3 * rsv[i]).round(2)
            d[i] = (2/3 * d[i-1] + 1/3 * k[i]).round(2)
            
        j = (3 * k - 2 * d).round(2)
        
        return k, d, j

    @staticmethod
    def dmi(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算DMI (Directional Movement Index) 指标
        Returns:
            PDI(+DI), MDI(-DI), ADX
        """
        # 计算真实波幅（TR）
        tr = pd.DataFrame(index=df.index)
        tr['hl'] = df['high'] - df['low']
        tr['hc'] = abs(df['high'] - df['close'].shift(1))
        tr['lc'] = abs(df['low'] - df['close'].shift(1))
        tr = tr.max(axis=1)
        
        # 计算方向变动（DM）
        pdm = df['high'] - df['high'].shift(1)
        mdm = df['low'].shift(1) - df['low']
        
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
        
        # 计算平滑值
        tr14 = tr.ewm(alpha=1/period, adjust=False).mean()
        pdm14 = pdm.ewm(alpha=1/period, adjust=False).mean()
        mdm14 = mdm.ewm(alpha=1/period, adjust=False).mean()
        
        # 计算DI
        pdi = (pdm14 / tr14 * 100).round(2)
        mdi = (mdm14 / tr14 * 100).round(2)
        
        # 计算DX和ADX
        dx = (abs(pdi - mdi) / (pdi + mdi) * 100).round(2)
        adx = dx.ewm(alpha=1/period, adjust=False).mean().round(2)
        
        return pdi, mdi, adx
