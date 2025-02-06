# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:33 PM
@File       : long_term_uptrend.py
@Description: 
"""

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class LongTermUpTrendStrategy(BaseStrategy):
    """
    长期上涨策略
    """

    def __init__(self):
        super().__init__(name="长期上涨策略", description="多头排列")
        self._params = {
            "ma_periods": [5, 10, 20, 30, 60, 120, 240],  # 多头排列均线
            "ma_period": 20,  # 回踩均线
            "continuous_days": 20,  # 连续多头排列的天数要求
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")

        # 1. 数据预处理 - 确保所有数值列都是float类型
        df = data.copy()

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in numeric_columns:
            df[column] = df[column].astype(float)

        # 生成条件
        signals = pd.DataFrame(index=df.index)

        # 2. 计算多头排列均线
        ma_periods = sorted(list(set(self._params["ma_periods"])))
        short_period = ma_periods[0]
        short_ma = CalIndicators.ema(df, short_period, 'close')
        conditions = pd.Series(True, index=df.index)
        for period in ma_periods[1:]:
            long_ma = CalIndicators.ema(df, period, 'close')
            conditions = conditions & (short_ma > long_ma)
            # 计算均线间的距离
            ma_distance = short_ma - long_ma
            # 计算均线间的距离的百分比
            ma_distance_percent = ma_distance / long_ma * 100
            signals[f'ma{short_period}_to_ma{period}'] = ma_distance_percent
            short_period = period
            short_ma = long_ma

        # 计算连续多头排列的天数
        continuous_trend = conditions.astype(int)
        continuous_days = continuous_trend.groupby((continuous_trend != continuous_trend.shift()).cumsum()).cumsum()
        df['trend'] = np.where(continuous_days >= self._params['continuous_days'], 1, 0)
        # 计算当前的连续趋势天数
        current_continuous_days = continuous_days.iloc[-1] if continuous_trend.iloc[-1] == 1 else 0

        basic_conditions = (
                (df['trend'] > 0) &
                (df['pe_ttm'] > 0) &
                (df['pb_mrq'] > 0)
        )

        # 计算当前股价到回踩均线距离的百分比
        df['ma_price'] = CalIndicators.ema(df, self._params['ma_period'], 'close')
        df['price_to_ma'] = ((df['close'] - df['ma_price']) / df['ma_price'] * 100).round(2)

        # 基础条件
        signals['signal'] = 0
        signals.loc[basic_conditions, 'signal'] = 1

        # 详细信息
        signals['trade_date'] = df['trade_date']
        signals['price'] = df['close']
        signals['ma_price'] = df['ma_price']
        signals['price_to_ma'] = df['price_to_ma']
        signals['pe_ttm'] = df['pe_ttm']
        signals['ps_ttm'] = df['ps_ttm']
        signals['pb_mrq'] = df['pb_mrq']
        signals['pcf_ncf_ttm'] = df['pcf_ncf_ttm']
        # 添加连续趋势天数信息
        signals['continuous_trend_days'] = current_continuous_days

        # 只返回最新一天的信息
        last_day = signals.iloc[-1]

        return last_day
