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

        # 2. 计算多头排列均线
        ma_periods = sorted(list(set(self._params["ma_periods"])))
        short_ma = CalIndicators.ema(df, ma_periods[0], 'close')
        conditions = pd.Series(True, index=df.index)
        for period in ma_periods[1:]:
            long_ma = CalIndicators.ema(df, period, 'close')
            conditions = conditions & (short_ma > long_ma)
        df['trend'] = np.where(conditions, 1, -1)

        basic_conditions = (
                (df['trend'] > 0) &
                (df['pe_ttm'] > 0) &
                (df['pb_mrq'] > 0)
        )

        # 计算股价到回踩均线距离
        df['ma_price'] = CalIndicators.ema(df, self._params['ma_period'], 'close')
        df['price_to_ma'] = ((df['close'] - df['ma_price']) / df['ma_price'] * 100).round(2)

        # 生成条件
        signals = pd.DataFrame(index=df.index)

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

        # 只返回最新一天的信息
        last_day = signals.iloc[-1]

        return last_day
