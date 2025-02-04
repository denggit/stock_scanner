#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:33 PM
@File       : double_up.py
@Description: 
"""

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy


class DoubleUpStrategy(BaseStrategy):
    """扫描翻倍股票"""

    def __init__(self):
        super().__init__(name="扫描翻倍股", description="扫描过去周期内曾经翻倍过的股票")
        self.params = {
            "double_period": 20,  # 观察期（交易日）
            "times": 2.00,  # 翻倍倍数
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")

        # 1. 数据预处理 - 确保所有数值列都是float类型
        df = data.copy()
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)

        # 2. 计算在观察期内的最大涨幅
        period = self._params['double_period']
        times = float(self._params['times'])

        max_returns = []
        start_dates = []
        end_dates = []

        # 对每个时间点，向前看period个交易日，找出最大涨幅
        for i in range(len(df)):
            if i < period:
                max_returns.append(np.nan)
                start_dates.append(None)
                end_dates.append(None)
                continue

            window = df.iloc[i - period:i + 1]
            current_price = window['close'].iloc[-1]

            # 计算窗口内所有可能的收益率
            returns = []
            for j in range(len(window) - 1):
                start_price = window['close'].iloc[j]
                return_rate = current_price / start_price
                returns.append((return_rate, window.index[j], window.index[-1]))

            # 找出最大收益率及其对应的起始和结束时间
            if returns:
                max_return, start_date, end_date = max(returns, key=lambda x: x[0])
                max_returns.append(max_return)
                start_dates.append(start_date)
                end_dates.append(end_date)
            else:
                max_returns.append(np.nan)
                start_dates.append(None)
                end_dates.append(None)

        # 3. 生成信号
        signals = pd.DataFrame(index=df.index)
        signals['max_return'] = max_returns
        signals['start_date'] = start_dates
        signals['end_date'] = end_dates

        # 找出满足翻倍条件的点
        double_condition = signals['max_return'] >= times

        # 如果有满足条件的点，找出最大涨幅点
        if double_condition.any():
            max_return_index = signals.loc[double_condition, 'max_return'].idxmax()
            result = pd.Series({
                'signal': 1,
                'start_date': data[data.index == signals.loc[max_return_index, 'start_date']].trade_date.iloc[
                    0].strftime("%Y-%m-%d"),
                'end_date': data[data.index == signals.loc[max_return_index, 'end_date']].trade_date.iloc[0].strftime(
                    "%Y-%m-%d"),
                'max_return': signals.loc[max_return_index, 'max_return'],
                'start_price': df.loc[signals.loc[max_return_index, 'start_date'], 'close'],
                'end_price': df.loc[signals.loc[max_return_index, 'end_date'], 'close']
            })
        else:
            result = pd.Series({
                'signal': 0,
                'start_date': None,
                'end_date': None,
                'max_return': np.nan,
                'start_price': np.nan,
                'end_price': np.nan
            })

        return result
