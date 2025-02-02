#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 6:33 PM
@File       : stock_data_fetcher.py
@Description: 
"""
import datetime
from typing import Optional, Dict

import pandas as pd

from backend.data.database import DatabaseManager
from backend.utils.indicators import CalIndicators


class StockDataFetcher:

    def __init__(self):
        self.db = DatabaseManager()

    def _calculate_indicators(self, df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """
        计算指标
        :param df: 数据
        :param indicators: 指标
        :return: 数据
        """
        keys = [key.lower() for key in list(indicators.keys())]
        if 'MA' in keys:
            for period in indicators['MA']:
                df[f'MA_{period}'] = CalIndicators.ema(df, period)

        if 'MACD' in keys:
            df['macd'], df['signal'], df['hist'] = CalIndicators.macd(df)

        if 'RSI' in keys:
            df['rsi'] = CalIndicators.rsi(df, period=indicators['RSI'])

    def fetch_stock_data(
            self,
            code: str, 
            period: str = 'daily',
            start_date: Optional[str] = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            end_date: Optional[str] = datetime.date.today().strftime("%Y-%m-%d"),
            indicators: Dict = None
    ) -> pd.DataFrame:
        """
        获取股票数据
        :param code: 股票代码
        :param period: 数据周期，可选值：daily, weekly, monthly
        :param start_date: 开始日期，格式为YYYY-MM-DD
        :param end_date: YYYY-MM-DD
        :param indicators:
        """
        if period.lower().startswith('d'):
            df = self.db.get_stock_daily(code=code, start_date=start_date, end_date=end_date)
        # elif period.lower().startswith('w'):
        #     df = self.db.get_stock_weekly(code=code, start_date=start_date, end_date=end_date)
        # elif period.lower().startswith('m'):
        #     df = self.db.get_stock_monthly(code=code, start_date=start_date, end_date=end_date)
        else:
            raise ValueError(f"Invalid period: {period}")
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        :return: 股票列表
        """
        return self.db.get_stock_list()
