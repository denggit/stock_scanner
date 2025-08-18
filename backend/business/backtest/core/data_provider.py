#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:30 PM
@File       : data_provider.py
@Description: 
"""
# backend/business/backtest/core/data_provider.py

from typing import List, Dict

import pandas as pd

# 最终修正：导入专用于数据获取的 StockDataFetcher，实现最小权限和最佳实践
from backend.business.data.data_fetcher import StockDataFetcher


class DataProvider:
    """
    统一数据提供者 (最终版)。

    职责:
    1. 使用专为读取设计的 StockDataFetcher 来加载数据，确保安全与解耦。
    2. 在回测开始时，预加载所有需要的行情数据。
    3. 提供标准化的API，供回测引擎的其他部分查询数据。
    """

    def __init__(self, stock_codes: List[str], start_date: str, end_date: str, adjust: str = '3'):
        """
        初始化DataProvider。

        Args:
            stock_codes (List[str]): 需要回测的股票代码列表。
            start_date (str): 回测开始日期 (e.g., "2020-01-01")。
            end_date (str): 回测结束日期 (e.g., "2023-12-31")。
            adjust (str): 复权类型, '1'后复权, '2'前复权, '3'不复权。默认为'3'。
        """
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.adjust = adjust

        self.market_data: pd.DataFrame = self._load_market_data()
        self.trading_days: pd.DatetimeIndex = self._get_trading_days()
        self.factor_data: Dict[str, pd.DataFrame] = {}
        print("DataProvider initialized: Market data loaded.")

    def _load_market_data(self) -> pd.DataFrame:
        """
        从数据库加载所有股票在指定时间范围内的日线行情数据。

        【核心修正】: 调用 StockDataFetcher 来获取数据。
        """
        print(f"Loading market data via StockDataFetcher for {len(self.stock_codes)} stocks...")

        # 实例化数据获取器
        fetcher = StockDataFetcher()
        all_data_dict = {}

        for code in self.stock_codes:
            # 调用 get_stock_daily 方法
            stock_df = fetcher.fetch_stock_data(code=code, start_date=self.start_date, end_date=self.end_date,
                                                adjust=self.adjust)
            if not stock_df.empty:
                # 确保 trade_date 是 datetime 类型，并设为索引
                stock_df['date'] = pd.to_datetime(stock_df['trade_date'])
                all_data_dict[code] = stock_df.set_index('date')

        fetcher.close()  # 完成查询后关闭连接

        if not all_data_dict:
            raise ValueError("未能从数据库加载任何股票数据，请检查日期范围和股票列表。")

        # 将所有DataFrame合并成一个大的、使用MultiIndex的DataFrame
        panel_data = pd.concat(all_data_dict.values(), keys=all_data_dict.keys(), names=['stock_code', 'date'])

        # 调整索引顺序为 (date, stock_code) 以优化按日期的查询
        panel_data = panel_data.swaplevel('stock_code', 'date').sort_index()

        print(f"Market data loaded successfully. Shape: {panel_data.shape}")
        return panel_data

    def _get_trading_days(self) -> pd.DatetimeIndex:
        """从已加载的数据中提取所有唯一的交易日。"""
        if self.market_data.empty:
            return pd.to_datetime([])

        trading_days = self.market_data.index.get_level_values('date').unique().sort_values()
        print(f"Found {len(trading_days)} trading days.")
        return trading_days

    def get_daily_bars(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        获取指定日期的所有股票的行情截面数据。
        """
        try:
            return self.market_data.loc[date]
        except KeyError:
            return pd.DataFrame()

    def get_factor_values(self, factor_name: str, date: pd.Timestamp) -> pd.Series:
        """获取指定日期的指定因子在所有股票上的截面值。"""
        # (待实现)
        pass

        # 请将此方法添加到 DataProvider 类的内部

    def get_history_data(self, stock_code: str, end_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
        """
        获取单个股票到指定日期为止的一段历史行情数据。

        Args:
            stock_code (str): 股票代码。
            end_date (pd.Timestamp): 历史数据的结束日期 (包含此日期)。
            lookback_days (int): 回溯的天数。

        Returns:
            pd.DataFrame: 包含历史行情数据的DataFrame，按日期升序排列。
        """
        try:
            # 高效地从大的MultiIndex DataFrame中切片出单个股票的所有历史数据
            stock_df = self.market_data.loc[(slice(None), stock_code), :]

            # 重置索引，让 'date' 成为普通列，以便进行日期比较
            stock_df = stock_df.reset_index(level='stock_code', drop=True)

            # 再次切片，获取end_date之前lookback_days天的数据
            end_loc = stock_df.index.get_loc(end_date)
            start_loc = max(0, end_loc - lookback_days + 1)

            return stock_df.iloc[start_loc: end_loc + 1]

        except KeyError:
            # 如果当天或该股票没有任何数据，返回空DataFrame
            return pd.DataFrame()
