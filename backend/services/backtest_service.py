#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:59 PM
@File       : backtest_service.py
@Description: 
"""

from backend.data.stock_data_fetcher import StockDataFetcher
from backend.utils.logger import setup_logger

logger = setup_logger("backtest_service")


class BacktestService:
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.strategies = {
        }

    def run_backtest(self, strategy: str, start_date: str, end_date: str, initial_capital: float, params: dict):
        """运行回测"""
        pass

    def get_backtest_results(self, backtest_id: str):
        """获取回测结果"""
        pass
