#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:41 PM
@File       : buy_and_hold.py.py
@Description: 
"""
# backend/business/backtest/strategies/implementations/buy_and_hold.py

import pandas as pd
from ..base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    一个最简单的买入并持有策略。

    逻辑:
    1. 在回测的第一天，将资金平均分配给股票池中的所有股票，并满仓买入。
    2. 此后不做任何操作，一直持有到回测结束。
    """

    def initialize(self):
        """初始化函数，这里我们设置一个标志位，确保只在第一天买入。"""
        self.context.stocks_bought = False
        print("BuyAndHoldStrategy initialized. Waiting for the first trading day to buy.")

    def handle_data(self, daily_bars: pd.DataFrame):
        """
        核心策略逻辑。
        """
        # 如果还没买过股票，并且当天有数据
        if not self.context.stocks_bought and not daily_bars.empty:

            # 计算每只股票应该分配多少资金
            num_stocks = len(daily_bars.index)
            if num_stocks == 0:
                return

            cash_per_stock = self.context.portfolio.initial_cash / num_stocks

            print(f"[{self.context.current_dt.date()}] First trading day. Attempting to buy {num_stocks} stocks.")

            # 遍历当天的所有股票并下单
            for stock_code in daily_bars.index:
                # 使用 order_value 函数，让 PortfolioManager 自动计算购买股数
                self.context.portfolio.order_value(stock_code, cash_per_stock)

            # 更新标志位，确保这个买入逻辑只执行一次
            self.context.stocks_bought = True