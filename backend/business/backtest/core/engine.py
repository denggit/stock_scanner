#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:37 PM
@File       : engine.py
@Description: 
"""
# backend/business/backtest/core/engine.py

from typing import Type, Dict, Any, List

from .data_provider import DataProvider
from .portfolio import PortfolioManager
from ..strategies.base import BaseStrategy


class BacktestEngine:
    """
    回测主引擎。

    职责:
    1. 初始化所有核心组件（DataProvider, PortfolioManager, Strategy）。
    2. 驱动整个回测流程，按交易日进行时间循环。
    3. 在每个交易日的正确时间点，调用策略的生命周期函数。
    4. 协调数据和投资组合的更新。
    5. 回测结束后，返回结果。
    """

    def __init__(self, start_date: str, end_date: str, initial_cash: float):
        """
        初始化回测引擎。

        Args:
            start_date (str): 回测开始日期, 格式 "YYYY-MM-DD"。
            end_date (str): 回测结束日期, 格式 "YYYY-MM-DD"。
            initial_cash (float): 初始资金。
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        print(f"BacktestEngine initialized for period: {start_date} to {end_date} with initial cash: {initial_cash}")

    def run(self,
            strategy_class: Type[BaseStrategy],
            stock_codes: List[str],
            strategy_params: Dict[str, Any] = None,
            adjust: str = '3'):
        """
        运行回测。

        Args:
            strategy_class (Type[BaseStrategy]): 要运行的策略类 (注意是类本身，不是实例)。
            stock_codes (List[str]): 股票池列表。
            strategy_params (Dict[str, Any], optional): 策略参数。Defaults to None.
            adjust (str, optional): 复权类型 ('1'后复权, '3'不复权)。Defaults to '3'.

        Returns:
            List[Dict]: 包含每日净值记录的列表。
        """
        strategy_params = strategy_params or {}

        # 1. 初始化所有组件
        print("--- Initializing components ---")
        data_provider = DataProvider(stock_codes, self.start_date, self.end_date, adjust)
        portfolio_manager = PortfolioManager(self.initial_cash, data_provider)
        strategy = strategy_class(portfolio_manager.context, strategy_params)

        # 2. 调用策略初始化
        print("--- Calling strategy.initialize() ---")
        strategy.initialize()

        # 3. --- 主事件循环 ---
        print("--- Starting main event loop ---")
        for current_date in data_provider.trading_days:
            try:
                # 更新上下文中的当前日期
                portfolio_manager.context.current_dt = current_date

                # a. 开盘前事件
                strategy.before_trading_start()

                # b. 获取当天数据并调用策略核心逻辑
                daily_bars = data_provider.get_daily_bars(current_date)

                # 只有当天有数据时才继续（处理停牌、或数据缺失的情况）
                if not daily_bars.empty:
                    strategy.handle_data(daily_bars)

                # c. 订单撮合与结算 (使用当天的数据)
                portfolio_manager.process_orders(daily_bars)

                # d. 收盘后事件
                strategy.after_trading_end()

                # e. 更新当日组合净值 (必须在所有交易完成后)
                portfolio_manager.update_portfolio_value(current_date, daily_bars)

                if (len(portfolio_manager.daily_net_values)) % 100 == 0:
                    print(
                        f"  Processed {len(portfolio_manager.daily_net_values)} days... Current Date: {current_date.date()}")
                        
            except Exception as e:
                print(f"ERROR: Failed to process date {current_date.date()}: {e}")
                # 继续处理下一个交易日，不中断整个回测
                continue

        # 4. 回测结束
        print("--- Backtest Finished ---")
        final_net_value = portfolio_manager.total_value
        print(f"Final net value: {final_net_value:.2f}")

        # 5. 返回结果 (暂时先返回每日净值)
        # 未来可以扩展为返回一个包含更多分析的Report对象
        return portfolio_manager.daily_net_values