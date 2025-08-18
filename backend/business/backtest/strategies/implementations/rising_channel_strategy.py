#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 10:07 PM
@File       : rising_channel_strategy.py
@Description: 
"""
# backend/business/backtest/strategies/implementations/rising_channel_strategy.py

import pandas as pd
from ..base import BaseStrategy

# 复用您项目中强大的因子计算引擎！
from backend.business.factor.core.engine.library.channel_analysis.rising_channel import AscendingChannelRegression


class RisingChannelStrategy(BaseStrategy):
    """
    上升通道策略 (新框架版)。

    逻辑:
    1. 在每日开盘前，筛选出所有A股。
    2. 对股票池中的每一只股票：
        a. 获取其历史数据。
        b. 调用已有的 RisingChannel 因子计算器判断其是否处于上升通道中。
    3. 买入/持仓逻辑：
        - 如果股票处于上升通道，并且我们还没有持仓，则买入。
    4. 卖出逻辑：
        - 如果我们持有某只股票，但它现在已经不再处于上升通道中，则卖出。
    """

    def initialize(self):
        """初始化策略参数。"""
        # 从传入的参数或默认值中获取参数
        self.lookback_period = self.params.get('lookback_period', 120)
        self.min_channel_width = self.params.get('min_channel_width', 0.1)
        # 初始化一个 RisingChannel 实例，用于后续计算
        self.channel_calculator = AscendingChannelRegression(
            lookback_period=self.lookback_period,
            min_channel_width=self.min_channel_width
        )
        # 初始化股票池为空列表
        self.context.stock_pool = []
        print("RisingChannelStrategy initialized.")

    def before_trading_start(self):
        """每日开盘前，准备好当天要交易的股票池。"""
        # 在 `handle_data` 中，daily_bars 的 index 就是当天的股票池
        # 这里可以预留做更复杂的股票池筛选，例如剔除ST、小市值等
        # 为简化起见，我们直接使用当天有行情的所有股票
        pass

    def handle_data(self, daily_bars: pd.DataFrame):
        """
        执行每日的交易逻辑。
        """
        print(f"--- {self.context.current_dt.date()} ---")

        # 1. 确定今天的股票池
        self.context.stock_pool = daily_bars.index.tolist()

        # 2. 检查并执行卖出操作
        # 使用 list(self.context.portfolio.positions.keys()) 来避免在遍历时修改字典
        for stock_code in list(self.context.portfolio.positions.keys()):
            is_in_channel = self._check_channel_status(stock_code)
            if not is_in_channel:
                print(f"  [SELL] {stock_code} is no longer in a rising channel. Selling.")
                self.context.portfolio.order_target(stock_code, 0)  # 目标股数设为0，即清仓

        # 3. 检查并执行买入操作
        # 假设我们等分资金
        positions_to_hold = len(self.context.portfolio.positions)
        target_positions = self.params.get('target_positions', 10)  # 假设最多持有10只

        if positions_to_hold < target_positions:
            for stock_code in self.context.stock_pool:
                # 如果已经持有，则不再买入
                if stock_code in self.context.portfolio.positions:
                    continue

                is_in_channel = self._check_channel_status(stock_code)
                if is_in_channel:
                    print(f"  [BUY] {stock_code} is in a rising channel. Buying.")
                    # 计算可用于购买新股票的资金
                    cash_for_new_buys = self.context.portfolio.total_value / target_positions
                    self.context.portfolio.order_target_value(stock_code, cash_for_new_buys)

                    # 如果持仓达到上限，则停止买入
                    if len(self.context.portfolio.positions) >= target_positions:
                        break

    def _check_channel_status(self, stock_code: str) -> bool:
        """
        调用因子计算器，判断单只股票当前是否处于上升通道。
        """
        try:
            # 从 DataProvider 获取计算所需的回溯历史数据
            # 注意: context 上的 data_provider 属性是在 PortfolioManager 初始化时设置的
            history_data = self.context.data_provider.get_history_data(
                stock_code, self.context.current_dt, self.lookback_period + 5  # 多取几天以防万一
            )

            if history_data is None or len(history_data) < self.lookback_period:
                return False

            # 调用因子计算器的核心方法
            channel_info = self.channel_calculator.calculate(history_data)

            # 根据因子计算结果判断状态
            # channel_info['state'] == 1 表示处于上升通道
            if channel_info is not None and not channel_info.empty:
                # 使用.iloc[-1]获取最后一天的状态
                return channel_info['state'].iloc[-1] == 1
            return False
        except Exception as e:
            # print(f"  Error checking channel for {stock_code}: {e}")
            return False