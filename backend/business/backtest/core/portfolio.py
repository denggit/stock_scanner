#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:36 PM
@File       : portfolio.py
@Description: 
"""
# backend/business/backtest/core/portfolio.py

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .context import Context
from .data_provider import DataProvider


@dataclass
class Position:
    """持仓对象"""
    stock_code: str
    amount: int = 0
    avg_cost: float = 0.0

    @property
    def market_value(self) -> float:
        # 市值需要外部价格来计算，这里仅为结构示例
        return 0.0


@dataclass
class Order:
    """订单对象"""
    stock_code: str
    amount: int  # 正数表示买入, 负数表示卖出
    created_dt: pd.Timestamp
    order_id: int = field(default_factory=lambda: id(object()))
    # 订单类型, 'market', 'limit' 等, 初期简化为市价单
    order_type: str = 'market'
    status: str = 'open'  # 'open', 'filled', 'canceled', 'pending'


@dataclass
class Trade:
    """成交记录对象"""
    stock_code: str
    trade_price: float
    amount: int
    trade_dt: pd.Timestamp
    commission: float = 0.0


class PortfolioManager:
    """
    投资组合管理器。

    职责:
    1. 管理账户状态：现金、持仓、总市值。
    2. 提供交易API给策略使用（通过Context对象）。
    3. 在引擎的驱动下，处理订单、模拟撮合、更新持仓。
    4. 记录每日的账户净值。
    """

    def __init__(self, initial_cash: float, data_provider: DataProvider):
        self.initial_cash = initial_cash
        self.available_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.total_value = initial_cash

        self.context = Context(data_provider, self)

        self._open_orders: List[Order] = []
        self.trade_records: List[Trade] = []
        self.daily_net_values: List[Dict] = []

    # --- 暴露给策略的交易API (通过context对象调用) ---
    def order(self, stock_code: str, amount: int):
        """按股数下单。正为买，负为卖。"""
        if amount == 0:
            return

        order = Order(
            stock_code=stock_code,
            amount=amount,
            created_dt=self.context.current_dt
        )
        self._open_orders.append(order)
        print(f"[{self.context.current_dt.date()}] Order created: {amount} shares of {stock_code}")
        return order

    def order_value(self, stock_code: str, value: float):
        """按价值下单。"""
        # 通过context获取当天的数据
        daily_bars = self.context.data_provider.get_daily_bars(self.context.current_dt)
        if stock_code not in daily_bars.index:
            print(
                f"WARNING: No market data for {stock_code} on {self.context.current_dt.date()}. Cannot place order by value.")
            return None

        price = daily_bars.loc[stock_code]['close']
        if price > 0:
            # 计算可以购买的股数 (向下取整到100股的倍数，A股交易规则)
            amount = int(value / price / 100) * 100
            # 调用基础的 order 方法来创建订单
            return self.order(stock_code, amount)
        return None

    def order_target(self, stock_code: str, amount: int):
        """调整目标仓位到指定股数。"""
        current_amount = self.positions.get(stock_code, Position(stock_code)).amount
        delta_amount = amount - current_amount
        return self.order(stock_code, delta_amount)

    def order_target_value(self, stock_code: str, value: float):
        """调整目标仓位到指定价值。"""
        # 通过context获取当天的数据
        daily_bars = self.context.data_provider.get_daily_bars(self.context.current_dt)
        if stock_code not in daily_bars.index:
            print(
                f"WARNING: No market data for {stock_code} on {self.context.current_dt.date()}. Cannot place order by target value.")
            return None

        price = daily_bars.loc[stock_code]['close']
        if price > 0:
            target_amount = int(value / price / 100) * 100
            return self.order_target(stock_code, target_amount)
        return None

    # --- 引擎内部调用的核心方法 ---
    def process_orders(self, daily_bars: pd.DataFrame):
        """
        在每日循环中，根据当天行情处理所有待处理订单。
        初期简化：假设所有订单都以当日收盘价成交。
        """
        if not self._open_orders:
            return

        for order in self._open_orders:
            if order.stock_code in daily_bars.index:
                trade_price = daily_bars.loc[order.stock_code]['close']

                # 简单手续费模型 (万分之三, 最低5元)
                commission = max(5, abs(order.amount * trade_price * 0.0003))

                # 检查现金是否足够
                if order.amount > 0 and self.available_cash < order.amount * trade_price + commission:
                    print(f"WARNING: Not enough cash to buy {order.amount} of {order.stock_code}. Order skipped.")
                    continue

                # 更新现金
                self.available_cash -= order.amount * trade_price
                self.available_cash -= commission

                # 更新持仓
                if order.stock_code not in self.positions:
                    self.positions[order.stock_code] = Position(stock_code=order.stock_code)

                pos = self.positions[order.stock_code]
                # 更新平均成本 (简化处理)
                new_total_cost = pos.avg_cost * pos.amount + order.amount * trade_price
                new_amount = pos.amount + order.amount
                pos.avg_cost = new_total_cost / new_amount if new_amount != 0 else 0
                pos.amount = new_amount

                # 如果持仓为0，则从字典中移除
                if pos.amount == 0:
                    del self.positions[order.stock_code]

                # 记录成交
                trade = Trade(
                    stock_code=order.stock_code,
                    trade_price=trade_price,
                    amount=order.amount,
                    trade_dt=self.context.current_dt,
                    commission=commission
                )
                self.trade_records.append(trade)
                order.status = 'filled'
            else:
                # 处理订单股票不在当日数据中的情况（如停牌）
                print(f"WARNING: Stock {order.stock_code} not available on {self.context.current_dt.date()}. Order kept for next day.")
                order.status = 'pending'  # 保持订单状态为待处理

        # 只清空已成交的订单，保留待处理的订单
        self._open_orders = [order for order in self._open_orders if order.status == 'pending']

    def update_portfolio_value(self, date: pd.Timestamp, daily_bars: pd.DataFrame):
        """在每日收盘后，更新组合总净值。"""
        market_value = 0.0
        for stock_code, position in self.positions.items():
            if stock_code in daily_bars.index:
                market_value += position.amount * daily_bars.loc[stock_code]['close']

        self.total_value = self.available_cash + market_value
        self.daily_net_values.append({'date': date, 'net_value': self.total_value})
