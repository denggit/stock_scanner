#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础策略类
使用策略模式设计，提供统一的策略接口
支持多股票持仓管理
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List

import backtrader as bt


class BaseStrategy(bt.Strategy):
    """
    基础策略抽象类
    所有交易策略都应该继承此类
    
    多持仓管理改进：
    - 使用字典 self.active_orders 管理多个并发订单
    - 支持同时持有多个不同股票的仓位
    - 以订单引用ID (order.ref) 作为键进行管理
    - 移除单持仓限制，支持多股票策略
    """

    def __init__(self):
        """初始化基础策略"""
        super().__init__()

        # 基础数据引用
        self._init_data_references()

        # 交易状态管理
        self._init_trading_state()

        # 交易记录
        self._init_trade_logger()

        # 策略参数验证
        self._validate_parameters()

    def _init_data_references(self):
        """初始化数据引用"""
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self.dataopen = self.datas[0].open

    def _init_trading_state(self):
        """初始化交易状态"""
        # 使用字典管理多个活跃订单，支持并发订单处理
        self.active_orders = {}
        # 使用字典存储每个持仓的买入价格，键为订单引用ID
        self.position_buy_prices = {}
        # 移除单持仓状态变量，使用backtrader的position对象管理

    def _init_trade_logger(self):
        """初始化交易记录器"""
        self.trades = []
        self.trade_count = 0

    def _validate_parameters(self):
        """验证策略参数"""
        # 子类可以重写此方法来验证参数
        pass

    @abstractmethod
    def next(self):
        """
        策略核心逻辑
        每个交易日都会调用此方法
        子类必须实现此方法
        """
        pass

    def execute_buy(self, size: Optional[int] = None, price: Optional[float] = None, signal: Optional[Dict[str, Any]] = None):
        """
        执行买入操作
        
        Args:
            size: 买入数量，None表示使用95%资金或从signal中获取
            price: 买入价格，None表示市价买入
            signal: 买入信号字典，可能包含size等信息
            
        Returns:
            bool: 是否成功创建订单
        """
        # 检查是否有未完成的订单（只检查订单，不检查持仓）
        if self.active_orders:
            return False

        # 优先使用signal中的size，其次使用传入的size参数
        if signal and 'size' in signal:
            size = signal['size']
        elif size is None:
            # 使用95%资金买入
            available_cash = self.broker.getcash() * 0.95
            size = int(available_cash / self.dataclose[0])

        if size <= 0:
            return False

        # 执行买入
        if price:
            order = self.buy(size=size, price=price)
        else:
            order = self.buy(size=size)

        # 将订单添加到活跃订单字典中
        if order:
            self.active_orders[order.ref] = order

            # 记录买入交易
            stock_code = signal.get('stock_code') if signal else None
            reason = signal.get('reason') if signal else ''
            self._log_trade("BUY", size, self.dataclose[0], stock_code=stock_code, reason=reason)
            return True
        
        return False

    def execute_sell(self, size: Optional[int] = None, price: Optional[float] = None):
        """
        执行卖出操作
        
        Args:
            size: 卖出数量，None表示全部卖出
            price: 卖出价格，None表示市价卖出
            
        Returns:
            bool: 是否成功创建订单
        """
        # 检查是否有未完成的订单（只检查订单，不检查持仓）
        if self.active_orders:
            return False

        # 检查是否有持仓可卖
        if not self.position:
            return False

        if size is None:
            size = self.position.size

        if size <= 0:
            return False

        # 执行卖出
        if price:
            order = self.sell(size=size, price=price)
        else:
            order = self.sell(size=size)

        # 将订单添加到活跃订单字典中
        if order:
            self.active_orders[order.ref] = order

            # 记录卖出交易（收益率将在notify_trade中计算并更新）
            self._log_trade("SELL", size, self.dataclose[0])
            return True
        
        return False

    def _log_trade(self, action: str, size: int, price: float, returns: float = None, 
                   stock_code: str = None, reason: str = '', buy_price: float = None, 
                   holding_days: int = None):
        """
        记录交易信息 - 使用标准字段名
        
        Args:
            action: 交易动作 (BUY/SELL)
            size: 交易数量
            price: 交易价格
            returns: 收益率（卖出时，百分比形式）
            stock_code: 股票代码
            reason: 交易原因
            buy_price: 买入价格（卖出时）
            holding_days: 持仓天数（卖出时）
        """
        trade = {
            # 标准字段名 - 统一使用英文
            "trade_id": self.trade_count,
            "date": self.data.datetime.date(),
            "action": action,
            "stock_code": stock_code,
            "size": size,
            "price": price,
            "value": price * size,
            "reason": reason,
            "cash": self.broker.getcash(),
            "portfolio_value": self.broker.getvalue()
        }

        # 卖出时添加收益相关信息
        if action == "SELL":
            if returns is not None:
                trade["returns"] = returns
            if buy_price is not None:
                trade["buy_price"] = buy_price
            if holding_days is not None:
                trade["holding_days"] = holding_days

        self.trades.append(trade)
        self.trade_count += 1

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            策略信息字典
        """
        return {
            "strategy_name": self.__class__.__name__,
            "parameters": self._get_parameters(),
            "trade_count": self.trade_count,
            "current_position": self.position.size if self.position else 0,
            "current_cash": self.broker.getcash(),
            "portfolio_value": self.broker.getvalue(),
            "active_orders_count": len(self.active_orders)
        }

    def _get_parameters(self) -> Dict[str, Any]:
        """
        获取策略参数
        
        Returns:
            策略参数字典
        """
        # 子类可以重写此方法来返回具体参数
        return {}

    def notify_order(self, order):
        """
        订单状态通知
        
        改进的订单管理：
        - 使用 order.ref 作为键管理活跃订单
        - 订单完成、取消或拒绝时从字典中移除
        - 支持多订单并发处理
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 检查订单是否在活跃订单字典中
        if order.ref not in self.active_orders:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                # 买入完成
                pass
            else:
                # 卖出完成
                pass

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 订单取消或拒绝
            pass

        # 从活跃订单字典中移除已完成的订单
        del self.active_orders[order.ref]

    def notify_trade(self, trade):
        """
        交易完成通知
        
        改进的交易价格记录和收益率计算：
        - 当交易开仓时，记录真实的成交价到position_buy_prices字典
        - 当交易平仓时，从字典中获取对应的买入价格计算收益率
        - 计算完成后，更新交易记录中的收益率信息
        """
        if trade.isopen:
            # 交易开仓时，记录真实的成交价
            self.position_buy_prices[trade.ref] = trade.price
        elif trade.isclosed:
            # 交易平仓时，计算收益率并更新交易记录
            if trade.ref in self.position_buy_prices:
                buy_price = self.position_buy_prices[trade.ref]
                returns = (trade.price - buy_price) / buy_price * 100
                
                # 更新最后一笔交易记录的收益率信息
                if self.trades and self.trades[-1]["action"] == "SELL":
                    self.trades[-1]["returns"] = returns
                    self.trades[-1]["buy_price"] = buy_price
                
                # 从字典中删除对应的买入价格记录
                del self.position_buy_prices[trade.ref]

    def has_active_orders(self) -> bool:
        """
        检查是否有活跃订单
        
        Returns:
            bool: 是否有未完成的订单
        """
        return len(self.active_orders) > 0

    def get_active_orders_count(self) -> int:
        """
        获取活跃订单数量
        
        Returns:
            int: 活跃订单数量
        """
        return len(self.active_orders)

    def get_active_orders_info(self) -> Dict[int, Dict[str, Any]]:
        """
        获取活跃订单信息
        
        Returns:
            Dict: 活跃订单信息字典，键为订单引用ID
        """
        orders_info = {}
        for ref, order in self.active_orders.items():
            orders_info[ref] = {
                "ref": order.ref,
                "status": order.status,
                "size": order.size,
                "price": order.price,
                "is_buy": order.isbuy(),
                "created": order.created
            }
        return orders_info

    def get_per_trade_cash(self, max_positions: int) -> float:
        """
        计算单笔交易的可用资金，避免前视偏差
        
        使用T-1日收盘后的总资产除以最大持仓数来计算每笔交易的资金分配，
        确保仓位计算只使用决策时刻已知的信息。
        
        Args:
            max_positions: 最大持仓数量
            
        Returns:
            float: 单笔交易的可用资金
        """
        if max_positions <= 0:
            return 0.0
            
        # 使用T-1日收盘后的总资产（self.broker.getvalue()）
        # 这避免了使用当天收盘价计算持股价值的前视偏差
        total_value = self.broker.getvalue()
        
        # 平均分配到每个持仓
        per_trade_cash = total_value / max_positions
        
        return per_trade_cash

    def get_position_info(self) -> Dict[str, Any]:
        """
        获取当前持仓信息
        
        Returns:
            Dict: 持仓信息字典
        """
        if not self.position:
            return {
                "has_position": False,
                "size": 0,
                "value": 0.0,
                "buy_prices": {}
            }
        
        return {
            "has_position": True,
            "size": self.position.size,
            "value": self.position.size * self.dataclose[0],
            "buy_prices": self.position_buy_prices.copy()
        }

    def can_buy_stock(self, stock_code: str = None) -> bool:
        """
        检查是否可以买入股票
        
        Args:
            stock_code: 股票代码（可选，用于检查是否已有该股票的订单）
            
        Returns:
            bool: 是否可以买入
        """
        # 检查是否有未完成的订单
        if self.active_orders:
            return False
        
        # 如果有指定股票代码，检查是否已有该股票的订单
        if stock_code:
            for order in self.active_orders.values():
                if hasattr(order.data, '_name') and order.data._name == stock_code:
                    return False
        
        return True

    def can_sell_stock(self, stock_code: str = None) -> bool:
        """
        检查是否可以卖出股票
        
        Args:
            stock_code: 股票代码（可选，用于检查是否已有该股票的订单）
            
        Returns:
            bool: 是否可以卖出
        """
        # 检查是否有未完成的订单
        if self.active_orders:
            return False
        
        # 检查是否有持仓可卖
        if not self.position:
            return False
        
        # 如果有指定股票代码，检查是否已有该股票的订单
        if stock_code:
            for order in self.active_orders.values():
                if hasattr(order.data, '_name') and order.data._name == stock_code:
                    return False
        
        return True
