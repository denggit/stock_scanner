#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础策略类
使用策略模式设计，提供统一的策略接口
"""

import backtrader as bt
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class BaseStrategy(bt.Strategy):
    """
    基础策略抽象类
    所有交易策略都应该继承此类
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
        self.order = None
        self.buy_price = None
        self.position_size = 0
    
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
    
    def execute_buy(self, size: Optional[int] = None, price: Optional[float] = None):
        """
        执行买入操作
        
        Args:
            size: 买入数量，None表示使用95%资金
            price: 买入价格，None表示市价买入
        """
        if self.order or self.position:
            return False
            
        if size is None:
            # 使用95%资金买入
            available_cash = self.broker.getcash() * 0.95
            size = int(available_cash / self.dataclose[0])
        
        if size <= 0:
            return False
        
        # 执行买入
        if price:
            self.order = self.buy(size=size, price=price)
        else:
            self.order = self.buy(size=size)
        
        self.buy_price = self.dataclose[0]
        self.position_size = size
        
        # 记录买入交易
        self._log_trade("BUY", size, self.dataclose[0])
        return True
    
    def execute_sell(self, size: Optional[int] = None, price: Optional[float] = None):
        """
        执行卖出操作
        
        Args:
            size: 卖出数量，None表示全部卖出
            price: 卖出价格，None表示市价卖出
        """
        if self.order or not self.position:
            return False
        
        if size is None:
            size = self.position.size
        
        if size <= 0:
            return False
        
        # 计算收益率
        returns = (self.dataclose[0] - self.buy_price) / self.buy_price * 100 if self.buy_price else 0
        
        # 执行卖出
        if price:
            self.order = self.sell(size=size, price=price)
        else:
            self.order = self.sell(size=size)
        
        # 记录卖出交易
        self._log_trade("SELL", size, self.dataclose[0], returns)
        
        # 重置状态
        self.buy_price = None
        self.position_size = 0
        return True
    
    def _log_trade(self, action: str, size: int, price: float, returns: float = None):
        """
        记录交易信息
        
        Args:
            action: 交易动作 (BUY/SELL)
            size: 交易数量
            price: 交易价格
            returns: 收益率（卖出时）
        """
        trade = {
            "id": self.trade_count,
            "date": self.data.datetime.date(),
            "action": action,
            "price": price,
            "size": size,
            "value": price * size,
            "cash": self.broker.getcash(),
            "portfolio_value": self.broker.getvalue()
        }
        
        if returns is not None:
            trade["returns"] = returns
            
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
            "portfolio_value": self.broker.getvalue()
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
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                pass  # 买入完成
            else:
                pass  # 卖出完成
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass  # 订单取消或拒绝
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return
        
        # 可以在这里添加交易完成后的逻辑
        pass 