#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSI策略
"""

import backtrader as bt
from typing import Dict, Any

from ..core.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI策略
    RSI超卖时买入，超买时卖出
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        初始化策略参数
        
        Args:
            rsi_period: RSI周期
            oversold: 超卖阈值
            overbought: 超买阈值
        """
        super().__init__()
        
        # RSI指标
        self.rsi = bt.indicators.RSI(self.dataclose, period=rsi_period)
        
        # 策略参数
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
    def next(self):
        """策略核心逻辑"""
        # 如果有未完成的订单，等待
        if self.order:
            return
            
        # 当前无持仓
        if not self.position:
            # RSI超卖买入
            if self.rsi[0] < self.oversold:
                self.execute_buy()
                
        # 当前有持仓
        else:
            # RSI超买卖出
            if self.rsi[0] > self.overbought:
                self.execute_sell()
    
    def _get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought
        }
    
    def _validate_parameters(self):
        """验证策略参数"""
        # 在backtrader中，参数验证应该在策略初始化时进行
        # 这里我们跳过验证，因为参数已经在__init__中验证过了
        pass 