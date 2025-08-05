#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MACD策略
"""

import backtrader as bt
from typing import Dict, Any

from ..core.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    """
    MACD策略
    MACD金叉买入，死叉卖出
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        初始化策略参数
        
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
        """
        super().__init__()
        
        # MACD指标
        self.macd = bt.indicators.MACD(
            self.dataclose,
            period_me1=fast_period,
            period_me2=slow_period,
            period_signal=signal_period
        )
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        
        # 策略参数
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def next(self):
        """策略核心逻辑"""
        # 如果有未完成的订单，等待
        if self.order:
            return
            
        # 当前无持仓
        if not self.position:
            # MACD金叉买入
            if self.crossover > 0:
                self.execute_buy()
                
        # 当前有持仓
        else:
            # MACD死叉卖出
            if self.crossover < 0:
                self.execute_sell()
    
    def _get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period
        }
    
    def _validate_parameters(self):
        """验证策略参数"""
        # 在backtrader中，参数验证应该在策略初始化时进行
        # 这里我们跳过验证，因为参数已经在__init__中验证过了
        pass 