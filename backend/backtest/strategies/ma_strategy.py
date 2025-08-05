#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
移动平均策略
"""

import backtrader as bt
from typing import Dict, Any

from ..core.base_strategy import BaseStrategy


class MAStrategy(BaseStrategy):
    """
    移动平均策略
    当短期均线上穿长期均线时买入，下穿时卖出
    """
    
    def __init__(self, short_period: int = 10, long_period: int = 30):
        """
        初始化策略参数
        
        Args:
            short_period: 短期移动平均周期
            long_period: 长期移动平均周期
        """
        super().__init__()
        
        # 验证参数
        if short_period >= long_period:
            raise ValueError("短期周期必须小于长期周期")
        
        # 移动平均线
        self.short_ma = bt.indicators.SMA(self.dataclose, period=short_period)
        self.long_ma = bt.indicators.SMA(self.dataclose, period=long_period)
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
        # 策略参数
        self.short_period = short_period
        self.long_period = long_period
        
    def next(self):
        """策略核心逻辑"""
        # 如果有未完成的订单，等待
        if self.order:
            return
            
        # 当前无持仓
        if not self.position:
            # 金叉买入
            if self.crossover > 0:
                self.execute_buy()
                
        # 当前有持仓
        else:
            # 死叉卖出
            if self.crossover < 0:
                self.execute_sell()
    
    def _get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            "short_period": self.short_period,
            "long_period": self.long_period
        }
    
    def _validate_parameters(self):
        """验证策略参数"""
        # 在backtrader中，参数验证应该在策略初始化时进行
        # 这里我们跳过验证，因为参数已经在__init__中验证过了
        pass 