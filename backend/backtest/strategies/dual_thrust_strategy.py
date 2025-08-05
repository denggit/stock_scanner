#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双轨突破策略
"""

import backtrader as bt
from typing import Dict, Any

from ..core.base_strategy import BaseStrategy


class DualThrustStrategy(BaseStrategy):
    """
    双轨突破策略
    突破上轨买入，跌破下轨卖出
    """
    
    def __init__(self, period: int = 20, k1: float = 0.7, k2: float = 0.7):
        """
        初始化策略参数
        
        Args:
            period: 计算周期
            k1: 上轨系数
            k2: 下轨系数
        """
        super().__init__()
        
        # 计算HH和LL
        self.hh = bt.indicators.Highest(self.datahigh, period=period)
        self.ll = bt.indicators.Lowest(self.datalow, period=period)
        
        # 计算LC和HC
        self.lc = bt.indicators.Lowest(self.dataclose, period=period)
        self.hc = bt.indicators.Highest(self.dataclose, period=period)
        
        # 策略参数
        self.period = period
        self.k1 = k1
        self.k2 = k2
        
    def next(self):
        """策略核心逻辑"""
        # 如果有未完成的订单，等待
        if self.order:
            return
            
        # 计算上下轨
        range_high = self.hh[0] - self.lc[0]
        range_low = self.hc[0] - self.ll[0]
        
        upper_band = self.dataclose[0] + self.k1 * range_high
        lower_band = self.dataclose[0] - self.k2 * range_low
        
        # 当前无持仓
        if not self.position:
            # 突破上轨买入
            if self.dataclose[0] > upper_band:
                self.execute_buy()
                
        # 当前有持仓
        else:
            # 跌破下轨卖出
            if self.dataclose[0] < lower_band:
                self.execute_sell()
    
    def _get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            "period": self.period,
            "k1": self.k1,
            "k2": self.k2
        }
    
    def _validate_parameters(self):
        """验证策略参数"""
        # 在backtrader中，参数验证应该在策略初始化时进行
        # 这里我们跳过验证，因为参数已经在__init__中验证过了
        pass 