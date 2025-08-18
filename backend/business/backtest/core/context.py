#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:36 PM
@File       : context.py
@Description: 
"""

from typing import TYPE_CHECKING
import pandas as pd

# 避免循环导入
if TYPE_CHECKING:
    from .data_provider import DataProvider
    from .portfolio import PortfolioManager

class Context:
    """
    上下文对象 (Context)。

    作为策略访问所有回测资源的“遥控器”。
    它在整个回测生命周期中只有一个实例，并被传递给策略的每一个生命周期函数。
    """
    def __init__(self, data_provider: 'DataProvider', portfolio_manager: 'PortfolioManager'):
        self.data_provider = data_provider
        self.portfolio = portfolio_manager
        self.current_dt: pd.Timestamp = None

    def __setattr__(self, name, value):
        # 允许策略开发者在context上自由设置自定义属性
        # 例如: context.my_variable = 100
        super().__setattr__(name, value)