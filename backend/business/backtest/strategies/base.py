#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:29 PM
@File       : base.py
@Description: 
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, Any
import pandas as pd

# 使用TYPE_CHECKING来避免循环导入，这是Python类型提示的标准做法
if TYPE_CHECKING:
    from ..core.context import Context


class BaseStrategy(ABC):
    """
    策略抽象基类 (Abstract Base Class)。

    所有策略类都必须继承自此类，并实现其定义的抽象方法。
    这为回测引擎提供了一个统一的策略接口。
    """

    def __init__(self, context: 'Context', params: Optional[Dict[str, Any]] = None):
        """
        构造函数，在策略被实例化时调用。

        Args:
            context (Context): 上下文对象，策略通过它与回测引擎的其他部分交互。
            params (Optional[Dict[str, Any]]): 策略的参数字典，用于传递可调参数。
        """
        self.context = context
        self.params = params or {}

    def initialize(self):
        """
        【可选实现】初始化函数。

        在回测开始前被调用，且整个回测周期只调用一次。
        适合用于设置策略的初始参数、加载模型、定义全局变量等。
        """
        pass

    def before_trading_start(self):
        """
        【可选实现】每日开盘前调用。

        在每个交易日的 `handle_data` 之前被调用。
        适合用于每日的股票池筛选、计算当天需要使用的技术指标、更新参数等。
        """
        pass

    @abstractmethod
    def handle_data(self, daily_bars: pd.DataFrame):
        """
        【必须实现】核心策略逻辑函数。

        每个交易日的数据会触发此函数的调用。
        这是策略产生交易信号和执行订单的核心位置。

        Args:
            daily_bars (pd.DataFrame):
                一个以股票代码为索引 (index) 的 DataFrame，
                包含了当天所有股票的行情数据 (open, high, low, close, volume等)。
        """
        raise NotImplementedError("策略必须实现 'handle_data' 方法！")

    def after_trading_end(self):
        """
        【可选实现】每日收盘后调用。

        在每个交易日的 `handle_data` 和订单处理之后被调用。
        适合用于当日的复盘、记录自定义日志、重置每日状态等。
        """
        pass