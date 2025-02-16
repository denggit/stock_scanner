#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:32 PM
@File       : base.py
@Description: 
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description
        self._params = {}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置策略参数"""
        self._params.update(params)

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self._params.copy()

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据是否符合策略要求"""
        required_columns = {"open", "high", "low", "close", "volume"}
        return required_columns.issubset(data.columns)

    def get_description(self) -> str:
        """获取策略描述"""
        return f"""
        策略名称: {self.name}

        策略说明: 
        {self.description}

        参数说明:
        {self._get_param_description()}
        """

    def _get_param_description(self) -> str:
        """生成参数说明"""
        if not self._params:
            return "无参数"

        desc = []
        for name, value in self._params.items():
            desc.append(f"- {name}: {value}")
        return "\n".join(desc)

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        生成交易信号

        参数:
        data: pd.DataFrame, 股票数据

        返回:
        pd.Series, 交易信号
            1: 买入
            0: 持有
            -1: 卖出
        """
        pass

    @staticmethod
    def calculate_metrics(returns: pd.Series) -> dict:
        """计算回测指标"""
        total_return = (returns.iloc[-1] - 1) * 100
        daily_returns = returns.pct_change()

        # 计算最大回撤
        cummax = returns.cummax()
        drawdown = (returns - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # 计算夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
