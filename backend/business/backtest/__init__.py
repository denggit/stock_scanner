#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测框架模块
提供基于backtrader的简单回测功能
使用设计模式构建，代码结构清晰
"""

# 核心模块
from .core import (
    BaseStrategy,
    BacktestEngine,
    DataManager,
    ResultAnalyzer,
    BacktestFactory
)

# 策略模块
from .strategies import (
    MAStrategy,
    RSIStrategy,
    MACDStrategy,
    DualThrustStrategy
)

# 工具模块
from .utils import (
    DataUtils,
    ReportUtils
)


# 便捷函数
def run_backtest(
        data,
        strategy_class,
        initial_cash: float = 100000.0,
        commission: float = 0.0003,
        strategy_params: dict = None,
        strategy_name: str = "策略",
        plot: bool = False
):
    """
    运行回测的便捷函数
    
    Args:
        data: 股票数据
        strategy_class: 策略类
        initial_cash: 初始资金
        commission: 手续费率
        strategy_params: 策略参数
        strategy_name: 策略名称
        plot: 是否绘制图表
        
    Returns:
        回测结果
    """
    return BacktestFactory.create_simple_backtest(
        data=data,
        strategy_class=strategy_class,
        initial_cash=initial_cash,
        commission=commission,
        strategy_params=strategy_params,
        strategy_name=strategy_name,
        plot=plot
    )


def run_multi_strategy_backtest(
        data,
        strategies: list,
        initial_cash: float = 100000.0,
        commission: float = 0.0003
):
    """
    运行多策略回测的便捷函数
    
    Args:
        data: 股票数据
        strategies: 策略列表，每个元素包含 name, class, params
        initial_cash: 初始资金
        commission: 手续费率
        
    Returns:
        多策略回测结果
    """
    return BacktestFactory.create_multi_strategy_backtest(
        data=data,
        strategies=strategies,
        initial_cash=initial_cash,
        commission=commission
    )


def optimize_parameters(
        data,
        strategy_class,
        parameter_ranges: dict,
        initial_cash: float = 100000.0,
        commission: float = 0.0003,
        optimization_target: str = "总收益率"
):
    """
    参数优化的便捷函数
    
    Args:
        data: 股票数据
        strategy_class: 策略类
        parameter_ranges: 参数范围字典
        initial_cash: 初始资金
        commission: 手续费率
        optimization_target: 优化目标指标
        
    Returns:
        参数优化结果
    """
    return BacktestFactory.create_parameter_optimization(
        data=data,
        strategy_class=strategy_class,
        parameter_ranges=parameter_ranges,
        initial_cash=initial_cash,
        commission=commission,
        optimization_target=optimization_target
    )


__all__ = [
    # 核心模块
    'BaseStrategy',
    'BacktestEngine',
    'DataManager',
    'ResultAnalyzer',
    'BacktestFactory',

    # 策略模块
    'MAStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'DualThrustStrategy',

    # 工具模块
    'DataUtils',
    'ReportUtils',

    # 便捷函数
    'run_backtest',
    'run_multi_strategy_backtest',
    'optimize_parameters'
]
