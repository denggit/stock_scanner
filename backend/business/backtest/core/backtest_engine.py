#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测引擎
使用工厂模式管理回测流程
"""

import backtrader as bt
from typing import Dict, Any, Optional, List, Type
import logging

from .data_manager import DataManager
from .result_analyzer import ResultAnalyzer
from .base_strategy import BaseStrategy


class BacktestEngine:
    """
    回测引擎
    负责协调整个回测流程
    """
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.0003):
        """
        初始化回测引擎
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = None
        self.data_manager = DataManager()
        self.result_analyzer = ResultAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # 初始化分析器
        self._init_analyzers()
    
    def _init_analyzers(self):
        """初始化分析器"""
        self.result_analyzer.add_analyzer('sharpe', bt.analyzers.SharpeRatio, _name='sharpe')
        self.result_analyzer.add_analyzer('drawdown', bt.analyzers.DrawDown, _name='drawdown')
        self.result_analyzer.add_analyzer('returns', bt.analyzers.Returns, _name='returns')
        self.result_analyzer.add_analyzer('trades', bt.analyzers.TradeAnalyzer, _name='trades')
    
    def add_data(self, data, name: str = "data") -> None:
        """
        添加数据到回测引擎
        
        Args:
            data: 股票数据（pandas DataFrame或已加载的数据源）
            name: 数据名称
        """
        if self.cerebro is None:
            self.cerebro = bt.Cerebro()
        
        # 如果输入是DataFrame，先加载
        if hasattr(data, 'columns'):  # 是DataFrame
            data_feed = self.data_manager.load_data(data, name)
        else:  # 已经是数据源
            data_feed = data
            name = getattr(data, 'name', name)
        
        self.cerebro.adddata(data_feed)
        self.logger.info(f"数据已添加: {name}")
    
    def add_strategy(self, strategy_class: Type[BaseStrategy], **kwargs) -> None:
        """
        添加策略到回测引擎
        
        Args:
            strategy_class: 策略类
            **kwargs: 策略参数
        """
        if self.cerebro is None:
            self.cerebro = bt.Cerebro()
        
        self.cerebro.addstrategy(strategy_class, **kwargs)
        self.logger.info(f"策略已添加: {strategy_class.__name__}")
    
    def add_analyzers(self) -> None:
        """添加分析器到引擎"""
        if self.cerebro is None:
            return
        
        for name, (analyzer_class, kwargs) in self.result_analyzer.analyzers.items():
            # 确保kwargs中不包含_name参数
            kwargs_copy = kwargs.copy()
            if '_name' in kwargs_copy:
                del kwargs_copy['_name']
            
            self.cerebro.addanalyzer(analyzer_class, _name=name, **kwargs_copy)
    
    def run(self, strategy_name: str = "策略") -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            回测结果字典
        """
        if self.cerebro is None:
            raise ValueError("请先添加数据和策略")
        
        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_cash)
        
        # 设置手续费
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # 添加分析器
        self.add_analyzers()
        
        # 运行回测
        self.logger.info("开始运行回测...")
        strategy_results = self.cerebro.run()
        
        # 分析结果
        results = self.result_analyzer.analyze(self.cerebro, strategy_results)
        
        # 生成报告
        report = self.result_analyzer.generate_report(results, strategy_name)
        self.logger.info(f"回测完成: {strategy_name}")
        
        return {
            **results,
            "report": report,
            "strategy_name": strategy_name
        }
    
    def plot(self, **kwargs):
        """绘制回测结果"""
        if self.cerebro is None:
            raise ValueError("请先运行回测")
        
        self.cerebro.plot(**kwargs)
    
    def get_data_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取数据信息
        
        Args:
            name: 数据名称
            
        Returns:
            数据信息字典
        """
        return self.data_manager.get_data_info(name)
    
    def list_cached_data(self) -> List[str]:
        """
        列出所有缓存的数据名称
        
        Returns:
            数据名称列表
        """
        return self.data_manager.list_cached_data()
    
    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            name: 数据名称，None表示清除所有缓存
        """
        self.data_manager.clear_cache(name)


class BacktestFactory:
    """
    回测工厂类
    使用工厂模式创建不同类型的回测
    """
    
    @staticmethod
    def create_simple_backtest(
        data,
        strategy_class: Type[BaseStrategy],
        initial_cash: float = 100000.0,
        commission: float = 0.0003,
        strategy_params: Dict = None,
        strategy_name: str = "策略",
        plot: bool = False
    ) -> Dict[str, Any]:
        """
        创建简单回测
        
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
        # 创建回测引擎
        engine = BacktestEngine(initial_cash, commission)
        
        # 添加数据
        engine.add_data(data)
        
        # 添加策略
        if strategy_params:
            engine.add_strategy(strategy_class, **strategy_params)
        else:
            engine.add_strategy(strategy_class)
        
        # 运行回测
        results = engine.run(strategy_name)
        
        # 绘制图表
        if plot:
            engine.plot()
        
        return results
    
    @staticmethod
    def create_multi_strategy_backtest(
        data,
        strategies: List[Dict],
        initial_cash: float = 100000.0,
        commission: float = 0.0003
    ) -> Dict[str, Any]:
        """
        创建多策略回测
        
        Args:
            data: 股票数据
            strategies: 策略列表，每个元素包含 name, class, params
            initial_cash: 初始资金
            commission: 手续费率
            
        Returns:
            多策略回测结果
        """
        results = {}
        
        for strategy_info in strategies:
            name = strategy_info['name']
            strategy_class = strategy_info['class']
            params = strategy_info.get('params', {})
            
            # 运行单个策略回测
            result = BacktestFactory.create_simple_backtest(
                data=data,
                strategy_class=strategy_class,
                initial_cash=initial_cash,
                commission=commission,
                strategy_params=params,
                strategy_name=name
            )
            
            results[name] = result
        
        # 生成比较报告
        analyzer = ResultAnalyzer()
        comparison_report = analyzer.compare_strategies(results)
        
        return {
            "strategy_results": results,
            "comparison_report": comparison_report
        }
    
    @staticmethod
    def create_parameter_optimization(
        data,
        strategy_class: Type[BaseStrategy],
        parameter_ranges: Dict[str, List],
        initial_cash: float = 100000.0,
        commission: float = 0.0003,
        optimization_target: str = "总收益率"
    ) -> Dict[str, Any]:
        """
        创建参数优化回测
        
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
        import itertools
        
        # 生成参数组合
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        optimization_results = []
        best_result = None
        best_params = None
        best_value = -float('inf')
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            
            # 运行回测
            result = BacktestFactory.create_simple_backtest(
                data=data,
                strategy_class=strategy_class,
                initial_cash=initial_cash,
                commission=commission,
                strategy_params=params,
                strategy_name=f"参数优化_{len(optimization_results)+1}"
            )
            
            # 记录结果
            optimization_result = {
                "parameters": params,
                "result": result,
                "target_value": result['metrics'].get(optimization_target, 0)
            }
            optimization_results.append(optimization_result)
            
            # 更新最优结果
            if optimization_result['target_value'] > best_value:
                best_value = optimization_result['target_value']
                best_params = params
                best_result = result
        
        return {
            "optimization_results": optimization_results,
            "best_params": best_params,
            "best_result": best_result,
            "best_value": best_value,
            "total_combinations": len(param_combinations)
        } 