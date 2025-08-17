#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测引擎
使用工厂模式管理回测流程
"""

from typing import Dict, Any, Optional, List, Type

import backtrader as bt
import logging

from backend.utils.logger import setup_logger
from .base_strategy import BaseStrategy
from .data_manager import DataManager
from .result_analyzer import ResultAnalyzer
from .trading_rules import AStockCommissionInfo, AShareTradingRules


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
            commission: 手续费率（作为后备配置；优先使用 A股自定义手续费模型）
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = None
        self.data_manager = DataManager()
        self.result_analyzer = ResultAnalyzer()
        # 统一使用backtest主日志记录器，便于全局日志管理和追踪
        self.logger = setup_logger("backtest")

        # 禁用backtrader的默认日志配置，防止重复日志
        self._disable_backtrader_logging()

        # 初始化分析器
        self._init_analyzers()

    def _disable_backtrader_logging(self):
        """禁用backtrader的默认日志配置"""
        try:
            # 禁用backtrader的默认日志处理器
            bt_logger = logging.getLogger('backtrader')
            bt_logger.disabled = True
            
            # 禁用backtrader相关的其他日志记录器
            for logger_name in ['backtrader', 'bt', 'cerebro']:
                logger = logging.getLogger(logger_name)
                logger.disabled = True
                # 清除可能存在的处理器
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    
            self.logger.info("已禁用backtrader默认日志配置")
        except Exception as e:
            self.logger.warning(f"禁用backtrader日志配置时出现警告: {e}")

    def _init_analyzers(self):
        """初始化分析器"""
        self.result_analyzer.add_analyzer('sharpe', bt.analyzers.SharpeRatio, _name='sharpe')
        self.result_analyzer.add_analyzer('drawdown', bt.analyzers.DrawDown, _name='drawdown')
        self.result_analyzer.add_analyzer('returns', bt.analyzers.Returns, _name='returns')
        self.result_analyzer.add_analyzer('trades', bt.analyzers.TradeAnalyzer, _name='trades')
        # 增加日频收益分析器，便于构建每日收益表
        self.result_analyzer.add_analyzer('timereturn', bt.analyzers.TimeReturn, _name='timereturn',
                                          timeframe=bt.TimeFrame.Days)

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
        self.logger.debug(f"数据已添加: {name}")

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

    def _configure_broker(self):
        """
        配置broker的资金与手续费模型
        优先使用A股的自定义手续费模型（佣金+过户费+印花税），否则退化为仅佣金费率。
        """
        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_cash)

        # 为了实现“同一交易日先卖后买”的现金可用性，启用 cheat-on-close 模式：
        # 在 next() 内发出的市价单将以当日收盘价成交，使得卖出释放的现金可立即用于当日买入。
        try:
            self.cerebro.broker.set_coc(True)
            self.logger.info("已启用 cheat-on-close，同日先卖后买顺序将按 next() 调用顺序执行")
        except Exception as e:
            self.logger.warning(f"启用 cheat-on-close 失败，将使用预算模拟：{e}")

        # 优先使用自定义的A股手续费模型
        try:
            comminfo = AStockCommissionInfo(
                commission=AShareTradingRules.COMMISSION_RATE,
                stamp_tax_rate=AShareTradingRules.STAMP_TAX_RATE,
                transfer_fee_rate=AShareTradingRules.TRANSFER_FEE_RATE,
                min_commission=AShareTradingRules.MIN_COMMISSION,
            )
            self.cerebro.broker.addcommissioninfo(comminfo)
            # 关闭默认滑点/其它影响（如需，可在外部另行设置）
            self.logger.info("已启用A股自定义手续费模型")
        except Exception as e:
            # 后备：仅设置佣金费率
            self.cerebro.broker.setcommission(commission=self.commission)
            self.logger.warning(f"启用A股手续费模型失败，退化为固定佣金率: {e}")

        self.logger.info(f"Broker配置完成 - 初始资金: {self.initial_cash}")

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

        # 配置资金与手续费
        self._configure_broker()

        # 添加分析器
        self.add_analyzers()

        # 运行回测
        self.logger.info("开始运行回测...")
        strategy_results = self.cerebro.run()

        # 保存策略实例以便后续访问
        self._last_run_results = strategy_results

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

        # 绘图
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
                strategy_name=f"参数优化_{len(optimization_results) + 1}"
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
