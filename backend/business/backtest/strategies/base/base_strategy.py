#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略基类
使用模板方法模式定义策略执行流程
提供可复用的基础策略功能
"""

import logging
from typing import Dict, Any, List

import backtrader as bt
import pandas as pd

from .data_manager import DataManager
from .position_manager import PositionManager
from .trade_logger import TradeLogger
from .trade_manager import TradeManager
from backend.business.backtest.core.trading_rules import is_trade_blocked_by_price_limit


class BaseStrategy(bt.Strategy):
    """
    策略基类
    
    使用模板方法模式定义策略执行流程：
    1. 数据准备 -> prepare_data()
    2. 信号生成 -> generate_signals()
    3. 风险控制 -> risk_control()
    4. 执行交易 -> execute_trades()
    5. 记录日志 -> log_results()
    """

    # 默认参数，子类可以覆盖
    params = (
        ('max_positions', 50),  # 最大持仓数量
        ('min_data_points', 60),  # 最小数据点数
        ('enable_logging', True),  # 是否启用日志
    )

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame] = None):
        """
        初始化策略基类
        
        Args:
            stock_data_dict: 股票数据字典 {股票代码: DataFrame}
        """
        super().__init__()

        # 设置日志记录器
        self.logger = logging.getLogger("backtest")

        # 初始化各个管理器
        self._init_managers()

        # 设置股票数据
        if stock_data_dict:
            self.data_manager.set_stock_data(stock_data_dict)

        # 当前日期
        self.current_date = None

        # 策略状态
        self._is_initialized = False

        self.logger.info(f"策略 {self.__class__.__name__} 初始化完成")

    def _init_managers(self):
        """初始化各个管理器"""
        self.position_manager = PositionManager()
        self.trade_manager = TradeManager(None, self.position_manager)  # broker稍后设置
        self.data_manager = DataManager()
        self.trade_logger = TradeLogger()

    def next(self):
        """
        策略主逻辑 - 模板方法
        定义执行流程，子类通过实现抽象方法来定制具体逻辑
        """
        # 获取当前日期
        self.current_date = self.data.datetime.date(0)

        # 跳过前min_data_points天的数据
        if len(self.data) < self.params.min_data_points:
            if self.params.enable_logging:
                self.logger.debug(f"跳过第 {len(self.data)} 天，等待足够的历史数据")
            return

        # 延迟初始化（首次运行时）
        if not self._is_initialized:
            self._delayed_initialization()
            self._is_initialized = True

        if self.params.enable_logging:
            self.logger.info(f"========回测交易日: {self.current_date}========")

        try:
            # 模板方法：执行策略流程
            self._execute_strategy_flow()

        except Exception as e:
            self.logger.error(f"策略执行失败: {e}")
            import traceback
            traceback.print_exc()

    def _execute_strategy_flow(self):
        """
        执行策略流程 - 模板方法
        定义标准的策略执行步骤
        """
        # 1. 数据准备
        self.prepare_data()

        # 2. 生成交易信号
        signals = self.generate_signals()

        # 3. 风险控制
        filtered_signals = self.risk_control(signals)

        # 4. 执行交易
        self.execute_trades(filtered_signals)

        # 5. 记录结果
        self.log_results()

    def prepare_data(self):
        """
        数据准备阶段
        子类应该实现具体的数据处理逻辑
        """
        # 默认实现：什么都不做
        pass

    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        生成交易信号
        子类必须实现此方法
        
        Returns:
            交易信号列表，每个信号包含：
            {
                'action': 'BUY'/'SELL',
                'stock_code': '股票代码',
                'price': 价格,
                'confidence': 信心度,
                'reason': '信号原因'
            }
        """
        # 默认实现：不生成任何信号
        return []

    def risk_control(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        风险控制
        默认实现基本的风险控制逻辑，子类可以覆盖
        
        Args:
            signals: 原始交易信号
            
        Returns:
            过滤后的交易信号
        """
        filtered_signals = []

        for signal in signals:
            # 基本风险检查
            if self._is_signal_valid(signal):
                filtered_signals.append(signal)
            else:
                if self.params.enable_logging:
                    self.logger.warning(f"信号被风险控制过滤: {signal}")

        return filtered_signals

    def execute_trades(self, signals: List[Dict[str, Any]]):
        """
        执行交易
        默认实现基本的交易执行逻辑，子类可以覆盖
        
        Args:
            signals: 经过风险控制的交易信号
        """
        for signal in signals:
            try:
                if signal['action'] == 'BUY':
                    self._execute_buy_signal(signal)
                elif signal['action'] == 'SELL':
                    self._execute_sell_signal(signal)

            except Exception as e:
                self.logger.error(f"执行交易信号失败: {signal}, 错误: {e}")

    def log_results(self):
        """
        记录结果
        默认实现基本的结果记录，子类可以覆盖
        """
        if self.params.enable_logging:
            position_count = self.position_manager.get_position_count()
            total_value = self.broker.getvalue()
            cash = self.broker.getcash()

            self.logger.debug(f"当前持仓: {position_count}, 总资产: {total_value:.2f}, 现金: {cash:.2f}")

    def _delayed_initialization(self):
        """
        延迟初始化
        在第一次next()调用时执行，此时backtrader环境已经完全初始化
        """
        self.logger.info("执行策略延迟初始化...")

        # 初始化交易管理器
        self.trade_manager.set_broker(self.broker)

        # 子类可以覆盖此方法进行自定义初始化
        self.on_delayed_init()

    def on_delayed_init(self):
        """
        延迟初始化钩子方法
        子类可以覆盖此方法进行自定义初始化
        """
        pass

    def _is_signal_valid(self, signal: Dict[str, Any]) -> bool:
        """
        检查交易信号是否有效
        
        Args:
            signal: 交易信号
            
        Returns:
            是否有效
        """
        required_fields = ['action', 'stock_code', 'price']

        # 检查必需字段
        for field in required_fields:
            if field not in signal:
                return False

        # 检查价格有效性
        if signal['price'] <= 0:
            return False

        # 检查持仓限制
        if signal['action'] == 'BUY':
            current_positions = self.position_manager.get_position_count()
            if current_positions >= self.params.max_positions:
                return False

        # 涨跌停拦截（基于前收）
        try:
            stock_code = signal['stock_code']
            price = float(signal['price'])
            # 获取该股票在 current_date 的前一交易日收盘价
            df = self.data_manager.get_stock_data_until(stock_code, self.current_date, min_data_points=2)
            if df is not None and len(df) >= 2:
                prev_close = float(df.iloc[-2]['close'])
                if is_trade_blocked_by_price_limit(price, prev_close, signal['action'], is_st=False):
                    return False
        except Exception:
            # 数据不足或异常时，不做强制拦截
            pass

        return True

    def _execute_buy_signal(self, signal: Dict[str, Any]):
        """
        执行买入信号
        
        Args:
            signal: 买入信号
        """
        stock_code = signal['stock_code']
        price = signal['price']

        # 计算买入数量（内部已按手、最小金额、费用覆盖）
        shares = self.trade_manager.calculate_buy_size(price, self.params.max_positions)

        if shares > 0:
            # 执行买入
            self.buy(size=shares)

            # 更新持仓
            self.position_manager.add_position(stock_code, shares, price, self.current_date)

            # 记录交易
            extra_fields = {k: v for k, v in signal.items() if k not in ['action', 'stock_code', 'price', 'reason', 'confidence']}
            self.trade_logger.log_trade(
                action='BUY',
                stock_code=stock_code,
                size=shares,
                price=price,
                date=self.current_date,
                reason=signal.get('reason', ''),
                confidence=signal.get('confidence', 0.0),
                **extra_fields
            )

            if self.params.enable_logging:
                self.logger.info(f"买入 {stock_code}: {shares}股 @ {price:.2f}")

    def _execute_sell_signal(self, signal: Dict[str, Any]):
        """
        执行卖出信号
        
        Args:
            signal: 卖出信号
        """
        stock_code = signal['stock_code']
        price = signal['price']

        # 获取持仓信息
        position_info = self.position_manager.get_position(stock_code)
        if not position_info:
            return

        # 涨跌停拦截（跌停禁卖）
        try:
            df = self.data_manager.get_stock_data_until(stock_code, self.current_date, min_data_points=2)
            if df is not None and len(df) >= 2:
                prev_close = float(df.iloc[-2]['close'])
                if is_trade_blocked_by_price_limit(price, prev_close, 'SELL', is_st=False):
                    return
        except Exception:
            pass

        shares = position_info['shares']

        # 执行卖出
        self.sell(size=shares)

        # 计算收益
        buy_price = position_info['buy_price']
        buy_date = position_info['buy_date']

        returns_pct = (price - buy_price) / buy_price * 100 if buy_price > 0 else 0
        profit_amount = (price - buy_price) * shares
        holding_days = (self.current_date - buy_date).days

        # 更新持仓
        self.position_manager.remove_position(stock_code)

        # 记录交易
        extra_fields = {k: v for k, v in signal.items() if k not in ['action', 'stock_code', 'price', 'reason', 'confidence']}
        self.trade_logger.log_trade(
            action='SELL',
            stock_code=stock_code,
            size=shares,
            price=price,
            date=self.current_date,
            reason=signal.get('reason', ''),
            confidence=signal.get('confidence', 0.0),
            returns=returns_pct,
            profit_amount=profit_amount,
            buy_price=buy_price,
            holding_days=holding_days,
            **extra_fields
        )

        if self.params.enable_logging:
            profit_sign = "+" if profit_amount >= 0 else ""
            self.logger.info(f"卖出 {stock_code}: {shares}股 @ {price:.2f}, "
                             f"收益: {profit_sign}{profit_amount:.2f}元 ({returns_pct:.2f}%), "
                             f"持仓{holding_days}天")
