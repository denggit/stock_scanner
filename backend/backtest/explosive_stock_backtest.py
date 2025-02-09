#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
爆发型股票策略回测
"""

import logging
from typing import Dict, Any

import backtrader as bt
import pandas as pd

from backend.strategies.explosive_stock import ExplosiveStockStrategy


class ExplosiveStockBacktest(bt.Strategy):
    """爆发型股票回测策略"""

    def __init__(self):
        """初始化回测指标"""
        # 创建策略实例
        self.strategy = ExplosiveStockStrategy()

        # 从策略中获取参数
        strategy_params = self.strategy.get_parameters()

        # 设置参数
        self.p.volume_ma = strategy_params.get('volume_ma', 20)
        self.p.rsi_period = strategy_params.get('rsi_period', 14)
        self.p.bb_period = strategy_params.get('bb_period', 20)
        self.p.bb_std = strategy_params.get('bb_std', 2)
        self.p.recent_days = strategy_params.get('recent_days', 5)
        self.p.weights = strategy_params.get('weights', {
            'volume': 0.35,
            'momentum': 0.30,
            'pattern': 0.20,
            'volatility': 0.15
        })

        # 记录交易状态
        self.order = None
        self.buy_price = None
        self.buy_signal_strength = None

        # 价格相关指标
        self.dataclose = self.datas[0].close
        self.volume = self.datas[0].volume

        # 技术指标
        self._init_indicators()

        # 记录交易日志
        self.trade_log = []

    def _init_indicators(self):
        """初始化技术指标"""
        # 使用与策略相同的技术指标参数
        self.volume_ma = bt.indicators.SMA(
            self.volume,
            period=self.p.volume_ma,
            plotname='Volume MA'
        )

        self.rsi = bt.indicators.RSI(
            self.dataclose,
            period=self.p.rsi_period,
            plotname='RSI'
        )

        self.bb = bt.indicators.BollingerBands(
            self.dataclose,
            period=self.p.bb_period,
            devfactor=self.p.bb_std,
            plotname='Bollinger Bands'
        )

        self.macd = bt.indicators.MACD(
            self.dataclose,
            plotname='MACD'
        )

    def next(self):
        """每个交易日的策略执行"""
        if self.order:
            return

        # 准备当前数据
        current_data = self._prepare_data()

        # 使用策略生成信号
        signal = self.strategy.generate_signal(current_data)

        # 处理信号
        self._process_signal(signal)

    def _prepare_data(self) -> pd.DataFrame:
        """准备用于策略分析的数据"""
        # 获取当前日期之前的所有数据
        data = {
            'close': self.dataclose.get(size=self.p.period),
            'volume': self.volume.get(size=self.p.period),
            'rsi': self.rsi.get(size=self.p.period),
            'macd': self.macd.macd.get(size=self.p.period),
            'macd_signal': self.macd.signal.get(size=self.p.period),
            'bb_upper': self.bb.top.get(size=self.p.period),
            'bb_middle': self.bb.mid.get(size=self.p.period),
            'bb_lower': self.bb.bot.get(size=self.p.period),
            'volume_ma': self.volume_ma.get(size=self.p.period)
        }

        return pd.DataFrame(data)

    def _process_signal(self, signal: pd.Series):
        """处理策略生成的信号"""
        signal_strength = signal.get('signal', 0)
        volume_ratio = signal.get('volume_ratio', 0)

        # 记录当前状态
        self._log_status(signal_strength, volume_ratio)

        # 当前无持仓，检查买入条件
        if not self.position:
            if self._check_buy_conditions(signal):
                self._execute_buy(signal)

        # 当前有持仓，检查卖出条件
        else:
            if self._check_sell_conditions(signal):
                self._execute_sell(signal)

    def _check_buy_conditions(self, signal: pd.Series) -> bool:
        """使用策略的买入条件"""
        buy_conditions = self.strategy._params.get('buy_conditions', {})

        return all([
            signal.get('signal', 0) >= buy_conditions.get('min_signal', 70),
            signal.get('volume_ratio', 0) >= buy_conditions.get('min_volume_ratio', 1.5),
            buy_conditions.get('rsi_range', (0, 100))[0] <= signal.get('rsi', 50) <=
            buy_conditions.get('rsi_range', (0, 100))[1],
            signal.get('explosion_probability', 0) >= buy_conditions.get('min_explosion_prob', 50)
        ])

    def _check_sell_conditions(self, signal: pd.Series) -> bool:
        """使用策略的卖出条件"""
        if not self.buy_price:
            return False

        sell_conditions = self.strategy._params.get('sell_conditions', {})
        returns = (self.dataclose[0] - self.buy_price) / self.buy_price * 100

        return any([
            returns >= sell_conditions.get('profit_target', 30),
            returns <= sell_conditions.get('stop_loss', -7),
            signal.get('rsi', 0) >= sell_conditions.get('max_rsi', 85),
            signal.get('signal', 0) < sell_conditions.get('min_signal', 40),
            signal.get('volume_ratio', 0) < sell_conditions.get('volume_shrink', 0.5)
        ])

    def _execute_buy(self, signal: pd.Series):
        """执行买入操作"""
        # 计算买入数量（使用95%的资金，预留5%作为缓冲）
        size = int(self.broker.getcash() * 0.95 / self.dataclose[0])

        # 创建买入订单
        self.order = self.buy(size=size)
        self.buy_price = self.dataclose[0]
        self.buy_signal_strength = signal.get('signal', 0)

        # 记录交易
        self._log_trade("BUY", size, self.dataclose[0], self.buy_signal_strength)

    def _execute_sell(self, signal: pd.Series):
        """执行卖出操作"""
        # 创建卖出订单
        self.order = self.sell(size=self.position.size)

        # 计算收益率
        returns = (self.dataclose[0] - self.buy_price) / self.buy_price * 100

        # 记录交易
        self._log_trade("SELL", self.position.size, self.dataclose[0], self.buy_signal_strength, returns)

        # 重置买入相关变量
        self.buy_price = None
        self.buy_signal_strength = None

    def _log_trade(self, action: str, size: int, price: float, signal_strength: float, returns: float = None):
        """记录交易信息"""
        trade_info = {
            "date": self.data.datetime.date(),
            "action": action,
            "price": price,
            "size": size,
            "signal": signal_strength,
            "value": price * size
        }

        if returns is not None:
            trade_info["returns"] = returns

        self.trade_log.append(trade_info)

    def _log_status(self, signal_strength: float, volume_ratio: float):
        """记录每日状态"""
        if self.position:
            returns = (self.dataclose[0] - self.buy_price) / self.buy_price * 100
            position_value = self.position.size * self.dataclose[0]
        else:
            returns = None
            position_value = 0

        status = {
            "date": self.data.datetime.date(),
            "close": self.dataclose[0],
            "signal": signal_strength,
            "volume_ratio": volume_ratio,
            "rsi": self.rsi[0],
            "position": bool(self.position),
            "position_value": position_value,
            "cash": self.broker.getcash(),
            "returns": returns
        }

        # 这里可以选择记录到文件或数据库中
        logging.debug(f"Status: {status}")


def run_explosive_stock_backtest(data, backtest_init_params: dict = None, **kwargs) -> \
        Dict[str, Any]:
    """运行多股票组合回测
    
    Args:
        data: Dict[str, pd.DataFrame] 股票数据字典，key为股票代码
        backtest_init_params: 回测初始化参数
        **kwargs: 其他参数
    """
    # 创建组合回测引擎
    cerebro = bt.Cerebro()

    # 获取回测初始化参数
    allocation_strategy = backtest_init_params.get("allocation_strategy")
    initial_capital = backtest_init_params.get("initial_capital")
    max_positions = backtest_init_params.get("max_positions")

    # 添加所有股票数据
    for stock_code, stock_data in data.items():
        data_feed = bt.feeds.PandasData(
            dataname=stock_data,
            name=stock_code
        )
        cerebro.adddata(data_feed)

    # 设置初始资金
    cerebro.broker.setcash(initial_capital)

    # 设置手续费
    cerebro.broker.setcommission(commission=0.0003)

    # 添加策略
    cerebro.addstrategy(
        ExplosiveStockBacktest,
        max_positions=max_positions,
        allocation_strategy=allocation_strategy,
        **kwargs
    )

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # 运行回测
    results = cerebro.run()
    strat = results[0]

    # 获取回测结果
    return _format_backtest_results(strat, data)


def _format_backtest_results(strat, data) -> Dict[str, Any]:
    """格式化回测结果"""
    # 获取分析器结果
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    # 计算指标
    metrics = {
        "总收益率": returns['rtot'] * 100,
        "最大回撤": drawdown['max']['drawdown'] * 100,
        "夏普比率": sharpe_ratio,
        "交易次数": trades.get('total', {}).get('total', 0),
        "胜率": trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1) * 100,
        "平均收益": trades.get('pnl', {}).get('average', 0),
        "最大单笔收益": trades.get('pnl', {}).get('max', 0),
        "最大单笔亏损": trades.get('pnl', {}).get('min', 0),
        "持仓时间": trades.get('len', {}).get('average', 0)
    }

    return {
        "summary": {
            "metrics": metrics,
            "trades": strat.trade_log,  # 使用我们自定义记录的交易日志
            "returns": returns['rtot'],
            "dates": data.index.strftime("%Y-%m-%d").tolist()
        }
    }
