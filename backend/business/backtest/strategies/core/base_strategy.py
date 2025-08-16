#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略基类
使用模板方法模式定义策略执行流程
提供可复用的基础策略功能
"""

from typing import Dict, Any, List

import backtrader as bt
import pandas as pd

from backend.business.backtest.core.trading_rules import is_trade_blocked_by_price_limit, AShareTradingRules
from backend.utils.logger import setup_logger
from .managers.data_manager import DataManager
from .managers.position_manager import PositionManager
from .managers.trade_logger import TradeLogger
from .managers.trade_manager import TradeManager


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

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame] = None, effective_start_date: str | None = None,
                 **kwargs):
        """
        初始化策略基类
        
        Args:
            stock_data_dict: 股票数据字典 {股票代码: DataFrame}
            **kwargs: backtrader 策略参数（通过params定义的参数）
        """
        # 调用父类初始化，传递所有参数给 backtrader
        super().__init__(**kwargs)

        # 设置日志记录器
        self.logger = setup_logger("backtest")

        # 禁用backtrader的默认日志配置，防止重复日志
        self._disable_backtrader_logging()

        # 初始化各个管理器
        self._init_managers()

        # 设置股票数据
        if stock_data_dict:
            self.data_manager.set_stock_data(stock_data_dict)

        # 当前日期
        self.current_date = None

        # 若Runner已扩展起始日期，记录真实生效日，避免策略再跳过
        self._effective_start_date = effective_start_date

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

        # 检查是否需要跳过早期日期：若Runner已扩展起始日期到足够历史，则不再跳过
        should_skip = self._should_skip_early_days()
        if should_skip:
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
        filtered_signals: List[Dict[str, Any]] = []

        # 计算当日预计持仓：当前持仓 - 当日卖出股票数（去重）
        try:
            current_positions = self.position_manager.get_position_count()
        except Exception:
            current_positions = 0

        sell_codes = {s.get('stock_code') for s in signals if s.get('action') == 'SELL' and s.get('stock_code')}
        projected_positions = max(0, current_positions - len(sell_codes))
        try:
            max_positions = int(getattr(self.params, 'max_positions', 0) or 0)
        except Exception:
            max_positions = 0
        allowed_buy_slots = max(0, max_positions - projected_positions)

        # 按顺序过滤：卖出信号不受持仓上限限制；买入信号受“当日可用空位”限制
        for signal in signals:
            is_valid, reasons = self._validate_signal_with_reasons(signal, projected_positions, allowed_buy_slots)

            if is_valid:
                if signal.get('action') == 'BUY':
                    # 占用一个买入空位
                    if allowed_buy_slots > 0:
                        filtered_signals.append(signal)
                        allowed_buy_slots -= 1
                    else:
                        # 理论上不应触发（因为 is_valid 已包含空位校验），兜底再记一条原因
                        reasons.append(f"当日可用买入空位不足；预计持仓={projected_positions}，上限={max_positions}")
                        if self.params.enable_logging:
                            self.logger.warning(f"信号被风险控制过滤: {signal} | 过滤原因: {'; '.join(reasons)}")
                else:
                    # SELL 信号直接通过
                    filtered_signals.append(signal)
            else:
                if self.params.enable_logging:
                    self.logger.warning(f"信号被风险控制过滤: {signal} | 过滤原因: {'; '.join(reasons)}")

        return filtered_signals

    def _validate_signal_with_reasons(
        self,
        signal: Dict[str, Any],
        projected_positions: int,
        allowed_buy_slots: int
    ) -> tuple[bool, List[str]]:
        """
        校验信号有效性并返回失败原因列表（为空表示通过）。
        
        规则说明：
        - 字段与价格有效性检查
        - 涨跌停限制：涨停禁买、跌停禁卖（基于前收）
        - 当日持仓容量：买入需占用“预计持仓”后的可用空位
        
        Args:
            signal: 单个交易信号
            projected_positions: 预计持仓（当前持仓-当日卖出数）
            allowed_buy_slots: 当日剩余可用买入空位
        Returns:
            (是否有效, 原因列表)
        """
        reasons: List[str] = []

        try:
            action = signal.get('action')
            stock_code = signal.get('stock_code')
            price = float(signal.get('price', 0) or 0)
        except Exception:
            reasons.append('信号字段解析失败')
            return False, reasons

        # 1) 必填字段与价格有效性
        for field in ['action', 'stock_code', 'price']:
            if field not in signal:
                reasons.append(f"缺少字段: {field}")
        if price <= 0:
            reasons.append('价格无效(<=0)')

        if reasons:
            return False, reasons

        # 2) 涨跌停限制（基于前一交易日收盘价）
        try:
            df = self.data_manager.get_stock_data_until(stock_code, self.current_date, min_data_points=2)
            if df is not None and len(df) >= 2:
                prev_close = float(df.iloc[-2]['close'])
                # 细化原因：计算上/下限（使用精确涨跌停价格计算）
                limit_info = AShareTradingRules.check_price_limit(price, prev_close, stock_code=stock_code, is_st=False)
                if action == 'BUY' and limit_info.get('is_upper_limit'):
                    reasons.append(
                        f"涨停禁买: 当前价={price:.4f} ≥ 涨停价={limit_info.get('upper_limit', 0):.4f} (实际涨幅{limit_info.get('actual_upper_rate', 0)*100:.2f}%)"
                    )
                if action == 'SELL' and limit_info.get('is_lower_limit'):
                    reasons.append(
                        f"跌停禁卖: 当前价={price:.4f} ≤ 跌停价={limit_info.get('lower_limit', 0):.4f} (实际跌幅{abs(limit_info.get('actual_lower_rate', 0))*100:.2f}%)"
                    )
        except Exception:
            # 数据不足或异常时不强制拦截，不记录为失败
            pass

        # 3) 当日可用空位（仅 BUY）
        if action == 'BUY':
            try:
                max_positions = int(getattr(self.params, 'max_positions', 0) or 0)
            except Exception:
                max_positions = 0
            if max_positions > 0:
                # 若已无空位，则添加原因
                if allowed_buy_slots <= 0:
                    reasons.append(
                        f"持仓上限: 预计持仓={projected_positions}，上限={max_positions}，当日可用空位=0"
                    )

        return (len(reasons) == 0), reasons

    def execute_trades(self, signals: List[Dict[str, Any]]):
        """
        执行交易
        默认实现基本的交易执行逻辑，子类可以覆盖
        
        Args:
            signals: 经过风险控制的交易信号
        """
        if self.params.enable_logging:
            self.logger.info(f"交易执行前 - 现金: {self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}")
        
        # 先拆分信号
        sell_signals = [s for s in signals if s.get('action') == 'SELL']
        buy_signals = [s for s in signals if s.get('action') == 'BUY']

        # A) 预估当日卖出可得净入金（用于当日预算，避免因结算时点导致买入受限）
        estimated_sell_proceeds = 0.0
        if sell_signals:
            from backend.business.backtest.core.trading_rules import is_trade_blocked_by_price_limit
            for s in sell_signals:
                try:
                    stock_code = s.get('stock_code')
                    price = float(s.get('price'))
                    pos = self.position_manager.get_position(stock_code)
                    if not pos:
                        continue
                    # 跌停禁卖则不计入预算
                    df = self.data_manager.get_stock_data_until(stock_code, self.current_date, min_data_points=2)
                    if df is not None and len(df) >= 2:
                        prev_close = float(df.iloc[-2]['close'])
                        if is_trade_blocked_by_price_limit(price, prev_close, 'SELL', stock_code=stock_code, is_st=False):
                            continue
                    est_net = self.trade_manager.estimate_sell_total_proceeds(pos['shares'], price)
                    estimated_sell_proceeds += est_net
                except Exception:
                    continue

        # B) 先执行卖出
        for signal in sell_signals:
            try:
                self._execute_sell_signal(signal)
            except Exception as e:
                self.logger.error(f"执行卖出信号失败: {signal}, 错误: {e}")

        # C) 再执行买入（带预算；预算= 当前现金 + 估算卖出净入金）
        if buy_signals:
            # 使用"现金 + 估算可得卖出净入金"作为当日等权分摊的总预算
            # 说明：卖出在 cheat-on-close 下于收盘撮合，撮合前现金未更新；
            # 因此这里并入估算可得净入金用于预算分摊，后续仍以费用覆盖/按手/等权上限等约束防超额。
            remaining_budget = float(self.broker.getcash()) + float(estimated_sell_proceeds)
            remaining_budget = max(0.0, remaining_budget)
            remaining_buys = len(buy_signals)
            
            if self.params.enable_logging:
                self.logger.info(
                    f"预算计算: 现金={self.broker.getcash():.2f}, 估算卖出收益={estimated_sell_proceeds:.2f}, 总预算={remaining_budget:.2f}"
                )

            for signal in buy_signals:
                try:
                    stock_code = signal['stock_code']
                    price = float(signal['price'])

                    if remaining_budget <= 0:
                        break

                    # 优先使用信号中预先计算好的size，避免前视偏差
                    if 'size' in signal:
                        shares = signal['size']
                        if self.params.enable_logging:
                            self.logger.info(f"使用信号中预先计算的买入数量: {shares}")
                    else:
                        # 回退到原有的预算分摊逻辑
                        shares = self.trade_manager.calculate_buy_size(
                            price, self.params.max_positions,
                            available_cash_override=remaining_budget,
                            intended_slots=remaining_buys
                        )
                    
                    if shares <= 0:
                        remaining_buys = max(0, remaining_buys - 1)
                        continue

                    est_cost = self.trade_manager.estimate_buy_total_cost(shares, price)
                    if est_cost > remaining_budget:
                        from backend.business.backtest.core.trading_rules import AShareTradingRules
                        # 按手递减，直至满足预算或无法下单
                        while shares > 0 and est_cost > remaining_budget:
                            shares = max(0, shares - AShareTradingRules.MIN_TRADE_UNITS)
                            est_cost = self.trade_manager.estimate_buy_total_cost(shares, price)
                        if shares <= 0 or est_cost > remaining_budget:
                            remaining_buys = max(0, remaining_buys - 1)
                            continue

                    # 更新信号中的size（如果预算调整了数量）
                    signal['size'] = shares
                    
                    # 执行买入操作
                    self._execute_buy_signal(signal, remaining_budget)
                    
                    remaining_budget -= est_cost
                    remaining_buys = max(0, remaining_buys - 1)

                except Exception as e:
                    self.logger.error(f"执行买入信号失败: {signal}, 错误: {e}")

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

    def _get_data_source_for_stock(self, stock_code: str):
        """
        获取股票对应的数据源
        
        Args:
            stock_code: 股票代码
            
        Returns:
            backtrader数据源，如果未找到返回None
        """
        try:
            # 遍历所有数据源，查找匹配的股票代码
            for data in self.datas:
                if hasattr(data, '_name') and data._name == stock_code:
                    return data
            
            # 如果没有找到完全匹配的，尝试其他匹配方式
            for data in self.datas:
                if hasattr(data, '_name') and stock_code in data._name:
                    return data
                    
            return None
        except Exception as e:
            self.logger.error(f"获取股票 {stock_code} 数据源失败: {e}")
            return None

    def _should_skip_early_days(self) -> bool:
        """
        判断是否应该跳过前min_data_points天的数据
        
        如果策略有缓存适配器且已预加载数据，则可以从第一天开始回测
        否则仍需等待足够的历史数据
        
        Returns:
            bool: True表示应该跳过，False表示可以开始回测
        """
        # 如果提供了有效生效日，且当前bar日期已达到或超过生效日，则不跳过
        try:
            if getattr(self, '_effective_start_date', None):
                eff_dt = pd.to_datetime(self._effective_start_date).date()
                if self.current_date and self.current_date >= eff_dt:
                    return False
        except Exception:
            pass

        # 检查是否有足够的历史数据
        if len(self.data) < self.params.min_data_points:
            # 检查是否有缓存适配器和预加载数据
            if hasattr(self, 'cache_adapter') and self.cache_adapter is not None:
                # 检查是否已经预加载了通道数据
                if hasattr(self, 'preloaded_channel_data') and self.preloaded_channel_data:
                    if self.params.enable_logging:
                        self.logger.info(f"使用缓存数据，从第 {len(self.data)} 天开始回测")
                    return False  # 有缓存数据，不跳过

                # 即使有缓存适配器，但如果没有预加载数据，仍需检查数据是否足够
                if self.params.enable_logging:
                    self.logger.debug(f"缓存适配器存在但无预加载数据，检查实时数据可用性")

            # 没有缓存或缓存无数据，按传统逻辑跳过
            return True

        # 有足够的历史数据，不跳过
        return False

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

        # 涨跌停拦截（基于前收）
        try:
            stock_code = signal['stock_code']
            price = float(signal['price'])
            # 获取该股票在 current_date 的前一交易日收盘价
            df = self.data_manager.get_stock_data_until(stock_code, self.current_date, min_data_points=2)
            if df is not None and len(df) >= 2:
                prev_close = float(df.iloc[-2]['close'])
                if is_trade_blocked_by_price_limit(price, prev_close, signal['action'], stock_code=stock_code, is_st=False):
                    return False
        except Exception:
            # 数据不足或异常时，不做强制拦截
            pass

        return True

    def _execute_buy_signal(self, signal: Dict[str, Any], remaining_budget: float = None):
        """
        执行买入信号
        
        Args:
            signal: 买入信号
            remaining_budget: 剩余预算（用于日志输出）
        """
        stock_code = signal['stock_code']
        price = signal['price']
        shares = signal.get('size', 0)

        if shares > 0:
            # 执行买入 - 修复：指定正确的数据源
            data_source = self._get_data_source_for_stock(stock_code)
            if data_source:
                self.buy(data=data_source, size=shares)
            else:
                self.logger.warning(f"未找到股票 {stock_code} 对应的数据源，跳过买入")
                return

            # 更新持仓
            self.position_manager.add_position(stock_code, shares, price, self.current_date)

            # 记录交易
            extra_fields = {k: v for k, v in signal.items() if
                            k not in ['action', 'stock_code', 'price', 'reason', 'confidence', 'size']}
            
            # 计算买入成本（预估）
            buy_cost = AShareTradingRules.calculate_total_fees(shares * price, is_buy=True)
            
            self.trade_logger.log_trade(
                action='BUY',
                stock_code=stock_code,
                size=shares,
                price=price,
                date=self.current_date,
                reason=signal.get('reason', ''),
                confidence=signal.get('confidence', 0.0),
                trade_cost=buy_cost,  # 添加买入成本
                **extra_fields
            )

            if self.params.enable_logging:
                budget_info = f"，预算剩余: {remaining_budget:.2f}" if remaining_budget is not None else ""
                self.logger.info(f"买入 {stock_code}: {shares}股 @ {price:.2f}{budget_info}")

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
                if is_trade_blocked_by_price_limit(price, prev_close, 'SELL', stock_code=stock_code, is_st=False):
                    return
        except Exception:
            pass

        shares = position_info['shares']

        # 执行卖出 - 修复：指定正确的数据源
        data_source = self._get_data_source_for_stock(stock_code)
        if data_source:
            self.sell(data=data_source, size=shares)
        else:
            self.logger.warning(f"未找到股票 {stock_code} 对应的数据源，跳过卖出")
            return

        # 计算收益
        buy_price = position_info['buy_price']
        buy_date = position_info['buy_date']

        returns_pct = (price - buy_price) / buy_price * 100 if buy_price > 0 else 0
        profit_amount = (price - buy_price) * shares
        holding_days = (self.current_date - buy_date).days

        # 更新持仓
        self.position_manager.remove_position(stock_code)

        # 记录交易
        extra_fields = {k: v for k, v in signal.items() if
                        k not in ['action', 'stock_code', 'price', 'reason', 'confidence']}
        
        # 计算卖出成本（预估）
        sell_cost = AShareTradingRules.calculate_total_fees(shares * price, is_buy=False)
        
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
            trade_cost=sell_cost,  # 使用计算出的卖出成本
            **extra_fields
        )

        if self.params.enable_logging:
            profit_sign = "+" if profit_amount >= 0 else ""
            self.logger.info(f"卖出 {stock_code}: {shares}股 @ {price:.2f}, "
                             f"收益: {profit_sign}{profit_amount:.2f}元 ({returns_pct:.2f}%), "
                             f"持仓{holding_days}天")

    def get_per_trade_cash(self, max_positions: int) -> float:
        """
        计算单笔交易的可用资金，避免前视偏差
        
        使用T-1日收盘后的总资产减去现金储备，然后除以最大持仓数来计算每笔交易的资金分配，
        确保仓位计算只使用决策时刻已知的信息，同时保持稳健的资金管理原则。
        
        Args:
            max_positions: 最大持仓数量
            
        Returns:
            float: 单笔交易的可用资金
        """
        if max_positions <= 0:
            return 0.0
            
        # 使用T-1日收盘后的总资产（self.broker.getvalue()）
        # 这避免了使用当天收盘价计算持股价值的前视偏差
        total_value = self.broker.getvalue()
        
        # 获取交易管理器中的现金储备比例
        min_cash_reserve_ratio = self.trade_manager.risk_params.get('min_cash_reserve', 0.05)
        
        # 计算需要保留的现金金额
        min_cash_reserve = total_value * min_cash_reserve_ratio
        
        # 计算可用于投资的总风险资本
        # 保护性检查：确保风险资本不会小于0
        risk_capital = max(0.0, total_value - min_cash_reserve)
        
        # 使用风险资本来分配给每笔交易的资金
        per_trade_cash = risk_capital / max_positions
        
        if self.params.enable_logging:
            self.logger.debug(f"资金分配计算: 总资产={total_value:.2f}, "
                             f"现金储备={min_cash_reserve:.2f}({min_cash_reserve_ratio*100:.1f}%), "
                             f"风险资本={risk_capital:.2f}, "
                             f"每笔交易={per_trade_cash:.2f}")
        
        return per_trade_cash

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            策略信息字典，包含策略名称、参数、当前状态等
        """
        return {
            "strategy_name": self.__class__.__name__,
            "parameters": self._get_parameters(),
            "current_positions": self.position_manager.get_position_count(),
            "current_cash": self.broker.getcash() if hasattr(self, 'broker') and self.broker else 0,
            "portfolio_value": self.broker.getvalue() if hasattr(self, 'broker') and self.broker else 0,
            "current_date": self.current_date.strftime('%Y-%m-%d') if self.current_date else None,
            "is_initialized": self._is_initialized
        }

    def _get_parameters(self) -> Dict[str, Any]:
        """
        获取策略参数
        
        Returns:
            参数字典
        """
        try:
            # 获取策略的参数定义
            if hasattr(self.__class__, 'params'):
                param_dict = {}
                # 安全地遍历类的参数定义
                try:
                    pairs = self.__class__.params._getpairs()
                    for pair in pairs:
                        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                            param_name, param_value = pair[0], pair[1]
                            # 获取实际参数值
                            actual_value = getattr(self.params, param_name, param_value)
                            param_dict[param_name] = actual_value
                        elif isinstance(pair, str):
                            # 如果只是参数名
                            param_dict[pair] = getattr(self.params, pair, None)
                except:
                    # 如果 _getpairs() 失败，尝试直接访问参数属性
                    for attr in dir(self.params):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(self.params, attr)
                                if not callable(value):
                                    param_dict[attr] = value
                            except:
                                continue
                return param_dict
            else:
                return {}
        except Exception as e:
            # 如果参数访问失败，返回空字典而不是崩溃
            return {"error": f"Unable to access parameters: {str(e)}"}

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取策略性能摘要
        
        Returns:
            性能摘要字典
        """
        if hasattr(self, 'trade_logger') and self.trade_logger:
            trades = self.trade_logger.get_trades()
        else:
            trades = []

        return {
            "total_trades": len(trades),
            "trades": trades,
            "current_positions": self.position_manager.get_position_count(),
            "portfolio_value": self.broker.getvalue() if hasattr(self, 'broker') and self.broker else 0
        }

    def _disable_backtrader_logging(self):
        """禁用backtrader的默认日志配置"""
        try:
            import logging
            # 禁用backtrader的默认日志处理器
            for logger_name in ['backtrader', 'bt', 'cerebro']:
                logger = logging.getLogger(logger_name)
                logger.disabled = True
                # 清除可能存在的处理器
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
        except Exception as e:
            self.logger.warning(f"禁用backtrader日志配置时出现警告: {e}")
