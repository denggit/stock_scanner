#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易管理器
负责交易执行、资金管理和风险控制
使用策略模式处理不同的资金分配策略
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from backend.business.backtest.core.trading_rules import AShareTradingRules
from backend.utils.logger import setup_logger
from .position_manager import PositionManager


class FundAllocationStrategy(ABC):
    """
    资金分配策略抽象基类
    使用策略模式实现不同的资金分配算法
    """

    @abstractmethod
    def calculate_allocation(self, available_cash: float, price: float,
                             max_positions: int, current_positions: int) -> int:
        """
        计算买入数量
        
        Args:
            available_cash: 可用资金
            price: 股票价格
            max_positions: 最大持仓数
            current_positions: 当前持仓数
            
        Returns:
            建议买入数量
        """
        pass


class EqualWeightAllocation(FundAllocationStrategy):
    """
    等权重资金分配策略（按剩余空位等权）
    将可用资金平均分配给“剩余可建仓的空位数”，而不是按 max_positions 直接等分。
    例如：初始 20 万、max_positions=20，已经持有 10 只且剩余现金 10 万，则单只目标金额≈ 10万 / (20-10) = 1 万。
    """

    def calculate_allocation(self, available_cash: float, price: float,
                             max_positions: int, current_positions: int,
                             intended_slots: Optional[int] = None) -> int:
        """
        等权重分配计算（按剩余空位）
        
        Args:
            available_cash: 可用资金
            price: 股票价格
            max_positions: 最大持仓数
            current_positions: 当前持仓数
            intended_slots: 当日计划买入的“剩余笔数”（覆盖默认的“剩余空位数”分母）。
                             若提供且 > 0，则按该值进行等分，从而避免“按全部空位”等分导致单笔过小。
            
        Returns:
            建议买入数量
        """
        if price <= 0 or max_positions <= 0:
            return 0

        # 预留少量手续费（粗略预留，精确校验在 TradeManager 内完成）
        commission_rate = AShareTradingRules.COMMISSION_RATE
        usable_cash = max(0.0, available_cash * (1 - commission_rate))

        # 优先使用“当日计划买入的剩余笔数”进行等分；若未提供则退回“剩余空位数”等分
        if intended_slots is not None and int(intended_slots) > 0:
            remaining_slots = int(intended_slots)
        else:
            remaining_slots = max(1, max_positions - current_positions)
        cash_per_stock = usable_cash / remaining_slots
        shares = int(cash_per_stock / price)

        return max(0, shares)


class TargetPercentAllocation(FundAllocationStrategy):
    """
    目标百分比资金分配策略
    每只股票占用总资金的固定百分比
    """

    def __init__(self, target_percent: float = 0.02):
        """
        初始化目标百分比分配策略
        
        Args:
            target_percent: 每只股票的目标占比（默认2%）
        """
        self.target_percent = target_percent

    def calculate_allocation(self, available_cash: float, price: float,
                             max_positions: int, current_positions: int) -> int:
        """
        目标百分比分配计算
        
        Args:
            available_cash: 可用资金
            price: 股票价格
            max_positions: 最大持仓数
            current_positions: 当前持仓数
            
        Returns:
            建议买入数量
        """
        if price <= 0:
            return 0

        # 计算目标投资金额
        target_amount = available_cash * self.target_percent
        shares = int(target_amount / price)

        return max(0, shares)


class TradeManager:
    """
    交易管理器
    
    功能：
    - 执行买入卖出交易
    - 管理资金分配
    - 风险控制
    - 交易统计
    """

    def __init__(self, broker=None, position_manager: PositionManager = None):
        """
        初始化交易管理器
        
        Args:
            broker: backtrader的broker对象
            position_manager: 仓位管理器
        """
        self.broker = broker
        self.position_manager = position_manager or PositionManager()
        self.logger = setup_logger("backtest")

        # 资金分配策略（默认使用等权重）
        self.allocation_strategy = EqualWeightAllocation()

        # 交易统计
        self.trade_stats = {
            'total_buys': 0,
            'total_sells': 0,
            'total_buy_value': 0.0,
            'total_sell_value': 0.0,
            'last_trade_date': None
        }

        # 风险控制参数
        self.risk_params = {
            'max_single_position_percent': 0.1,  # 单只股票最大占比10%
            'min_cash_reserve': 0.05,  # 最低现金储备5%
            'max_daily_trades': 20,  # 每日最大交易次数
            'min_price': 2.0,  # 最低股价
            'max_price': 1000.0  # 最高股价
        }

    def set_broker(self, broker):
        """设置broker对象"""
        self.broker = broker

    def set_allocation_strategy(self, strategy: FundAllocationStrategy):
        """
        设置资金分配策略
        
        Args:
            strategy: 资金分配策略实例
        """
        self.allocation_strategy = strategy
        self.logger.info(f"切换资金分配策略: {strategy.__class__.__name__}")

    def _apply_ashare_constraints(self, shares: int, price: float, available_cash: float) -> int:
        """
        应用A股交易约束：按手取整、最小金额校验、费用覆盖校验
        
        Args:
            shares: 原始建议买入数量
            price: 当前价格
            available_cash: 可用现金
        
        Returns:
            满足A股规则后的买入股数
        """
        if shares <= 0 or price <= 0:
            return 0

        # 1) 按手取整
        shares = AShareTradingRules.adjust_trade_quantity(shares)
        if shares <= 0:
            return 0

        # 2) 最小成交金额校验（如不足则尝试提升到满足最小金额的最近整数手）
        trade_value = shares * price
        if trade_value < AShareTradingRules.MIN_TRADE_AMOUNT:
            import math
            min_lots = math.ceil(AShareTradingRules.MIN_TRADE_AMOUNT / price / AShareTradingRules.MIN_TRADE_UNITS)
            shares = max(shares, min_lots * AShareTradingRules.MIN_TRADE_UNITS)
            trade_value = shares * price

        # 3) 费用覆盖校验（买入：佣金+过户费）
        total_fees = AShareTradingRules.calculate_total_fees(trade_value, is_buy=True)
        total_cost = trade_value + total_fees
        if total_cost > available_cash:
            affordable_shares = int(available_cash / price)
            affordable_shares = AShareTradingRules.adjust_trade_quantity(affordable_shares)
            if affordable_shares <= 0:
                return 0
            trade_value = affordable_shares * price
            total_fees = AShareTradingRules.calculate_total_fees(trade_value, is_buy=True)
            total_cost = trade_value + total_fees
            if total_cost > available_cash:
                affordable_shares -= AShareTradingRules.MIN_TRADE_UNITS
                affordable_shares = max(0, affordable_shares)
                if affordable_shares <= 0:
                    return 0
                trade_value = affordable_shares * price
                total_fees = AShareTradingRules.calculate_total_fees(trade_value, is_buy=True)
                total_cost = trade_value + total_fees
                if total_cost > available_cash:
                    return 0
            shares = affordable_shares

        return shares

    def calculate_buy_size(self, price: float, max_positions: int,
                           available_cash_override: Optional[float] = None,
                           intended_slots: Optional[int] = None) -> int:
        """
        计算买入数量
        
        Args:
            price: 股票价格
            max_positions: 最大持仓数量
            available_cash_override: 可选，外部传入的预算现金，用于同一交易日多笔下单时防止超额
            intended_slots: 可选，当日剩余计划买入的笔数（用于等权资金分配的分母）
            
        Returns:
            建议买入数量
        """
        if not self.broker:
            return 0

        # 获取可用资金
        available_cash = self.broker.getcash() if available_cash_override is None else max(0.0, float(
            available_cash_override))
        current_positions = self.position_manager.get_position_count()

        # 使用当前的资金分配策略
        shares = self.allocation_strategy.calculate_allocation(
            available_cash, price, max_positions, current_positions,
            intended_slots=intended_slots
        )

        # 风险控制检查（传入max_positions用于等权上限约束）
        shares = self._apply_risk_control(shares, price, available_cash, max_positions)

        # A股约束：按手、最小金额与费用覆盖
        shares = self._apply_ashare_constraints(shares, price, available_cash)

        return shares

    def estimate_buy_total_cost(self, shares: int, price: float) -> float:
        """
        估算买入总成本（含费用）
        
        Args:
            shares: 买入股数
            price: 价格
        Returns:
            总成本=成交额+佣金/过户费（买入不含印花税）
        """
        if shares <= 0 or price <= 0:
            return 0.0
        trade_value = float(shares) * float(price)
        total_fees = AShareTradingRules.calculate_total_fees(trade_value, is_buy=True)
        return trade_value + total_fees

    def estimate_sell_total_proceeds(self, shares: int, price: float) -> float:
        """
        估算卖出净入金（扣除佣金/过户费/印花税）
        
        Args:
            shares: 卖出股数
            price: 卖出价格
        Returns:
            卖出后实际增加的现金（>=0）。无效输入返回0。
        """
        if shares <= 0 or price <= 0:
            return 0.0
        trade_value = float(shares) * float(price)
        total_fees = AShareTradingRules.calculate_total_fees(trade_value, is_buy=False)
        net_proceeds = trade_value - total_fees
        return max(0.0, net_proceeds)

    def _apply_risk_control(self, shares: int, price: float, available_cash: float, max_positions: int) -> int:
        """
        应用风险控制规则
        
        Args:
            shares: 原始买入数量
            price: 股票价格
            available_cash: 可用资金
            
        Returns:
            经过风险控制的买入数量
        """
        if shares <= 0:
            return 0

        # 检查价格范围
        if price < self.risk_params['min_price'] or price > self.risk_params['max_price']:
            self.logger.warning(f"价格 {price} 超出允许范围，跳过交易")
            return 0

        # 统一以“最严格的限制”为准：
        # - 单票最大占比（基于总资产）
        # - 单票预算不超过 总资产 / max_positions（等权仓位上限，若提供）
        # - 最低现金留存（基于总资产）
        total_value = self.broker.getvalue() if self.broker else available_cash
        min_cash_reserve = total_value * self.risk_params['min_cash_reserve']
        max_trade_value_by_percent = total_value * self.risk_params['max_single_position_percent']
        # 等权仓位上限：若未设置/无效，则不生效
        try:
            mp = int(max_positions) if max_positions is not None else 0
        except Exception:
            mp = 0
        max_trade_value_by_equal_weight = (total_value / mp) if mp and mp > 0 else float('inf')
        max_trade_value_by_reserve = max(0.0, available_cash - min_cash_reserve)

        # 允许的最大交易金额取二者较小值
        max_allowable_trade_value = min(max_trade_value_by_percent, max_trade_value_by_equal_weight, max_trade_value_by_reserve)

        # 若当前建议超过限制，则下调
        current_trade_value = shares * price
        if current_trade_value > max_allowable_trade_value:
            limited_shares = int(max_allowable_trade_value / price)
            # 仅在改变时输出一次日志
            if current_trade_value > max_trade_value_by_percent:
                self.logger.info(f"单只股票占比过高，调整买入数量至 {limited_shares}")
            if current_trade_value > max_trade_value_by_equal_weight:
                self.logger.info(f"按等权上限(总资产/最大持仓数)限制，调整买入数量至 {limited_shares}")
            if current_trade_value > max_trade_value_by_reserve:
                self.logger.info(f"为保持现金储备，调整买入数量至 {limited_shares}")
            shares = limited_shares

        # 若不可交易金额为0，则清零
        if max_allowable_trade_value <= 0:
            shares = 0
            self.logger.warning("资金不足，无法满足现金/占比约束")

        return shares

    def execute_buy(self, stock_code: str, shares: int, price: float, date,
                    strategy_instance) -> bool:
        """
        执行买入交易
        
        Args:
            stock_code: 股票代码
            shares: 买入数量
            price: 买入价格
            date: 交易日期
            strategy_instance: 策略实例（用于执行backtrader交易）
            
        Returns:
            是否成功执行
        """
        if shares <= 0:
            return False

        try:
            # 执行backtrader买入
            strategy_instance.buy(size=shares)

            # 更新持仓
            self.position_manager.add_position(stock_code, shares, price, date)

            # 更新统计
            self._update_buy_stats(shares, price, date)

            self.logger.info(f"执行买入: {stock_code} {shares}股 @ {price:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"买入执行失败: {e}")
            return False

    def execute_sell(self, stock_code: str, date, strategy_instance) -> Optional[Dict[str, Any]]:
        """
        执行卖出交易
        
        Args:
            stock_code: 股票代码
            date: 交易日期
            strategy_instance: 策略实例（用于执行backtrader交易）
            
        Returns:
            交易信息字典，如果失败返回None
        """
        # 获取持仓信息
        position_info = self.position_manager.get_position(stock_code)
        if not position_info:
            self.logger.warning(f"尝试卖出未持有的股票: {stock_code}")
            return None

        shares = position_info['shares']

        try:
            # 执行backtrader卖出
            strategy_instance.sell(size=shares)

            # 移除持仓
            removed_position = self.position_manager.remove_position(stock_code)

            # 更新统计（需要当前价格，这里暂时用买入价作为占位符）
            self._update_sell_stats(shares, position_info['buy_price'], date)

            self.logger.info(f"执行卖出: {stock_code} {shares}股")

            return {
                'stock_code': stock_code,
                'shares': shares,
                'buy_price': position_info['buy_price'],
                'buy_date': position_info['buy_date'],
                'sell_date': date
            }

        except Exception as e:
            self.logger.error(f"卖出执行失败: {e}")
            return None

    def _update_buy_stats(self, shares: int, price: float, date):
        """更新买入统计"""
        self.trade_stats['total_buys'] += 1
        self.trade_stats['total_buy_value'] += shares * price
        self.trade_stats['last_trade_date'] = date

    def _update_sell_stats(self, shares: int, price: float, date):
        """更新卖出统计"""
        self.trade_stats['total_sells'] += 1
        self.trade_stats['total_sell_value'] += shares * price
        self.trade_stats['last_trade_date'] = date

    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        获取交易统计信息
        
        Returns:
            交易统计字典
        """
        stats = self.trade_stats.copy()

        # 添加计算字段
        if self.trade_stats['total_buys'] > 0:
            stats['avg_buy_value'] = self.trade_stats['total_buy_value'] / self.trade_stats['total_buys']
        else:
            stats['avg_buy_value'] = 0.0

        if self.trade_stats['total_sells'] > 0:
            stats['avg_sell_value'] = self.trade_stats['total_sell_value'] / self.trade_stats['total_sells']
        else:
            stats['avg_sell_value'] = 0.0

        stats['total_trades'] = self.trade_stats['total_buys'] + self.trade_stats['total_sells']

        return stats

    def update_risk_params(self, **params):
        """
        更新风险控制参数
        
        Args:
            **params: 风险参数键值对
        """
        for key, value in params.items():
            if key in self.risk_params:
                old_value = self.risk_params[key]
                self.risk_params[key] = value
                self.logger.info(f"更新风险参数 {key}: {old_value} -> {value}")
            else:
                self.logger.warning(f"未知的风险参数: {key}")

    def get_current_exposure(self) -> Dict[str, float]:
        """
        获取当前风险敞口
        
        Returns:
            风险敞口字典
        """
        if not self.broker:
            return {}

        total_value = self.broker.getvalue()
        cash = self.broker.getcash()
        position_value = self.position_manager.get_total_position_value()

        return {
            'total_value': total_value,
            'cash': cash,
            'position_value': position_value,
            'cash_ratio': cash / total_value if total_value > 0 else 0,
            'position_ratio': position_value / total_value if total_value > 0 else 0,
            'position_count': self.position_manager.get_position_count()
        }
