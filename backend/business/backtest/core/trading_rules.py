#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股交易规则配置
包含A股市场的各种交易限制和规则
"""

from typing import Dict, Any

# 为 backtrader 手续费模型提供支持
try:
    import backtrader as bt  # type: ignore
except Exception:  # 在纯单元测试环境可能没有 backtrader
    bt = None  # type: ignore


class AShareTradingRules:
    """
    A股交易规则类
    包含A股市场的各种交易限制和规则
    """

    # 基础交易规则
    MIN_TRADE_UNITS = 100  # 最小交易单位（股）
    MIN_TRADE_AMOUNT = 1000  # 最小交易金额（元）

    # 手续费规则
    COMMISSION_RATE = 0.0003  # 佣金费率（万分之三）
    MIN_COMMISSION = 5.0  # 最低佣金（元）
    MAX_COMMISSION = 0.0003  # 最高佣金费率（保留字段）

    # 印花税规则
    STAMP_TAX_RATE = 0.001  # 印花税率（千分之一，仅卖出时收取）

    # 过户费规则
    TRANSFER_FEE_RATE = 0.00002  # 过户费率（十万分之二）

    # 涨跌停规则
    PRICE_LIMIT_RATE = 0.10  # 涨跌停幅度（10%）
    ST_PRICE_LIMIT_RATE = 0.05  # ST股票涨跌停幅度（5%）

    # 交易时间规则
    TRADING_HOURS = {
        "morning": ("09:30", "11:30"),
        "afternoon": ("13:00", "15:00")
    }

    # 停牌规则
    SUSPENSION_RULES = {
        "daily_limit": True,  # 涨跌停停牌
        "news_announcement": True,  # 重大事项停牌
        "trading_halt": True  # 交易异常停牌
    }

    @classmethod
    def calculate_commission(cls, trade_amount: float, is_buy: bool = True) -> float:
        """
        计算交易佣金
        
        Args:
            trade_amount: 交易金额
            is_buy: 是否为买入交易（保留参数）
            
        Returns:
            佣金金额（不含印花税与过户费）
        """
        commission = trade_amount * cls.COMMISSION_RATE
        commission = max(commission, cls.MIN_COMMISSION)
        return commission

    @classmethod
    def calculate_stamp_tax(cls, trade_amount: float) -> float:
        """
        计算印花税（仅卖出时收取）
        
        Args:
            trade_amount: 交易金额
            
        Returns:
            印花税金额
        """
        return trade_amount * cls.STAMP_TAX_RATE

    @classmethod
    def calculate_transfer_fee(cls, trade_amount: float) -> float:
        """
        计算过户费
        
        Args:
            trade_amount: 交易金额
            
        Returns:
            过户费金额
        """
        return trade_amount * cls.TRANSFER_FEE_RATE

    @classmethod
    def calculate_total_fees(cls, trade_amount: float, is_buy: bool = True) -> float:
        """
        计算总交易费用
        
        Args:
            trade_amount: 交易金额
            is_buy: 是否为买入交易
            
        Returns:
            总费用金额（佣金+过户费，若卖出则加印花税）
        """
        commission = cls.calculate_commission(trade_amount, is_buy)
        transfer_fee = cls.calculate_transfer_fee(trade_amount)

        total_fees = commission + transfer_fee

        # 卖出时加收印花税
        if not is_buy:
            stamp_tax = cls.calculate_stamp_tax(trade_amount)
            total_fees += stamp_tax

        return total_fees

    @classmethod
    def adjust_trade_quantity(cls, quantity: int) -> int:
        """
        调整交易数量，确保符合A股规则（按手）
        
        Args:
            quantity: 原始交易数量
            
        Returns:
            调整后的交易数量（向下取整为100的整数倍；若>0且不足100，则返回100）
        """
        # 确保是100的整数倍
        adjusted_quantity = (quantity // cls.MIN_TRADE_UNITS) * cls.MIN_TRADE_UNITS

        # 如果调整后为0，但原始数量大于0，则设为最小交易单位
        if adjusted_quantity == 0 and quantity > 0:
            adjusted_quantity = cls.MIN_TRADE_UNITS

        return adjusted_quantity

    @classmethod
    def check_price_limit(cls, current_price: float, reference_price: float, is_st: bool = False) -> Dict[str, Any]:
        """
        检查价格是否触及涨跌停
        
        Args:
            current_price: 当前价格
            reference_price: 参考价格（通常是前一日收盘价）
            is_st: 是否为ST股票
            
        Returns:
            检查结果字典
        """
        limit_rate = cls.ST_PRICE_LIMIT_RATE if is_st else cls.PRICE_LIMIT_RATE

        upper_limit = reference_price * (1 + limit_rate)
        lower_limit = reference_price * (1 - limit_rate)

        return {
            "is_upper_limit": current_price >= upper_limit,
            "is_lower_limit": current_price <= lower_limit,
            "upper_limit": upper_limit,
            "lower_limit": lower_limit,
            "limit_rate": limit_rate
        }

    @classmethod
    def get_trading_rules_summary(cls) -> Dict[str, Any]:
        """
        获取交易规则摘要
        
        Returns:
            交易规则摘要字典
        """
        return {
            "min_trade_units": cls.MIN_TRADE_UNITS,
            "min_trade_amount": cls.MIN_TRADE_AMOUNT,
            "commission_rate": cls.COMMISSION_RATE,
            "min_commission": cls.MIN_COMMISSION,
            "stamp_tax_rate": cls.STAMP_TAX_RATE,
            "transfer_fee_rate": cls.TRANSFER_FEE_RATE,
            "price_limit_rate": cls.PRICE_LIMIT_RATE,
            "st_price_limit_rate": cls.ST_PRICE_LIMIT_RATE,
            "trading_hours": cls.TRADING_HOURS,
            "suspension_rules": cls.SUSPENSION_RULES
        }


def is_trade_blocked_by_price_limit(current_price: float, reference_price: float, action: str,
                                    is_st: bool = False) -> bool:
    """
    纯函数：基于涨跌停规则判断是否应拦截交易
    
    Args:
        current_price: 当前价格
        reference_price: 参考价格（通常昨日收盘价）
        action: 交易方向，'BUY' 或 'SELL'
        is_st: 是否ST股票
    
    Returns:
        True 表示应拦截（涨停禁买/跌停禁卖），False 表示不拦截
    """
    result = AShareTradingRules.check_price_limit(current_price, reference_price, is_st=is_st)
    if action == 'BUY' and result["is_upper_limit"]:
        return True
    if action == 'SELL' and result["is_lower_limit"]:
        return True
    return False


# 在 backtrader 中实现 A股手续费模型，合并佣金、过户费和（卖出时）印花税
if bt is not None:
    class AStockCommissionInfo(bt.CommInfoBase):
        """
        A股股票手续费模型（backtrader CommissionInfo）
        
        计算规则：
        - 佣金：max(金额 * commission, min_commission)
        - 过户费：金额 * transfer_fee_rate（买卖双方）
        - 印花税：金额 * stamp_tax_rate（仅卖出）
        
        参数:
        - commission: 佣金费率（如 0.0003）
        - stamp_tax_rate: 印花税率（如 0.001）
        - transfer_fee_rate: 过户费率（如 0.00002）
        - min_commission: 单笔最低佣金（如 5 元）
        """

        params = (
            ("commission", AShareTradingRules.COMMISSION_RATE),
            ("stamp_tax_rate", AShareTradingRules.STAMP_TAX_RATE),
            ("transfer_fee_rate", AShareTradingRules.TRANSFER_FEE_RATE),
            ("min_commission", AShareTradingRules.MIN_COMMISSION),
            ("stocklike", True),
            ("commtype", bt.CommInfoBase.COMM_PERC),
        )

        def getcommission(self, size, price, pseudoexec=False):
            """
            计算 backtrader 单笔交易的费用
            
            Args:
                size: 交易数量（买入为正，卖出为负）
                price: 交易价格
                pseudoexec: 保留参数
            
            Returns:
                费用总额（正数）
            """
            trade_value = abs(size) * float(price)
            # 佣金（含最低佣金）
            commission = max(trade_value * float(self.p.commission), float(self.p.min_commission))
            # 过户费
            transfer_fee = trade_value * float(self.p.transfer_fee_rate)
            # 印花税：仅卖出时收取
            stamp_tax = trade_value * float(self.p.stamp_tax_rate) if size < 0 else 0.0
            return commission + transfer_fee + stamp_tax

# 全局交易规则实例
ASHARE_RULES = AShareTradingRules()
