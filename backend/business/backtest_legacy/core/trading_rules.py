#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股交易规则配置
包含A股市场的各种交易限制和规则
"""

from typing import Dict, Any
import re

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

    # 涨跌停规则 - 默认值（将被动态方法覆盖）
    PRICE_LIMIT_RATE = 0.10  # 默认涨跌停幅度（10%）
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
    def get_price_limit_rate_by_code(cls, stock_code: str, is_st: bool = False) -> float:
        """
        根据股票代码确定涨跌停幅度
        
        Args:
            stock_code: 股票代码（如：'000001'、'300001'、'688001'等）
            is_st: 是否为ST股票
            
        Returns:
            涨跌停幅度（如：0.10表示10%）
            
        Note:
            涨跌停规则：
            - 沪市主板（60xxxx）：±10%
            - 深市主板（000xxx、001xxx）：±10%
            - 创业板（300xxx）：±20%
            - 科创板（688xxx）：±20%
            - 北交所（83xxxx、87xxxx、43xxxx）：±30%（新股前5日无涨跌幅限制）
            - ST股票：在对应板块基础上减半
        """
        # 清理股票代码，移除市场前缀
        clean_code = stock_code.upper().replace('SH', '').replace('SZ', '').replace('BJ', '')
        
        # 根据代码段判断涨跌停幅度
        if re.match(r'^60\d{4}$', clean_code):
            # 沪市主板
            base_rate = 0.10
        elif re.match(r'^000\d{3}$|^001\d{3}$', clean_code):
            # 深市主板
            base_rate = 0.10
        elif re.match(r'^300\d{3}$', clean_code):
            # 创业板
            base_rate = 0.20
        elif re.match(r'^688\d{3}$', clean_code):
            # 科创板
            base_rate = 0.20
        elif re.match(r'^83\d{4}$|^87\d{4}$|^43\d{4}$', clean_code):
            # 北交所
            base_rate = 0.30
        else:
            # 默认使用主板规则
            base_rate = 0.10
        
        # ST股票涨跌停幅度减半
        if is_st:
            return base_rate * 0.5
        
        return base_rate

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
    def calculate_precise_price_limits(cls, reference_price: float, stock_code: str = None, is_st: bool = False) -> Dict[str, Any]:
        """
        计算精确的涨跌停价格，考虑A股最小变动单位（0.01元）和四舍五入规则
        
        Args:
            reference_price: 参考价格（通常是前一日收盘价）
            stock_code: 股票代码（用于确定涨跌停幅度）
            is_st: 是否为ST股票
            
        Returns:
            包含精确涨跌停价格的字典
            
        Note:
            A股涨跌停价格计算规则：
            1. 理论涨跌停价格 = 参考价格 × (1 ± 涨跌停幅度)
            2. 实际涨跌停价格 = 理论价格按最小变动单位（0.01元）四舍五入
            3. 实际涨跌停幅度可能小于理论幅度
        """
        # 获取涨跌停幅度
        if stock_code:
            limit_rate = cls.get_price_limit_rate_by_code(stock_code, is_st)
        else:
            limit_rate = cls.ST_PRICE_LIMIT_RATE if is_st else cls.PRICE_LIMIT_RATE
        
        # 计算理论涨跌停价格
        theoretical_upper = reference_price * (1 + limit_rate)
        theoretical_lower = reference_price * (1 - limit_rate)
        
        # 按最小变动单位（0.01元）四舍五入
        precise_upper = round(theoretical_upper / 0.01) * 0.01
        precise_lower = round(theoretical_lower / 0.01) * 0.01
        
        # 计算实际涨跌停幅度
        actual_upper_rate = (precise_upper - reference_price) / reference_price
        actual_lower_rate = (precise_lower - reference_price) / reference_price
        
        return {
            "theoretical_upper": theoretical_upper,
            "theoretical_lower": theoretical_lower,
            "precise_upper": precise_upper,
            "precise_lower": precise_lower,
            "theoretical_rate": limit_rate,
            "actual_upper_rate": actual_upper_rate,
            "actual_lower_rate": actual_lower_rate,
            "reference_price": reference_price,
            "stock_code": stock_code,
            "is_st": is_st
        }

    @classmethod
    def check_price_limit(cls, current_price: float, reference_price: float, 
                         stock_code: str = None, is_st: bool = False) -> Dict[str, Any]:
        """
        检查价格是否触及涨跌停（使用精确价格计算）
        
        Args:
            current_price: 当前价格
            reference_price: 参考价格（通常是前一日收盘价）
            stock_code: 股票代码（用于确定涨跌停幅度）
            is_st: 是否为ST股票
            
        Returns:
            检查结果字典
            
        Note:
            跌停处理规则：
            - 涨停：当前价格 >= 涨停价时无法买入
            - 跌停：当前价格在跌停价上下各一分钱范围内时无法卖出（按跌停处理）
        """
        # 获取精确的涨跌停价格
        price_limits = cls.calculate_precise_price_limits(reference_price, stock_code, is_st)
        
        # 涨停判断：当前价格 >= 涨停价
        is_upper_limit = current_price >= price_limits["precise_upper"]
        
        # 跌停判断：当前价格在跌停价上下各一分钱范围内
        # 即：跌停价 - 0.01 <= 当前价格 <= 跌停价 + 0.01
        # 注意：跌停价上下各一分钱范围内都无法卖出，按跌停处理
        lower_limit_price = price_limits["precise_lower"]
        is_lower_limit = (lower_limit_price - 0.01) <= current_price <= (lower_limit_price + 0.01)
        
        return {
            "is_upper_limit": is_upper_limit,
            "is_lower_limit": is_lower_limit,
            "upper_limit": price_limits["precise_upper"],
            "lower_limit": price_limits["precise_lower"],
            "lower_limit_range": (lower_limit_price - 0.01, lower_limit_price + 0.01),
            "theoretical_upper": price_limits["theoretical_upper"],
            "theoretical_lower": price_limits["theoretical_lower"],
            "limit_rate": price_limits["theoretical_rate"],
            "actual_upper_rate": price_limits["actual_upper_rate"],
            "actual_lower_rate": price_limits["actual_lower_rate"],
            "stock_code": stock_code,
            "is_st": is_st
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
            "suspension_rules": cls.SUSPENSION_RULES,
            "price_limit_rules": {
                "sh_main": "60xxxx: ±10%",
                "sz_main": "000xxx/001xxx: ±10%",
                "gem": "300xxx: ±20%",
                "star": "688xxx: ±20%",
                "beijing": "83xxxx/87xxxx/43xxxx: ±30%",
                "st_stocks": "ST股票: 对应板块基础上减半"
            }
        }


def is_trade_blocked_by_price_limit(current_price: float, reference_price: float, action: str,
                                    stock_code: str = None, is_st: bool = False) -> bool:
    """
    纯函数：基于涨跌停规则判断是否应拦截交易
    
    Args:
        current_price: 当前价格
        reference_price: 参考价格（通常昨日收盘价）
        action: 交易方向，'BUY' 或 'SELL'
        stock_code: 股票代码（用于确定涨跌停幅度）
        is_st: 是否ST股票
    
    Returns:
        True 表示应拦截（涨停禁买/跌停禁卖），False 表示不拦截
    """
    result = AShareTradingRules.check_price_limit(current_price, reference_price, 
                                                 stock_code=stock_code, is_st=is_st)
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
