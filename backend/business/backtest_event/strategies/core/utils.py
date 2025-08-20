#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具模块
提供所有策略都可以使用的基础工具函数
"""

from typing import Dict, Any, List, Optional, Union


class SignalUtils:
    """
    信号处理工具类
    提供创建买入/卖出信号的标准化方法，适用于所有策略
    """

    @staticmethod
    def create_buy_signal(
            stock_code: str,
            price: float,
            reason: str,
            confidence: float = 1.0,
            extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建标准化的买入信号
        
        Args:
            stock_code: 股票代码
            price: 买入价格
            reason: 买入理由
            confidence: 信心度 (0.0-1.0)
            extra: 额外信息字典（如技术指标信息）
            
        Returns:
            买入信号字典
        """
        signal = {
            'action': 'BUY',
            'stock_code': stock_code,
            'price': price,
            'reason': reason,
            'confidence': confidence
        }

        if extra:
            signal.update(extra)

        return signal

    @staticmethod
    def create_sell_signal(
            stock_code: str,
            price: float,
            reason: str,
            confidence: float = 1.0,
            extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建标准化的卖出信号
        
        Args:
            stock_code: 股票代码
            price: 卖出价格
            reason: 卖出理由
            confidence: 信心度 (0.0-1.0)
            extra: 额外信息字典（如技术指标信息）
            
        Returns:
            卖出信号字典
        """
        signal = {
            'action': 'SELL',
            'stock_code': stock_code,
            'price': price,
            'reason': reason,
            'confidence': confidence
        }

        if extra:
            signal.update(extra)

        return signal

    @staticmethod
    def validate_signal(signal: Dict[str, Any]) -> bool:
        """
        验证信号的基本有效性
        
        Args:
            signal: 信号字典
            
        Returns:
            是否有效
        """
        required_fields = ['action', 'stock_code', 'price', 'reason']

        # 检查必需字段
        for field in required_fields:
            if field not in signal:
                return False

        # 检查action有效性
        if signal['action'] not in ['BUY', 'SELL']:
            return False

        # 检查价格有效性
        try:
            price = float(signal['price'])
            if price <= 0:
                return False
        except (ValueError, TypeError):
            return False

        # 检查信心度有效性
        if 'confidence' in signal:
            try:
                confidence = float(signal['confidence'])
                if not (0.0 <= confidence <= 1.0):
                    return False
            except (ValueError, TypeError):
                return False

        return True

    @staticmethod
    def format_signal_for_log(signal: Dict[str, Any]) -> str:
        """
        格式化信号用于日志输出
        
        Args:
            signal: 信号字典
            
        Returns:
            格式化的字符串
        """
        action = signal.get('action', 'UNKNOWN')
        stock_code = signal.get('stock_code', 'UNKNOWN')
        price = signal.get('price', 0)
        reason = signal.get('reason', '')
        confidence = signal.get('confidence', 0)

        base_info = f"{action} {stock_code} @ {price:.2f} (信心度: {confidence:.2f})"

        if reason:
            base_info += f" - {reason}"

        # 添加额外信息（排除基本字段）
        excluded_fields = {'action', 'stock_code', 'price', 'reason', 'confidence'}
        extra_info = []

        for key, value in signal.items():
            if key not in excluded_fields and value is not None:
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        extra_info.append(f"{key}: {value:.2f}")
                    else:
                        extra_info.append(f"{key}: {value}")
                else:
                    extra_info.append(f"{key}: {value}")

        if extra_info:
            base_info += f" [{', '.join(extra_info)}]"

        return base_info


class ParameterUtils:
    """
    参数处理工具类
    提供策略参数处理的通用方法
    """

    @staticmethod
    def parse_bounds(
            min_val: Optional[float],
            max_val: Optional[float],
            bounds_range: Optional[Union[List, tuple]],
            param_name: str = "参数"
    ) -> tuple[Optional[float], Optional[float]]:
        """
        解析参数区间的通用函数
        
        Args:
            min_val: 最小值
            max_val: 最大值  
            bounds_range: 区间 [min, max]，优先使用此参数
            param_name: 参数名称，用于日志
            
        Returns:
            (min_val, max_val) 元组，任一端为None表示不限制该端
        """
        try:
            if isinstance(bounds_range, (list, tuple)) and len(bounds_range) == 2:
                return bounds_range[0], bounds_range[1]
        except Exception:
            pass
        return min_val, max_val

    @staticmethod
    def merge_strategy_params(
            base_params: Dict[str, Any],
            strategy_specific_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合并基础参数和策略特定参数
        
        Args:
            base_params: 基础参数字典
            strategy_specific_params: 策略特定参数字典
            
        Returns:
            合并后的参数字典
        """
        merged = base_params.copy()
        merged.update(strategy_specific_params)
        return merged

    @staticmethod
    def validate_param_ranges(
            params: Dict[str, Any],
            param_ranges: Dict[str, Dict[str, float]]
    ) -> bool:
        """
        验证参数是否在指定范围内
        
        Args:
            params: 参数字典
            param_ranges: 参数范围字典，格式：{param_name: {'min': min_val, 'max': max_val}}
            
        Returns:
            是否所有参数都在范围内
        """
        for param_name, ranges in param_ranges.items():
            if param_name in params:
                value = params[param_name]
                min_val = ranges.get('min')
                max_val = ranges.get('max')

                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False

        return True


class PriceUtils:
    """
    价格处理工具类
    提供价格相关的通用计算方法
    """

    @staticmethod
    def calculate_percentage_distance(
            current_price: float,
            reference_price: float
    ) -> float:
        """
        计算价格间的百分比距离
        
        Args:
            current_price: 当前价格
            reference_price: 参考价格
            
        Returns:
            百分比距离 (current - reference) / reference * 100
        """
        if reference_price <= 0:
            return 0.0

        return (current_price - reference_price) / reference_price * 100

    @staticmethod
    def is_price_valid(price: Any) -> bool:
        """
        检查价格是否有效
        
        Args:
            price: 价格值
            
        Returns:
            是否有效
        """
        try:
            price_float = float(price)
            return price_float > 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def safe_float_conversion(
            value: Any,
            default: float = 0.0
    ) -> float:
        """
        安全的浮点数转换
        
        Args:
            value: 待转换的值
            default: 转换失败时的默认值
            
        Returns:
            转换后的浮点数
        """
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default


class DataUtils:
    """
    数据处理工具类
    提供通用的数据处理方法
    """

    @staticmethod
    def format_analysis_extras(
            analysis_result: Any,
            field_mapping: Dict[str, str],
            numeric_precision: int = 4
    ) -> Dict[str, Any]:
        """
        格式化分析结果的额外字段（通用版本）
        
        Args:
            analysis_result: 分析结果对象
            field_mapping: 字段映射 {目标字段名: 源字段名}
            numeric_precision: 数值精度
            
        Returns:
            格式化后的额外字段字典
        """
        extras = {}

        if analysis_result is None:
            return extras

        for target_field, source_field in field_mapping.items():
            try:
                value = getattr(analysis_result, source_field, None)
                if value is not None:
                    # 特殊处理某些字段类型
                    if hasattr(value, 'value'):  # 如枚举类型
                        extras[target_field] = value.value
                    elif isinstance(value, (int, float)):
                        if isinstance(value, float):
                            extras[target_field] = round(value, numeric_precision)
                        else:
                            extras[target_field] = value
                    else:
                        extras[target_field] = value
            except Exception:
                # 忽略字段提取错误，继续处理其他字段
                continue

        return extras

    @staticmethod
    def safe_calculate_distance(
            current_value: Any,
            reference_value: Any,
            fallback_value: float = 0.0
    ) -> float:
        """
        安全的距离计算（通用版本）
        
        Args:
            current_value: 当前值
            reference_value: 参考值
            fallback_value: 计算失败时的后备值
            
        Returns:
            计算的距离值
        """
        try:
            current = PriceUtils.safe_float_conversion(current_value)
            reference = PriceUtils.safe_float_conversion(reference_value)

            if reference <= 0:
                return fallback_value

            if current <= 0:
                return fallback_value

            return PriceUtils.calculate_percentage_distance(current, reference)

        except Exception:
            return fallback_value


# 保持向后兼容的工厂函数
def create_buy_signal(stock_code: str, price: float, reason: str,
                      confidence: float = 1.0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """创建买入信号的工厂函数"""
    return SignalUtils.create_buy_signal(stock_code, price, reason, confidence, extra)


def create_sell_signal(stock_code: str, price: float, reason: str,
                       confidence: float = 1.0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """创建卖出信号的工厂函数"""
    return SignalUtils.create_sell_signal(stock_code, price, reason, confidence, extra)
