#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通道分析专用工具模块
提供上升通道、回归分析等策略专用的工具函数
"""

from typing import Dict, Any, Tuple, List, Optional, Union

from ...core.utils import PriceUtils, DataUtils


class ChannelAnalysisUtils:
    """
    通道分析工具类
    专门为通道分析类策略（如上升通道、下降通道等）提供工具函数
    """

    @staticmethod
    def parse_r2_bounds(
            r2_min: Optional[float],
            r2_max: Optional[float],
            r2_range: Optional[Union[List, Tuple]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        解析R²区间参数的工具函数（专用于回归分析类策略）
        
        Args:
            r2_min: R²最小值
            r2_max: R²最大值  
            r2_range: R²区间 [min, max]，优先使用此参数
            
        Returns:
            (r2_min, r2_max) 元组，任一端为None表示不限制该端
        """
        try:
            if isinstance(r2_range, (list, tuple)) and len(r2_range) == 2:
                return r2_range[0], r2_range[1]
        except Exception:
            pass
        return r2_min, r2_max

    @staticmethod
    def calculate_distance_with_channel_fallback(
            current_value: Any,
            reference_value: Any,
            distance_config: Dict[str, float],
            config_key_prefix: str = 'fallback_distance'
    ) -> float:
        """
        带通道特定后备方案的距离计算
        
        Args:
            current_value: 当前值
            reference_value: 参考值
            distance_config: 距离计算配置（通道策略专用）
            config_key_prefix: 配置键前缀
            
        Returns:
            计算的距离值
        """
        # 转换为浮点数
        current = PriceUtils.safe_float_conversion(current_value)
        reference = PriceUtils.safe_float_conversion(reference_value)

        # 参考值无效的情况
        if reference <= 0:
            return distance_config.get(f'{config_key_prefix}_invalid', 10.0)

        # 当前值无效的情况  
        if current <= 0:
            return 0.1  # 固定后备值，不再依赖配置参数

        # 计算实际距离
        distance = PriceUtils.calculate_percentage_distance(current, reference)

        # 如果距离为负数，返回0（新策略只处理通道内股票）
        if distance < 0:
            return 0.0

        return distance

    @staticmethod
    def format_channel_analysis_extras(
            analysis_result: Any,
            field_mapping: Dict[str, str],
            include_channel_width: bool = True,
            price_precision: int = 2
    ) -> Dict[str, Any]:
        """
        格式化通道分析结果的额外字段
        
        Args:
            analysis_result: 通道分析结果对象
            field_mapping: 字段映射 {目标字段名: 源字段名}
            include_channel_width: 是否包含通道宽度计算
            price_precision: 价格精度
            
        Returns:
            格式化后的额外字段字典
        """
        extras = DataUtils.format_analysis_extras(analysis_result, field_mapping)

        # 通道特定的额外处理
        if include_channel_width and analysis_result:
            try:
                upper = getattr(analysis_result, 'upper_today', None)
                lower = getattr(analysis_result, 'lower_today', None)
                if upper is not None and lower is not None:
                    extras['通道宽度'] = round(float(upper - lower), price_precision)
            except Exception:
                pass

        return extras

    @staticmethod
    def validate_channel_state(channel_state, required_fields: List[str] = None) -> bool:
        """
        验证通道状态的有效性
        
        Args:
            channel_state: 通道状态对象
            required_fields: 必需字段列表
            
        Returns:
            是否有效
        """
        if channel_state is None:
            return False

        if required_fields is None:
            required_fields = ['channel_status', 'upper_today', 'lower_today']

        for field in required_fields:
            if not hasattr(channel_state, field):
                return False

        return True

    @staticmethod
    def is_price_in_channel(
            current_price: float,
            channel_state,
            strict_bounds: bool = True
    ) -> bool:
        """
        检查价格是否在通道内（通用版本）
        
        Args:
            current_price: 当前价格
            channel_state: 通道状态对象
            strict_bounds: 是否严格检查边界（True=严格大于下沿，False=大于等于）
            
        Returns:
            是否在通道内
        """
        if not ChannelAnalysisUtils.validate_channel_state(channel_state):
            return False

        try:
            lower = getattr(channel_state, 'lower_today', None)
            upper = getattr(channel_state, 'upper_today', None)

            if lower is None or upper is None:
                return False

            lower_val = float(lower)
            upper_val = float(upper)

            if strict_bounds:
                lower_ok = current_price > lower_val
            else:
                lower_ok = current_price >= lower_val

            upper_ok = current_price <= upper_val

            return lower_ok and upper_ok

        except (ValueError, TypeError, AttributeError):
            return False


class RegressionUtils:
    """
    回归分析工具类
    专门为基于回归分析的策略提供工具函数
    """

    @staticmethod
    def validate_regression_quality(
            r2: float,
            min_r2: Optional[float] = None,
            max_r2: Optional[float] = None
    ) -> bool:
        """
        验证回归质量
        
        Args:
            r2: R²值
            min_r2: 最小R²阈值
            max_r2: 最大R²阈值
            
        Returns:
            是否满足质量要求
        """
        try:
            r2_val = float(r2)

            if min_r2 is not None and r2_val < min_r2:
                return False

            if max_r2 is not None and r2_val > max_r2:
                return False

            return True

        except (ValueError, TypeError):
            return False

    @staticmethod
    def format_regression_info(
            analysis_result: Any,
            include_statistics: bool = True
    ) -> Dict[str, Any]:
        """
        格式化回归分析信息
        
        Args:
            analysis_result: 回归分析结果
            include_statistics: 是否包含统计信息
            
        Returns:
            格式化的回归信息
        """
        info = {}

        if analysis_result is None:
            return info

        # 基础回归信息
        regression_fields = {
            'R²': 'r2',
            '斜率β': 'beta',
            '截距α': 'alpha',
            '标准误差': 'std_error'
        }

        for display_name, field_name in regression_fields.items():
            try:
                value = getattr(analysis_result, field_name, None)
                if value is not None:
                    if isinstance(value, float):
                        info[display_name] = round(value, 4)
                    else:
                        info[display_name] = value
            except Exception:
                continue

        # 统计信息
        if include_statistics:
            stats_fields = {
                'P值': 'p_value',
                'F统计量': 'f_statistic',
                '样本数': 'sample_size'
            }

            for display_name, field_name in stats_fields.items():
                try:
                    value = getattr(analysis_result, field_name, None)
                    if value is not None:
                        info[display_name] = value
                except Exception:
                    continue

        return info


# 保持向后兼容的工厂函数
def parse_r2_bounds(r2_min: Optional[float], r2_max: Optional[float],
                    r2_range: Optional[Union[List, Tuple]]) -> Tuple[Optional[float], Optional[float]]:
    """解析R²区间的工厂函数（向后兼容）"""
    return ChannelAnalysisUtils.parse_r2_bounds(r2_min, r2_max, r2_range)
