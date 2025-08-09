#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重构后的上升通道策略
使用策略框架的所有组件，代码简洁且功能完整

策略逻辑：
1. 找出所有上升通道为NORMAL的股票
2. 按股价距离下沿的百分比距离排序（从小到大）
3. 平均买入前N只股票至满仓
4. 每天检查持仓股票状态，非NORMAL状态则卖出
5. 当未满N只股票时，重新选股并买入至N只
"""

from typing import Dict, Any, List

import pandas as pd

from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus
from .base import (
    BaseStrategy,
    ChannelAnalyzerManager
)


class RisingChannelStrategy(BaseStrategy):
    """
    重构后的上升通道策略
    
    继承自BaseStrategy，实现具体的上升通道交易逻辑
    使用组合模式整合各个管理器组件
    """

    # 策略参数
    params = (
        ('max_positions', 50),  # 最大持仓数量
        ('min_data_points', 60),  # 最小数据点数
        ('min_channel_score', 60.0),  # 最小通道评分
        ('enable_logging', True),  # 是否启用日志

        # 通道分析参数
        ('k', 2.0),  # 通道斜率参数
        ('L_max', 120),  # 最大回看天数
        ('delta_cut', 5),  # 切割参数
        ('pivot_m', 3),  # 枢轴参数
        ('gain_trigger', 0.30),  # 收益触发阈值
        ('beta_delta', 0.15),  # beta增量
        ('break_days', 3),  # 突破天数
        ('reanchor_fail_max', 2),  # 重锚定失败最大次数
        ('R2_min', 0.20),  # 最小R2值
        ('width_pct_min', 0.04),  # 最小宽度百分比
        ('width_pct_max', 0.20),  # 最大宽度百分比
    )

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame] = None):
        """
        初始化上升通道策略
        
        Args:
            stock_data_dict: 股票数据字典
        """
        # 调用父类初始化
        super().__init__(stock_data_dict)

        # 通道分析器管理器
        self.channel_manager = None

        # 当前分析结果缓存
        self.current_analysis_results = {}

    def on_delayed_init(self):
        """
        延迟初始化
        在backtrader环境完全初始化后调用
        """
        # 初始化通道分析器管理器
        channel_params = self._get_channel_params()
        self.channel_manager = ChannelAnalyzerManager(
            analyzer_type='real',  # 使用真实分析器
            **channel_params
        )

        self.logger.info("上升通道策略延迟初始化完成")

    def prepare_data(self):
        """
        数据准备阶段
        记录当前交易日的数据
        """
        # 获取当前主数据源信息
        current_data = {
            'open': self.data.open[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'close': self.data.close[0],
            'volume': getattr(self.data, 'volume', [0])[0]
        }

        # 获取当前股票代码
        main_stock_code = self._get_main_stock_code()

        # 记录数据到数据管理器
        self.data_manager.record_current_data(
            main_stock_code,
            self.current_date,
            current_data
        )

    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        生成交易信号
        
        Returns:
            交易信号列表
        """
        signals = []

        # 1. 检查当前持仓，生成卖出信号
        sell_signals = self._generate_sell_signals()
        signals.extend(sell_signals)

        # 2. 如果持仓不足，生成买入信号
        current_positions = self.position_manager.get_position_count()
        if current_positions < self.params.max_positions:
            buy_signals = self._generate_buy_signals()
            signals.extend(buy_signals)

        return signals

    def _generate_sell_signals(self) -> List[Dict[str, Any]]:
        """
        生成卖出信号
        检查持仓股票是否仍符合NORMAL通道状态
        
        Returns:
            卖出信号列表
        """
        sell_signals = []

        # 获取所有持仓股票
        held_stocks = self.position_manager.get_position_codes()

        for stock_code in held_stocks:
            # 获取股票历史数据
            stock_data = self.data_manager.get_stock_data_until(
                stock_code,
                self.current_date,
                self.params.min_data_points
            )

            if stock_data is None:
                # 数据不足，卖出
                signals = self._create_sell_signal(stock_code, "数据不足")
                if signals:
                    sell_signals.append(signals)
                continue

            # 分析通道状态
            channel_state = self.channel_manager.analyze_channel(
                stock_code, stock_data, self.current_date
            )

            # 如果不是NORMAL状态，卖出
            if (not channel_state or
                    not hasattr(channel_state, 'channel_status') or
                    channel_state.channel_status != ChannelStatus.NORMAL):

                reason = f"通道状态变化: {channel_state.channel_status.value if channel_state else 'None'}"
                current_price = self.data_manager.get_stock_price(stock_code, self.current_date)
                extras = {
                    '通道状态': getattr(getattr(channel_state, 'channel_status', None), 'value', None),
                    '斜率β': getattr(channel_state, 'beta', None),
                    'R²': getattr(channel_state, 'r2', None),
                    '今日中轴': getattr(channel_state, 'mid_today', None),
                    '今日上沿': getattr(channel_state, 'upper_today', None),
                    '今日下沿': getattr(channel_state, 'lower_today', None),
                    '距下沿(%)': self._calculate_distance_to_lower(current_price, channel_state) if current_price and channel_state else None,
                    '通道宽度': (getattr(channel_state, 'upper_today', None) - getattr(channel_state, 'lower_today', None)) if getattr(channel_state, 'upper_today', None) is not None and getattr(channel_state, 'lower_today', None) is not None else None,
                }
                signal = self._create_sell_signal(stock_code, reason, extra=extras)
                if signal:
                    sell_signals.append(signal)

        return sell_signals

    def _generate_buy_signals(self) -> List[Dict[str, Any]]:
        """
        生成买入信号
        选择符合条件的NORMAL通道股票，且要求股价位于通道内（>下沿 且 ≤上沿）
        
        Returns:
            买入信号列表
        """
        buy_signals = []

        # 批量分析所有股票的通道状态
        self._update_channel_analysis()

        # 筛选NORMAL状态的股票
        normal_stocks = self.channel_manager.filter_normal_channels(
            self.current_analysis_results,
            self.params.min_channel_score
        )

        if not normal_stocks:
            return buy_signals

        # 计算每只股票距离下沿的百分比距离，并排序
        normal_stocks_with_distance = []

        for stock_info in normal_stocks:
            stock_code = stock_info['stock_code']
            channel_state = stock_info['channel_state']

            # 获取当前价格
            current_price = self.data_manager.get_stock_price(stock_code, self.current_date)
            if current_price <= 0:
                continue

            # 要求价格在通道内：严格大于下沿，且不超过上沿
            lower_ok = hasattr(channel_state, 'lower_today') and current_price > getattr(channel_state, 'lower_today', 0)
            upper_ok = hasattr(channel_state, 'upper_today') and current_price <= getattr(channel_state, 'upper_today', float('inf'))
            if not (lower_ok and upper_ok):
                # 不在通道内，跳过
                continue

            # 计算距离下沿的百分比距离
            distance_to_lower = self._calculate_distance_to_lower(
                current_price, channel_state
            )

            normal_stocks_with_distance.append({
                'stock_code': stock_code,
                'current_price': current_price,
                'distance_to_lower': distance_to_lower,
                'channel_state': channel_state,
                'score': stock_info['score']
            })

        # 按距离下沿排序（从小到大）
        normal_stocks_with_distance.sort(key=lambda x: x['distance_to_lower'])

        # 计算需要买入的数量
        current_positions = self.position_manager.get_position_count()
        need_to_buy = min(
            self.params.max_positions - current_positions,
            len(normal_stocks_with_distance)
        )

        # 生成买入信号
        for i in range(need_to_buy):
            stock_info = normal_stocks_with_distance[i]
            stock_code = stock_info['stock_code']

            # 检查是否已经持仓
            if not self.position_manager.has_position(stock_code):
                chs = stock_info['channel_state']
                extras = {
                    '通道状态': getattr(getattr(chs, 'channel_status', None), 'value', None),
                    '通道评分': round(float(stock_info['score']), 2) if 'score' in stock_info and stock_info['score'] is not None else None,
                    '斜率β': getattr(chs, 'beta', None),
                    'R²': getattr(chs, 'r2', None),
                    '今日中轴': getattr(chs, 'mid_today', None),
                    '今日上沿': getattr(chs, 'upper_today', None),
                    '今日下沿': getattr(chs, 'lower_today', None),
                    '距下沿(%)': round(float(stock_info['distance_to_lower']), 2) if 'distance_to_lower' in stock_info and stock_info['distance_to_lower'] is not None else None,
                    '通道宽度': (getattr(chs, 'upper_today', None) - getattr(chs, 'lower_today', None)) if getattr(chs, 'upper_today', None) is not None and getattr(chs, 'lower_today', None) is not None else None,
                }
                signal = self._create_buy_signal(
                    stock_code,
                    stock_info['current_price'],
                    f"通道NORMAL，价格位于通道内，距离下沿{stock_info['distance_to_lower']:.2f}% ，评分{stock_info['score']:.1f}",
                    stock_info['score'] / 100.0,
                    extra=extras
                )
                buy_signals.append(signal)

        return buy_signals

    def _update_channel_analysis(self):
        """更新所有股票的通道分析结果"""
        # 获取所有股票数据
        stock_data_dict = {}

        for stock_code in self.data_manager.get_stock_codes_list():
            stock_data = self.data_manager.get_stock_data_until(
                stock_code,
                self.current_date,
                self.params.min_data_points
            )
            if stock_data is not None:
                stock_data_dict[stock_code] = stock_data

        # 批量分析
        self.current_analysis_results = self.channel_manager.batch_analyze(
            stock_data_dict,
            self.current_date
        )

    def _calculate_distance_to_lower(self, current_price: float, channel_state) -> float:
        """
        计算股价距离下沿的百分比距离
        
        Args:
            current_price: 当前价格
            channel_state: 通道状态对象
            
        Returns:
            距离下沿的百分比距离
        """
        distance_config = RisingChannelConfig.get_distance_config()

        # 无通道或缺少下沿字段
        if channel_state is None or not hasattr(channel_state, 'lower_today'):
            return distance_config['fallback_distance_no_state']

        # 取值并做None/类型保护
        try:
            lower_price_raw = getattr(channel_state, 'lower_today', None)
            lower_price = float(lower_price_raw) if lower_price_raw is not None else None
        except Exception:
            lower_price = None

        try:
            current = float(current_price) if current_price is not None else None
        except Exception:
            current = None

        # 下沿无效
        if lower_price is None or lower_price <= 0:
            return distance_config['fallback_distance_invalid']

        # 当前价格无效
        if current is None or current <= 0:
            return distance_config['min_distance_below_lower']

        # 计算距离
        distance = (current - lower_price) / lower_price * 100
        if distance < 0:
            distance = distance_config['min_distance_below_lower']
        return distance

    def _create_buy_signal(self, stock_code: str, price: float,
                           reason: str, confidence: float, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        创建买入信号
        
        Args:
            stock_code: 股票代码
            price: 价格
            reason: 买入原因
            confidence: 信心度
            extra: 额外字段（如通道信息）
            
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

    def _create_sell_signal(self, stock_code: str, reason: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        创建卖出信号
        
        Args:
            stock_code: 股票代码
            reason: 卖出原因
            extra: 额外字段（如通道信息）
            
        Returns:
            卖出信号字典，如果无法获取价格则返回None
        """
        price = self.data_manager.get_stock_price(stock_code, self.current_date)
        if price <= 0:
            self.logger.warning(f"无法获取股票 {stock_code} 的价格，跳过卖出")
            return None

        signal = {
            'action': 'SELL',
            'stock_code': stock_code,
            'price': price,
            'reason': reason,
            'confidence': 1.0  # 卖出信号通常是高信心度的
        }
        if extra:
            signal.update(extra)
        return signal

    def _get_main_stock_code(self) -> str:
        """获取主数据源股票代码"""
        # 从数据源名称获取股票代码
        data_name = getattr(self.data, '_name', 'data')
        if data_name != 'data':
            return data_name
        else:
            # 如果没有明确的数据名称，使用默认值
            return "main_stock"

    def _get_channel_params(self) -> Dict[str, Any]:
        """获取通道分析器参数"""
        return {
            "k": self.params.k,
            "L_max": self.params.L_max,
            "delta_cut": self.params.delta_cut,
            "pivot_m": self.params.pivot_m,
            "gain_trigger": self.params.gain_trigger,
            "beta_delta": self.params.beta_delta,
            "break_days": self.params.break_days,
            "reanchor_fail_max": self.params.reanchor_fail_max,
            "min_data_points": self.params.min_data_points,
            "R2_min": self.params.R2_min,
            "width_pct_min": self.params.width_pct_min,
            "width_pct_max": self.params.width_pct_max
        }

    def _get_parameters(self) -> Dict[str, Any]:
        """获取策略参数（覆盖父类方法）"""
        base_params = super()._get_parameters()

        # 添加上升通道特有参数
        channel_params = {
            'min_channel_score': self.params.min_channel_score,
            'k': self.params.k,
            'L_max': self.params.L_max,
            'delta_cut': self.params.delta_cut,
            'pivot_m': self.params.pivot_m,
            'gain_trigger': self.params.gain_trigger,
            'beta_delta': self.params.beta_delta,
            'break_days': self.params.break_days,
            'reanchor_fail_max': self.params.reanchor_fail_max,
            'R2_min': self.params.R2_min,
            'width_pct_min': self.params.width_pct_min,
            'width_pct_max': self.params.width_pct_max
        }

        base_params.update(channel_params)
        return base_params

    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息（覆盖父类方法）"""
        base_info = super().get_strategy_info()

        # 添加通道分析统计
        if self.channel_manager:
            base_info['channel_analysis_stats'] = self.channel_manager.get_analysis_statistics()

        # 添加当前分析结果统计
        if self.current_analysis_results:
            normal_count = len(self.channel_manager.filter_normal_channels(
                self.current_analysis_results,
                self.params.min_channel_score
            ))

            base_info['current_analysis'] = {
                'total_analyzed': len(self.current_analysis_results),
                'normal_channels': normal_count,
                'analysis_date': self.current_date
            }

        return base_info


# 为了保持与原有代码的兼容性，保留原来的类名
RisingChannelBacktestStrategy = RisingChannelStrategy


def create_rising_channel_strategy(max_positions: int = 50,
                                   min_channel_score: float = 60.0,
                                   **params) -> RisingChannelStrategy:
    """
    工厂函数：创建上升通道策略实例
    
    Args:
        max_positions: 最大持仓数量
        min_channel_score: 最小通道评分
        **params: 其他策略参数
        
    Returns:
        策略实例
    """
    # 构建参数字典
    strategy_params = {
        'max_positions': max_positions,
        'min_channel_score': min_channel_score,
        **params
    }

    # 创建策略实例
    strategy = RisingChannelStrategy()

    # 设置参数
    for param_name, param_value in strategy_params.items():
        if hasattr(strategy.params, param_name):
            setattr(strategy.params, param_name, param_value)

    return strategy
