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

from typing import Dict, Any, List, Tuple

import pandas as pd

from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus
from ...core import BaseStrategy, SignalUtils, ParameterUtils, DataUtils
from ...analyzers.channel import ChannelAnalyzerManager, ChannelAnalysisUtils, parse_r2_bounds


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
        ('R2_max', None),  # 最大R2值（仅用于选股过滤；None 表示不设上限）
        ('R2_range', None),  # 参数优化时可传入 [R2_min, R2_max]，两者均可为 None
        ('width_pct_min', 0.04),  # 最小宽度百分比
        ('width_pct_max', 0.20),  # 最大宽度百分比
        ('max_distance_from_lower', 15.0),  # 买入时距离通道下沿的最大百分比距离（%）
    )

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame] = None, cache_adapter=None, **kwargs):
        """
        初始化上升通道策略
        
        Args:
            stock_data_dict: 股票数据字典
            cache_adapter: 缓存适配器实例
            **kwargs: backtrader 策略参数（通过params定义的参数）
        """
        # 调用父类初始化
        super().__init__(stock_data_dict, **kwargs)

        # 缓存适配器
        self.cache_adapter = cache_adapter

        # 通道分析器管理器
        self.channel_manager = None

        # 当前分析结果缓存
        self.current_analysis_results = {}
        
        # 预加载的通道历史数据
        self.preloaded_channel_data = {}

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

        # 预加载缓存通道数据（如果有缓存适配器）
        if self.cache_adapter is not None:
            self._preload_channel_data()

        self.logger.info("上升通道策略延迟初始化完成")

    def _preload_channel_data(self):
        """
        预加载通道历史数据到内存
        """
        try:
            self.logger.info("开始预加载通道历史数据...")
            
            channel_params = self._extract_channel_params()
            
            # 使用缓存适配器获取历史数据
            self.preloaded_channel_data = self.cache_adapter.get_channel_history_data(
                stock_data_dict=self.stock_data_dict,
                params=channel_params
            )
            
            self.logger.info(f"成功预加载 {len(self.preloaded_channel_data)} 只股票的通道历史数据")
            
        except Exception as e:
            self.logger.error(f"预加载通道历史数据失败: {e}")
            self.preloaded_channel_data = {}

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
            if self._should_sell_stock(channel_state):
                reason = f"通道状态变化: {channel_state.channel_status.value if channel_state else 'None'}"
                current_price = self.data_manager.get_stock_price(stock_code, self.current_date)
                extras = self._format_channel_extras(channel_state, current_price)
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
        r2_min, r2_max = self._get_effective_r2_bounds()
        normal_stocks = self.channel_manager.filter_normal_channels(
            self.current_analysis_results,
            self.params.min_channel_score,
            r2_min=r2_min,
            r2_max=r2_max
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

            # 检查价格是否在通道内
            if not self._is_price_in_channel(current_price, channel_state):
                continue

            # 计算距离下沿的百分比距离
            distance_to_lower = self._calculate_distance_to_lower(
                current_price, channel_state
            )

            # 检查距离是否超过最大允许值
            if distance_to_lower > self.params.max_distance_from_lower:
                if self.params.enable_logging:
                    self.logger.debug(f"股票 {stock_code} 距离下沿 {distance_to_lower:.2f}% "
                                      f"超过最大允许值 {self.params.max_distance_from_lower:.2f}%，跳过")
                continue

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
                extras = self._format_channel_extras(
                    chs, 
                    stock_info['current_price'],
                    score=stock_info.get('score'),
                    distance_to_lower=stock_info.get('distance_to_lower')
                )
                signal = self._create_buy_signal(
                    stock_code,
                    stock_info['current_price'],
                    f"通道NORMAL，价格位于通道内，距离下沿{stock_info['distance_to_lower']:.2f}%（≤{self.params.max_distance_from_lower:.1f}%），评分{stock_info['score']:.1f}",
                    stock_info['score'] / 100.0,
                    extra=extras
                )
                buy_signals.append(signal)

        return buy_signals

    def _update_channel_analysis(self):
        """更新所有股票的通道分析结果"""
        # 如果有预加载的通道数据，优先使用缓存
        if self.preloaded_channel_data:
            self._update_channel_analysis_from_cache()
        else:
            self._update_channel_analysis_traditional()

    def _update_channel_analysis_from_cache(self):
        """从预加载的缓存数据更新通道分析结果"""
        self.current_analysis_results = {}
        current_date_str = self.current_date.strftime('%Y-%m-%d')
        
        for stock_code in self.data_manager.get_stock_codes_list():
            try:
                # 从预加载数据中获取对应日期的通道状态
                if stock_code in self.preloaded_channel_data:
                    channel_df = self.preloaded_channel_data[stock_code]
                    channel_df['trade_date'] = pd.to_datetime(channel_df['trade_date'])
                    
                    # 找到当前日期或最近的通道数据
                    target_data = channel_df[channel_df['trade_date'] <= self.current_date]
                    if not target_data.empty:
                        latest_data = target_data.iloc[-1]
                        
                        # 构建通道状态对象
                        if not pd.isna(latest_data.get('beta')):
                            channel_state = self._build_channel_state_from_cache(latest_data)
                            
                            # 计算通道评分（使用缓存时不受min_data_points限制）
                            stock_data = self.data_manager.get_stock_data_until(
                                stock_code, self.current_date, min_data_points=1  # 使用缓存时只需要当前数据
                            )
                            if stock_data is not None:
                                score = self.channel_manager.calculate_channel_score(
                                    stock_code, channel_state, stock_data, self.current_date
                                )
                                
                                self.current_analysis_results[stock_code] = {
                                    'channel_state': channel_state,
                                    'score': score,
                                    'is_cached': True
                                }
                            else:
                                # 即使没有足够的股票数据，也可以使用缓存的通道状态
                                # 给一个基础评分，避免因数据不足而错失机会
                                base_score = 60.0  # 基础分数
                                self.current_analysis_results[stock_code] = {
                                    'channel_state': channel_state,
                                    'score': base_score,
                                    'is_cached': True
                                }
                        
            except Exception as e:
                self.logger.debug(f"从缓存获取股票 {stock_code} 通道数据失败: {e}")
                continue

    def _update_channel_analysis_traditional(self):
        """传统方式更新通道分析结果"""
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
        
        # 标记为非缓存数据
        for stock_code in self.current_analysis_results:
            self.current_analysis_results[stock_code]['is_cached'] = False

    def _build_channel_state_from_cache(self, cache_data: pd.Series):
        """从缓存数据构建通道状态对象"""
        from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelState, ChannelStatus
        
        # 创建一个简化的通道状态对象，只包含必要的字段
        class CachedChannelState:
            def __init__(self, cache_data):
                self.anchor_date = pd.to_datetime(cache_data.get('anchor_date'))
                self.anchor_price = cache_data.get('anchor_price')
                self.beta = cache_data.get('beta')
                self.sigma = cache_data.get('sigma')
                self.r2 = cache_data.get('r2')
                self.mid_today = cache_data.get('mid_today')
                self.upper_today = cache_data.get('upper_today')
                self.lower_today = cache_data.get('lower_today')
                self.mid_tomorrow = cache_data.get('mid_tomorrow')
                self.upper_tomorrow = cache_data.get('upper_tomorrow')
                self.lower_tomorrow = cache_data.get('lower_tomorrow')
                self.cumulative_gain = cache_data.get('cumulative_gain')
                
                # 通道状态
                status_value = cache_data.get('channel_status', 1)  # 默认为NORMAL
                if isinstance(status_value, str):
                    self.channel_status = ChannelStatus(int(status_value)) if status_value.isdigit() else ChannelStatus.NORMAL
                else:
                    self.channel_status = ChannelStatus(status_value) if status_value is not None else ChannelStatus.NORMAL
        
        return CachedChannelState(cache_data)

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

        # 获取下沿价格
        lower_price = getattr(channel_state, 'lower_today', None)
        
        # 使用通道专用工具计算距离
        return ChannelAnalysisUtils.calculate_distance_with_channel_fallback(
            current_price, 
            lower_price, 
            distance_config,
            'fallback_distance'
        )

    def _format_channel_extras(
        self, 
        channel_state, 
        current_price: float,
        score: float = None,
        distance_to_lower: float = None
    ) -> Dict[str, Any]:
        """
        格式化通道分析的额外信息
        
        Args:
            channel_state: 通道状态对象
            current_price: 当前价格
            score: 通道评分（可选）
            distance_to_lower: 距下沿距离（可选）
            
        Returns:
            格式化后的额外信息字典
        """
        # 基础字段映射
        field_mapping = {
            '通道状态': 'channel_status',
            '斜率β': 'beta',
            'R²': 'r2',
            '今日中轴': 'mid_today',
            '今日上沿': 'upper_today',
            '今日下沿': 'lower_today'
        }
        
        extras = ChannelAnalysisUtils.format_channel_analysis_extras(
            channel_state, field_mapping, include_channel_width=False
        )
        
        # 添加评分信息
        if score is not None:
            extras['通道评分'] = round(float(score), 2)
        
        # 添加距离信息
        if distance_to_lower is not None:
            extras['距下沿(%)'] = round(float(distance_to_lower), 2)
        elif current_price and channel_state:
            calculated_distance = self._calculate_distance_to_lower(current_price, channel_state)
            extras['距下沿(%)'] = round(calculated_distance, 2)
        
        # 计算通道宽度
        upper = getattr(channel_state, 'upper_today', None) if channel_state else None
        lower = getattr(channel_state, 'lower_today', None) if channel_state else None
        if upper is not None and lower is not None:
            extras['通道宽度'] = upper - lower
            
        return extras

    def _extract_channel_params(self) -> Dict[str, Any]:
        """
        提取通道算法参数（用于缓存键生成）
        
        只包含影响通道计算结果的算法参数，不包含策略层面的参数
        
        Returns:
            通道算法参数字典
        """
        try:
            # 从通道分析器管理器获取实际使用的算法参数
            if hasattr(self, 'channel_manager') and self.channel_manager is not None:
                analyzer = self.channel_manager.get_analyzer()
                if hasattr(analyzer, '_analyzer') and analyzer._analyzer is not None:
                    # 从真实的AscendingChannelRegression实例获取参数
                    return analyzer._analyzer._get_config_dict()
            
            # 如果无法从管理器获取，直接创建AscendingChannelRegression实例
            from backend.business.factor.core.engine.library.channel_analysis.rising_channel import AscendingChannelRegression
            temp_analyzer = AscendingChannelRegression()
            return temp_analyzer._get_config_dict()
            
        except Exception as e:
            self.logger.warning(f"无法获取通道算法参数，使用默认值: {e}")
            
            # 回退到硬编码的算法参数
            return {
                'k': 2.0,
                'L_max': 120,
                'delta_cut': 5,
                'pivot_m': 3,
                'gain_trigger': 0.30,
                'beta_delta': 0.15,
                'break_days': 3,
                'reanchor_fail_max': 2,
                'min_data_points': 60,
                'R2_min': 0.20,
                'width_pct_min': 0.04,
                'width_pct_max': 0.12
            }

    def _should_sell_stock(self, channel_state) -> bool:
        """
        判断是否应该卖出股票
        
        Args:
            channel_state: 通道状态对象
            
        Returns:
            是否应该卖出
        """
        return (not channel_state or
                not hasattr(channel_state, 'channel_status') or
                channel_state.channel_status != ChannelStatus.NORMAL)

    def _is_price_in_channel(self, current_price: float, channel_state) -> bool:
        """
        检查价格是否在通道内
        
        Args:
            current_price: 当前价格
            channel_state: 通道状态对象
            
        Returns:
            是否在通道内
        """
        return ChannelAnalysisUtils.is_price_in_channel(
            current_price, channel_state, strict_bounds=True
        )

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
        return SignalUtils.create_buy_signal(stock_code, price, reason, confidence, extra)

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

        return SignalUtils.create_sell_signal(
            stock_code, price, reason, 
            confidence=1.0,  # 卖出信号通常是高信心度的
            extra=extra
        )

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
        # 注意：若用户将 R2_min 设为 None，我们不传入该键，让底层分析器使用其默认值（避免回归有效性检查出错）
        params = {
            "k": self.params.k,
            "L_max": self.params.L_max,
            "delta_cut": self.params.delta_cut,
            "pivot_m": self.params.pivot_m,
            "gain_trigger": self.params.gain_trigger,
            "beta_delta": self.params.beta_delta,
            "break_days": self.params.break_days,
            "reanchor_fail_max": self.params.reanchor_fail_max,
            "min_data_points": self.params.min_data_points,
            "width_pct_min": self.params.width_pct_min,
            "width_pct_max": self.params.width_pct_max
        }
        if self.params.R2_min is not None:
            params["R2_min"] = self.params.R2_min
        return params

    def _get_parameters(self) -> Dict[str, Any]:
        """获取策略参数（覆盖父类方法）"""
        base_params = super()._get_parameters()

        # 添加上升通道特有参数
        channel_params = self._extract_channel_params()
        
        return ParameterUtils.merge_strategy_params(base_params, channel_params)

    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息（覆盖父类方法）"""
        base_info = super().get_strategy_info()

        # 添加通道分析统计
        if self.channel_manager:
            base_info['channel_analysis_stats'] = self.channel_manager.get_analysis_statistics()

        # 添加当前分析结果统计
        if self.current_analysis_results:
            r2_min, r2_max = self._get_effective_r2_bounds()
            normal_count = len(self.channel_manager.filter_normal_channels(
                self.current_analysis_results,
                self.params.min_channel_score,
                r2_min=r2_min,
                r2_max=r2_max
            ))

            base_info['current_analysis'] = {
                'total_analyzed': len(self.current_analysis_results),
                'normal_channels': normal_count,
                'analysis_date': self.current_date
            }

        return base_info

    def _get_effective_r2_bounds(self) -> tuple[float | None, float | None]:
        """
        解析当前有效的 R² 过滤区间。
        优先使用 `R2_range=[min, max]`，若未提供，则使用 `R2_min` 与 `R2_max`。
        两端为 None 表示不做该侧限制。
        """
        return parse_r2_bounds(
            getattr(self.params, 'R2_min', None),
            getattr(self.params, 'R2_max', None),
            getattr(self.params, 'R2_range', None)
        )


# R²区间解析函数已移至 signal_utils.py 中的 ParameterUtils.parse_r2_bounds


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
