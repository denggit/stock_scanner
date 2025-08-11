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

from typing import Dict, Any, List, Optional

import pandas as pd

from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus
from ...analyzers.channel import ChannelAnalyzerManager, ChannelAnalysisUtils, parse_r2_bounds
from ...core import BaseStrategy, SignalUtils, ParameterUtils


class RisingChannelStrategy(BaseStrategy):
    """
    上升通道策略 #1 - 基于技术分析的量化交易策略
    
    策略核心思想：
    通过识别股票价格在上升通道中的运行状态，在合适的时机买入和卖出股票，
    实现趋势跟踪和风险控制的平衡。
    
    策略逻辑流程：
    1. 数据准备阶段：记录当前交易日的股票数据
    2. 信号生成阶段：
       a) 检查当前持仓股票，如果通道状态不再是NORMAL则卖出
       b) 如果持仓数量不足max_positions，则寻找新的买入机会
    3. 风险控制阶段：验证交易信号的合理性
    4. 交易执行阶段：执行买入和卖出操作
    5. 日志记录阶段：记录交易详情和分析结果
    
    买入条件（同时满足）：
    1. 股票通道状态为NORMAL（正常上升通道）
    2. 通道评分 >= min_channel_score（默认60分）
    3. 当前价格位于通道内（>下沿 且 ≤上沿）
    4. 距离通道下沿的百分比距离 <= max_distance_from_lower（默认10%）
    5. 当前持仓数量 < max_positions
    6. R²值在有效范围内（R2_min <= R² <= R2_max）
    7. 通道宽度在合理范围内（width_pct_min <= 宽度 <= width_pct_max）
    
    卖出条件（满足任一）：
    1. 股票通道状态不再是NORMAL（通道失效）
    2. 股票数据不足（无法进行有效分析）
    3. 通道分析出现异常
    
    选股排序规则：
    1. 首先筛选出所有满足买入条件的股票
    2. 按距离通道下沿的百分比距离从小到大排序
    3. 优先买入距离下沿最近的股票（风险相对较小）
    
    仓位管理：
    1. 最大持仓数量由max_positions参数控制
    2. 平均分配资金到每只股票
    3. 动态调整：卖出后立即寻找新的买入机会
    
    技术指标参数：
    - k: 通道斜率参数，控制通道的陡峭程度
    - L_max: 最大回看天数，限制历史数据范围
    - delta_cut: 切割参数，用于通道重构
    - pivot_m: 枢轴参数，影响关键点的识别
    - gain_trigger: 收益触发阈值，控制通道重构时机
    - beta_delta: Beta变化阈值，检测通道斜率变化
    - break_days: 突破天数，确认通道突破的有效性
    - R2_min/R2_max: R²值范围，控制通道拟合质量
    - width_pct_min/width_pct_max: 通道宽度范围，控制通道的稳定性
    
    风险控制：
    1. 通道状态监控：实时跟踪通道有效性
    2. 价格位置控制：确保买入价格在通道内
    3. 距离限制：避免买入距离下沿过远的股票
    4. 评分过滤：只买入通道质量较高的股票
    5. 数据质量检查：确保分析数据的完整性
    
    适用场景：
    - 趋势明显的上升市场
    - 个股技术面分析
    - 中短期交易策略
    - 风险偏好适中的投资者
    
    注意事项：
    1. 策略对市场趋势敏感，震荡市可能表现不佳
    2. 需要足够的历史数据支持通道分析
    3. 参数调优对策略表现影响较大
    4. 建议在实盘前进行充分的回测验证
    
    继承自BaseStrategy，使用组合模式整合各个管理器组件，
    实现了策略逻辑与框架功能的解耦，便于维护和扩展。
    """

    # 策略参数 - 从配置文件动态获取
    @classmethod
    def get_default_params(cls):
        """从配置文件获取默认参数"""
        config_params = RisingChannelConfig.get_strategy_params()
        # 转换为backtrader需要的元组格式
        return tuple((key, value) for key, value in config_params.items())

    # 在类定义时获取默认参数
    _default_config_params = RisingChannelConfig.get_strategy_params()
    params = tuple((key, value) for key, value in _default_config_params.items())

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

        # 不再使用预加载数据，按需逐日读取
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

        # 不再进行预加载

        self.logger.info("上升通道策略延迟初始化完成")

    # 预加载入口已移除

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

        # 2. 需要买入时再生成买入信号
        current_positions = self.position_manager.get_position_count()
        # 计算执行卖出后的预期持仓数量（去重统计待卖股票）
        sell_codes = {s.get('stock_code') for s in sell_signals if s.get('stock_code')}
        projected_positions = max(0, current_positions - len(sell_codes))
        if projected_positions < self.params.max_positions:
            available_slots = self.params.max_positions - projected_positions
            buy_signals = self._generate_buy_signals(available_slots=available_slots)
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

    def _generate_buy_signals(self, available_slots: Optional[int] = None) -> List[Dict[str, Any]]:
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

        # 计算需要买入的数量（若传入 available_slots，则以其为准）
        if available_slots is None:
            current_positions = self.position_manager.get_position_count()
            need_to_buy = min(
                self.params.max_positions - current_positions,
                len(normal_stocks_with_distance)
            )
        else:
            need_to_buy = min(available_slots, len(normal_stocks_with_distance))

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
        # 仅在需要买入时才触发通道获取/计算，以减少无谓的数据库访问
        # 判断：卖出后的空位数 > 0，才进行当日通道获取
        current_positions = self.position_manager.get_position_count()

        # 避免重复生成卖出信号：仅获取用于数量估计的代码集合
        held_codes = set(self.position_manager.get_position_codes())
        sell_candidate_count = 0
        try:
            # 轻量级地基于持仓与通道状态检查是否存在潜在卖出，不生成完整信号结构
            for stock_code in held_codes:
                stock_data = self.data_manager.get_stock_data_until(
                    stock_code,
                    self.current_date,
                    self.params.min_data_points
                )
                if stock_data is None:
                    sell_candidate_count += 1
                    continue
                ch_state = None
                if self.channel_manager:
                    try:
                        ch_state = self.channel_manager.analyze_channel(stock_code, stock_data, self.current_date)
                    except Exception:
                        ch_state = None
                if self._should_sell_stock(ch_state):
                    sell_candidate_count += 1
        except Exception:
            # 回退到原有逻辑
            sell_signals = self._generate_sell_signals()
            sell_candidate_count = len({s.get('stock_code') for s in sell_signals if s.get('stock_code')})

        projected_positions = max(0, current_positions - sell_candidate_count)
        if projected_positions < self.params.max_positions:
            self._update_channel_analysis_from_db_or_compute()
        else:
            # 无需买入时，仅保留卖出判定用的 channel_state（可空）
            self.current_analysis_results = {}

    def _update_channel_analysis_from_db_or_compute(self):
        """按需从数据库获取当日通道，缺失则计算并写回，随后构建分析结果。"""
        self.current_analysis_results = {}
        if self.cache_adapter is None:
            # 无DB适配器则退回传统纯计算
            return self._update_channel_analysis_traditional()

        try:
            channel_params = self._extract_channel_params()
            stock_dict = {}
            codes = self.data_manager.get_stock_codes_list()
            for code in codes:
                df = self.data_manager.get_stock_data_until(code, self.current_date, self.params.min_data_points)
                if df is not None:
                    stock_dict[code] = df

            if not stock_dict:
                return

            # 从DB获取当日数据，缺失将自动计算并写回
            daily_map = self.cache_adapter.get_channels_for_date(
                stock_data_dict=stock_dict,
                params=channel_params,
                target_date=self.current_date.strftime('%Y-%m-%d'),
                min_window_size=self.params.min_data_points,
            )

            for stock_code, ch_df in (daily_map or {}).items():
                try:
                    if ch_df is None or ch_df.empty:
                        continue
                    last = ch_df.iloc[-1]
                    if pd.isna(last.get('beta')):
                        continue

                    channel_state = self._build_channel_state_from_cache(last)

                    stock_data = stock_dict.get(stock_code)
                    if stock_data is not None:
                        score = self.channel_manager.calculate_channel_score(
                            stock_code, channel_state, stock_data, self.current_date
                        )
                    else:
                        score = 60.0

                    self.current_analysis_results[stock_code] = {
                        'channel_state': channel_state,
                        'score': score,
                        'is_cached': True,
                    }
                except Exception as e:
                    self.logger.debug(f"构建股票 {stock_code} 通道状态失败: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"按需获取当日通道失败，退回纯计算: {e}")
            return self._update_channel_analysis_traditional()

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
        from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus

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

                # 通道状态：严格基于当日价格与上下沿即时判定（不做任何旧值兼容映射）
                try:
                    close_val = cache_data.get('close', None)
                    upper_val = self.upper_today
                    lower_val = self.lower_today
                    if close_val is not None and upper_val is not None and lower_val is not None:
                        if float(close_val) > float(upper_val):
                            self.channel_status = ChannelStatus.BREAKOUT
                        elif float(close_val) < float(lower_val):
                            self.channel_status = ChannelStatus.BREAKDOWN
                        else:
                            self.channel_status = ChannelStatus.NORMAL
                    else:
                        self.channel_status = ChannelStatus.NORMAL
                except Exception:
                    self.channel_status = ChannelStatus.NORMAL

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
            from backend.business.factor.core.engine.library.channel_analysis.rising_channel import \
                AscendingChannelRegression
            temp_analyzer = AscendingChannelRegression()
            return temp_analyzer._get_config_dict()

        except Exception as e:
            self.logger.warning(f"无法获取通道算法参数，使用默认值: {e}")

            # 回退到硬编码的算法参数（仅保留必要项）
            return {
                'k': 2.0,
                'L_max': 120,
                'delta_cut': 5,
                'pivot_m': 3,
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
