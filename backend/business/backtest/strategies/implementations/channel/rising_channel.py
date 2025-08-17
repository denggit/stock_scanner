#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道策略实现
基于技术分析的量化交易策略
"""

from typing import Dict, Any, List, Optional

import pandas as pd

from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.backtest.core.trading_rules import AShareTradingRules
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
    
    买入策略（T日预筛选 + T+1日买入）：
    
    T日预筛选条件（同时满足）：
    1. 股票通道状态为NORMAL（正常上升通道）
    2. 通道评分 >= min_channel_score（默认60分）
    3. 当前价格位于通道内（>下沿 且 ≤上沿）
    4. R²值在有效范围内（R2_min <= R² <= R2_max）
    5. 通道宽度在合理范围内（width_pct_min <= 宽度 <= width_pct_max）
    6. 股价在中轴到下沿中间位置下方（几何位置判断）
    7. 股价距离通道下沿百分比距离 <= max_distance_to_lower（默认3%）
    
    根据股价距离通道下沿百分比距离从小到大排序，选出max_position只股票。
    
    T+1日买入条件（同时满足）：
    1. 依然是上升通道为NORMAL状态的股票
    2. 几何位置要求：股价在中轴到下沿中间位置的下方
    3. close > open的股票
    4. 当日涨幅 >= min_daily_gain（默认0.5%）
    5. 当日交易量大于过去五日平均交易量的1.2倍（可配置）
    6. 根据当日交易量对比过去五日平均交易量的增长比来从大到小排序
    
    依次买入挑选出来的股票直到满仓或无满足条件的股票
    
     卖出条件（满足任一）：
     1. 上升通道状态不为 NORMAL
     2. 当日最高价 high 大于通道上沿 upper_today
    
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

    # 策略参数定义 - 使用开发环境的默认值
    # 这些参数可以被 backtrader 优化引擎动态修改
    params = (
        # 策略基础参数
        ('max_positions', 20),  # 最大持仓数量（开发环境默认值）
        ('min_data_points', 60),  # 最小数据点数
        ('min_channel_score', 60.0),  # 最小通道评分（开发环境默认值）
        ('enable_logging', True),  # 是否启用日志

        # 数据预处理参数
        ('adjust', 1),  # 复权类型：1-后复权，2-前复权，3-不复权

        # 卖出规则参数
        ('sell_on_close_breakout', True),  # 是否使用收盘价突破通道上沿作为卖出条件
        ('breakout_pullback_threshold', 3.0),  # BREAKOUT状态回撤阈值（%）

        # 通道分析参数
        ('k', 2.0),  # 通道斜率参数
        ('L_max', 120),  # 最大回看天数
        ('delta_cut', 5),  # 切割参数
        ('pivot_m', 3),  # 枢轴参数
        ('R2_min', 0.35),  # 最小R²值（用于通道有效性判定）
        ('R2_max', 0.95),  # 最大R²值上限（仅用于选股过滤）
        ('R2_range', None),  # 参数优化时可传入 [R2_min, R2_max]，两者均可为 None
        ('width_pct_min', 0.05),  # 最小通道宽度
        ('width_pct_max', 0.12),  # 最大通道宽度
        
        # 成交量过滤参数
        ('min_volume_ratio', 1.2),  # 最小成交量比（当日成交量/5日平均成交量）
        
        # T日预筛选参数
        ('max_distance_to_lower', 3.0),  # T日预筛选时股价距离通道下沿的最大百分比距离
        
        # T+1日买入参数
        ('min_daily_gain', 0.5),  # T+1日买入时的最小当日涨幅要求（%）
    )

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame] = None, cache_adapter=None, **kwargs):
        """
        初始化上升通道策略
        
        Args:
            stock_data_dict: 股票数据字典
            cache_adapter: 缓存适配器实例
            **kwargs: backtrader 策略参数（通过params定义的参数）
        """
        # 获取配置文件中的策略参数
        from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
        config = RisingChannelConfig()
        config_params = config.get_strategy_params()
        
        # 合并配置参数和传入的kwargs，kwargs优先级更高
        merged_params = config_params.copy()
        merged_params.update(kwargs)
        
        # 调用父类初始化，使用合并后的参数
        super().__init__(stock_data_dict, **merged_params)

        # 缓存适配器
        self.cache_adapter = cache_adapter

        # 通道分析器管理器
        self.channel_manager = None

        # 当前分析结果缓存
        self.current_analysis_results = {}

        # 不再使用预加载数据，按需逐日读取
        self.preloaded_channel_data = {}

        # 记录个股是否曾进入过 BREAKOUT 状态，用于“回归 NORMAL 后卖出”的规则
        self._breakout_flag: dict[str, bool] = {}
        
        # 记录BREAKOUT状态的最高价，用于回撤策略
        self._breakout_high_prices: dict[str, float] = {}
        
        # T日预筛选的股票池，用于T+1日买入
        self._preselected_stocks: List[str] = []

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
        
        策略逻辑：
        1. T日：完成当日买卖后，如果不满仓，进行预筛选
        2. T+1日：从预筛选股票中买入
        
        Returns:
            交易信号列表
        """
        signals = []

        # 1. 检查当前持仓，生成卖出信号
        sell_signals = self._generate_sell_signals()
        signals.extend(sell_signals)

        # 2. 计算执行卖出后的预期持仓数量
        current_positions = self.position_manager.get_position_count()
        sell_codes = {s.get('stock_code') for s in sell_signals if s.get('stock_code')}
        projected_positions = max(0, current_positions - len(sell_codes))
        
        # 3. 如果不满仓，进行T日预筛选
        if projected_positions < self.params.max_positions:
            self._preselect_stocks_t_day()
            
            # 4. 生成买入信号（T+1日买入）
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

            # 若符合卖出条件，生成卖出信号
            if self._should_sell_stock(stock_code, channel_state, stock_data):
                # 获取具体的卖出原因
                sell_reason = self._get_sell_reason(stock_code, channel_state, stock_data)
                current_price = self.data_manager.get_stock_price(stock_code, self.current_date)
                extras = self._format_channel_extras(channel_state, current_price)
                # 在extras中添加卖出原因
                extras['卖出原因'] = sell_reason
                signal = self._create_sell_signal(stock_code, sell_reason, extra=extras)
                if signal:
                    sell_signals.append(signal)

        return sell_signals

    def _preselect_stocks_t_day(self) -> None:
        """
        T日预筛选股票 - 根据通道状态和价格位置筛选候选股票
        
        筛选条件：
        1. 通道状态要求：股票通道状态为NORMAL（正常上升通道）
        2. 通道评分要求：通道评分 >= min_channel_score（默认60分）
        3. 价格位置要求：当前价格位于通道内（>下沿 且 ≤上沿）
        4. R²质量要求：R²值在有效范围内（R2_min <= R² <= R2_max）
        5. 通道宽度要求：通道宽度在合理范围内（width_pct_min <= 宽度 <= width_pct_max）
        6. 几何位置要求：股价在中轴到下沿中间位置的下方
        
        根据股价距离通道下沿百分比距离从小到大排序，选出max_position只股票
        """
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
            self._preselected_stocks = []
            if self.params.enable_logging:
                self.logger.info("T日预筛选：没有找到符合条件的股票")
            return

        # 记录符合条件的股票
        qualified_stocks = []

        for stock_info in normal_stocks:
            stock_code = stock_info['stock_code']
            channel_state = stock_info['channel_state']

            # 校验通道宽度有效性
            if not self._is_channel_width_valid(channel_state):
                if self.params.enable_logging:
                    self.logger.debug(
                        f"T日预筛选 - 股票 {stock_code} 通道宽度超出范围，跳过"
                    )
                continue

            # 获取当前价格
            current_price = self.data_manager.get_stock_price(stock_code, self.current_date)
            if current_price <= 0:
                continue

            # 检查价格是否在通道内
            if not self._is_price_in_channel(current_price, channel_state):
                continue

            # 计算通道几何结构
            mid_price = getattr(channel_state, 'mid_today', None)
            lower_price = getattr(channel_state, 'lower_today', None)

            if mid_price is None or lower_price is None:
                continue

            # 计算中轴到下沿的中间位置
            mid_to_lower_midpoint = (mid_price + lower_price) / 2.0

            # 检查股价是否在中轴到下沿中间位置的下方
            if current_price > mid_to_lower_midpoint:
                if self.params.enable_logging:
                    self.logger.debug(
                        f"T日预筛选 - 股票 {stock_code} 股价在中轴到下沿中间位置上方，跳过"
                    )
                continue

            # 计算距离下沿的百分比距离
            distance_to_lower = self._calculate_distance_to_lower(
                current_price, channel_state
            )

            # 检查股价距离通道下沿是否在允许范围内
            if distance_to_lower > self.params.max_distance_to_lower:
                if self.params.enable_logging:
                    self.logger.debug(
                        f"T日预筛选 - 股票 {stock_code} 距离下沿 {distance_to_lower:.2f}% > {self.params.max_distance_to_lower}%，跳过"
                    )
                continue

            # 记录符合条件的股票
            qualified_stocks.append({
                'stock_code': stock_code,
                'current_price': current_price,
                'distance_to_lower': distance_to_lower,
                'channel_state': channel_state,
                'score': stock_info['score']
            })

        # 按距离下沿百分比从小到大排序，选出max_position只股票
        qualified_stocks.sort(key=lambda x: x['distance_to_lower'])
        self._preselected_stocks = [stock['stock_code'] for stock in qualified_stocks[:self.params.max_positions]]

        if self.params.enable_logging:
            self.logger.info(f"T日预筛选完成：从{len(qualified_stocks)}只股票中选出{len(self._preselected_stocks)}只候选股票")
            for i, stock_code in enumerate(self._preselected_stocks[:5]):  # 只显示前5只
                stock_info = next(s for s in qualified_stocks if s['stock_code'] == stock_code)
                self.logger.info(f"  {i + 1}. {stock_code}: 距离下沿 {stock_info['distance_to_lower']:.2f}%, 评分 {stock_info['score']:.1f}")

    def _generate_buy_signals(self, available_slots: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        生成买入信号 - T+1日从预筛选股票中买入
        
        T+1日筛选条件：
        1. 依然是上升通道为NORMAL状态的股票
        2. 几何位置要求：股价在中轴到下沿中间位置的下方
        3. close > open的股票
        4. 当日交易量大于过去五日平均交易量
        5. 根据当日交易量对比过去五日平均交易量的增长比来从大到小排序
        
        依次买入挑选出来的股票直到满仓或无满足条件的股票
        
        Args:
            available_slots: 可用买入槽位数
            
        Returns:
            买入信号列表
        """
        buy_signals = []

        # 检查是否有预筛选的股票
        if not self._preselected_stocks:
            if self.params.enable_logging:
                self.logger.info("T+1日买入：没有预筛选的股票，跳过买入")
            return buy_signals

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
            if self.params.enable_logging:
                self.logger.info("T+1日买入：没有NORMAL状态的股票，跳过买入")
            return buy_signals

        # 从预筛选股票中筛选符合条件的股票
        qualified_stocks = []

        for stock_code in self._preselected_stocks:
            # 检查是否在NORMAL状态股票中
            stock_info = next((s for s in normal_stocks if s['stock_code'] == stock_code), None)
            if not stock_info:
                if self.params.enable_logging:
                    self.logger.debug(f"T+1日买入 - 股票 {stock_code} 不再是NORMAL状态，跳过")
                continue

            channel_state = stock_info['channel_state']

            # 获取当前价格
            current_price = self.data_manager.get_stock_price(stock_code, self.current_date)
            if current_price <= 0:
                continue

            # 检查收盘价 > 开盘价
            current_open = self.data_manager.get_stock_open_price(stock_code, self.current_date)
            if current_open <= 0:
                if self.params.enable_logging:
                    self.logger.debug(f"T+1日买入 - 股票 {stock_code} 开盘价无效，跳过")
                continue
            
            if current_price <= current_open:
                if self.params.enable_logging:
                    self.logger.debug(f"T+1日买入 - 股票 {stock_code} 收盘价 {current_price:.2f} <= 开盘价 {current_open:.2f}，跳过")
                continue

            # 计算当日涨幅
            daily_gain = ((current_price - current_open) / current_open) * 100
            
            # 检查当日涨幅是否满足要求
            if daily_gain < self.params.min_daily_gain:
                if self.params.enable_logging:
                    self.logger.debug(f"T+1日买入 - 股票 {stock_code} 当日涨幅 {daily_gain:.2f}% < {self.params.min_daily_gain}%，跳过")
                continue

            # 检查几何位置要求：股价在中轴到下沿中间位置的下方
            mid_price = getattr(channel_state, 'mid_today', None)
            lower_price = getattr(channel_state, 'lower_today', None)

            if mid_price is None or lower_price is None:
                continue

            # 计算中轴到下沿的中间位置
            mid_to_lower_midpoint = (mid_price + lower_price) / 2.0

            # 检查股价是否在中轴到下沿中间位置的下方
            if current_price > mid_to_lower_midpoint:
                if self.params.enable_logging:
                    self.logger.debug(
                        f"T+1日买入 - 股票 {stock_code} 股价在中轴到下沿中间位置上方，跳过"
                    )
                continue

            # 检查成交量要求：当日交易量大于过去五日平均交易量
            current_volume = self.data_manager.get_stock_volume(stock_code, self.current_date)
            avg_volume = self.data_manager.get_stock_avg_volume(stock_code, self.current_date, days=5)
            
            if current_volume <= 0 or avg_volume <= 0:
                if self.params.enable_logging:
                    self.logger.debug(f"T+1日买入 - 股票 {stock_code} 成交量数据无效，跳过")
                continue
            
            # 检查成交量比要求：当日交易量大于过去五日平均交易量的min_volume_ratio倍
            volume_ratio = current_volume / avg_volume
            if volume_ratio < self.params.min_volume_ratio:
                if self.params.enable_logging:
                    self.logger.debug(f"T+1日买入 - 股票 {stock_code} 成交量比 {volume_ratio:.2f} < {self.params.min_volume_ratio}，跳过")
                continue

            # 计算成交量增长比
            volume_ratio = current_volume / avg_volume

            # 记录符合条件的股票
            qualified_stocks.append({
                'stock_code': stock_code,
                'current_price': current_price,
                'channel_state': channel_state,
                'score': stock_info['score'],
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'daily_gain': daily_gain
            })

        if not qualified_stocks:
            if self.params.enable_logging:
                self.logger.info("T+1日买入：没有满足所有条件的股票，跳过买入")
            return buy_signals

        # 根据成交量增长比从大到小排序
        qualified_stocks.sort(key=lambda x: x['volume_ratio'], reverse=True)

        if self.params.enable_logging:
            self.logger.info(f"T+1日买入：找到 {len(qualified_stocks)} 只满足条件的股票，按成交量增长比排序:")
            for i, stock in enumerate(qualified_stocks[:5]):  # 只显示前5只
                self.logger.info(f"  {i + 1}. {stock['stock_code']}: 成交量比 {stock['volume_ratio']:.2f}, "
                                 f"当日成交量 {stock['current_volume']:.0f}, 5日平均 {stock['avg_volume']:.0f}, "
                                 f"当日涨幅 {stock['daily_gain']:.2f}%")

        # 计算需要买入的数量
        if available_slots is None:
            current_positions = self.position_manager.get_position_count()
            need_to_buy = min(
                self.params.max_positions - current_positions,
                len(qualified_stocks)
            )
        else:
            need_to_buy = min(available_slots, len(qualified_stocks))

        # 生成买入信号
        for i in range(need_to_buy):
            stock_info = qualified_stocks[i]
            stock_code = stock_info['stock_code']

            # 检查是否已经持仓
            if not self.position_manager.has_position(stock_code):
                chs = stock_info['channel_state']
                extras = self._format_channel_extras(
                    chs,
                    stock_info['current_price'],
                    score=stock_info.get('score')
                )

                # 添加成交量信息
                extras.update({
                    'current_volume': stock_info['current_volume'],
                    'avg_volume': stock_info['avg_volume'],
                    'volume_ratio': stock_info['volume_ratio'],
                    'daily_gain': stock_info['daily_gain'],
                    # 添加中文字段名，确保与报告生成兼容
                    '当日成交量': stock_info['current_volume'],
                    '5日平均成交量': stock_info['avg_volume'],
                    '成交量比': stock_info['volume_ratio'],
                    '当日涨幅': stock_info['daily_gain']
                })

                # 计算买入数量
                per_trade_cash = self.get_per_trade_cash(self.params.max_positions)
                current_price = stock_info['current_price']

                # 计算理论买入数量
                theoretical_size = int(per_trade_cash / current_price)

                # 使用A股交易规则调整数量为100的整数倍
                adjusted_size = AShareTradingRules.adjust_trade_quantity(theoretical_size)

                # 如果调整后数量为0，跳过这只股票
                if adjusted_size <= 0:
                    if self.params.enable_logging:
                        self.logger.debug(f"T+1日买入 - 股票 {stock_code} 调整后买入数量为0，跳过")
                    continue

                signal = self._create_buy_signal(
                    stock_code,
                    stock_info['current_price'],
                    f"T+1日买入: 成交量比{stock_info['volume_ratio']:.2f}, "
                    f"当日成交量{stock_info['current_volume']:.0f}, 5日平均{stock_info['avg_volume']:.0f}, "
                    f"当日涨幅{stock_info['daily_gain']:.2f}%, 评分{stock_info['score']:.1f}",
                    stock_info['score'] / 100.0,
                    extra=extras
                )

                # 将计算好的买入数量添加到信号中
                signal['size'] = adjusted_size

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
                if self._should_sell_stock(stock_code, ch_state, stock_data):
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
        """
        传统方式更新通道分析结果（优化版本）
        
        修改说明：
        - 数据净化：在数据进入预筛选模块之前，先执行.dropna()操作
        - 数据验证：检查清理后的数据长度是否满足min_data_points要求
        - 防止污染：确保预筛选和通道分析接收到的都是不含NaN值的干净数据
        """
        # 获取所有股票数据
        stock_data_dict = {}

        for stock_code in self.data_manager.get_stock_codes_list():
            stock_data = self.data_manager.get_stock_data_until(
                stock_code,
                self.current_date,
                self.params.min_data_points
            )
            if stock_data is not None:
                # 数据净化：移除NaN值，确保数据干净
                cleaned_data = stock_data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                
                # 检查清理后的数据长度是否满足要求
                if len(cleaned_data) >= self.params.min_data_points:
                    stock_data_dict[stock_code] = cleaned_data
                else:
                    self.logger.debug(f"股票 {stock_code} 清理后数据不足: {len(cleaned_data)} < {self.params.min_data_points}")

        if not stock_data_dict:
            self.logger.warning("没有找到满足条件的股票数据")
            self.current_analysis_results = {}
            return

        # 性能优化：预筛选股票，减少通道分析的计算量
        from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
        prefilter_config = RisingChannelConfig.get_prefilter_config()

        if (prefilter_config['enable_prefilter'] and
                len(stock_data_dict) > prefilter_config['min_stocks_for_prefilter']):
            self.logger.info(f"开始预筛选股票，原始数量: {len(stock_data_dict)}")

            # 导入预筛选工具
            from backend.business.backtest.utils.data_utils import DataUtils

            # 预筛选参数
            prefilter_params = {
                'min_data_points': self.params.min_data_points,
                'ma_period': prefilter_config['ma_period'],
                'lookback_days': prefilter_config['lookback_days'],
                'volume_threshold': prefilter_config['volume_threshold'],
                'min_conditions_met': prefilter_config['min_conditions_met'],
                'enable_volume_check': prefilter_config['enable_volume_check']
            }

            # 执行预筛选（现在接收到的都是干净的数据）
            filtered_stock_codes = DataUtils.prefilter_stocks(
                stock_data_dict, **prefilter_params
            )

            # 构建筛选后的股票数据字典
            filtered_stock_data_dict = {
                code: stock_data_dict[code]
                for code in filtered_stock_codes
                if code in stock_data_dict
            }

            self.logger.info(f"预筛选完成，筛选后数量: {len(filtered_stock_data_dict)} "
                             f"(筛选率: {len(filtered_stock_data_dict) / len(stock_data_dict) * 100:.1f}%)")

            # 使用筛选后的数据进行通道分析
            stock_data_dict = filtered_stock_data_dict

        # 批量分析（现在接收到的都是干净的数据）
        self.current_analysis_results = self.channel_manager.batch_analyze(
            stock_data_dict,
            self.current_date
        )

        # 标记为非缓存数据
        for stock_code in self.current_analysis_results:
            self.current_analysis_results[stock_code]['is_cached'] = False

    def _build_channel_state_from_cache(self, cache_data: pd.Series):
        """从缓存数据构建通道状态对象 - 完全使用数据库字段，不重新计算"""
        from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus

        # 创建一个简化的通道状态对象，直接使用数据库中的所有字段
        class CachedChannelState:
            def __init__(self, cache_data):
                # 直接使用数据库中的所有字段，不重新计算
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

                # 其他数据库字段
                self.break_cnt_up = cache_data.get('break_cnt_up', 0)
                self.break_cnt_down = cache_data.get('break_cnt_down', 0)
                self.reanchor_fail_up = cache_data.get('reanchor_fail_up', 0)
                self.reanchor_fail_down = cache_data.get('reanchor_fail_down', 0)
                self.window_size = cache_data.get('window_size', 0)
                self.days_since_anchor = cache_data.get('days_since_anchor', 0)
                self.break_reason = cache_data.get('break_reason', None)
                self.width_pct = cache_data.get('width_pct', None)
                self.slope_deg = cache_data.get('slope_deg', None)
                self.volatility = cache_data.get('volatility', None)

                # 通道状态：直接使用数据库中的状态，不重新计算
                try:
                    db_status = cache_data.get('channel_status', None)
                    if db_status is not None:
                        # 数据库中有状态，直接使用
                        self.channel_status = ChannelStatus(db_status)
                    else:
                        # 数据库中没有状态，使用默认值
                        self.channel_status = ChannelStatus.OTHER
                except Exception:
                    # 异常情况下使用默认值
                    self.channel_status = ChannelStatus.OTHER

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

    def _is_channel_width_valid(self, channel_state) -> bool:
        """
        通道宽度有效性校验 - 直接使用数据库中的width_pct字段

        优先使用数据库中的width_pct字段，避免重新计算，
        要求 width_pct_min <= 宽度 <= width_pct_max。

        Args:
            channel_state: 通道状态对象
        Returns:
            bool: 宽度是否在有效范围
        """
        try:
            if channel_state is None:
                return False

            # 优先使用数据库中的width_pct字段
            width_pct = getattr(channel_state, 'width_pct', None)
            if width_pct is not None:
                # 数据库中有宽度数据，直接使用
                return (width_pct >= float(self.params.width_pct_min)) and (
                            width_pct <= float(self.params.width_pct_max))

            # 数据库中没有宽度数据，回退到重新计算
            upper = getattr(channel_state, 'upper_today', None)
            lower = getattr(channel_state, 'lower_today', None)
            mid = getattr(channel_state, 'mid_today', None)
            if upper is None or lower is None or mid is None or float(mid) <= 0:
                return False
            width_pct = (float(upper) - float(lower)) / float(mid)
            return (width_pct >= float(self.params.width_pct_min)) and (width_pct <= float(self.params.width_pct_max))
        except Exception:
            return False

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

        # 直接使用数据库中的通道宽度字段
        width_pct = getattr(channel_state, 'width_pct', None) if channel_state else None
        if width_pct is not None:
            extras['通道宽度(%)'] = round(float(width_pct) * 100, 2)
        else:
            # 回退到重新计算
            upper = getattr(channel_state, 'upper_today', None) if channel_state else None
            lower = getattr(channel_state, 'lower_today', None) if channel_state else None
            if upper is not None and lower is not None:
                extras['通道宽度'] = upper - lower

        return extras

    def _extract_channel_params(self) -> Dict[str, Any]:
        """
        提取通道算法参数（用于缓存键生成）
        
        只包含影响通道计算结果的算法参数，不包含策略层面的参数
        包含adjust参数，因为不同复权类型会产生不同的通道计算结果
        
        Returns:
            通道算法参数字典
        """
        try:
            # 从通道分析器管理器获取实际使用的算法参数
            if hasattr(self, 'channel_manager') and self.channel_manager is not None:
                analyzer = self.channel_manager.get_analyzer()
                if hasattr(analyzer, '_analyzer') and analyzer._analyzer is not None:
                    # 从真实的AscendingChannelRegression实例获取参数
                    params = analyzer._analyzer._get_config_dict()
                    # 确保包含adjust参数（用于缓存键生成）
                    params['adjust'] = getattr(self.params, 'adjust', 1)
                    return params

            # 如果无法从管理器获取，直接创建AscendingChannelRegression实例
            from backend.business.factor.core.engine.library.channel_analysis.rising_channel import \
                AscendingChannelRegression
            temp_analyzer = AscendingChannelRegression()
            params = temp_analyzer._get_config_dict()
            # 确保包含adjust参数（用于缓存键生成）
            params['adjust'] = getattr(self.params, 'adjust', 1)
            return params

        except Exception as e:
            self.logger.warning(f"无法获取通道算法参数，使用默认值: {e}")

            # 回退到硬编码的算法参数（仅保留必要项）
            return {
                'k': 2.0,
                'L_max': 120,
                'delta_cut': 5,
                'pivot_m': 3,
                'min_data_points': 60,
                'R2_min': 0.6,
                'width_pct_min': 0.04,
                'width_pct_max': 0.12,
                'adjust': getattr(self.params, 'adjust', 1)  # 添加adjust参数（用于缓存键生成）
            }

    def _get_sell_reason(self, stock_code: str, channel_state, stock_data: Optional[pd.DataFrame] = None) -> str:
        """
        获取卖出原因
        
        Args:
            stock_code: 股票代码
            channel_state: 通道状态对象
            stock_data: 股票数据
            
        Returns:
            卖出原因字符串
        """
        try:
            # 1. 检查通道状态
            if channel_state is None:
                return "通道状态为空"
            
            if not hasattr(channel_state, 'channel_status'):
                return "通道状态字段缺失"
            
            current_status = channel_state.channel_status
            
            # 2. BREAKOUT状态特殊处理
            if current_status == ChannelStatus.BREAKOUT:
                sell_on_close = getattr(self.params, 'sell_on_close_breakout', True)
                
                if sell_on_close and stock_code in self._breakout_high_prices:
                    # 获取当前价格
                    current_price = None
                    if stock_data is not None and len(stock_data) >= 1:
                        try:
                            current_price = float(stock_data.iloc[-1]['close'])
                        except Exception:
                            current_price = None
                    
                    if current_price is not None:
                        high_price = self._breakout_high_prices[stock_code]
                        pullback_threshold = getattr(self.params, 'breakout_pullback_threshold', 3.0)
                        
                        if high_price > 0:
                            pullback_pct = ((high_price - current_price) / high_price) * 100
                            
                            # 如果回撤超过阈值，返回回撤卖出原因
                            if pullback_pct >= pullback_threshold:
                                return f"BREAKOUT回撤卖出: 最高价{high_price:.2f}, 当前价{current_price:.2f}, 回撤{pullback_pct:.2f}%"
                
                # 如果不是回撤卖出，返回一般BREAKOUT原因
                return f"通道状态异常: {current_status.value}"
            
            # 3. 其他非NORMAL状态
            if current_status != ChannelStatus.NORMAL:
                return f"通道状态异常: {current_status.value}"
            
            # 4. NORMAL状态：检查价格突破通道上沿
            sell_on_close = getattr(self.params, 'sell_on_close_breakout', True)
            
            # 获取当日价格
            day_price = None
            if stock_data is not None and len(stock_data) >= 1:
                try:
                    if sell_on_close:
                        day_price = float(stock_data.iloc[-1]['close'])
                    else:
                        day_price = float(stock_data.iloc[-1]['high'])
                except Exception:
                    day_price = None
            
            # 获取通道上沿
            upper_today = getattr(channel_state, 'upper_today', None)
            if day_price is not None and upper_today is not None:
                try:
                    if float(day_price) > float(upper_today):
                        price_type = "收盘价" if sell_on_close else "最高价"
                        return f"{price_type}突破通道上沿: {day_price:.2f} > {upper_today:.2f}"
                except Exception:
                    pass
            
            # 5. 其他情况
            return "其他卖出条件"
            
        except Exception as e:
            return f"卖出原因分析异常: {str(e)}"

    def _should_sell_stock(self, stock_code: str, channel_state, stock_data: Optional[pd.DataFrame] = None) -> bool:
        """
        判断是否应该卖出股票（优化后的规则）

        卖出规则（满足任一条件即卖出）：
        1) 上升通道状态不为 NORMAL（除了BREAKOUT状态的特殊处理）
        2) 价格突破通道上沿（根据 sell_on_close_breakout 参数决定使用收盘价还是最高价）
        3) BREAKOUT状态：如果sell_on_close_breakout=True，则从最高点回撤3%再卖出

        Args:
            stock_code: 股票代码
            channel_state: 通道状态对象（需包含 channel_status 与 upper_today 字段）
            stock_data: 截止当前交易日的历史数据（用于获取当日价格）
        Returns:
            是否应该卖出
        """
        try:
            # 1) 通道状态检查
            if (channel_state is None) or (not hasattr(channel_state, 'channel_status')):
                return True
            
            current_status = channel_state.channel_status
            
            # 获取当前价格
            current_price = None
            if stock_data is not None and len(stock_data) >= 1:
                try:
                    current_price = float(stock_data.iloc[-1]['close'])
                except Exception:
                    current_price = None
            
            # 2) BREAKOUT状态特殊处理
            if current_status == ChannelStatus.BREAKOUT:
                sell_on_close = getattr(self.params, 'sell_on_close_breakout', True)
                
                if sell_on_close and current_price is not None:
                    # 记录或更新BREAKOUT状态的最高价
                    if stock_code not in self._breakout_high_prices:
                        self._breakout_high_prices[stock_code] = current_price
                    else:
                        self._breakout_high_prices[stock_code] = max(
                            self._breakout_high_prices[stock_code], 
                            current_price
                        )
                    
                    # 计算回撤百分比
                    high_price = self._breakout_high_prices[stock_code]
                    pullback_threshold = getattr(self.params, 'breakout_pullback_threshold', 3.0)
                    
                    if high_price > 0:
                        pullback_pct = ((high_price - current_price) / high_price) * 100
                        
                        # 如果回撤超过阈值，则卖出
                        if pullback_pct >= pullback_threshold:
                            if self.params.enable_logging:
                                self.logger.info(f"BREAKOUT回撤卖出: {stock_code}, 最高价: {high_price:.2f}, "
                                                f"当前价: {current_price:.2f}, 回撤: {pullback_pct:.2f}%")
                            return True
                    
                    # 未达到回撤阈值，继续持有
                    return False
                else:
                    # sell_on_close=False 或无法获取价格，按原逻辑处理
                    return True
            
            # 3) 其他非NORMAL状态：直接卖出
            if current_status != ChannelStatus.NORMAL:
                return True

            # 4) NORMAL状态：检查价格突破通道上沿
            sell_on_close = getattr(self.params, 'sell_on_close_breakout', True)

            # 获取当日价格
            day_price = None
            if stock_data is not None and len(stock_data) >= 1:
                try:
                    if sell_on_close:
                        # 使用收盘价判断突破（更稳健，避免盘中波动）
                        day_price = float(stock_data.iloc[-1]['close'])
                    else:
                        # 使用最高价判断突破（更敏感，快速响应）
                        day_price = float(stock_data.iloc[-1]['high'])
                except Exception:
                    day_price = None

            # 获取通道上沿
            upper_today = getattr(channel_state, 'upper_today', None)
            if (day_price is not None) and (upper_today is not None):
                try:
                    if float(day_price) > float(upper_today):
                        return True
                except Exception:
                    # 若比较失败，按未触发第二条处理
                    pass

            # 其余情况不卖出
            return False

        except Exception:
            # 任意异常，保守处理：不触发卖出（以避免异常导致过度平仓）
            return False

    def _execute_sell_signal(self, signal: Dict[str, Any]):
        """
        覆盖父类：执行卖出后，若持仓已清空则清理 BREAKOUT 标记。

        Args:
            signal: 卖出信号
        """
        # 先执行父类逻辑（包含跌停禁卖、清仓、记录等）
        super()._execute_sell_signal(signal)

        # 若卖出成功（不再持仓），清理 BREAKOUT 标记和最高价记录
        try:
            stock_code = signal.get('stock_code')
            if stock_code and (not self.position_manager.has_position(stock_code)):
                self._breakout_flag.pop(stock_code, None)
                self._breakout_high_prices.pop(stock_code, None)
        except Exception:
            pass

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
        # 优先使用配置文件中的参数
        from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
        config = RisingChannelConfig()
        config_params = config.get_strategy_params()
        
        # 构建通道分析器参数
        params = {
            "k": config_params.get('k', self.params.k),
            "L_max": config_params.get('L_max', self.params.L_max),
            "delta_cut": config_params.get('delta_cut', self.params.delta_cut),
            "pivot_m": config_params.get('pivot_m', self.params.pivot_m),
            "min_data_points": config_params.get('min_data_points', self.params.min_data_points),
            "width_pct_min": config_params.get('width_pct_min', self.params.width_pct_min),
            "width_pct_max": config_params.get('width_pct_max', self.params.width_pct_max),
            "adjust": config_params.get('adjust', self.params.adjust),  # 复权类型参数，通道缓存需要包含
            "min_volume_ratio": config_params.get('min_volume_ratio', self.params.min_volume_ratio),  # 最小成交量比参数
            "max_distance_to_lower": config_params.get('max_distance_to_lower', self.params.max_distance_to_lower),  # T日预筛选距离参数
            "min_daily_gain": config_params.get('min_daily_gain', self.params.min_daily_gain),  # T+1日买入涨幅参数
        }

        # 添加策略层面的质量筛选参数
        r2_min = config_params.get('R2_min', self.params.R2_min)
        r2_max = config_params.get('R2_max', self.params.R2_max)
        
        if r2_min is not None:
            params["R2_min"] = r2_min
        if r2_max is not None:
            params["R2_max"] = r2_max
            
        min_channel_score = config_params.get('min_channel_score', getattr(self.params, 'min_channel_score', None))
        if min_channel_score is not None:
            params["min_channel_score"] = min_channel_score

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
