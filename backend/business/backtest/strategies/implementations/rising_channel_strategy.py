#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道策略实现 - 新框架版本
基于技术分析的量化交易策略

本策略完全适配新的事件驱动回测框架，忠实复现旧策略的所有逻辑。
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from backend.business.backtest.strategies.base import BaseStrategy
from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.factor.core.engine.library.channel_analysis.rising_channel import AscendingChannelRegression
from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus
from backend.utils.logger import setup_logger


class RisingChannelStrategy(BaseStrategy):
    """
    上升通道策略 - 新框架版本
    
    策略核心思想：
    通过识别股票价格在上升通道中的运行状态，在合适的时机买入和卖出股票，
    实现趋势跟踪和风险控制的平衡。
    
    策略逻辑流程：
    1. 初始化阶段：设置参数和状态变量
    2. 每日开盘前：更新持仓股票的通道分析
    3. 交易执行：检查卖出条件，执行T+1日买入逻辑
    4. 风险控制：验证交易信号的合理性
    
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
    
    仓位管理：
    1. 最大持仓数量由max_positions参数控制
    2. 平均分配资金到每只股票
    3. 动态调整：卖出后立即寻找新的买入机会
    """

    def __init__(self, context, params: Optional[Dict[str, Any]] = None):
        """
        初始化上升通道策略
        
        Args:
            context: 回测上下文对象
            params: 策略参数字典
        """
        # 获取配置文件中的策略参数
        config = RisingChannelConfig()
        config_params = config.get_strategy_params()
        
        # 合并配置参数和传入的params，params优先级更高
        merged_params = config_params.copy()
        if params:
            merged_params.update(params)
        
        super().__init__(context, merged_params)
        
        # 设置日志
        self.logger = setup_logger("backtest")
        
        # 策略状态变量
        self.stock_data = {}  # 存储每只股票的计算结果和状态
        self._breakout_high_prices = {}  # 记录BREAKOUT状态的最高价
        self._preselected_stocks = []  # T日预筛选的股票池
        
        # 通道分析器
        self.channel_analyzer = None
        
        # 当前分析结果缓存
        self.current_analysis_results = {}

    def initialize(self):
        """
        策略初始化
        在回测开始前调用，且只调用一次
        """
        # 初始化通道分析器
        channel_params = self._get_channel_params()
        self.channel_analyzer = AscendingChannelRegression(**channel_params)
        
        self.logger.info("上升通道策略初始化完成")

    def before_trading_start(self):
        """
        每日开盘前调用
        更新持仓股票的通道分析
        """
        # 获取当前持仓股票
        held_stocks = list(self.context.portfolio.positions.keys())
        
        # 更新持仓股票的通道分析
        for stock_code in held_stocks:
            self._update_stock_channel_analysis(stock_code)

    def handle_data(self, daily_bars: pd.DataFrame):
        """
        核心策略逻辑函数
        每个交易日调用一次
        
        Args:
            daily_bars: 当日所有股票的行情数据
        """
        # 1. 检查当前持仓，生成卖出信号
        sell_signals = self._generate_sell_signals(daily_bars)
        
        # 2. 执行卖出操作
        for signal in sell_signals:
            self._execute_sell_signal(signal)
        
        # 3. 计算执行卖出后的预期持仓数量
        current_positions = len(self.context.portfolio.positions)
        sell_codes = {s.get('stock_code') for s in sell_signals if s.get('stock_code')}
        projected_positions = max(0, current_positions - len(sell_codes))
        
        # 4. 如果不满仓，进行T日预筛选
        if projected_positions < self.params.get('max_positions', 20):
            self._preselect_stocks_t_day(daily_bars)
            
            # 5. 生成买入信号（T+1日买入）
            available_slots = self.params.get('max_positions', 20) - projected_positions
            buy_signals = self._generate_buy_signals(daily_bars, available_slots)
            
            # 6. 执行买入操作
            for signal in buy_signals:
                self._execute_buy_signal(signal)

    def after_trading_end(self):
        """
        每日收盘后调用
        清理BREAKOUT标记
        """
        # 清理已清仓股票的BREAKOUT标记
        current_positions = set(self.context.portfolio.positions.keys())
        for stock_code in list(self._breakout_high_prices.keys()):
            if stock_code not in current_positions:
                self._breakout_high_prices.pop(stock_code, None)

    def _get_channel_params(self) -> Dict[str, Any]:
        """
        获取通道分析器参数
        
        Returns:
            通道分析器参数字典
        """
        return {
            'k': self.params.get('k', 2.0),
            'L_max': self.params.get('L_max', 120),
            'delta_cut': self.params.get('delta_cut', 5),
            'pivot_m': self.params.get('pivot_m', 3),
            'min_data_points': self.params.get('min_data_points', 60),
            'R2_min': self.params.get('R2_min', 0.35),
            'width_pct_min': self.params.get('width_pct_min', 0.05),
            'width_pct_max': self.params.get('width_pct_max', 0.12)
        }

    def _update_stock_channel_analysis(self, stock_code: str):
        """
        更新单只股票的通道分析
        
        Args:
            stock_code: 股票代码
        """
        try:
            # 获取历史数据
            end_date = self.context.current_dt
            lookback_days = self.params.get('min_data_points', 60)
            stock_data = self.context.data_provider.get_history_data(
                stock_code, end_date, lookback_days
            )
            
            if stock_data.empty:
                return
            
            # 分析通道状态
            channel_state = self.channel_analyzer.fit_channel(stock_data)
            
            # 计算通道评分
            score = self._calculate_channel_score(channel_state, stock_data)
            
            # 存储分析结果
            self.stock_data[stock_code] = {
                'channel_state': channel_state,
                'score': score,
                'last_update': self.context.current_dt
            }
            
        except Exception as e:
            self.logger.warning(f"更新股票 {stock_code} 通道分析失败: {e}")

    def _calculate_channel_score(self, channel_state, stock_data: pd.DataFrame) -> float:
        """
        计算通道评分
        
        Args:
            channel_state: 通道状态对象
            stock_data: 股票数据
            
        Returns:
            通道评分
        """
        try:
            score = 60.0  # 基础评分
            
            # 根据R²值调整评分
            if hasattr(channel_state, 'r2') and channel_state.r2 is not None:
                r2 = channel_state.r2
                if r2 >= 0.8:
                    score += 20
                elif r2 >= 0.6:
                    score += 10
                elif r2 >= 0.4:
                    score += 5
                elif r2 < 0.2:
                    score -= 20
            
            # 根据通道宽度调整评分
            if hasattr(channel_state, 'width_pct') and channel_state.width_pct is not None:
                width_pct = channel_state.width_pct
                if 0.05 <= width_pct <= 0.12:
                    score += 10
                elif width_pct < 0.03 or width_pct > 0.2:
                    score -= 10
            
            # 根据斜率调整评分
            if hasattr(channel_state, 'beta') and channel_state.beta is not None:
                beta = channel_state.beta
                if 0.01 <= beta <= 0.05:
                    score += 10
                elif beta < 0.005 or beta > 0.1:
                    score -= 10
            
            return max(0, min(100, score))
            
        except Exception:
            return 60.0

    def _generate_sell_signals(self, daily_bars: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        生成卖出信号
        检查持仓股票是否仍符合NORMAL通道状态
        
        Args:
            daily_bars: 当日行情数据
            
        Returns:
            卖出信号列表
        """
        sell_signals = []
        held_stocks = list(self.context.portfolio.positions.keys())
        
        for stock_code in held_stocks:
            # 获取股票数据
            if stock_code not in self.stock_data:
                continue
                
            stock_info = self.stock_data[stock_code]
            channel_state = stock_info['channel_state']
            
            # 获取当日价格
            if stock_code not in daily_bars.index:
                continue
                
            current_price = daily_bars.loc[stock_code]['close']
            
            # 检查是否应该卖出
            if self._should_sell_stock(stock_code, channel_state, current_price, daily_bars):
                sell_reason = self._get_sell_reason(stock_code, channel_state, current_price, daily_bars)
                signal = {
                    'stock_code': stock_code,
                    'price': current_price,
                    'reason': sell_reason,
                    'type': 'sell'
                }
                sell_signals.append(signal)
        
        return sell_signals

    def _should_sell_stock(self, stock_code: str, channel_state, current_price: float, 
                          daily_bars: pd.DataFrame) -> bool:
        """
        判断是否应该卖出股票
        
        Args:
            stock_code: 股票代码
            channel_state: 通道状态对象
            current_price: 当前价格
            daily_bars: 当日行情数据
            
        Returns:
            是否应该卖出
        """
        try:
            # 1. 通道状态检查
            if channel_state is None or not hasattr(channel_state, 'channel_status'):
                return True
            
            current_status = channel_state.channel_status
            
            # 2. BREAKOUT状态特殊处理
            if current_status == ChannelStatus.BREAKOUT:
                sell_on_close = self.params.get('sell_on_close_breakout', True)
                
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
                    pullback_threshold = self.params.get('breakout_pullback_threshold', 3.0)
                    
                    if high_price > 0:
                        pullback_pct = ((high_price - current_price) / high_price) * 100
                        
                        # 如果回撤超过阈值，则卖出
                        if pullback_pct >= pullback_threshold:
                            return True
                    
                    # 未达到回撤阈值，继续持有
                    return False
                else:
                    # sell_on_close=False，按原逻辑处理
                    return True
            
            # 3. 其他非NORMAL状态：直接卖出
            if current_status != ChannelStatus.NORMAL:
                return True

            # 4. NORMAL状态：检查价格突破通道上沿
            sell_on_close = self.params.get('sell_on_close_breakout', True)
            
            # 获取当日价格
            if stock_code not in daily_bars.index:
                return False
                
            if sell_on_close:
                day_price = daily_bars.loc[stock_code]['close']
            else:
                day_price = daily_bars.loc[stock_code]['high']

            # 获取通道上沿
            upper_today = getattr(channel_state, 'upper_today', None)
            if day_price is not None and upper_today is not None:
                if float(day_price) > float(upper_today):
                    return True

            # 其余情况不卖出
            return False

        except Exception:
            # 任意异常，保守处理：不触发卖出
            return False

    def _get_sell_reason(self, stock_code: str, channel_state, current_price: float, 
                        daily_bars: pd.DataFrame) -> str:
        """
        获取卖出原因
        
        Args:
            stock_code: 股票代码
            channel_state: 通道状态对象
            current_price: 当前价格
            daily_bars: 当日行情数据
            
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
                sell_on_close = self.params.get('sell_on_close_breakout', True)
                
                if sell_on_close and stock_code in self._breakout_high_prices:
                    high_price = self._breakout_high_prices[stock_code]
                    pullback_threshold = self.params.get('breakout_pullback_threshold', 3.0)
                    
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
            sell_on_close = self.params.get('sell_on_close_breakout', True)
            
            # 获取当日价格
            if stock_code not in daily_bars.index:
                return "无当日数据"
                
            if sell_on_close:
                day_price = daily_bars.loc[stock_code]['close']
            else:
                day_price = daily_bars.loc[stock_code]['high']
            
            # 获取通道上沿
            upper_today = getattr(channel_state, 'upper_today', None)
            if day_price is not None and upper_today is not None:
                if float(day_price) > float(upper_today):
                    price_type = "收盘价" if sell_on_close else "最高价"
                    return f"{price_type}突破通道上沿: {day_price:.2f} > {upper_today:.2f}"
            
            # 5. 其他情况
            return "其他卖出条件"
            
        except Exception as e:
            return f"卖出原因分析异常: {str(e)}"

    def _preselect_stocks_t_day(self, daily_bars: pd.DataFrame):
        """
        T日预筛选股票 - 根据通道状态和价格位置筛选候选股票
        
        Args:
            daily_bars: 当日行情数据
        """
        # 批量分析所有股票的通道状态
        self._update_all_stocks_channel_analysis(daily_bars)
        
        # 筛选NORMAL状态的股票
        r2_min, r2_max = self._get_effective_r2_bounds()
        normal_stocks = self._filter_normal_channels(r2_min, r2_max)
        
        if not normal_stocks:
            self._preselected_stocks = []
            if self.params.get('enable_logging', True):
                self.logger.info("T日预筛选：没有找到符合条件的股票")
            return
        
        # 记录符合条件的股票
        qualified_stocks = []
        
        for stock_info in normal_stocks:
            stock_code = stock_info['stock_code']
            channel_state = stock_info['channel_state']
            
            # 校验通道宽度有效性
            if not self._is_channel_width_valid(channel_state):
                continue
            
            # 获取当前价格
            if stock_code not in daily_bars.index:
                continue
            current_price = daily_bars.loc[stock_code]['close']
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
                continue
            
            # 计算距离下沿的百分比距离
            distance_to_lower = self._calculate_distance_to_lower(current_price, channel_state)
            
            # 检查股价距离通道下沿是否在允许范围内
            max_distance = self.params.get('max_distance_to_lower', 3.0)
            if distance_to_lower > max_distance:
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
        max_positions = self.params.get('max_positions', 20)
        self._preselected_stocks = [stock['stock_code'] for stock in qualified_stocks[:max_positions]]
        
        if self.params.get('enable_logging', True):
            self.logger.info(f"T日预筛选完成：从{len(qualified_stocks)}只股票中选出{len(self._preselected_stocks)}只候选股票")

    def _generate_buy_signals(self, daily_bars: pd.DataFrame, available_slots: int) -> List[Dict[str, Any]]:
        """
        生成买入信号 - T+1日从预筛选股票中买入
        
        Args:
            daily_bars: 当日行情数据
            available_slots: 可用买入槽位数
            
        Returns:
            买入信号列表
        """
        buy_signals = []
        
        # 检查是否有预筛选的股票
        if not self._preselected_stocks:
            if self.params.get('enable_logging', True):
                self.logger.info("T+1日买入：没有预筛选的股票，跳过买入")
            return buy_signals
        
        # 筛选NORMAL状态的股票
        r2_min, r2_max = self._get_effective_r2_bounds()
        normal_stocks = self._filter_normal_channels(r2_min, r2_max)
        
        if not normal_stocks:
            if self.params.get('enable_logging', True):
                self.logger.info("T+1日买入：没有NORMAL状态的股票，跳过买入")
            return buy_signals
        
        # 从预筛选股票中筛选符合条件的股票
        qualified_stocks = []
        
        for stock_code in self._preselected_stocks:
            # 检查是否在NORMAL状态股票中
            stock_info = next((s for s in normal_stocks if s['stock_code'] == stock_code), None)
            if not stock_info:
                continue
            
            channel_state = stock_info['channel_state']
            
            # 获取当前价格
            if stock_code not in daily_bars.index:
                continue
            current_price = daily_bars.loc[stock_code]['close']
            if current_price <= 0:
                continue
            
            # 检查收盘价 > 开盘价
            current_open = daily_bars.loc[stock_code]['open']
            if current_open <= 0:
                continue
            
            if current_price <= current_open:
                continue
            
            # 计算当日涨幅
            daily_gain = ((current_price - current_open) / current_open) * 100
            
            # 检查当日涨幅是否满足要求
            min_daily_gain = self.params.get('min_daily_gain', 0.5)
            if daily_gain < min_daily_gain:
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
                continue
            
            # 检查成交量要求：当日交易量大于过去五日平均交易量
            current_volume = daily_bars.loc[stock_code]['volume']
            
            # 获取5日平均成交量
            avg_volume = self._get_avg_volume(stock_code, 5)
            
            if current_volume <= 0 or avg_volume <= 0:
                continue
            
            # 检查成交量比要求
            volume_ratio = current_volume / avg_volume
            min_volume_ratio = self.params.get('min_volume_ratio', 1.2)
            if volume_ratio < min_volume_ratio:
                continue
            
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
            if self.params.get('enable_logging', True):
                self.logger.info("T+1日买入：没有满足所有条件的股票，跳过买入")
            return buy_signals
        
        # 根据成交量增长比从大到小排序
        qualified_stocks.sort(key=lambda x: x['volume_ratio'], reverse=True)
        
        # 计算需要买入的数量
        need_to_buy = min(available_slots, len(qualified_stocks))
        
        # 生成买入信号
        for i in range(need_to_buy):
            stock_info = qualified_stocks[i]
            stock_code = stock_info['stock_code']
            
            # 检查是否已经持仓
            if stock_code in self.context.portfolio.positions:
                continue
            
            # 计算买入数量
            per_trade_cash = self._get_per_trade_cash()
            current_price = stock_info['current_price']
            
            # 计算理论买入数量
            theoretical_size = int(per_trade_cash / current_price)
            
            # 调整数量为100的整数倍
            adjusted_size = (theoretical_size // 100) * 100
            
            # 如果调整后数量为0，跳过这只股票
            if adjusted_size <= 0:
                continue
            
            signal = {
                'stock_code': stock_code,
                'price': current_price,
                'size': adjusted_size,
                'reason': f"T+1日买入: 成交量比{stock_info['volume_ratio']:.2f}, "
                         f"当日涨幅{stock_info['daily_gain']:.2f}%, 评分{stock_info['score']:.1f}",
                'type': 'buy'
            }
            buy_signals.append(signal)
        
        return buy_signals

    def _update_all_stocks_channel_analysis(self, daily_bars: pd.DataFrame):
        """
        更新所有股票的通道分析结果
        
        Args:
            daily_bars: 当日行情数据
        """
        self.current_analysis_results = {}
        
        for stock_code in daily_bars.index:
            try:
                # 获取历史数据
                end_date = self.context.current_dt
                lookback_days = self.params.get('min_data_points', 60)
                stock_data = self.context.data_provider.get_history_data(
                    stock_code, end_date, lookback_days
                )
                
                if stock_data.empty:
                    continue
                
                # 分析通道状态
                channel_state = self.channel_analyzer.fit_channel(stock_data)
                
                # 计算通道评分
                score = self._calculate_channel_score(channel_state, stock_data)
                
                self.current_analysis_results[stock_code] = {
                    'channel_state': channel_state,
                    'score': score
                }
                
            except Exception as e:
                self.logger.debug(f"分析股票 {stock_code} 通道状态失败: {e}")
                continue

    def _filter_normal_channels(self, r2_min: float, r2_max: float) -> List[Dict[str, Any]]:
        """
        筛选NORMAL状态的股票
        
        Args:
            r2_min: 最小R²值
            r2_max: 最大R²值
            
        Returns:
            NORMAL状态股票列表
        """
        normal_stocks = []
        min_score = self.params.get('min_channel_score', 60.0)
        
        for stock_code, result in self.current_analysis_results.items():
            channel_state = result['channel_state']
            score = result['score']
            
            # 检查通道状态
            if not hasattr(channel_state, 'channel_status'):
                continue
            if channel_state.channel_status != ChannelStatus.NORMAL:
                continue
            
            # 检查评分
            if score < min_score:
                continue
            
            # 检查R²值
            if hasattr(channel_state, 'r2') and channel_state.r2 is not None:
                r2 = channel_state.r2
                if r2_min is not None and r2 < r2_min:
                    continue
                if r2_max is not None and r2 > r2_max:
                    continue
            
            normal_stocks.append({
                'stock_code': stock_code,
                'channel_state': channel_state,
                'score': score
            })
        
        return normal_stocks

    def _get_effective_r2_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """
        获取有效的R²边界
        
        Returns:
            (r2_min, r2_max) 元组
        """
        r2_min = self.params.get('R2_min', None)
        r2_max = self.params.get('R2_max', None)
        r2_range = self.params.get('R2_range', None)
        
        if r2_range is not None and isinstance(r2_range, (list, tuple)) and len(r2_range) == 2:
            return r2_range[0], r2_range[1]
        
        return r2_min, r2_max

    def _is_channel_width_valid(self, channel_state) -> bool:
        """
        检查通道宽度是否有效
        
        Args:
            channel_state: 通道状态对象
            
        Returns:
            宽度是否在有效范围
        """
        try:
            if channel_state is None:
                return False
            
            # 优先使用数据库中的width_pct字段
            width_pct = getattr(channel_state, 'width_pct', None)
            if width_pct is not None:
                width_pct_min = self.params.get('width_pct_min', 0.05)
                width_pct_max = self.params.get('width_pct_max', 0.12)
                return width_pct_min <= width_pct <= width_pct_max
            
            # 回退到重新计算
            upper = getattr(channel_state, 'upper_today', None)
            lower = getattr(channel_state, 'lower_today', None)
            mid = getattr(channel_state, 'mid_today', None)
            
            if upper is None or lower is None or mid is None or float(mid) <= 0:
                return False
            
            width_pct = (float(upper) - float(lower)) / float(mid)
            width_pct_min = self.params.get('width_pct_min', 0.05)
            width_pct_max = self.params.get('width_pct_max', 0.12)
            return width_pct_min <= width_pct <= width_pct_max
            
        except Exception:
            return False

    def _is_price_in_channel(self, current_price: float, channel_state) -> bool:
        """
        检查价格是否在通道内
        
        Args:
            current_price: 当前价格
            channel_state: 通道状态对象
            
        Returns:
            是否在通道内
        """
        try:
            if channel_state is None:
                return False
            
            lower_today = getattr(channel_state, 'lower_today', None)
            upper_today = getattr(channel_state, 'upper_today', None)
            
            if lower_today is None or upper_today is None:
                return False
            
            return float(lower_today) < float(current_price) <= float(upper_today)
            
        except Exception:
            return False

    def _calculate_distance_to_lower(self, current_price: float, channel_state) -> float:
        """
        计算股价距离下沿的百分比距离
        
        Args:
            current_price: 当前价格
            channel_state: 通道状态对象
            
        Returns:
            距离下沿的百分比距离
        """
        try:
            if channel_state is None or not hasattr(channel_state, 'lower_today'):
                return 100.0  # 默认值
            
            lower_price = getattr(channel_state, 'lower_today', None)
            if lower_price is None or float(lower_price) <= 0:
                return 100.0
            
            distance = ((current_price - float(lower_price)) / float(lower_price)) * 100
            return distance
            
        except Exception:
            return 100.0

    def _get_avg_volume(self, stock_code: str, days: int) -> float:
        """
        获取股票的平均成交量
        
        Args:
            stock_code: 股票代码
            days: 天数
            
        Returns:
            平均成交量
        """
        try:
            end_date = self.context.current_dt
            stock_data = self.context.data_provider.get_history_data(
                stock_code, end_date, days + 1
            )
            
            if stock_data.empty or len(stock_data) < days:
                return 0.0
            
            # 取最近days天的成交量平均值
            recent_data = stock_data.tail(days)
            return recent_data['volume'].mean()
            
        except Exception:
            return 0.0

    def _get_per_trade_cash(self) -> float:
        """
        获取每笔交易的资金
        
        Returns:
            每笔交易资金
        """
        max_positions = self.params.get('max_positions', 20)
        total_cash = self.context.portfolio.available_cash
        return total_cash / max_positions

    def _execute_sell_signal(self, signal: Dict[str, Any]):
        """
        执行卖出信号
        
        Args:
            signal: 卖出信号
        """
        stock_code = signal['stock_code']
        price = signal['price']
        
        # 获取当前持仓
        if stock_code not in self.context.portfolio.positions:
            return
        
        position = self.context.portfolio.positions[stock_code]
        amount = position.amount
        
        # 执行卖出
        self.context.portfolio.order(stock_code, -amount)
        
        if self.params.get('enable_logging', True):
            self.logger.info(f"卖出 {stock_code}: {amount}股, 价格{price:.2f}, 原因: {signal['reason']}")

    def _execute_buy_signal(self, signal: Dict[str, Any]):
        """
        执行买入信号
        
        Args:
            signal: 买入信号
        """
        stock_code = signal['stock_code']
        price = signal['price']
        size = signal['size']
        
        # 执行买入
        self.context.portfolio.order(stock_code, size)
        
        if self.params.get('enable_logging', True):
            self.logger.info(f"买入 {stock_code}: {size}股, 价格{price:.2f}, 原因: {signal['reason']}")


# 为了保持与原有代码的兼容性，保留原来的类名
RisingChannelBacktestStrategy = RisingChannelStrategy


def create_rising_channel_strategy(max_positions: int = 50,
                                   min_channel_score: float = 60.0,
                                   **params) -> Dict[str, Any]:
    """
    工厂函数：创建上升通道策略参数
    
    Args:
        max_positions: 最大持仓数量
        min_channel_score: 最小通道评分
        **params: 其他策略参数
        
    Returns:
        策略参数字典
    """
    # 构建参数字典
    strategy_params = {
        'max_positions': max_positions,
        'min_channel_score': min_channel_score,
        **params
    }
    
    return strategy_params
