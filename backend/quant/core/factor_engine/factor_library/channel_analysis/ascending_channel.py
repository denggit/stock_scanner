#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : AI Assistant
@Date       : 2024-12-19
@File       : ascending_channel.py
@Description: 上升通道回归模块

该模块实现了上升通道回归分析算法，主要功能包括：
1. 自动检测和维护上升回归通道
2. 提供通道突破和重锚机制
3. 支持多种通道状态监控
4. 实时计算通道边界和预测值

算法流程：
初始化 → 找首个 anchor → 每日更新 → 检查重锚触发 → 状态管理
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .channel_state import ChannelState, ChannelStatus
from .pivot_detector import PivotDetector

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """数据不足异常"""
    pass


class AscendingChannelRegression:
    """
    上升通道回归分析类
    
    实现上升通道回归算法，自动检测、维护并输出最新上升回归通道信息
    """

    def __init__(self, config_path: Optional[str] = None, **params):
        """
        初始化上升通道回归分析器
        """
        self.config = self._load_config(config_path)
        self.k = params.get('k', self.config['ascending_channel']['k'])
        self.L_max = params.get('L_max', self.config['ascending_channel']['L_max'])
        self.delta_cut = params.get('delta_cut', self.config['ascending_channel']['delta_cut'])
        self.pivot_m = params.get('pivot_m', self.config['ascending_channel']['pivot_m'])
        self.gain_trigger = params.get('gain_trigger', self.config['ascending_channel']['gain_trigger'])
        self.beta_delta = params.get('beta_delta', self.config['ascending_channel']['beta_delta'])
        self.break_days = params.get('break_days', self.config['ascending_channel']['break_days'])
        self.reanchor_fail_max = params.get('reanchor_fail_max', self.config['ascending_channel']['reanchor_fail_max'])
        self.min_data_points = params.get('min_data_points', self.config['ascending_channel']['min_data_points'])
        self.R2_min = params.get('R2_min', self.config['ascending_channel'].get('R2_min', 0.20))
        self.width_pct_min = params.get('width_pct_min', self.config['ascending_channel'].get('width_pct_min', 0.04))
        self.width_pct_max = params.get('width_pct_max', self.config['ascending_channel'].get('width_pct_max', 0.12))
        self.pivot_detector = PivotDetector(pivot_m=self.pivot_m)
        self.state: Optional[ChannelState] = None
        logger.info("上升通道回归分析器初始化完成")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path (Optional[str]): 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        if config_path is None:
            # 使用默认配置文件
            config_path = Path(__file__).parent.parent / "configs" / "channel_config.yaml"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            # 返回默认配置
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'ascending_channel': {
                'k': 2.0,
                'L_max': 120,
                'delta_cut': 5,
                'pivot_m': 3,
                'gain_trigger': 0.30,
                'beta_delta': 0.15,
                'break_days': 3,
                'reanchor_fail_max': 2,
                'min_data_points': 60
            }
        }

    def fit_channel(self, df: pd.DataFrame) -> ChannelState:
        """
        拟合上升通道（仅做上升通道，严格全局pivot_low规则）
        """
        if len(df) < self.min_data_points:
            return ChannelState(
                anchor_date=None, anchor_price=None, window_df=pd.DataFrame(),
                beta=None, sigma=None,
                mid_today=None, upper_today=None, lower_today=None,
                mid_tomorrow=None, upper_tomorrow=None, lower_tomorrow=None,
                channel_status=ChannelStatus.BROKEN
            )
        df = df.sort_values('trade_date').reset_index(drop=True)
        # 只在最后L_max天内找pivot_low
        search_df = df.tail(self.L_max).copy()
        # 找所有pivot_low
        pivots = self.pivot_detector.get_anchor_candidates(search_df, n_candidates=20)
        # 过滤：只要距离最新日期>=min_data_points的pivot_low
        if not pivots:
            return ChannelState(
                anchor_date=None, anchor_price=None, window_df=pd.DataFrame(),
                beta=None, sigma=None,
                mid_today=None, upper_today=None, lower_today=None,
                mid_tomorrow=None, upper_tomorrow=None, lower_tomorrow=None,
                channel_status=ChannelStatus.BROKEN
            )
        today = search_df['trade_date'].iloc[-1]
        valid_pivots = [(d, p) for d, p in pivots if (today - pd.to_datetime(d)).days >= self.min_data_points]
        if not valid_pivots:
            return ChannelState(
                anchor_date=None, anchor_price=None, window_df=pd.DataFrame(),
                beta=None, sigma=None,
                mid_today=None, upper_today=None, lower_today=None,
                mid_tomorrow=None, upper_tomorrow=None, lower_tomorrow=None,
                channel_status=ChannelStatus.BROKEN
            )
        # 选最靠前的最低pivot_low
        anchor_date, anchor_price = min(valid_pivots, key=lambda x: x[1])
        anchor_date = pd.to_datetime(anchor_date)
        window_df = df[df['trade_date'] >= anchor_date].copy()
        # 回归
        beta, sigma, r2 = self._calculate_regression(window_df, anchor_date)
        if beta is None or beta <= 0 or r2 < self.R2_min:
            state = ChannelState(
                anchor_date=anchor_date, anchor_price=anchor_price, window_df=window_df,
                beta=beta, sigma=sigma,
                mid_today=None, upper_today=None, lower_today=None,
                mid_tomorrow=None, upper_tomorrow=None, lower_tomorrow=None,
                channel_status=ChannelStatus.BROKEN
            )
            state.r2 = r2
            return state
        state = ChannelState(
            anchor_date=anchor_date, anchor_price=anchor_price, window_df=window_df,
            beta=beta, sigma=sigma,
            mid_today=0.0, upper_today=0.0, lower_today=0.0,
            mid_tomorrow=0.0, upper_tomorrow=0.0, lower_tomorrow=0.0
        )
        state.update_channel_boundaries(beta, sigma, self.k)
        width_pct = (state.upper_today - state.lower_today) / state.mid_today if state.mid_today else None
        if width_pct is None or width_pct < self.width_pct_min or width_pct > self.width_pct_max:
            state.channel_status = ChannelStatus.BROKEN
            state.r2 = r2
            return state
        state.cumulative_gain = (window_df['close'].iloc[-1] - anchor_price) / anchor_price
        state.channel_status = ChannelStatus.NORMAL
        state.r2 = r2
        return state

    def update(self, state: ChannelState, bar: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新通道状态
        
        Args:
            state (ChannelState): 当前通道状态
            bar (Dict[str, Any]): 新的K线数据，包含 trade_date, open, high, low, close, volume
            
        Returns:
            Dict[str, Any]: 更新后的通道信息
        """
        if state is None:
            raise ValueError("通道状态不能为空")

        # 更新状态对象
        self.state = state

        # 添加新数据到窗口
        new_row = pd.DataFrame([bar])
        self.state.window_df = pd.concat([self.state.window_df, new_row], ignore_index=True)

        # 更新最后更新时间
        self.state.last_update = pd.to_datetime(bar['trade_date'])

        # 检查窗口长度，必要时滑动窗口
        if len(self.state.window_df) > self.L_max:
            self._slide_window()

        # 重新计算回归参数
        beta, sigma, r2 = self._calculate_regression(self.state.window_df, self.state.anchor_date)

        # 更新通道边界
        self.state.update_channel_boundaries(beta, sigma, self.k)

        # 更新累计涨幅
        self.state.cumulative_gain = (bar['close'] - self.state.anchor_price) / self.state.anchor_price

        # 检查价格位置并更新计数器
        self.state.update_break_counters(bar['close'], self.state.upper_today, self.state.lower_today)

        # 检查重锚触发条件
        if self._should_reanchor():
            self._reanchor()

        # 检查极端状态
        self._check_extreme_states()

        # 返回更新后的信息
        return self.state.to_dict()

    def _calculate_regression(self, df: pd.DataFrame, anchor_date: pd.Timestamp):
        """
        计算线性回归参数，返回(beta, sigma, r2)
        """
        dates = pd.to_datetime(df['trade_date'])
        anchor_date = pd.to_datetime(anchor_date)
        days_since_anchor = (dates - anchor_date).dt.days
        prices = df['close'].values
        if len(days_since_anchor) < 2 or np.std(days_since_anchor) == 0 or np.std(prices) == 0:
            return None, None, None
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(days_since_anchor, prices)
            r2 = r_value ** 2
            predicted = slope * days_since_anchor + intercept
            residuals = prices - predicted
            sigma = np.std(residuals)
            return slope, sigma, r2
        except Exception:
            return None, None, None

    def _slide_window(self) -> None:
        """滑动窗口，删除最早的数据"""
        # 删除最早的 delta_cut 天数据
        self.state.window_df = self.state.window_df.iloc[self.delta_cut:].reset_index(drop=True)
        logger.debug(f"窗口滑动，删除最早 {self.delta_cut} 天数据")

    def _detect_trend(self, df: pd.DataFrame, lookback_days: int = 20) -> str:
        """
        检测价格趋势
        
        Args:
            df (pd.DataFrame): 价格数据
            lookback_days (int): 回看天数
            
        Returns:
            str: 趋势类型 ('up', 'down', 'sideways')
        """
        if len(df) < lookback_days:
            return 'sideways'
        
        # 取最近lookback_days天的数据
        recent_df = df.tail(lookback_days)
        
        # 计算线性回归斜率
        dates = pd.to_datetime(recent_df['trade_date'])
        prices = recent_df['close'].values
        
        if len(dates) < 2:
            return 'sideways'
        
        # 计算天数差
        days_since_start = (dates - dates.iloc[0]).dt.days
        
        try:
            slope, _, _, _, _ = stats.linregress(days_since_start, prices)
            
            # 计算价格变化百分比
            price_change = (prices[-1] - prices[0]) / prices[0]
            
            # 判断趋势
            if slope > 0.005 and price_change > 0.01:  # 降低阈值：斜率大于0.005且涨幅大于1%
                return 'up'
            elif slope < -0.005 and price_change < -0.01:  # 降低阈值：斜率小于-0.005且跌幅大于1%
                return 'down'
            else:
                return 'sideways'
                
        except (ValueError, RuntimeWarning):
            return 'sideways'

    def _select_anchor_strategy(self, trend: str) -> str:
        """
        根据趋势选择锚点策略
        
        Args:
            trend (str): 价格趋势
            
        Returns:
            str: 锚点选择策略
        """
        if trend == 'up':
            return 'pivot_low'  # 上升趋势选择低点作为锚点
        elif trend == 'down':
            return 'pivot_high'  # 下降趋势选择高点作为锚点
        else:
            return 'pivot_low'  # 横盘趋势默认选择低点

    def _should_reanchor(self) -> bool:
        """
        检查是否应该重锚（pivot_low规则）
        """
        if not self.state.can_reanchor():
            return False
        # 只保留原有条件，去除趋势分流
        conditions = [
            len(self.state.window_df) > self.L_max,
            self.state.cumulative_gain >= self.gain_trigger,
            self.state.break_cnt_up >= self.break_days or self.state.break_cnt_down >= self.break_days,
            self.state.channel_status == ChannelStatus.BROKEN,
            self._check_beta_change()
        ]
        return any(conditions)

    def _check_beta_change(self) -> bool:
        """
        检查斜率变化
        
        Returns:
            bool: 斜率是否发生显著变化
        """
        # 这里需要保存历史斜率进行比较
        # 简化实现：暂时返回 False
        return False

    def _reanchor(self) -> None:
        """
        执行重锚操作（pivot_low规则）
        """
        logger.info("开始重锚操作(pivot_low)")
        new_anchor = self.pivot_detector.find_new_anchor(
            self.state.window_df,
            self.state.anchor_date,
            strategy='pivot_low'
        )
        if new_anchor is None:
            logger.warning("重锚失败：未找到新的有效锚点")
            if self.state.cumulative_gain > 0:
                self.state.reanchor_fail_up += 1
            else:
                self.state.reanchor_fail_down += 1
            return
        new_anchor_date, new_anchor_price = new_anchor
        self.state.anchor_date = new_anchor_date
        self.state.anchor_price = new_anchor_price
        self.state.window_df = self.state.window_df[
            self.state.window_df['trade_date'] >= new_anchor_date
        ].reset_index(drop=True)
        beta, sigma, r2 = self._calculate_regression(self.state.window_df, new_anchor_date)
        self.state.beta = beta
        self.state.sigma = sigma
        self.state.update_channel_boundaries(beta, sigma, self.k)
        self.state.reset_break_counters()
        self.state.reanchor_fail_up = 0
        self.state.reanchor_fail_down = 0
        if self.state.channel_status == ChannelStatus.BROKEN:
            self.state.channel_status = ChannelStatus.NORMAL
            logger.info("通道状态重置为正常")
        logger.info(f"重锚成功: 新锚点 {new_anchor_date} @ {new_anchor_price:.2f}, 窗口大小: {len(self.state.window_df)}")

    def _check_trend_reversal(self) -> bool:
        """
        检查趋势是否发生反转（已废弃，恒返False）
        """
        return False

    def _check_prolonged_failure(self) -> bool:
        """
        检查是否长时间通道失效
        
        Returns:
            bool: 是否长时间失效
        """
        # 如果通道状态为BROKEN且持续时间较长，触发重锚
        if self.state.channel_status == ChannelStatus.BROKEN:
            # 检查连续突破下沿的天数
            if self.state.break_cnt_down >= self.break_days:  # 降低阈值，只要达到break_days就触发
                logger.info(f"检测到长时间通道失效: 连续突破 {self.state.break_cnt_down} 天")
                return True
        
        # 检查连续重锚失败
        if self.state.reanchor_fail_up >= 1 or self.state.reanchor_fail_down >= 1:  # 降低阈值
            logger.info(f"检测到重锚失败: 上沿失败 {self.state.reanchor_fail_up} 次，下沿失败 {self.state.reanchor_fail_down} 次")
            return True
        
        return False

    def _check_extreme_states(self) -> None:
        """检查并更新极端状态"""
        # 检查连续重锚失败
        if self.state.reanchor_fail_up >= self.reanchor_fail_max:
            self.state.channel_status = ChannelStatus.ACCEL_BREAKOUT
            logger.warning(f"进入加速突破状态: 连续重锚失败 {self.state.reanchor_fail_up} 次")

        elif self.state.reanchor_fail_down >= self.reanchor_fail_max:
            self.state.channel_status = ChannelStatus.BREAKDOWN
            logger.warning(f"进入跌破状态: 连续重锚失败 {self.state.reanchor_fail_down} 次")

        # 检查通道失效
        elif (self.state.break_cnt_down >= self.break_days and
              self.state.channel_status != ChannelStatus.BROKEN):
            self.state.channel_status = ChannelStatus.BROKEN
            logger.warning("通道失效: 连续跌破下沿")

    def force_reanchor(self, state: ChannelState) -> ChannelState:
        """
        强制重锚
        
        Args:
            state (ChannelState): 当前状态
            
        Returns:
            ChannelState: 更新后的状态
        """
        self.state = state
        self._reanchor()
        return self.state

    def get_channel_info(self, state: Optional[ChannelState] = None) -> Dict[str, Any]:
        """
        获取通道信息
        
        Args:
            state (Optional[ChannelState]): 通道状态，如果为None则使用当前状态
            
        Returns:
            Dict[str, Any]: 通道信息字典
        """
        if state is None:
            state = self.state

        if state is None:
            return {}

        return state.to_dict()

    def fit_channel_history(self, df: pd.DataFrame, min_window_size: int = 60) -> pd.DataFrame:
        """
        计算历史上升通道数据（全局pivot_low规则，严格上升通道）
        """
        if len(df) < min_window_size + 20:
            raise InsufficientDataError(
                f"数据不足，至少需要 {min_window_size + 20} 个数据点，当前只有 {len(df)} 个"
            )
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        history_data = []
        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            for i in range(min_window_size, len(df)):
                current_df = df.iloc[:i+1].copy()
                current_date = current_df.iloc[-1]['trade_date']
                current_close = current_df.iloc[-1]['close']
                base_record = {
                    'trade_date': current_date,
                    'close': current_close,
                    'beta': None,
                    'sigma': None,
                    'mid_today': None,
                    'upper_today': None,
                    'lower_today': None,
                    'mid_tomorrow': None,
                    'upper_tomorrow': None,
                    'lower_tomorrow': None,
                    'channel_status': None,
                    'anchor_date': None,
                    'anchor_price': None,
                    'break_cnt_up': None,
                    'break_cnt_down': None,
                    'reanchor_fail_up': None,
                    'reanchor_fail_down': None,
                    'cumulative_gain': None,
                    'window_size': None,
                    'days_since_anchor': None
                }
                # 全局pivot_low规则
                search_df = current_df.tail(self.L_max).copy()
                pivots = self.pivot_detector.get_anchor_candidates(search_df, n_candidates=20)
                today = search_df['trade_date'].iloc[-1]
                valid_pivots = [(d, p) for d, p in pivots if (today - pd.to_datetime(d)).days >= self.min_data_points]
                if not valid_pivots:
                    base_record['channel_status'] = ChannelStatus.BROKEN.value
                    history_data.append(base_record)
                    continue
                anchor_date, anchor_price = min(valid_pivots, key=lambda x: x[1])
                anchor_date = pd.to_datetime(anchor_date)
                window_df = current_df[current_df['trade_date'] >= anchor_date].copy()
                beta, sigma, r2 = self._calculate_regression(window_df, anchor_date)
                if beta is None or beta <= 0 or r2 < self.R2_min:
                    base_record.update({
                        'anchor_date': anchor_date,
                        'anchor_price': anchor_price,
                        'beta': beta,
                        'sigma': sigma,
                        'r2': r2,
                        'channel_status': ChannelStatus.BROKEN.value,
                        'window_size': len(window_df),
                        'days_since_anchor': (current_date - anchor_date).days
                    })
                    history_data.append(base_record)
                    continue
                # 计算通道边界
                state = ChannelState(
                    anchor_date=anchor_date, anchor_price=anchor_price, window_df=window_df,
                    beta=beta, sigma=sigma,
                    mid_today=0.0, upper_today=0.0, lower_today=0.0,
                    mid_tomorrow=0.0, upper_tomorrow=0.0, lower_tomorrow=0.0
                )
                state.update_channel_boundaries(beta, sigma, self.k)
                width_pct = (state.upper_today - state.lower_today) / state.mid_today if state.mid_today else None
                if width_pct is None or width_pct < self.width_pct_min or width_pct > self.width_pct_max:
                    base_record.update({
                        'anchor_date': anchor_date,
                        'anchor_price': anchor_price,
                        'beta': beta,
                        'sigma': sigma,
                        'r2': r2,
                        'mid_today': state.mid_today,
                        'upper_today': state.upper_today,
                        'lower_today': state.lower_today,
                        'channel_status': ChannelStatus.BROKEN.value,
                        'window_size': len(window_df),
                        'days_since_anchor': (current_date - anchor_date).days
                    })
                    history_data.append(base_record)
                    continue
                # 正常通道
                base_record.update({
                    'anchor_date': anchor_date,
                    'anchor_price': anchor_price,
                    'beta': beta,
                    'sigma': sigma,
                    'r2': r2,
                    'mid_today': state.mid_today,
                    'upper_today': state.upper_today,
                    'lower_today': state.lower_today,
                    'mid_tomorrow': state.mid_tomorrow,
                    'upper_tomorrow': state.upper_tomorrow,
                    'lower_tomorrow': state.lower_tomorrow,
                    'channel_status': ChannelStatus.NORMAL.value,
                    'window_size': len(window_df),
                    'days_since_anchor': (current_date - anchor_date).days,
                    'cumulative_gain': (current_close - anchor_price) / anchor_price
                })
                history_data.append(base_record)
        finally:
            logging.getLogger().setLevel(original_level)
        if not history_data:
            raise InsufficientDataError("无法计算任何历史通道数据")
        history_df = pd.DataFrame(history_data)
        return history_df

    def fit_channel_history_optimized(self, df: pd.DataFrame, min_window_size: int = 60, step_days: int = 1) -> pd.DataFrame:
        """
        优化的历史通道数据计算（全局pivot_low规则，严格上升通道）
        """
        if len(df) < min_window_size + 20:
            raise InsufficientDataError(
                f"数据不足，至少需要 {min_window_size + 20} 个数据点，当前只有 {len(df)} 个"
            )
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        history_data = []
        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            for i in range(min_window_size, len(df), step_days):
                current_df = df.iloc[:i+1].copy()
                current_date = current_df.iloc[-1]['trade_date']
                current_close = current_df.iloc[-1]['close']
                base_record = {
                    'trade_date': current_date,
                    'close': current_close,
                    'beta': None,
                    'sigma': None,
                    'mid_today': None,
                    'upper_today': None,
                    'lower_today': None,
                    'mid_tomorrow': None,
                    'upper_tomorrow': None,
                    'lower_tomorrow': None,
                    'channel_status': None,
                    'anchor_date': None,
                    'anchor_price': None,
                    'break_cnt_up': None,
                    'break_cnt_down': None,
                    'reanchor_fail_up': None,
                    'reanchor_fail_down': None,
                    'cumulative_gain': None,
                    'window_size': None,
                    'days_since_anchor': None
                }
                search_df = current_df.tail(self.L_max).copy()
                pivots = self.pivot_detector.get_anchor_candidates(search_df, n_candidates=20)
                today = search_df['trade_date'].iloc[-1]
                valid_pivots = [(d, p) for d, p in pivots if (today - pd.to_datetime(d)).days >= self.min_data_points]
                if not valid_pivots:
                    base_record['channel_status'] = ChannelStatus.BROKEN.value
                    history_data.append(base_record)
                    continue
                anchor_date, anchor_price = min(valid_pivots, key=lambda x: x[1])
                anchor_date = pd.to_datetime(anchor_date)
                window_df = current_df[current_df['trade_date'] >= anchor_date].copy()
                beta, sigma, r2 = self._calculate_regression(window_df, anchor_date)
                if beta is None or beta <= 0 or r2 < self.R2_min:
                    base_record.update({
                        'anchor_date': anchor_date,
                        'anchor_price': anchor_price,
                        'beta': beta,
                        'sigma': sigma,
                        'channel_status': ChannelStatus.BROKEN.value,
                        'window_size': len(window_df),
                        'days_since_anchor': (current_date - anchor_date).days
                    })
                    history_data.append(base_record)
                    continue
                state = ChannelState(
                    anchor_date=anchor_date, anchor_price=anchor_price, window_df=window_df,
                    beta=beta, sigma=sigma,
                    mid_today=0.0, upper_today=0.0, lower_today=0.0,
                    mid_tomorrow=0.0, upper_tomorrow=0.0, lower_tomorrow=0.0
                )
                state.update_channel_boundaries(beta, sigma, self.k)
                width_pct = (state.upper_today - state.lower_today) / state.mid_today if state.mid_today else None
                if width_pct is None or width_pct < self.width_pct_min or width_pct > self.width_pct_max:
                    base_record.update({
                        'anchor_date': anchor_date,
                        'anchor_price': anchor_price,
                        'beta': beta,
                        'sigma': sigma,
                        'mid_today': state.mid_today,
                        'upper_today': state.upper_today,
                        'lower_today': state.lower_today,
                        'channel_status': ChannelStatus.BROKEN.value,
                        'window_size': len(window_df),
                        'days_since_anchor': (current_date - anchor_date).days
                    })
                    history_data.append(base_record)
                    continue
                base_record.update({
                    'anchor_date': anchor_date,
                    'anchor_price': anchor_price,
                    'beta': beta,
                    'sigma': sigma,
                    'mid_today': state.mid_today,
                    'upper_today': state.upper_today,
                    'lower_today': state.lower_today,
                    'mid_tomorrow': state.mid_tomorrow,
                    'upper_tomorrow': state.upper_tomorrow,
                    'lower_tomorrow': state.lower_tomorrow,
                    'channel_status': ChannelStatus.NORMAL.value,
                    'window_size': len(window_df),
                    'days_since_anchor': (current_date - anchor_date).days,
                    'cumulative_gain': (current_close - anchor_price) / anchor_price
                })
                history_data.append(base_record)
        finally:
            logging.getLogger().setLevel(original_level)
        if not history_data:
            raise InsufficientDataError("无法计算任何历史通道数据")
        history_df = pd.DataFrame(history_data)
        return history_df

    def update_channel_history_incremental(self, history_df: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        增量更新历史通道数据
        
        基于已有的历史通道数据，添加新的数据并更新通道状态，避免重新计算
        
        Args:
            history_df (pd.DataFrame): 已有的历史通道数据
            new_data (pd.DataFrame): 新的数据，包含 trade_date, open, high, low, close, volume
            
        Returns:
            pd.DataFrame: 更新后的历史通道数据
        """
        if history_df.empty:
            raise ValueError("历史通道数据不能为空")
        
        # 确保数据按时间排序
        new_data = new_data.sort_values('trade_date').reset_index(drop=True)
        new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])
        
        # 获取最后一条历史记录的状态
        last_record = history_df.iloc[-1]
        
        # 检查最后一条记录是否有效
        if pd.isna(last_record['beta']):
            logger.warning("最后一条历史记录无效，无法进行增量更新")
            return history_df
        
        # 重建最后的状态对象
        last_state = ChannelState(
            anchor_date=pd.to_datetime(last_record['anchor_date']),
            anchor_price=last_record['anchor_price'],
            window_df=pd.DataFrame(),  # 临时占位，将在下面重建
            beta=last_record['beta'],
            sigma=last_record['sigma'],
            mid_today=last_record['mid_today'],
            upper_today=last_record['upper_today'],
            lower_today=last_record['lower_today'],
            mid_tomorrow=last_record['mid_tomorrow'],
            upper_tomorrow=last_record['upper_tomorrow'],
            lower_tomorrow=last_record['lower_tomorrow']
        )
        
        # 设置其他状态
        last_state.break_cnt_up = last_record['break_cnt_up']
        last_state.break_cnt_down = last_record['break_cnt_down']
        last_state.reanchor_fail_up = last_record['reanchor_fail_up']
        last_state.reanchor_fail_down = last_record['reanchor_fail_down']
        last_state.cumulative_gain = last_record['cumulative_gain']
        last_state.channel_status = ChannelStatus(last_record['channel_status'])
        last_state.last_update = pd.to_datetime(last_record['trade_date'])
        
        # 重建窗口数据 - 基于历史数据重建
        # 获取历史数据中最后window_size天的数据
        window_size = last_record['window_size']
        if pd.isna(window_size) or window_size <= 0:
            logger.warning("无法重建窗口数据，window_size无效")
            return history_df
        
        # 从历史数据中重建窗口
        # 这里我们需要原始的价格数据来重建窗口
        # 由于无法完全重建，我们使用一个简化的方法
        # 在实际使用中，建议使用 update_channel_history_with_state 方法
        
        # 存储新的历史记录
        new_history_data = []
        
        # 临时禁用某些日志输出
        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            # 逐日更新
            for _, row in new_data.iterrows():
                try:
                    # 准备新的K线数据
                    new_bar = {
                        'trade_date': row['trade_date'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    
                    # 使用update方法更新状态
                    updated_info = self.update(last_state, new_bar)
                    
                    # 更新当前状态
                    last_state = self.state
                    
                    # 构建新的历史记录
                    new_record = {
                        'trade_date': row['trade_date'],
                        'close': row['close'],
                        'beta': last_state.beta,
                        'sigma': last_state.sigma,
                        'mid_today': last_state.mid_today,
                        'upper_today': last_state.upper_today,
                        'lower_today': last_state.lower_today,
                        'mid_tomorrow': last_state.mid_tomorrow,
                        'upper_tomorrow': last_state.upper_tomorrow,
                        'lower_tomorrow': last_state.lower_tomorrow,
                        'channel_status': last_state.channel_status.value,
                        'anchor_date': last_state.anchor_date,
                        'anchor_price': last_state.anchor_price,
                        'break_cnt_up': last_state.break_cnt_up,
                        'break_cnt_down': last_state.break_cnt_down,
                        'reanchor_fail_up': last_state.reanchor_fail_up,
                        'reanchor_fail_down': last_state.reanchor_fail_down,
                        'cumulative_gain': last_state.cumulative_gain,
                        'window_size': len(last_state.window_df),
                        'days_since_anchor': (row['trade_date'] - last_state.anchor_date).days
                    }
                    
                    new_history_data.append(new_record)
                    
                except Exception as e:
                    # 如果某个时点更新失败，记录日志并保留基础记录
                    logger.warning(f"增量更新时点 {row['trade_date']} 失败: {e}")
                    
                    # 构建基础记录（保留价格信息，通道数据为None）
                    base_record = {
                        'trade_date': row['trade_date'],
                        'close': row['close'],
                        'beta': None,
                        'sigma': None,
                        'mid_today': None,
                        'upper_today': None,
                        'lower_today': None,
                        'mid_tomorrow': None,
                        'upper_tomorrow': None,
                        'lower_tomorrow': None,
                        'channel_status': None,
                        'anchor_date': None,
                        'anchor_price': None,
                        'break_cnt_up': None,
                        'break_cnt_down': None,
                        'reanchor_fail_up': None,
                        'reanchor_fail_down': None,
                        'cumulative_gain': None,
                        'window_size': None,
                        'days_since_anchor': None
                    }
                    
                    new_history_data.append(base_record)
                    
                    # 重置状态以便重新建立
                    last_state = None
                    continue
        finally:
            # 恢复日志级别
            logging.getLogger().setLevel(original_level)
        
        # 合并历史数据和新数据
        if new_history_data:
            new_history_df = pd.DataFrame(new_history_data)
            updated_history_df = pd.concat([history_df, new_history_df], ignore_index=True)
            
            logger.info(f"增量更新完成: 添加了 {len(new_history_data)} 条新记录")
            return updated_history_df
        else:
            logger.warning("增量更新失败: 没有成功添加新记录")
            return history_df
    
    def update_channel_history_with_state(self, history_df: pd.DataFrame, 
                                         last_state: ChannelState, 
                                         new_data: pd.DataFrame) -> pd.DataFrame:
        """
        基于完整状态对象的增量更新（推荐使用）
        
        需要保存完整的ChannelState对象，可以更准确地重建状态
        
        Args:
            history_df (pd.DataFrame): 已有的历史通道数据
            last_state (ChannelState): 最后的状态对象
            new_data (pd.DataFrame): 新的数据
            
        Returns:
            pd.DataFrame: 更新后的历史通道数据
        """
        if history_df.empty:
            raise ValueError("历史通道数据不能为空")
        
        if last_state is None:
            raise ValueError("最后状态对象不能为空")
        
        # 确保数据按时间排序
        new_data = new_data.sort_values('trade_date').reset_index(drop=True)
        new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])
        
        # 存储新的历史记录
        new_history_data = []
        
        # 使用最后的状态作为起始状态
        current_state = last_state
        
        # 逐日更新
        for _, row in new_data.iterrows():
            try:
                # 准备新的K线数据
                new_bar = {
                    'trade_date': row['trade_date'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                
                # 使用update方法更新状态
                updated_info = self.update(current_state, new_bar)
                
                # 更新当前状态
                current_state = self.state
                
                # 构建新的历史记录
                new_record = {
                    'trade_date': row['trade_date'],
                    'close': row['close'],
                    'beta': current_state.beta,
                    'sigma': current_state.sigma,
                    'mid_today': current_state.mid_today,
                    'upper_today': current_state.upper_today,
                    'lower_today': current_state.lower_today,
                    'mid_tomorrow': current_state.mid_tomorrow,
                    'upper_tomorrow': current_state.upper_tomorrow,
                    'lower_tomorrow': current_state.lower_tomorrow,
                    'channel_status': current_state.channel_status.value,
                    'anchor_date': current_state.anchor_date,
                    'anchor_price': current_state.anchor_price,
                    'break_cnt_up': current_state.break_cnt_up,
                    'break_cnt_down': current_state.break_cnt_down,
                    'reanchor_fail_up': current_state.reanchor_fail_up,
                    'reanchor_fail_down': current_state.reanchor_fail_down,
                    'cumulative_gain': current_state.cumulative_gain,
                    'window_size': len(current_state.window_df),
                    'days_since_anchor': (row['trade_date'] - current_state.anchor_date).days
                }
                
                new_history_data.append(new_record)
                
            except Exception as e:
                logger.warning(f"增量更新时点 {row['trade_date']} 失败: {e}")
                continue
        
        # 合并历史数据和新数据
        if new_history_data:
            new_history_df = pd.DataFrame(new_history_data)
            updated_history_df = pd.concat([history_df, new_history_df], ignore_index=True)
            
            logger.info(f"增量更新完成: 添加了 {len(new_history_data)} 条新记录")
            return updated_history_df
        else:
            logger.warning("增量更新失败: 没有成功添加新记录")
            return history_df

    def update_channel_history_incremental_improved(self, history_df: pd.DataFrame, 
                                                   original_df: pd.DataFrame, 
                                                   new_data: pd.DataFrame) -> pd.DataFrame:
        """
        改进的增量更新历史通道数据（推荐使用）
        
        基于原始数据和已有历史数据，正确重建状态并进行增量更新
        
        Args:
            history_df (pd.DataFrame): 已有的历史通道数据
            original_df (pd.DataFrame): 原始价格数据（用于重建状态）
            new_data (pd.DataFrame): 新的数据，包含 trade_date, open, high, low, close, volume
            
        Returns:
            pd.DataFrame: 更新后的历史通道数据
        """
        if history_df.empty:
            raise ValueError("历史通道数据不能为空")
        
        # 确保数据按时间排序
        original_df = original_df.sort_values('trade_date').reset_index(drop=True)
        original_df['trade_date'] = pd.to_datetime(original_df['trade_date'])
        
        new_data = new_data.sort_values('trade_date').reset_index(drop=True)
        new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])
        
        # 获取最后一条历史记录的状态
        last_record = history_df.iloc[-1]
        
        # 检查最后一条记录是否有效
        if pd.isna(last_record['beta']):
            logger.warning("最后一条历史记录无效，无法进行增量更新")
            return history_df
        
        # 重建最后的状态对象
        last_state = ChannelState(
            anchor_date=pd.to_datetime(last_record['anchor_date']),
            anchor_price=last_record['anchor_price'],
            window_df=pd.DataFrame(),  # 临时占位
            beta=last_record['beta'],
            sigma=last_record['sigma'],
            mid_today=last_record['mid_today'],
            upper_today=last_record['upper_today'],
            lower_today=last_record['lower_today'],
            mid_tomorrow=last_record['mid_tomorrow'],
            upper_tomorrow=last_record['upper_tomorrow'],
            lower_tomorrow=last_record['lower_tomorrow']
        )
        
        # 设置其他状态
        last_state.break_cnt_up = last_record['break_cnt_up']
        last_state.break_cnt_down = last_record['break_cnt_down']
        last_state.reanchor_fail_up = last_record['reanchor_fail_up']
        last_state.reanchor_fail_down = last_record['reanchor_fail_down']
        last_state.cumulative_gain = last_record['cumulative_gain']
        last_state.channel_status = ChannelStatus(last_record['channel_status'])
        last_state.last_update = pd.to_datetime(last_record['trade_date'])
        
        # 重建窗口数据 - 从原始数据中重建
        window_size = last_record['window_size']
        if pd.isna(window_size) or window_size <= 0:
            logger.warning("无法重建窗口数据，window_size无效")
            return history_df
        
        # 从原始数据中重建窗口
        # 找到最后一条历史记录对应的原始数据位置
        last_history_date = last_record['trade_date']
        last_history_idx = original_df[original_df['trade_date'] == last_history_date].index
        
        if len(last_history_idx) == 0:
            logger.warning(f"无法在原始数据中找到历史记录日期: {last_history_date}")
            return history_df
        
        last_history_idx = last_history_idx[0]
        
        # 重建窗口数据 - 取最后window_size天的数据
        # 这样可以确保窗口大小正确
        start_idx = max(0, last_history_idx - window_size + 1)
        window_df = original_df.iloc[start_idx:last_history_idx + 1].copy()
        
        if len(window_df) != window_size:
            logger.warning(f"窗口大小不匹配: 期望 {window_size}, 实际 {len(window_df)}")
            # 使用实际重建的窗口大小
            window_size = len(window_df)
        
        last_state.window_df = window_df
        
        # 存储新的历史记录
        new_history_data = []
        
        # 临时禁用某些日志输出
        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            # 逐日更新
            for _, row in new_data.iterrows():
                try:
                    # 准备新的K线数据
                    new_bar = {
                        'trade_date': row['trade_date'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    
                    # 使用update方法更新状态
                    updated_info = self.update(last_state, new_bar)
                    
                    # 更新当前状态
                    last_state = self.state
                    
                    # 构建新的历史记录
                    new_record = {
                        'trade_date': row['trade_date'],
                        'close': row['close'],
                        'beta': last_state.beta,
                        'sigma': last_state.sigma,
                        'mid_today': last_state.mid_today,
                        'upper_today': last_state.upper_today,
                        'lower_today': last_state.lower_today,
                        'mid_tomorrow': last_state.mid_tomorrow,
                        'upper_tomorrow': last_state.upper_tomorrow,
                        'lower_tomorrow': last_state.lower_tomorrow,
                        'channel_status': last_state.channel_status.value,
                        'anchor_date': last_state.anchor_date,
                        'anchor_price': last_state.anchor_price,
                        'break_cnt_up': last_state.break_cnt_up,
                        'break_cnt_down': last_state.break_cnt_down,
                        'reanchor_fail_up': last_state.reanchor_fail_up,
                        'reanchor_fail_down': last_state.reanchor_fail_down,
                        'cumulative_gain': last_state.cumulative_gain,
                        'window_size': len(last_state.window_df),
                        'days_since_anchor': (row['trade_date'] - last_state.anchor_date).days
                    }
                    
                    new_history_data.append(new_record)
                    
                except Exception as e:
                    # 如果某个时点更新失败，记录日志并保留基础记录
                    logger.warning(f"增量更新时点 {row['trade_date']} 失败: {e}")
                    
                    # 构建基础记录（保留价格信息，通道数据为None）
                    base_record = {
                        'trade_date': row['trade_date'],
                        'close': row['close'],
                        'beta': None,
                        'sigma': None,
                        'mid_today': None,
                        'upper_today': None,
                        'lower_today': None,
                        'mid_tomorrow': None,
                        'upper_tomorrow': None,
                        'lower_tomorrow': None,
                        'channel_status': None,
                        'anchor_date': None,
                        'anchor_price': None,
                        'break_cnt_up': None,
                        'break_cnt_down': None,
                        'reanchor_fail_up': None,
                        'reanchor_fail_down': None,
                        'cumulative_gain': None,
                        'window_size': None,
                        'days_since_anchor': None
                    }
                    
                    new_history_data.append(base_record)
                    
                    # 重置状态以便重新建立
                    last_state = None
                    continue
        finally:
            # 恢复日志级别
            logging.getLogger().setLevel(original_level)
        
        # 合并历史数据和新数据
        if new_history_data:
            new_history_df = pd.DataFrame(new_history_data)
            updated_history_df = pd.concat([history_df, new_history_df], ignore_index=True)
            
            logger.info(f"改进增量更新完成: 添加了 {len(new_history_data)} 条新记录")
            return updated_history_df
        else:
            logger.warning("改进增量更新失败: 没有成功添加新记录")
            return history_df


# 因子注册装饰器
def register_ascending_channel_factor(func):
    """注册上升通道因子"""
    func.__factor_name__ = 'ascending_channel_regression'
    func.__factor_type__ = 'channel_analysis'
    return func


@register_ascending_channel_factor
def ascending_channel_regression(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    上升通道回归因子
    
    Args:
        df (pd.DataFrame): 价格数据
        **kwargs: 其他参数
        
    Returns:
        pd.Series: 因子值序列
    """
    # 这里可以实现批量计算逻辑
    # 暂时返回空序列
    return pd.Series(index=df.index, dtype=float)
