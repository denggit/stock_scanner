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

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化上升通道回归分析器
        
        Args:
            config_path (Optional[str]): 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化参数
        self.k = self.config['ascending_channel']['k']
        self.L_max = self.config['ascending_channel']['L_max']
        self.delta_cut = self.config['ascending_channel']['delta_cut']
        self.pivot_m = self.config['ascending_channel']['pivot_m']
        self.gain_trigger = self.config['ascending_channel']['gain_trigger']
        self.beta_delta = self.config['ascending_channel']['beta_delta']
        self.break_days = self.config['ascending_channel']['break_days']
        self.reanchor_fail_max = self.config['ascending_channel']['reanchor_fail_max']
        self.min_data_points = self.config['ascending_channel']['min_data_points']

        # 初始化组件
        self.pivot_detector = PivotDetector(pivot_m=self.pivot_m)

        # 状态对象
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
        拟合上升通道
        
        Args:
            df (pd.DataFrame): 价格数据，包含 trade_date, open, high, low, close, volume
            
        Returns:
            ChannelState: 通道状态对象
            
        Raises:
            InsufficientDataError: 数据不足时抛出
        """
        # 数据验证
        if len(df) < self.min_data_points:
            raise InsufficientDataError(
                f"数据不足，至少需要 {self.min_data_points} 个数据点，当前只有 {len(df)} 个"
            )

        # 确保数据按时间排序
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 查找初始锚点
        anchor_result = self.pivot_detector.find_initial_anchor(df, strategy='pivot_low')
        if anchor_result is None:
            raise InsufficientDataError("无法找到有效的初始锚点")

        anchor_date, anchor_price = anchor_result

        # 创建窗口数据（从锚点到最新）
        window_df = df[df['trade_date'] >= anchor_date].copy()

        # 计算初始回归参数
        beta, sigma = self._calculate_regression(window_df, anchor_date)

        # 创建状态对象
        self.state = ChannelState(
            anchor_date=anchor_date,
            anchor_price=anchor_price,
            window_df=window_df,
            beta=beta,
            sigma=sigma,
            mid_today=0.0,  # 将在 update_channel_boundaries 中计算
            upper_today=0.0,
            lower_today=0.0,
            mid_tomorrow=0.0,
            upper_tomorrow=0.0,
            lower_tomorrow=0.0
        )

        # 更新通道边界
        self.state.update_channel_boundaries(beta, sigma, self.k)

        # 计算累计涨幅
        self.state.cumulative_gain = (window_df['close'].iloc[-1] - anchor_price) / anchor_price

        logger.info(f"通道拟合完成: 锚点 {anchor_date} @ {anchor_price:.2f}, "
                    f"β={beta:.4f}, σ={sigma:.4f}, 涨幅={self.state.cumulative_gain:.2%}")

        return self.state

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
        beta, sigma = self._calculate_regression(self.state.window_df, self.state.anchor_date)

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

    def _calculate_regression(self, df: pd.DataFrame, anchor_date: pd.Timestamp) -> Tuple[float, float]:
        """
        计算线性回归参数
        
        Args:
            df (pd.DataFrame): 窗口数据
            anchor_date (pd.Timestamp): 锚点日期
        
        Returns:
            Tuple[float, float]: (斜率, 标准差)
        """
        # 创建时间序列（以天为单位）
        dates = pd.to_datetime(df['trade_date'])
        anchor_date = pd.to_datetime(anchor_date)  # 修复：确保类型一致
        days_since_anchor = (dates - anchor_date).dt.days

        # 使用收盘价进行回归
        prices = df['close'].values

        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(days_since_anchor, prices)

        # 计算残差标准差
        predicted = slope * days_since_anchor + intercept
        residuals = prices - predicted
        sigma = np.std(residuals)

        return slope, sigma

    def _slide_window(self) -> None:
        """滑动窗口，删除最早的数据"""
        # 删除最早的 delta_cut 天数据
        self.state.window_df = self.state.window_df.iloc[self.delta_cut:].reset_index(drop=True)
        logger.debug(f"窗口滑动，删除最早 {self.delta_cut} 天数据")

    def _should_reanchor(self) -> bool:
        """
        检查是否应该重锚
        
        Returns:
            bool: 是否应该重锚
        """
        if not self.state.can_reanchor():
            return False

        # 检查各种重锚触发条件
        conditions = [
            # 1. 窗口超长
            len(self.state.window_df) > self.L_max,

            # 2. 累计涨幅触发
            self.state.cumulative_gain >= self.gain_trigger,

            # 3. 连续突破
            self.state.break_cnt_up >= self.break_days or self.state.break_cnt_down >= self.break_days,

            # 4. 斜率变化（需要历史数据）
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
        """执行重锚操作"""
        logger.info("开始重锚操作")

        # 查找新的锚点
        new_anchor = self.pivot_detector.find_new_anchor(
            self.state.window_df,
            self.state.anchor_date,
            strategy='pivot_low'
        )

        if new_anchor is None:
            logger.warning("重锚失败：未找到新的有效锚点")
            return

        new_anchor_date, new_anchor_price = new_anchor

        # 更新锚点信息
        self.state.anchor_date = new_anchor_date
        self.state.anchor_price = new_anchor_price

        # 更新窗口数据
        self.state.window_df = self.state.window_df[
            self.state.window_df['trade_date'] >= new_anchor_date
            ].reset_index(drop=True)

        # 重置计数器
        self.state.reset_break_counters()

        logger.info(f"重锚成功: 新锚点 {new_anchor_date} @ {new_anchor_price:.2f}")

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
