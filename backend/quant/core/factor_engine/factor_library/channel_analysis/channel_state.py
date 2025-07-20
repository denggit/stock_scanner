#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : AI Assistant
@Date       : 2024-12-19
@File       : channel_state.py
@Description: 通道状态管理类

该类负责管理上升通道回归分析的状态信息，包括：
1. 锚点信息（日期和价格）
2. 窗口数据
3. 回归参数（斜率和标准差）
4. 通道边界（上沿、中轴、下沿）
5. 突破计数器
6. 通道状态
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ChannelStatus(Enum):
    """通道状态枚举"""
    NORMAL = "NORMAL"              # 通道正常
    ACCEL_BREAKOUT = "ACCEL_BREAKOUT"  # 加速突破
    BREAKDOWN = "BREAKDOWN"        # 跌破下沿
    BROKEN = "BROKEN"              # 通道失效


@dataclass
class ChannelState:
    """
    通道状态数据类
    
    存储上升通道回归分析的所有状态信息
    """
    
    # 锚点信息
    anchor_date: pd.Timestamp
    anchor_price: float
    
    # 窗口数据
    window_df: pd.DataFrame
    
    # 回归参数
    beta: float  # 斜率 β_t
    sigma: float  # 标准差 σ_t
    
    # 通道边界（当前）
    mid_today: float
    upper_today: float
    lower_today: float
    
    # 通道边界（明日预测）
    mid_tomorrow: float
    upper_tomorrow: float
    lower_tomorrow: float
    
    # 突破计数器
    break_cnt_up: int = 0      # 连续突破上沿次数
    break_cnt_down: int = 0    # 连续突破下沿次数
    
    # 重锚失败计数器
    reanchor_fail_up: int = 0   # 重锚失败（上沿）
    reanchor_fail_down: int = 0 # 重锚失败（下沿）
    
    # 通道状态
    channel_status: ChannelStatus = ChannelStatus.NORMAL
    
    # 累计涨幅
    cumulative_gain: float = 0.0
    
    # 最后更新时间
    last_update: pd.Timestamp = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保anchor_date和last_update都是pd.Timestamp类型
        if not isinstance(self.anchor_date, pd.Timestamp):
            self.anchor_date = pd.to_datetime(self.anchor_date)
        if self.last_update is None:
            self.last_update = pd.Timestamp.now()
        elif not isinstance(self.last_update, pd.Timestamp):
            self.last_update = pd.to_datetime(self.last_update)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将状态转换为字典格式
        
        Returns:
            Dict[str, Any]: 状态字典
        """
        return {
            "beta": self.beta,
            "mid_today": self.mid_today,
            "upper_today": self.upper_today,
            "lower_today": self.lower_today,
            "mid_tomorrow": self.mid_tomorrow,
            "upper_tomorrow": self.upper_tomorrow,
            "lower_tomorrow": self.lower_tomorrow,
            "channel_status": self.channel_status.value,
            "anchor_date": pd.Timestamp(self.anchor_date).isoformat(),
            "anchor_price": self.anchor_price,
            "break_cnt_up": self.break_cnt_up,
            "break_cnt_down": self.break_cnt_down,
            "reanchor_fail_up": self.reanchor_fail_up,
            "reanchor_fail_down": self.reanchor_fail_down,
            "cumulative_gain": self.cumulative_gain,
            "last_update": pd.Timestamp(self.last_update).isoformat()
        }
    
    def update_break_counters(self, price: float, upper: float, lower: float) -> None:
        """
        更新突破计数器
        
        Args:
            price (float): 当前价格
            upper (float): 上沿价格
            lower (float): 下沿价格
        """
        if price > upper:
            self.break_cnt_up += 1
            self.break_cnt_down = 0  # 重置下沿计数器
        elif price < lower:
            self.break_cnt_down += 1
            self.break_cnt_up = 0    # 重置上沿计数器
        else:
            # 价格在通道内，重置计数器
            self.break_cnt_up = 0
            self.break_cnt_down = 0
    
    def reset_break_counters(self) -> None:
        """重置突破计数器"""
        self.break_cnt_up = 0
        self.break_cnt_down = 0
    
    def update_reanchor_fail_counters(self, direction: str) -> None:
        """
        更新重锚失败计数器
        
        Args:
            direction (str): 方向，"up" 或 "down"
        """
        if direction == "up":
            self.reanchor_fail_up += 1
        elif direction == "down":
            self.reanchor_fail_down += 1
    
    def reset_reanchor_fail_counters(self) -> None:
        """重置重锚失败计数器"""
        self.reanchor_fail_up = 0
        self.reanchor_fail_down = 0
    
    def update_channel_boundaries(self, beta: float, sigma: float, k: float) -> None:
        """
        更新通道边界
        """
        self.beta = beta
        self.sigma = sigma
        # 确保last_update和anchor_date类型一致
        if not isinstance(self.anchor_date, pd.Timestamp):
            self.anchor_date = pd.to_datetime(self.anchor_date)
        if not isinstance(self.last_update, pd.Timestamp):
            self.last_update = pd.to_datetime(self.last_update)
        self.mid_today = self._calculate_mid_price()
        self.upper_today = self.mid_today + k * sigma
        self.lower_today = self.mid_today - k * sigma
        self.mid_tomorrow = self.mid_today + beta
        self.upper_tomorrow = self.mid_tomorrow + k * sigma
        self.lower_tomorrow = self.mid_tomorrow - k * sigma
    
    def _calculate_mid_price(self) -> float:
        """
        计算中轴价格
        """
        # 确保last_update和anchor_date类型一致
        if not isinstance(self.anchor_date, pd.Timestamp):
            self.anchor_date = pd.to_datetime(self.anchor_date)
        if not isinstance(self.last_update, pd.Timestamp):
            self.last_update = pd.to_datetime(self.last_update)
        days_since_anchor = (self.last_update - self.anchor_date).days
        return self.anchor_price + self.beta * days_since_anchor
    
    def is_extreme_state(self) -> bool:
        """
        判断是否处于极端状态
        
        Returns:
            bool: 是否为极端状态
        """
        return self.channel_status in [
            ChannelStatus.ACCEL_BREAKOUT,
            ChannelStatus.BREAKDOWN,
            ChannelStatus.BROKEN
        ]
    
    def can_reanchor(self) -> bool:
        """
        判断是否可以重锚
        
        Returns:
            bool: 是否可以重锚
        """
        return not self.is_extreme_state()
    
    def get_status_description(self) -> str:
        """
        获取状态描述
        
        Returns:
            str: 状态描述
        """
        status_descriptions = {
            ChannelStatus.NORMAL: "通道正常",
            ChannelStatus.ACCEL_BREAKOUT: "加速突破",
            ChannelStatus.BREAKDOWN: "跌破下沿",
            ChannelStatus.BROKEN: "通道失效"
        }
        return status_descriptions.get(self.channel_status, "未知状态") 