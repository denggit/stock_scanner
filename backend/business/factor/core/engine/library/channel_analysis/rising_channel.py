#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : AI Assistant
@Date       : 2024-12-19
@File       : rising_channel.py
@Description: 上升通道回归模块

该模块实现了上升通道回归分析算法，主要功能包括：
1. 自动检测和维护上升回归通道
2. 提供通道突破和重锚机制
3. 支持多种通道状态监控
4. 实时计算通道边界和预测值

算法流程：
初始化 → 找首个 anchor → 每日更新 → 检查重锚触发 → 状态管理

重构版本：使用策略模式和模板方法模式优化代码结构，消除冗余
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from backend.utils.logger import setup_logger
from .channel_state import ChannelState, ChannelStatus
from .pivot_detector import PivotDetector

logger = setup_logger(__name__)


@dataclass
class ChannelCalculationResult:
    """通道计算结果封装"""
    is_valid: bool
    anchor_date: Optional[pd.Timestamp]
    anchor_price: Optional[float]
    window_df: pd.DataFrame
    beta: Optional[float]
    sigma: Optional[float]
    r2: Optional[float]
    state: Optional[ChannelState]
    break_reason: Optional[str] = None


class ChannelCalculationStrategy(ABC):
    """通道计算策略抽象基类"""

    @abstractmethod
    def calculate_for_date(self, df: pd.DataFrame, current_date: pd.Timestamp,
                           current_close: float, config: Dict[str, Any]) -> ChannelCalculationResult:
        """为特定日期计算通道"""
        pass


class StandardChannelStrategy(ChannelCalculationStrategy):
    """标准通道计算策略"""

    def __init__(self, pivot_detector: PivotDetector):
        self.pivot_detector = pivot_detector

    def calculate_for_date(self, df: pd.DataFrame, current_date: pd.Timestamp,
                           current_close: float, config: Dict[str, Any]) -> ChannelCalculationResult:
        """实现标准的v0.2通道计算逻辑"""
        # 1. 检查数据量
        if len(df) < config['min_data_points']:
            return ChannelCalculationResult(
                is_valid=False, anchor_date=None, anchor_price=None,
                window_df=pd.DataFrame(), beta=None, sigma=None, r2=None, state=None,
                break_reason="insufficient_data"
            )

        # 2. 全局pivot_low锚点查找
        search_df = df.tail(config['L_max'])
        pivots = self.pivot_detector.get_anchor_candidates(search_df, n_candidates=20)
        today = search_df['trade_date'].iloc[-1]
        valid_pivots = [(d, p) for d, p in pivots if (today - pd.to_datetime(d)).days >= config['min_data_points']]

        if not valid_pivots:
            return ChannelCalculationResult(
                is_valid=False, anchor_date=None, anchor_price=None,
                window_df=pd.DataFrame(), beta=None, sigma=None, r2=None, state=None,
                break_reason="no_valid_anchor"
            )

        # 3. 选择最早最低锚点
        anchor_date, anchor_price = min(valid_pivots, key=lambda x: x[1])
        anchor_date = pd.to_datetime(anchor_date)
        window_df = df[df['trade_date'] >= anchor_date]

        # 4. 回归计算
        beta, sigma, r2 = self._calculate_regression(window_df, anchor_date)

        # 5. 回归有效性检查（包括斜率为负的检查）
        if beta is None or beta <= 0 or r2 < config['R2_min']:
            return ChannelCalculationResult(
                is_valid=False, anchor_date=anchor_date, anchor_price=anchor_price,
                window_df=window_df, beta=beta, sigma=sigma, r2=r2, state=None,
                break_reason="invalid_regression"
            )

        # 6. 构建通道状态
        state = ChannelState(
            anchor_date=anchor_date, anchor_price=anchor_price, window_df=window_df,
            beta=beta, sigma=sigma, last_update=current_date,
            mid_today=0.0, upper_today=0.0, lower_today=0.0,
            mid_tomorrow=0.0, upper_tomorrow=0.0, lower_tomorrow=0.0
        )
        state.update_channel_boundaries(beta, sigma, config['k'])

        # 基于斜率和价格位置的状态判定
        if beta <= 0:
            # 斜率为负，标记为OTHER
            state.channel_status = ChannelStatus.OTHER
        else:
            # 斜率为正，根据价格位置判定状态
            if current_close > state.upper_today:
                state.channel_status = ChannelStatus.BREAKOUT
            elif current_close < state.lower_today:
                state.channel_status = ChannelStatus.BREAKDOWN
            else:
                state.channel_status = ChannelStatus.NORMAL

        # 7. 通道宽度检查（保持：宽度无效则记为无效，但带回已计算状态与边界）
        width_pct = (state.upper_today - state.lower_today) / state.mid_today if state.mid_today else None
        if width_pct is None or width_pct < config['width_pct_min'] or width_pct > config['width_pct_max']:
            return ChannelCalculationResult(
                is_valid=False, anchor_date=anchor_date, anchor_price=anchor_price,
                window_df=window_df, beta=beta, sigma=sigma, r2=r2, state=state,
                break_reason="invalid_width"
            )

        # 8. 设置其他字段
        state.cumulative_gain = (current_close - anchor_price) / anchor_price
        state.r2 = r2

        return ChannelCalculationResult(
            is_valid=True, anchor_date=anchor_date, anchor_price=anchor_price,
            window_df=window_df, beta=beta, sigma=sigma, r2=r2, state=state
        )

    def _calculate_regression(self, df: pd.DataFrame, anchor_date: pd.Timestamp):
        """计算线性回归参数，返回(beta, sigma, r2)"""
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


class AscendingChannelRegression:
    """
    上升通道回归分析类 - 重构版
    
    使用策略模式和模板方法模式优化的上升通道回归算法
    """

    def __init__(self, config_path: Optional[str] = None, **params):
        """初始化上升通道回归分析器"""
        # 配置加载与参数初始化
        self.config = self._load_config(config_path)
        self._init_parameters(params)

        # 组件初始化
        self.pivot_detector = PivotDetector(pivot_m=self.pivot_m)
        self.strategy = StandardChannelStrategy(self.pivot_detector)

        # 状态管理
        self.state: Optional[ChannelState] = None

        logger.info("上升通道回归分析器初始化完成")

    def _init_parameters(self, params: Dict[str, Any]) -> None:
        """初始化参数"""
        config_section = self.config['ascending_channel']
        self.k = params.get('k', config_section['k'])
        self.L_max = params.get('L_max', config_section['L_max'])
        self.delta_cut = params.get('delta_cut', config_section['delta_cut'])
        self.pivot_m = params.get('pivot_m', config_section['pivot_m'])
        self.min_data_points = params.get('min_data_points', config_section['min_data_points'])
        self.R2_min = params.get('R2_min', config_section.get('R2_min', 0.35))
        self.width_pct_min = params.get('width_pct_min', config_section.get('width_pct_min', 0.04))
        self.width_pct_max = params.get('width_pct_max', config_section.get('width_pct_max', 0.12))

    def _get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            'k': self.k,
            'L_max': self.L_max,
            'delta_cut': self.delta_cut,
            'pivot_m': self.pivot_m,
            'min_data_points': self.min_data_points,
            'R2_min': self.R2_min,
            'width_pct_min': self.width_pct_min,
            'width_pct_max': self.width_pct_max
        }

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "channel_config.yaml"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'ascending_channel': {
                'k': 2.0,
                'L_max': 120,
                'delta_cut': 5,
                'pivot_m': 3,
                'min_data_points': 60,
                'R2_min': 0.6,
                'width_pct_min': 0.04,
                'width_pct_max': 0.15
            }
        }

    def fit_channel(self, df: pd.DataFrame) -> ChannelState:
        """
        拟合上升通道（单点计算）
        
        该方法负责计算上升通道并完成状态判定，包括：
        1. 通道计算
        2. 质量检查（R²、通道宽度等）
        3. 状态判定（NORMAL、BREAKOUT、BREAKDOWN、ASCENDING_WEAK、OTHER）
        
        Args:
            df (pd.DataFrame): 股票数据
            
        Returns:
            ChannelState: 通道状态对象
        """
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        current_date = df.iloc[-1]['trade_date']
        current_close = df.iloc[-1]['close']

        # 使用策略计算
        result = self.strategy.calculate_for_date(df, current_date, current_close, self._get_config_dict())

        if not result.is_valid:
            # 处理无效结果
            if result.state is not None:
                # 若返回了state（如invalid_width），需要重新判定状态
                return self._determine_channel_status(result.state, current_close, result.r2, result.break_reason)
            else:
                # 构造一个最小可用状态并标记为 OTHER
                cumulative_gain = None
                if result.anchor_price is not None and result.anchor_price > 0:
                    cumulative_gain = (current_close - result.anchor_price) / result.anchor_price
                state = ChannelState(
                    anchor_date=result.anchor_date, anchor_price=result.anchor_price,
                    window_df=result.window_df if result.window_df is not None else pd.DataFrame(),
                    beta=result.beta if result.beta is not None else 0.0,
                    sigma=result.sigma if result.sigma is not None else 0.0,
                    mid_today=0.0, upper_today=0.0, lower_today=0.0,
                    mid_tomorrow=0.0, upper_tomorrow=0.0, lower_tomorrow=0.0,
                    r2=result.r2,
                    channel_status=ChannelStatus.OTHER,
                    cumulative_gain=cumulative_gain
                )
                return state

        # 有效结果，进行状态判定
        return self._determine_channel_status(result.state, current_close, result.r2, None)

    def _determine_channel_status(self, state: ChannelState, current_close: float,
                                  r2: Optional[float], break_reason: Optional[str]) -> ChannelState:
        """
        确定通道状态
        
        按照优先级顺序确定通道状态：
        1. 如果斜率为负：OTHER（不是上升通道）
        2. 如果斜率为正：
           a) 检查通道质量（宽度和R²）
           b) 如果质量不达标：ASCENDING_WEAK
           c) 如果质量达标：根据价格位置判定 NORMAL/BREAKOUT/BREAKDOWN
        
        Args:
            state (ChannelState): 通道状态对象
            current_close (float): 当前收盘价
            r2 (Optional[float]): 回归拟合优度
            break_reason (Optional[str]): 失效原因
            
        Returns:
            ChannelState: 更新后的通道状态对象
        """
        # 1. 首先判断斜率方向
        if state.beta is None or state.beta <= 0:
            # 斜率为负，标记为OTHER状态
            state.channel_status = ChannelStatus.OTHER
            return state

        # 2. 斜率为正，检查通道质量
        config = self._get_config_dict()
        quality_check_passed = self._check_channel_quality(state, r2, config)

        if not quality_check_passed:
            # 质量检查不通过，标记为弱上升通道
            state.channel_status = ChannelStatus.ASCENDING_WEAK
            return state

        # 3. 质量检查通过，根据价格位置判定状态
        if current_close > state.upper_today:
            # 股价在通道上沿以上
            state.channel_status = ChannelStatus.BREAKOUT
        elif current_close < state.lower_today:
            # 股价在通道下沿以下
            state.channel_status = ChannelStatus.BREAKDOWN
        else:
            # 股价在通道内
            state.channel_status = ChannelStatus.NORMAL

        return state

    def _check_channel_quality(self, state: ChannelState, r2: Optional[float],
                               config: Dict[str, Any]) -> bool:
        """
        检查通道质量
        
        检查R²和通道宽度是否满足要求
        
        Args:
            state (ChannelState): 通道状态对象
            r2 (Optional[float]): 回归拟合优度
            config (Dict[str, Any]): 配置参数
            
        Returns:
            bool: 质量检查是否通过
        """
        # 检查R²
        r2_min = config.get('R2_min', 0.6)
        r2_max = config.get('R2_max', 0.95)
        if r2 is None or r2 < r2_min or r2 > r2_max:
            return False

        # 检查通道宽度
        width_pct_min = config.get('width_pct_min', 0.04)
        width_pct_max = config.get('width_pct_max', 0.12)

        if state.mid_today and state.mid_today > 0:
            width_pct = (state.upper_today - state.lower_today) / state.mid_today
            if width_pct < width_pct_min or width_pct > width_pct_max:
                return False
        else:
            return False

        return True

    # 保留原有的动态更新相关方法，确保向后兼容
    def update(self, state: ChannelState, bar: Dict[str, Any]) -> Dict[str, Any]:
        """更新通道状态"""
        if state is None:
            raise ValueError("通道状态不能为空")

        self.state = state
        new_row = pd.DataFrame([bar])
        self.state.window_df = pd.concat([self.state.window_df, new_row], ignore_index=True)
        self.state.last_update = pd.to_datetime(bar['trade_date'])

        if len(self.state.window_df) > self.L_max:
            self._slide_window()

        beta, sigma, r2 = self._calculate_regression(self.state.window_df, self.state.anchor_date)
        self.state.update_channel_boundaries(beta, sigma, self.k)
        self.state.r2 = r2  # 更新r2字段
        self.state.cumulative_gain = (bar['close'] - self.state.anchor_price) / self.state.anchor_price

        # 基于斜率和质量检查的状态判定（即时）
        if self.state.beta <= 0:
            # 斜率为负，标记为OTHER
            self.state.channel_status = ChannelStatus.OTHER
        else:
            # 斜率为正，检查通道质量
            config = self._get_config_dict()
            quality_check_passed = self._check_channel_quality(self.state, r2, config)

            if not quality_check_passed:
                # 质量检查不通过，标记为弱上升通道
                self.state.channel_status = ChannelStatus.ASCENDING_WEAK
            else:
                # 质量检查通过，根据价格位置判定状态
                if bar['close'] > self.state.upper_today:
                    self.state.channel_status = ChannelStatus.BREAKOUT
                elif bar['close'] < self.state.lower_today:
                    self.state.channel_status = ChannelStatus.BREAKDOWN
                else:
                    self.state.channel_status = ChannelStatus.NORMAL

        if self._should_reanchor():
            self._reanchor()

        return self.state.to_dict()

    def _calculate_regression(self, df: pd.DataFrame, anchor_date: pd.Timestamp):
        """计算线性回归参数"""
        return self.strategy._calculate_regression(df, anchor_date)

    def _slide_window(self) -> None:
        """滑动窗口"""
        self.state.window_df = self.state.window_df.iloc[self.delta_cut:].reset_index(drop=True)
        logger.debug(f"窗口滑动，删除最早 {self.delta_cut} 天数据")

    def _should_reanchor(self) -> bool:
        """检查是否应该重锚"""
        if not self.state.can_reanchor():
            return False
        # 仅基于窗口长度判断是否重锚
        return len(self.state.window_df) > self.L_max

    def _reanchor(self) -> None:
        """执行重锚操作"""
        logger.info("开始重锚操作(pivot_low)")
        new_anchor = self.pivot_detector.find_new_anchor(
            self.state.window_df, self.state.anchor_date, strategy='pivot_low'
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
        self.state.r2 = r2  # 更新r2字段
        self.state.update_channel_boundaries(beta, sigma, self.k)
        self.state.reset_break_counters()
        self.state.reanchor_fail_up = 0
        self.state.reanchor_fail_down = 0

        # 重锚后重新判定状态（基于新的斜率和价格位置）
        # 注意：这里需要当前价格，但重锚时可能没有，所以暂时保持原有逻辑
        # 实际使用时会在下一次update时重新判定状态

        logger.info(
            f"重锚成功: 新锚点 {new_anchor_date} @ {new_anchor_price:.2f}, 窗口大小: {len(self.state.window_df)}")


# 因子注册装饰器
def register_ascending_channel_factor(func):
    """注册上升通道因子"""
    func.__factor_name__ = 'ascending_channel_regression'
    func.__factor_type__ = 'channel_analysis'
    return func


@register_ascending_channel_factor
def ascending_channel_regression(df: pd.DataFrame, **kwargs) -> pd.Series:
    """上升通道回归因子"""
    return pd.Series(index=df.index, dtype=float)
