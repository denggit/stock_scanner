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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

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

        # 5. 回归有效性检查
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

        # 基于当日价格的三态判定
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


class HistoryCalculationTemplate:
    """历史计算模板方法类"""

    def __init__(self, strategy: ChannelCalculationStrategy, config: Dict[str, Any]):
        self.strategy = strategy
        self.config = config

    def calculate_history(self, df: pd.DataFrame, min_window_size: int,
                          step_days: int = 1) -> pd.DataFrame:
        """模板方法：计算历史通道数据"""
        # 预处理
        df = self._preprocess_data(df, min_window_size)

        # 初始化结果容器
        history_data = []

        # 抑制日志
        with self._suppress_logs():
            # 主循环
            for i in range(min_window_size, len(df), step_days):
                current_df = df.iloc[:i + 1]
                current_date = current_df.iloc[-1]['trade_date']
                current_close = current_df.iloc[-1]['close']

                # 使用策略计算通道
                result = self.strategy.calculate_for_date(current_df, current_date, current_close, self.config)

                # 构建记录
                record = self._build_record_from_result(result, current_date, current_close)
                history_data.append(record)

        # 后处理
        return self._postprocess_results(history_data)

    def _preprocess_data(self, df: pd.DataFrame, min_window_size: int) -> pd.DataFrame:
        """预处理数据"""
        if len(df) < min_window_size + 20:
            raise InsufficientDataError(
                f"数据不足，至少需要 {min_window_size + 20} 个数据点，当前只有 {len(df)} 个"
            )
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    def _suppress_logs(self):
        """抑制日志输出上下文管理器"""

        class LogSuppressor:
            def __enter__(self):
                self.original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.CRITICAL)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                logging.getLogger().setLevel(self.original_level)

        return LogSuppressor()

    def _build_record_from_result(self, result: ChannelCalculationResult,
                                  current_date: pd.Timestamp, current_close: float) -> Dict[str, Any]:
        """从计算结果构建记录"""
        base_record = {
            'trade_date': current_date,
            'close': current_close,
            'beta': None,
            'sigma': None,
            'r2': None,  # 添加r2字段
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
            'days_since_anchor': None,
            'break_reason': None,  # 添加break_reason字段
            'width_pct': None,  # 添加通道宽度百分比
            'slope_deg': None,  # 添加斜率角度（度）
            'volatility': None  # 添加波动率
        }

        if not result.is_valid:
            # 无效通道
            if result.anchor_date is not None:
                # 计算累计涨幅（如果有anchor_price）
                cumulative_gain = None
                if result.anchor_price is not None and result.anchor_price > 0:
                    cumulative_gain = (current_close - result.anchor_price) / result.anchor_price
                base_record.update({
                    'anchor_date': result.anchor_date,
                    'anchor_price': result.anchor_price,
                    'beta': result.beta,
                    'sigma': result.sigma,
                    'r2': result.r2,
                    'window_size': len(result.window_df) if not result.window_df.empty else None,
                    'days_since_anchor': (current_date - result.anchor_date).days if result.anchor_date else None,
                    'break_reason': result.break_reason,
                    'cumulative_gain': cumulative_gain
                })
                if result.state:
                    base_record.update({
                        'mid_today': result.state.mid_today,
                        'upper_today': result.state.upper_today,
                        'lower_today': result.state.lower_today,
                    })
                    # 计算通道宽度百分比
                    if result.state.mid_today and result.state.mid_today > 0:
                        base_record['width_pct'] = (
                                                           result.state.upper_today - result.state.lower_today) / result.state.mid_today
                    # 计算斜率角度
                    if result.beta:
                        base_record['slope_deg'] = np.degrees(np.arctan(result.beta))
                    # 计算波动率
                    if result.sigma and result.state.mid_today and result.state.mid_today > 0:
                        base_record['volatility'] = result.sigma / result.state.mid_today
                
                # 状态判定：根据是否有state和break_reason来确定状态
                if result.state is not None:
                    # 有state但质量检查不通过，可能是ASCENDING_WEAK
                    if result.break_reason in ["invalid_width", "invalid_regression"]:
                        base_record['channel_status'] = ChannelStatus.ASCENDING_WEAK.value
                    else:
                        base_record['channel_status'] = ChannelStatus.BREAKDOWN.value
                else:
                    base_record['channel_status'] = ChannelStatus.BREAKDOWN.value
        else:
            # 有效通道
            state = result.state
            base_record.update({
                'anchor_date': result.anchor_date,
                'anchor_price': result.anchor_price,
                'beta': result.beta,
                'sigma': result.sigma,
                'r2': result.r2,
                'mid_today': state.mid_today,
                'upper_today': state.upper_today,
                'lower_today': state.lower_today,
                'mid_tomorrow': state.mid_tomorrow,
                'upper_tomorrow': state.upper_tomorrow,
                'lower_tomorrow': state.lower_tomorrow,
                'channel_status': ChannelStatus.NORMAL.value,
                'window_size': len(result.window_df),
                'days_since_anchor': (current_date - result.anchor_date).days,
                'cumulative_gain': state.cumulative_gain,
                'break_cnt_up': state.break_cnt_up,
                'break_cnt_down': state.break_cnt_down,
                'reanchor_fail_up': state.reanchor_fail_up,
                'reanchor_fail_down': state.reanchor_fail_down,
                'break_reason': None,  # 有效通道没有break_reason
                'width_pct': (
                                     state.upper_today - state.lower_today) / state.mid_today if state.mid_today and state.mid_today > 0 else None,
                'slope_deg': np.degrees(np.arctan(result.beta)) if result.beta else None,
                'volatility': result.sigma / state.mid_today if result.sigma and state.mid_today and state.mid_today > 0 else None
            })

        return base_record

    def _postprocess_results(self, history_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """后处理结果"""
        if not history_data:
            raise InsufficientDataError("无法计算任何历史通道数据")
        return pd.DataFrame(history_data)


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
        self.history_calculator = HistoryCalculationTemplate(self.strategy, self._get_config_dict())

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
        self.R2_min = params.get('R2_min', config_section.get('R2_min', 0.20))
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
                'R2_min': 0.20,
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
        3. 状态判定（NORMAL、BREAKOUT、BREAKDOWN、ASCENDING_WEAK）
        
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
                # 构造一个最小可用状态并标记为 BREAKDOWN
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
                    channel_status=ChannelStatus.BREAKDOWN,
                    cumulative_gain=cumulative_gain
                )
                return state

        # 有效结果，进行状态判定
        return self._determine_channel_status(result.state, current_close, result.r2, None)

    def _determine_channel_status(self, state: ChannelState, current_close: float, 
                                 r2: Optional[float], break_reason: Optional[str]) -> ChannelState:
        """
        确定通道状态
        
        根据通道质量和当前价格位置，确定最终的通道状态：
        - 如果质量检查不通过但趋势向上：ASCENDING_WEAK
        - 如果质量检查通过：根据价格位置判定 NORMAL/BREAKOUT/BREAKDOWN
        
        Args:
            state (ChannelState): 通道状态对象
            current_close (float): 当前收盘价
            r2 (Optional[float]): 回归拟合优度
            break_reason (Optional[str]): 失效原因
            
        Returns:
            ChannelState: 更新后的通道状态对象
        """
        config = self._get_config_dict()
        
        # 质量检查：检查R²和通道宽度
        quality_check_passed = self._check_channel_quality(state, r2, config)
        
        if not quality_check_passed:
            # 质量检查不通过，但趋势向上，标记为弱上升通道
            state.channel_status = ChannelStatus.ASCENDING_WEAK
            return state
        
        # 质量检查通过，根据价格位置判定状态
        if current_close > state.upper_today:
            state.channel_status = ChannelStatus.BREAKOUT
        elif current_close < state.lower_today:
            state.channel_status = ChannelStatus.BREAKDOWN
        else:
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
        if r2 is None or r2 < config['R2_min']:
            return False
            
        # 检查通道宽度
        if state.mid_today and state.mid_today > 0:
            width_pct = (state.upper_today - state.lower_today) / state.mid_today
            if width_pct < config['width_pct_min'] or width_pct > config['width_pct_max']:
                return False
        else:
            return False
            
        return True

    def fit_channel_history(self, df: pd.DataFrame, min_window_size: int = 60) -> pd.DataFrame:
        """计算历史上升通道数据（批量计算）"""
        return self.history_calculator.calculate_history(df, min_window_size, step_days=1)

    def fit_channel_history_optimized(self, df: pd.DataFrame, min_window_size: int = 60,
                                      step_days: int = 1) -> pd.DataFrame:
        """优化的历史通道数据计算（批量计算，可配置步长）"""
        return self.history_calculator.calculate_history(df, min_window_size, step_days)

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

        # 基于当日价格的三态判定（即时）
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

    def _check_beta_change(self) -> bool:
        """检查斜率变化"""
        return False

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

        if self.state.channel_status == ChannelStatus.BREAKDOWN:
            self.state.channel_status = ChannelStatus.NORMAL
            logger.info("通道状态从 BREAKDOWN 重置为 NORMAL")

        logger.info(
            f"重锚成功: 新锚点 {new_anchor_date} @ {new_anchor_price:.2f}, 窗口大小: {len(self.state.window_df)}")

    def _check_extreme_states(self) -> None:
        """检查并更新极端状态"""
        if self.state.reanchor_fail_up >= self.reanchor_fail_max:
            self.state.channel_status = ChannelStatus.BREAKOUT
            logger.debug(f"进入加速突破状态: 连续重锚失败 {self.state.reanchor_fail_up} 次")
        elif self.state.reanchor_fail_down >= self.reanchor_fail_max:
            self.state.channel_status = ChannelStatus.BREAKDOWN
            logger.debug(f"进入跌破状态: 连续重锚失败 {self.state.reanchor_fail_down} 次")
        elif (self.state.break_cnt_down >= self.break_days):
            self.state.channel_status = ChannelStatus.BREAKDOWN
            logger.debug("通道失效: 连续跌破下沿 → 标记为 BREAKDOWN")

    # 向后兼容方法
    def force_reanchor(self, state: ChannelState) -> ChannelState:
        """强制重锚"""
        self.state = state
        self._reanchor()
        return self.state

    def get_channel_info(self, state: Optional[ChannelState] = None) -> Dict[str, Any]:
        """获取通道信息"""
        if state is None:
            state = self.state
        if state is None:
            return {}
        return state.to_dict()

    # 增量更新方法保持不变
    def update_channel_history_incremental(self, history_df: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """增量更新历史通道数据"""
        if history_df.empty:
            raise ValueError("历史通道数据不能为空")

        new_data = new_data.sort_values('trade_date').reset_index(drop=True)
        new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])

        last_record = history_df.iloc[-1]

        if pd.isna(last_record['beta']):
            logger.warning("最后一条历史记录无效，无法进行增量更新")
            return history_df

        last_state = ChannelState(
            anchor_date=pd.to_datetime(last_record['anchor_date']),
            anchor_price=last_record['anchor_price'],
            window_df=pd.DataFrame(),
            beta=last_record['beta'],
            sigma=last_record['sigma'],
            mid_today=last_record['mid_today'],
            upper_today=last_record['upper_today'],
            lower_today=last_record['lower_today'],
            mid_tomorrow=last_record['mid_tomorrow'],
            upper_tomorrow=last_record['upper_tomorrow'],
            lower_tomorrow=last_record['lower_tomorrow']
        )

        last_state.break_cnt_up = last_record['break_cnt_up']
        last_state.break_cnt_down = last_record['break_cnt_down']
        last_state.reanchor_fail_up = last_record['reanchor_fail_up']
        last_state.reanchor_fail_down = last_record['reanchor_fail_down']
        last_state.cumulative_gain = last_record['cumulative_gain']
        last_state.channel_status = ChannelStatus(last_record['channel_status'])
        last_state.last_update = pd.to_datetime(last_record['trade_date'])

        window_size = last_record['window_size']
        if pd.isna(window_size) or window_size <= 0:
            logger.warning("无法重建窗口数据，window_size无效")
            return history_df

        new_history_data = []

        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

        try:
            for _, row in new_data.iterrows():
                try:
                    new_bar = {
                        'trade_date': row['trade_date'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }

                    updated_info = self.update(last_state, new_bar)
                    last_state = self.state

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
                    logger.warning(f"增量更新时点 {row['trade_date']} 失败: {e}")

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
                    last_state = None
                    continue
        finally:
            logging.getLogger().setLevel(original_level)

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
        """基于完整状态对象的增量更新"""
        if history_df.empty:
            raise ValueError("历史通道数据不能为空")

        if last_state is None:
            raise ValueError("最后状态对象不能为空")

        new_data = new_data.sort_values('trade_date').reset_index(drop=True)
        new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])

        new_history_data = []
        current_state = last_state

        for _, row in new_data.iterrows():
            try:
                new_bar = {
                    'trade_date': row['trade_date'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }

                updated_info = self.update(current_state, new_bar)
                current_state = self.state

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
        """改进的增量更新历史通道数据"""
        if history_df.empty:
            raise ValueError("历史通道数据不能为空")

        original_df = original_df.sort_values('trade_date').reset_index(drop=True)
        original_df['trade_date'] = pd.to_datetime(original_df['trade_date'])

        new_data = new_data.sort_values('trade_date').reset_index(drop=True)
        new_data['trade_date'] = pd.to_datetime(new_data['trade_date'])

        last_record = history_df.iloc[-1]

        if pd.isna(last_record['beta']):
            logger.warning("最后一条历史记录无效，无法进行增量更新")
            return history_df

        last_state = ChannelState(
            anchor_date=pd.to_datetime(last_record['anchor_date']),
            anchor_price=last_record['anchor_price'],
            window_df=pd.DataFrame(),
            beta=last_record['beta'],
            sigma=last_record['sigma'],
            mid_today=last_record['mid_today'],
            upper_today=last_record['upper_today'],
            lower_today=last_record['lower_today'],
            mid_tomorrow=last_record['mid_tomorrow'],
            upper_tomorrow=last_record['upper_tomorrow'],
            lower_tomorrow=last_record['lower_tomorrow']
        )

        last_state.break_cnt_up = last_record['break_cnt_up']
        last_state.break_cnt_down = last_record['break_cnt_down']
        last_state.reanchor_fail_up = last_record['reanchor_fail_up']
        last_state.reanchor_fail_down = last_record['reanchor_fail_down']
        last_state.cumulative_gain = last_record['cumulative_gain']
        last_state.channel_status = ChannelStatus(last_record['channel_status'])
        last_state.last_update = pd.to_datetime(last_record['trade_date'])

        window_size = last_record['window_size']
        if pd.isna(window_size) or window_size <= 0:
            logger.warning("无法重建窗口数据，window_size无效")
            return history_df

        last_history_date = last_record['trade_date']
        last_history_idx = original_df[original_df['trade_date'] == last_history_date].index

        if len(last_history_idx) == 0:
            logger.warning(f"无法在原始数据中找到历史记录日期: {last_history_date}")
            return history_df

        last_history_idx = last_history_idx[0]
        start_idx = max(0, last_history_idx - window_size + 1)
        window_df = original_df.iloc[start_idx:last_history_idx + 1]

        if len(window_df) != window_size:
            logger.warning(f"窗口大小不匹配: 期望 {window_size}, 实际 {len(window_df)}")
            window_size = len(window_df)

        last_state.window_df = window_df
        new_history_data = []

        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

        try:
            for _, row in new_data.iterrows():
                try:
                    new_bar = {
                        'trade_date': row['trade_date'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }

                    updated_info = self.update(last_state, new_bar)
                    last_state = self.state

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
                    logger.warning(f"增量更新时点 {row['trade_date']} 失败: {e}")

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
                    last_state = None
                    continue
        finally:
            logging.getLogger().setLevel(original_level)

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
    """上升通道回归因子"""
    return pd.Series(index=df.index, dtype=float)
