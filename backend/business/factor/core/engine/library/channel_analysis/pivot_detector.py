#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : AI Assistant
@Date       : 2024-12-19
@File       : pivot_detector.py
@Description: 锚点检测器

该类负责检测价格序列中的关键锚点，用于上升通道回归分析：
1. 检测 pivot low（局部最低点）
2. 检测 pivot high（局部最高点）
3. 验证锚点的有效性
4. 提供锚点选择策略
"""

import logging
from typing import Optional, Tuple, List

import pandas as pd

logger = logging.getLogger(__name__)


class PivotDetector:
    """
    锚点检测器类
    
    用于检测价格序列中的关键锚点，支持多种检测策略
    """

    def __init__(self, pivot_m: int = 3):
        """
        初始化锚点检测器
        
        Args:
            pivot_m (int): 判断 pivot 的宽度参数（m 左 m 右更高/更低）
        """
        self.pivot_m = pivot_m

    def find_pivot_low(self, df: pd.DataFrame, column: str = 'low', silent: bool = False) -> Optional[
        Tuple[pd.Timestamp, float]]:
        """
        查找 pivot low（局部最低点）
        
        Args:
            df (pd.DataFrame): 价格数据，包含 trade_date 和指定列
            column (str): 用于检测的列名，默认为 'low'
            silent (bool): 是否静默模式，减少日志输出
            
        Returns:
            Optional[Tuple[pd.Timestamp, float]]: (日期, 价格) 或 None
        """
        if len(df) < 2 * self.pivot_m + 1:
            if not silent:
                logger.warning(f"数据长度不足，无法检测 pivot low: {len(df)} < {2 * self.pivot_m + 1}")
            return None

        prices = df[column].values
        dates = df['trade_date'].values

        # 寻找局部最低点
        pivot_lows = []

        for i in range(self.pivot_m, len(prices) - self.pivot_m):
            current_price = prices[i]

            # 检查左侧 m 个点是否都更高
            left_higher = all(prices[j] > current_price for j in range(i - self.pivot_m, i))

            # 检查右侧 m 个点是否都更高
            right_higher = all(prices[j] > current_price for j in range(i + 1, i + self.pivot_m + 1))

            if left_higher and right_higher:
                pivot_lows.append((dates[i], current_price))

        if not pivot_lows:
            if not silent:
                logger.warning("未找到有效的 pivot low")
            return None

        # 选择最新的 pivot low
        latest_pivot = max(pivot_lows, key=lambda x: x[0])
        if not silent:
            logger.info(f"找到 pivot low: {latest_pivot[0]} @ {latest_pivot[1]:.2f}")

        return latest_pivot

    def find_pivot_high(self, df: pd.DataFrame, column: str = 'high', silent: bool = False) -> Optional[
        Tuple[pd.Timestamp, float]]:
        """
        查找 pivot high（局部最高点）
        
        Args:
            df (pd.DataFrame): 价格数据，包含 trade_date 和指定列
            column (str): 用于检测的列名，默认为 'high'
            silent (bool): 是否静默模式，减少日志输出
            
        Returns:
            Optional[Tuple[pd.Timestamp, float]]: (日期, 价格) 或 None
        """
        if len(df) < 2 * self.pivot_m + 1:
            if not silent:
                logger.warning(f"数据长度不足，无法检测 pivot high: {len(df)} < {2 * self.pivot_m + 1}")
            return None

        prices = df[column].values
        dates = df['trade_date'].values

        # 寻找局部最高点
        pivot_highs = []

        for i in range(self.pivot_m, len(prices) - self.pivot_m):
            current_price = prices[i]

            # 检查左侧 m 个点是否都更低
            left_lower = all(prices[j] < current_price for j in range(i - self.pivot_m, i))

            # 检查右侧 m 个点是否都更低
            right_lower = all(prices[j] < current_price for j in range(i + 1, i + self.pivot_m + 1))

            if left_lower and right_lower:
                pivot_highs.append((dates[i], current_price))

        if not pivot_highs:
            if not silent:
                logger.warning("未找到有效的 pivot high")
            return None

        # 选择最新的 pivot high
        latest_pivot = max(pivot_highs, key=lambda x: x[0])
        if not silent:
            logger.info(f"找到 pivot high: {latest_pivot[0]} @ {latest_pivot[1]:.2f}")

        return latest_pivot

    def find_initial_anchor(self, df: pd.DataFrame, strategy: str = 'pivot_low',
                            silent: bool = False) -> Optional[Tuple[pd.Timestamp, float]]:
        """
        查找初始锚点
        
        Args:
            df (pd.DataFrame): 价格数据
            strategy (str): 锚点选择策略
                - 'pivot_low': 选择 pivot low 作为锚点
                - 'pivot_high': 选择 pivot high 作为锚点
                - 'min_price': 选择最低价格作为锚点
                - 'max_price': 选择最高价格作为锚点
            silent (bool): 是否静默模式，减少日志输出
                
        Returns:
            Optional[Tuple[pd.Timestamp, float]]: (日期, 价格) 或 None
        """
        if len(df) < 60:  # 最小数据要求
            if not silent:
                logger.error(f"数据不足，至少需要60个数据点，当前只有 {len(df)} 个")
            return None

        if strategy == 'pivot_low':
            return self.find_pivot_low(df, 'low', silent=silent)
        elif strategy == 'pivot_high':
            return self.find_pivot_high(df, 'high', silent=silent)
        elif strategy == 'min_price':
            min_idx = df['low'].idxmin()
            return (df.loc[min_idx, 'trade_date'], df.loc[min_idx, 'low'])
        elif strategy == 'max_price':
            max_idx = df['high'].idxmax()
            return (df.loc[max_idx, 'trade_date'], df.loc[max_idx, 'high'])
        else:
            if not silent:
                logger.error(f"未知的锚点选择策略: {strategy}")
            return None

    def find_new_anchor(self, df: pd.DataFrame, current_anchor_date: pd.Timestamp,
                        strategy: str = 'pivot_low') -> Optional[Tuple[pd.Timestamp, float]]:
        """
        查找新的锚点（用于重锚）
        
        Args:
            df (pd.DataFrame): 价格数据
            current_anchor_date (pd.Timestamp): 当前锚点日期
            strategy (str): 锚点选择策略
            
        Returns:
            Optional[Tuple[pd.Timestamp, float]]: (日期, 价格) 或 None
        """
        # 只考虑当前锚点之后的数据
        recent_df = df[df['trade_date'] > current_anchor_date]

        if len(recent_df) < self.pivot_m * 2:
            logger.warning("重锚数据不足")
            return None

        return self.find_initial_anchor(recent_df, strategy)

    def validate_anchor(self, df: pd.DataFrame, anchor_date: pd.Timestamp,
                        anchor_price: float, min_gain: float = 0.05) -> bool:
        """
        验证锚点的有效性
        
        Args:
            df (pd.DataFrame): 价格数据
            anchor_date (pd.Timestamp): 锚点日期
            anchor_price (float): 锚点价格
            min_gain (float): 最小涨幅要求
            
        Returns:
            bool: 锚点是否有效
        """
        # 检查锚点后的价格走势
        future_df = df[df['trade_date'] > anchor_date]

        if len(future_df) < 10:  # 至少需要10个交易日的数据
            return False

        # 计算最大涨幅
        max_price = future_df['high'].max()
        max_gain = (max_price - anchor_price) / anchor_price

        # 检查是否有足够的上涨
        if max_gain < min_gain:
            logger.info(f"锚点验证失败：最大涨幅 {max_gain:.2%} < {min_gain:.2%}")
            return False

        # 检查价格是否主要向上
        price_trend = future_df['close'].iloc[-1] - anchor_price
        if price_trend <= 0:
            logger.info(f"锚点验证失败：价格趋势向下 {price_trend:.2f}")
            return False

        logger.info(f"锚点验证通过：最大涨幅 {max_gain:.2%}, 趋势 {price_trend:.2f}")
        return True

    def get_anchor_candidates(self, df: pd.DataFrame, n_candidates: int = 5) -> List[Tuple[pd.Timestamp, float]]:
        """
        获取锚点候选列表
        
        Args:
            df (pd.DataFrame): 价格数据
            n_candidates (int): 候选数量
            
        Returns:
            List[Tuple[pd.Timestamp, float]]: 锚点候选列表
        """
        candidates = []

        # 查找多个 pivot low
        prices = df['low'].values
        dates = df['trade_date'].values

        for i in range(self.pivot_m, len(prices) - self.pivot_m):
            current_price = prices[i]

            # 检查是否为局部最低点
            left_higher = all(prices[j] > current_price for j in range(i - self.pivot_m, i))
            right_higher = all(prices[j] > current_price for j in range(i + 1, i + self.pivot_m + 1))

            if left_higher and right_higher:
                candidates.append((dates[i], current_price))

        # 按日期排序，选择最新的 n_candidates 个
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:n_candidates]
