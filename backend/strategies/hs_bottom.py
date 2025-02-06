# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Your Name
@Date       : Current Date
@File       : hs_bottom.py
@Description: 头肩底形态选股策略
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class HSBottom(BaseStrategy):
    """
    头肩底形态策略
    识别市场中的头肩底形态，这是一个重要的反转形态，表明下跌趋势即将结束，转为上涨趋势
    """

    def __init__(self):
        super().__init__(name="头肩底形态策略", description="识别头肩底反转形态")
        self._params = {
            "lookback_period": 120,  # 回看的天数
            "volume_ratio": 1.5,  # 突破颈线时的成交量要求（相对于平均成交量的倍数）
            "shoulder_height_diff": 0.1,  # 左右肩高度差异容忍度（百分比）
            "neckline_slope_max": 0.1,  # 颈线斜率最大值（过陡的颈线可能不可靠）
            "min_pattern_points": 15,  # 整个形态的最小点数
            "max_pattern_points": 60,  # 整个形态的最大点数
        }

    @staticmethod
    def _find_local_extrema(prices: pd.Series, window: int = 5) -> Tuple[List[int], List[int]]:
        """
        找出局部最大值和最小值的位置
        """
        highs = []
        lows = []

        for i in range(window, len(prices) - window):
            if all(prices[i] <= prices[i - j] for j in range(1, window + 1)) and \
                    all(prices[i] <= prices[i + j] for j in range(1, window + 1)):
                lows.append(i)
            if all(prices[i] >= prices[i - j] for j in range(1, window + 1)) and \
                    all(prices[i] >= prices[i + j] for j in range(1, window + 1)):
                highs.append(i)

        return highs, lows

    def _validate_pattern(self, df: pd.DataFrame, left_shoulder: int,
                          head: int, right_shoulder: int, neckline_points: Tuple[int, int]) -> bool:
        """
        验证头肩底形态是否有效
        """
        # 检查点的顺序
        if not (left_shoulder < head < right_shoulder):
            return False

        # 检查左右肩的高度相似性
        left_shoulder_price = float(df['low'].iloc[left_shoulder])
        right_shoulder_price = float(df['low'].iloc[right_shoulder])
        head_price = float(df['low'].iloc[head])

        # 头部必须显著低于双肩（至少3%）
        left_head_diff = (left_shoulder_price - head_price) / left_shoulder_price
        right_head_diff = (right_shoulder_price - head_price) / right_shoulder_price
        if left_head_diff < 0.03 or right_head_diff < 0.03:
            return False

        shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price
        if shoulder_diff > self._params['shoulder_height_diff']:
            return False

        # 检查右肩成交量是否显著小于左肩（至少缩减30%）
        left_shoulder_volume = df['volume'].iloc[left_shoulder-2:left_shoulder+3].mean()  # 取左肩附近5天平均成交量
        right_shoulder_volume = df['volume'].iloc[right_shoulder-2:right_shoulder+3].mean()  # 取右肩附近5天平均成交量
        if right_shoulder_volume > left_shoulder_volume * 0.7:  # 右肩成交量需缩减30%以上
            return False

        # 检查双肩时间对称性
        left_shoulder_duration = head - left_shoulder
        right_shoulder_duration = right_shoulder - head
        if right_shoulder_duration > left_shoulder_duration * 1.5:  # 右肩形成时间不超过左肩的1.5倍
            return False

        # 检查颈线斜率
        neckline_slope = abs((df['high'].iloc[neckline_points[1]] - df['high'].iloc[neckline_points[0]]) /
                             (neckline_points[1] - neckline_points[0]))
        if neckline_slope > self._params['neckline_slope_max']:
            return False

        # 检查形态的总体大小
        pattern_length = right_shoulder - left_shoulder
        if not (self._params['min_pattern_points'] <= pattern_length <= self._params['max_pattern_points']):
            return False

        return True

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")

        # 数据预处理
        df = data.copy().tail(self._params['lookback_period'])
        df = df.reset_index(drop=True)

        # 计算日涨跌幅
        df['pct_chg'] = df['close'].pct_change() * 100

        # 寻找局部极值点
        highs, lows = self._find_local_extrema(df['low'])
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['trade_date'] = df['trade_date']
        signals['price'] = df['close']
        signals['pattern_start'] = None
        signals['pattern_end'] = None
        signals['neckline_price'] = None
        signals['volume_ratio'] = None
        signals['price_to_neck'] = None
        signals['neckline_left_date'] = None   # 新增：颈线左侧点日期
        signals['neckline_right_date'] = None  # 新增：颈线右侧点日期
        signals['neckline_left_price'] = None  # 新增：颈线左侧点价格
        signals['neckline_right_price'] = None # 新增：颈线右侧点价格
        
        # 如果没有足够的极值点，返回空信号
        if len(lows) < 3:
            return signals.iloc[-1]

        # 计算过去50天的平均成交量
        avg_volume = df['volume'].tail(50).mean()

        # 遍历可能的头肩底形态
        for i in range(len(lows)-2):
            left_shoulder = lows[i]
            head = lows[i+1]
            right_shoulder = lows[i+2]

            # 确保右肩已经形成（至少需要几天确认）
            if right_shoulder > len(df) - 5:  # 要求右肩形成后至少有5天
                continue

            # 找到左右肩对应的高点，用于确定颈线
            left_peak_idx = df['high'].iloc[left_shoulder:head].idxmax()
            right_peak_idx = df['high'].iloc[head:right_shoulder].idxmax()
            left_peak = df['high'].iloc[left_peak_idx]
            right_peak = df['high'].iloc[right_peak_idx]
            
            # 颈线点
            neckline_points = (left_peak_idx, right_peak_idx)

            # 验证形态
            if self._validate_pattern(df, left_shoulder, head, right_shoulder, neckline_points):
                # 计算颈线斜率和最后一个点的颈线价格
                neckline_slope = (right_peak - left_peak) / (right_peak_idx - left_peak_idx)
                last_point_distance = len(df) - 1 - left_peak_idx
                neckline_last_price = float(left_peak + neckline_slope * last_point_distance)
                
                # 计算价格距离颈线的百分比
                current_price = float(df['close'].iloc[-1])
                price_to_neck = round(((current_price - neckline_last_price) / neckline_last_price * 100), 2)
                
                # 未突破颈线
                if price_to_neck < 0:
                    signals.loc[signals.index[-1], 'signal'] = -1
                
                # 已突破颈线
                elif price_to_neck > 0:
                    # 检查最近三天的情况
                    recent_prices = df['close'].tail(3)
                    recent_volumes = df['volume'].tail(3)
                    recent_pct_chgs = df['pct_chg'].tail(3)

                    # 检查最近三天是否都上涨超过2%
                    if all(pct >= 2 for pct in recent_pct_chgs):
                        # 第一天突破
                        if len(recent_prices) >= 2 and float(recent_prices.iloc[-2]) <= neckline_last_price:
                            if recent_volumes.iloc[-1] >= avg_volume * 1.5:  # 放量1.5倍以上
                                signals.loc[signals.index[-1], 'signal'] = 1
                        # 第二天
                        elif len(recent_prices) >= 3 and float(recent_prices.iloc[-3]) <= neckline_last_price:
                            signals.loc[signals.index[-1], 'signal'] = 2
                        # 第三天
                        elif len(recent_prices) >= 4 and float(recent_prices.iloc[-4]) <= neckline_last_price:
                            signals.loc[signals.index[-1], 'signal'] = 3
                
                if signals.loc[signals.index[-1], 'signal'] != 0:
                    signals.loc[signals.index[-1], 'pattern_start'] = df['trade_date'].iloc[left_shoulder]
                    signals.loc[signals.index[-1], 'pattern_end'] = df['trade_date'].iloc[right_shoulder]
                    signals.loc[signals.index[-1], 'neckline_price'] = neckline_last_price
                    signals.loc[signals.index[-1], 'volume_ratio'] = (df['volume'].iloc[-1] / avg_volume).round(2)
                    signals.loc[signals.index[-1], 'price_to_neck'] = price_to_neck
                    signals.loc[signals.index[-1], 'neckline_left_date'] = df['trade_date'].iloc[left_peak_idx]
                    signals.loc[signals.index[-1], 'neckline_right_date'] = df['trade_date'].iloc[right_peak_idx]
                    signals.loc[signals.index[-1], 'neckline_left_price'] = float(left_peak)
                    signals.loc[signals.index[-1], 'neckline_right_price'] = float(right_peak)
                    break

        return signals.iloc[-1]
