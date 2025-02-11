# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Your Name
@Date       : Current Date
@File       : hs_bottom.py
@Description: 头肩底形态选股策略
"""

from typing import Tuple

import pandas as pd

from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class HSBottom(BaseStrategy):
    def __init__(self):
        super().__init__(name="头肩底形态策略", description="识别头肩底反转形态")
        self._params = {
            "lookback_period": 120,  # 回看的天数
            "volume_ratio": 1.5,  # 突破颈线时的成交量要求
            "shoulder_height_diff": 0.1,  # 左右肩高度差异容忍度
            "neckline_slope_range": (-0.087, 0.268),  # 颈线斜率范围（约-5度到15度）
            "head_drop_range": (15, 35),  # 头部跌幅范围（百分比）
            "min_pattern_days": 30,  # 头肩底最小形成天数
            "min_pattern_points": 15,  # 整个形态的最小点数
            "max_pattern_points": 60,  # 整个形态的最大点数
            "shoulder_time_diff": 0.2,  # 左右肩时间差异容忍度（20%）
            "head_depth": 0.03,  # 头部相对双肩的最小深度（3%）
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")

        # 数据预处理和计算指标
        df = data.copy().tail(self._params['lookback_period'])
        df = df.reset_index(drop=True)
        df = self._calculate_indicators(df)

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
        signals['neckline_left_date'] = None
        signals['neckline_right_date'] = None
        signals['neckline_left_price'] = None
        signals['neckline_right_price'] = None
        signals['signal_strength'] = None
        signals['resistance_to_neck'] = None  # 新增：压力位距离颈线的百分比
        signals['macd'] = None  # 新增：MACD值
        signals['rsi'] = None  # 新增：RSI值

        # 寻找局部极小值点
        lows = []
        for i in range(1, len(df) - 1):
            if df['low'].iloc[i] <= df['low'].iloc[i - 1] and df['low'].iloc[i] <= df['low'].iloc[i + 1]:
                lows.append(i)

        # 如果没有足够的极值点，返回空信号
        if len(lows) < 3:
            return signals.iloc[-1]

        # 计算过去50天的平均成交量
        avg_volume = df['volume'].tail(50).mean()

        # 遍历可能的头肩底形态
        for i in range(len(lows) - 2):
            left_shoulder = lows[i]
            head = lows[i + 1]
            right_shoulder = lows[i + 2]

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
                    # 如果当前价格超过颈线价格10%，就不显示了
                    if price_to_neck > 10:
                        continue

                    breakthrough_idx = None
                    for j in range(len(df) - 2, -1, -1):
                        if float(df['close'].iloc[j]) <= neckline_last_price:
                            breakthrough_idx = j + 1
                            break

                    if breakthrough_idx is not None:
                        # 计算颈线斜率
                        neckline_slope = (float(right_peak) - float(left_peak)) / (right_peak_idx - left_peak_idx)

                        # 检查颈线斜率是否在允许范围内
                        if not (self._params['neckline_slope_range'][0] <= neckline_slope <=
                                self._params['neckline_slope_range'][1]):
                            continue

                        high_after_break = float(df['high'].iloc[breakthrough_idx:].max())
                        low_after_break = float(df['low'].iloc[breakthrough_idx:].min())
                        breakthrough_price = float(df['close'].iloc[breakthrough_idx])

                        breakthrough_range = high_after_break - breakthrough_price
                        pullback_range = high_after_break - low_after_break

                        # 如果回踩幅度超过突破幅度的38.2%，则不显示
                        if breakthrough_range > 0 and pullback_range / breakthrough_range > 0.382:
                            continue

                        # 计算前期压力位距离颈线的百分比
                        resistance_price = float(df['resistance'].iloc[breakthrough_idx])
                        resistance_to_neck = round((resistance_price - neckline_last_price) / neckline_last_price * 100,
                                                   2)

                        # 获取最新的技术指标值
                        current_macd = float(df['macd'].iloc[-1])
                        current_rsi = float(df['rsi'].iloc[-1])

                        # 检查突破后的收盘价是否始终保持在颈线之上
                        all_closes_above_neckline = True
                        for k in range(breakthrough_idx, len(df)):
                            if float(df['close'].iloc[k]) < neckline_last_price:
                                all_closes_above_neckline = False
                                break

                        # 第一天突破
                        if len(df) - breakthrough_idx == 1:
                            # 检查突破时的成交量是否放大
                            if df['volume'].iloc[-1] >= avg_volume * self._params['volume_ratio']:
                                signals.loc[signals.index[-1], 'signal'] = 1
                        # 第二天和第三天
                        elif len(df) - breakthrough_idx in [2, 3]:
                            if all_closes_above_neckline:
                                signals.loc[signals.index[-1], 'signal'] = len(df) - breakthrough_idx

                        if signals.loc[signals.index[-1], 'signal'] != 0:
                            # 计算信号强度
                            volume_ratio = float(df['volume'].iloc[breakthrough_idx] / avg_volume)
                            breakthrough_pct = float((float(
                                df['close'].iloc[breakthrough_idx]) - neckline_last_price) / neckline_last_price * 100)

                            # 计算颈线斜率的百分比
                            neckline_slope_pct = abs(neckline_slope * 100)  # 转换为百分比

                            # 归一化处理
                            volume_score = min(volume_ratio / 3, 1.0)
                            breakthrough_score = min(breakthrough_pct / 5, 1.0)
                            slope_score = max(0, 1 - neckline_slope_pct / 10)

                            signal_strength = round(0.3 * slope_score + 0.4 * volume_score + 0.3 * breakthrough_score,
                                                    2)

                            # 更新信号信息
                            signals.loc[signals.index[-1], 'pattern_start'] = df['trade_date'].iloc[left_shoulder]
                            signals.loc[signals.index[-1], 'pattern_end'] = df['trade_date'].iloc[right_shoulder]
                            signals.loc[signals.index[-1], 'neckline_price'] = neckline_last_price
                            signals.loc[signals.index[-1], 'volume_ratio'] = (df['volume'].iloc[-1] / avg_volume).round(
                                2)
                            signals.loc[signals.index[-1], 'price_to_neck'] = price_to_neck
                            signals.loc[signals.index[-1], 'neckline_left_date'] = df['trade_date'].iloc[left_peak_idx]
                            signals.loc[signals.index[-1], 'neckline_right_date'] = df['trade_date'].iloc[
                                right_peak_idx]
                            signals.loc[signals.index[-1], 'neckline_left_price'] = float(left_peak)
                            signals.loc[signals.index[-1], 'neckline_right_price'] = float(right_peak)
                            signals.loc[signals.index[-1], 'signal_strength'] = signal_strength
                            signals.loc[signals.index[-1], 'resistance_to_neck'] = resistance_to_neck
                            signals.loc[signals.index[-1], 'macd'] = current_macd
                            signals.loc[signals.index[-1], 'rsi'] = current_rsi
                            break

        return signals.iloc[-1]

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = CalIndicators.macd(df, fast_period=12, slow_period=26,
                                                                            signal_period=9)

        # 计算RSI
        df['rsi'] = CalIndicators.rsi(df, period=14)

        # 计算前期压力位（使用前期高点）
        df['resistance'] = CalIndicators.resistance(df, window=60)

        return df

    def _validate_pattern(self, df: pd.DataFrame, left_shoulder: int,
                          head: int, right_shoulder: int, neckline_points: Tuple[int, int]) -> bool:
        """验证头肩底形态是否有效"""
        # 检查点的顺序
        if not (left_shoulder < head < right_shoulder):
            return False

        # 检查形态形成时间是否足够长
        if right_shoulder - left_shoulder < self._params['min_pattern_days']:
            return False

        # 检查左右肩的高度相似性
        left_shoulder_price = float(df['low'].iloc[left_shoulder])
        right_shoulder_price = float(df['low'].iloc[right_shoulder])
        head_price = float(df['low'].iloc[head])

        # 检查头部是否显著低于双肩
        left_head_diff = (left_shoulder_price - head_price) / left_shoulder_price
        right_head_diff = (right_shoulder_price - head_price) / right_shoulder_price
        if left_head_diff < self._params['head_depth'] or right_head_diff < self._params['head_depth']:
            return False

        # 检查头部跌幅是否在指定范围内
        head_drop = (df['high'].iloc[left_shoulder:head].max() - head_price) / df['high'].iloc[
                                                                               left_shoulder:head].max() * 100
        if not (self._params['head_drop_range'][0] <= head_drop <= self._params['head_drop_range'][1]):
            return False

        # 检查左右肩时间对称性
        left_shoulder_duration = head - left_shoulder
        right_shoulder_duration = right_shoulder - head
        time_diff_ratio = abs(right_shoulder_duration - left_shoulder_duration) / left_shoulder_duration
        if time_diff_ratio > self._params['shoulder_time_diff']:
            return False

        # 检查成交量特征
        left_shoulder_volume = df['volume'].iloc[left_shoulder - 2:left_shoulder + 3].mean()
        head_volume = df['volume'].iloc[head - 2:head + 3].mean()
        right_shoulder_volume = df['volume'].iloc[right_shoulder - 2:right_shoulder + 3].mean()

        # 左肩放量，头部缩量，右肩放量
        if not (left_shoulder_volume > head_volume and right_shoulder_volume > head_volume):
            return False

        return True
