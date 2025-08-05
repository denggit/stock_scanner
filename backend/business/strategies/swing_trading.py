#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:32 PM
@File       : swing_trading.py
@Description: 
"""

import numpy as np
import pandas as pd

from backend.business.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class SwingTradingStrategy(BaseStrategy):
    """波段交易策略"""

    def __init__(self):
        super().__init__(name="波段交易策略", description="基于趋势、振幅和技术指标的波段交易策略")
        self._params = {
            # 多头排列
            "bullish": True,  # 是否多头排列
            "ma_periods": [5, 10, 20],  # 均线周期

            # 时间周期参数
            "lookback_period": 10,  # 回溯周期
            "short_ma_period": 5,  # 短期均线周期
            "long_ma_period": 20,  # 长期均线周期

            # MACD参数
            "macd_fast_period": 12,  # 快速均线周期
            "macd_slow_period": 26,  # 慢速均线周期
            "macd_signal_period": 9,  # 信号均线周期

            # RSI参数
            "rsi_period": 14,  # RSI周期
            "rsi_overbought": 70,  # 超买阈值
            "rsi_oversold": 30,  # 超卖阈值

            # 振幅与波动参数
            "amplitude_threshold": 0.05,  # 振幅阈值
            "volatility_threshold": 0.02,  # 波动阈值
            "bollinger_k": 2,  # 布林带宽度

            # 权重参数
            "weights": {  # 信号强度评分权重
                "price": 0.3,  # 价格权重
                "volatility": 0.2,  # 波动权重
                "rsi": 0.15,  # RSI权重
                "macd": 0.15,  # MACD权重
                "trend": 0.2  # 趋势权重
            }
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        """
        if not self.validate_data(data):
            raise ValueError("数据格式错误")

        # 1. 数据预处理 - 确保所有数值列都是float类型
        df = data.copy()

        # 2. 趋势识别
        df['ma_short'] = CalIndicators.ema(df, self._params['short_ma_period'], 'close')
        df['ma_long'] = CalIndicators.ema(df, self._params['long_ma_period'], 'close')
        ma_periods = sorted(list(set(self._params['ma_periods'])))
        if self._params['bullish'] and len(ma_periods) > 1:
            short_ma = CalIndicators.ema(df, ma_periods[0], 'close')
            conditions = pd.Series(True, index=df.index)
            for period in ma_periods[1:]:
                long_ma = CalIndicators.ema(df, period, 'close')
                conditions = (conditions & (short_ma > long_ma))
            df['trend'] = np.where(conditions, 1, -1)
        else:
            df['trend'] = np.where(df['ma_short'] > df['ma_long'], 1, -1)

        # 3. 计算MACD
        macd_fast = self._params['macd_fast_period']
        macd_slow = self._params['macd_slow_period']
        macd_signal = self._params['macd_signal_period']
        df['macd'], df['macd_signal'], df['macd_hist'] = CalIndicators.macd(df, fast_period=macd_fast,
                                                                            slow_period=macd_slow,
                                                                            signal_period=macd_signal)

        # 4. 计算振幅和波动率
        df['amplitude'] = CalIndicators.amplitude(df, self._params['lookback_period'])
        if len(df) >= 250:
            df['volatility'] = CalIndicators.volatility(df, self._params['lookback_period'], annualized=True)
        else:
            df['volatility'] = CalIndicators.volatility(df, self._params['lookback_period'], annualized=False)

        # 5. 计算布林带
        short_period = self._params['short_ma_period']
        k = float(self._params['bollinger_k'])
        df['bb_upper'], df['bb_mid'], df['bb_lower'] = CalIndicators.bollinger_bands(df, ma_period=short_period,
                                                                                     bollinger_k=k)

        # 6. 计算RSI
        df['rsi'] = CalIndicators.rsi(df, self._params['rsi_period'])

        # 7. 支撑位和阻力位
        df['support'] = CalIndicators.support(df, self._params['lookback_period'])
        df['resistance'] = CalIndicators.resistance(df, self._params['lookback_period'])

        # 确保所有计算用的数值都是float类型
        for col in df.select_dtypes(include=['float32', 'float64']).columns:
            df[col] = df[col].astype(float)

        df['support_distance'] = (df['close'].astype(float) - df['support'].astype(float)) / df['support'].astype(float)
        df['resistance_distance'] = (df['close'].astype(float) - df['resistance'].astype(float)) / df[
            'resistance'].astype(float)

        # 8. 综合评分
        df['price_score'] = self._calculate_price_score(df)
        df['volatility_score'] = self._calculate_volatility_score(df)
        df['trend_score'] = self._calculate_trend_score(df)
        df['macd_score'] = self._calculate_macd_score(df)
        df['rsi_score'] = self._calculate_rsi_score(df)

        # 计算总分
        weights = self._params['weights']
        df['total_score'] = (df['price_score'] * weights['price'] +
                             df['volatility_score'] * weights['volatility'] +
                             df['trend_score'] * weights['trend'] +
                             df['macd_score'] * weights['macd'] +
                             df['rsi_score'] * weights['rsi']).round(2)

        # 9. 生成信号
        signals = pd.DataFrame(index=df.index)

        # 基本条件
        basic_conditions = (
                (df['amplitude'] >= self._params['amplitude_threshold']) &
                (df['volatility'] >= self._params['volatility_threshold']) &
                (df['total_score'].notna())
        )

        # 买入条件
        buy_conditions = (
                basic_conditions &
                (df['trend'] > 0) &
                (df['rsi'] <= self._params['rsi_oversold']) &
                (df['macd'] > df['macd_signal'])
        )

        # 卖出条件
        sell_conditions = (
                basic_conditions &
                (df['trend'] < 0) &
                (df['rsi'] >= self._params['rsi_overbought']) &
                (df['macd'] < df['macd_signal'])
        )

        # 设置信号强度
        signals['signal'] = 0
        signals.loc[buy_conditions, 'signal'] = 1
        signals.loc[sell_conditions, 'signal'] = -1
        signals['signal_strength'] = (df['total_score'] / 2).round().clip(1, 5)

        # 添加评分明细
        signals['price_score'] = df['price_score'].round(2)
        signals['volatility_score'] = df['volatility_score'].round(2)
        signals['trend_score'] = df['trend_score'].round(2)
        signals['macd_score'] = df['macd_score'].round(2)
        signals['rsi_score'] = df['rsi_score'].round(2)
        signals['total_score'] = df['total_score'].round(2)
        signals['rsi'] = df['rsi'].round(2)
        signals['macd'] = df['macd'].round(2)
        signals['macd_signal'] = df['macd_signal'].round(2)
        signals['amplitude'] = df['amplitude'].round(2)
        signals['volatility'] = df['volatility'].round(2)
        signals['support'] = df['support'].round(2)
        signals['resistance'] = df['resistance'].round(2)
        signals['support_distance'] = df['support_distance'].round(2)
        signals['resistance_distance'] = df['resistance_distance'].round(2)

        # 只返回最新一天的价格
        last_day = signals.iloc[-1]

        return last_day

    def _calculate_price_score(self, data: pd.DataFrame) -> pd.Series:
        """计算价格距离支撑位的得分（0-10分）
        距离越近得分越高
        - 距离 <= 0.5%: 10分
        - 距离 2%: 5分
        - 距离 4% 或以上: 0分
        """
        support_distance = data['support_distance'].abs()
        score = pd.Series(0.0, index=data.index)  # 初始化为浮点数

        # 线性插值计算分数
        score = np.where(support_distance <= 0.005, 10.0,  # 距离<=0.5%
                         np.where(support_distance <= 0.02,  # 距离<=2%
                                  10.0 - (support_distance - 0.005) * (5.0 / 0.015),  # 线性插值
                                  np.where(support_distance <= 0.04,  # 距离<=4%
                                           5.0 - (support_distance - 0.02) * (5.0 / 0.02),  # 线性插值
                                           0.0)))
        return pd.Series(score, index=data.index)

    def _calculate_volatility_score(self, data: pd.DataFrame) -> pd.Series:
        """计算波动性得分（0-10分）
        波动越大得分越高
        - 波动 >= 5%: 10分
        - 波动 2%: 5分
        - 波动 1% 或以下: 0分
        """
        volatility = data['volatility']
        score = pd.Series(0.0, index=data.index)  # 初始化为浮点数

        # 线性插值计算分数
        score = np.where(volatility >= 0.05, 10.0,  # 波动>=5%
                         np.where(volatility >= 0.02,  # 波动>=2%
                                  5.0 + (volatility - 0.02) * (5.0 / 0.03),  # 线性插值
                                  np.where(volatility >= 0.01,  # 波动>=1%
                                           (volatility - 0.01) * (5.0 / 0.01),  # 线性插值
                                           0.0)))
        return pd.Series(score, index=data.index)

    def _calculate_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """计算趋势得分（0-10分）
        趋势越强得分越高
        - 趋势向上: 10分
        - 趋势向下: 0分
        """
        trend = data['trend']
        ma_short = data['ma_short']
        ma_long = data['ma_long']

        # 计算均线斜率
        ma_short_slope = ma_short.diff() / ma_short.shift(1)
        ma_long_slope = ma_long.diff() / ma_long.shift(1)

        score = pd.Series(0.0, index=data.index)  # 初始化为浮点数

        # 根据趋势方向和均线斜率计算分数
        score = np.where(trend > 0,
                         10.0 * (1.0 + ma_short_slope) * (1.0 + ma_long_slope),  # 上升趋势加权
                         0.0)  # 下降趋势

        # 限制分数范围在0-10之间
        return pd.Series(np.clip(score, 0.0, 10.0), index=data.index)

    def _calculate_macd_score(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD得分（0-10分）
        - 金叉：10分
        - 死叉：0分        
        """
        macd = data['macd']
        signal = data['macd_signal']
        hist = data['macd_hist']

        score = pd.Series(0.0, index=data.index)  # 初始化为浮点数

        # MACD柱状图由负变正为金叉，由正变负为死叉
        golden_cross = (hist > 0) & (hist.shift(1) <= 0)
        death_cross = (hist < 0) & (hist.shift(1) >= 0)

        # 根据MACD柱状图的强度计算分数
        score = np.where(golden_cross, 10.0,
                         np.where(death_cross, 0.0,
                                  np.where(hist > 0,
                                           5.0 + 5.0 * (hist / hist.abs().max()),  # 正柱状图
                                           5.0 * (1.0 + hist / hist.abs().max()))))  # 负柱状图

        return pd.Series(np.clip(score, 0.0, 10.0), index=data.index)

    def _calculate_rsi_score(self, data: pd.DataFrame) -> pd.Series:
        """计算RSI得分（0-10分）
        - RSI 小于 oversold: 10分
        - RSI 大于 overbought: 0分
        - 其他情况：5分
        """
        rsi = data['rsi']
        oversold = self._params['rsi_oversold']
        overbought = self._params['rsi_overbought']

        score = pd.Series(5.0, index=data.index)  # 默认中性得分，初始化为浮点数

        # 根据RSI值计算分数
        score = np.where(rsi <= oversold, 10.0,  # 超卖区间
                         np.where(rsi >= overbought, 0.0,  # 超买区间
                                  # 中性区间，线性插值
                                  5.0 + 5.0 * (overbought - rsi) / (overbought - oversold)))

        return pd.Series(np.clip(score, 0.0, 10.0), index=data.index)
