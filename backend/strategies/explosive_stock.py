# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description: 爆发式选股策略 - 寻找20个交易日内可能暴涨30%的股票
"""
import logging

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class ExplosiveStockStrategy(BaseStrategy):
    """
    爆发式选股策略
    寻找短期爆发潜力的股票
    """

    def __init__(self):
        super().__init__(name="爆发式选股策略", description="寻找20个交易日内可能暴涨30%的股票")
        self._params = {
            "volume_ma": 20,  # 成交量均线周期
            "rsi_period": 14,  # RSI周期
            "bb_period": 20,  # 布林带周期
            "bb_std": 2,  # 布林带标准差倍数
            "recent_days": 5,  # 近期趋势分析天数
            "volume_weight": 0.35,  # 成交量分析权重
            "momentum_weight": 0.30,  # 动量分析权重
            "pattern_weight": 0.20,  # 形态分析权重
            "volatility_weight": 0.15  # 波动性分析权重
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")

        # 数据预处理
        df = data.copy()
        for column in ['open', 'high', 'low', 'close', 'volume']:
            df[column] = df[column].astype(float)

        # 初始化信号DataFrame
        signals = pd.DataFrame(index=df.index)

        try:
            # 1. 计算技术指标
            df = self._calculate_indicators(df)

            # 2. 计算各维度得分
            volume_score = self._analyze_volume(df)
            momentum_score = self._analyze_momentum(df)
            pattern_score = self._analyze_pattern(df)
            volatility_score = self._analyze_volatility(df)

            # 3. 计算综合得分
            final_score = (
                    volume_score * self._params['volume_weight'] +
                    momentum_score * self._params['momentum_weight'] +
                    pattern_score * self._params['pattern_weight'] +
                    volatility_score * self._params['volatility_weight']
            )

            # 4. 生成信号和详细信息
            signals['signal'] = round(final_score * 100, 2)
            signals['trade_date'] = df['trade_date']
            signals['price'] = df['close']
            signals['volume_ratio'] = (df['volume'] / df['volume_ma20']).replace([np.inf, -np.inf], np.nan).fillna(1)
            signals['rsi'] = df['rsi'].replace([np.inf, -np.inf], np.nan).fillna(50)
            signals['price_to_ma'] = ((df['close'] - df['ma20']) / df['ma20'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)
            signals['macd_hist'] = df['macd_hist'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # 获取最新一天的信号并确保所有值都是有效的JSON值
            latest_signal = signals.iloc[-1]
            # 将所有无效值替换为0
            latest_signal = latest_signal.replace([np.inf, -np.inf, np.nan], 0)
            
            return latest_signal

        except Exception as e:
            print(f"计算过程出错: {str(e)}")
            return pd.Series({'signal': 0})

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算变化率
        df['volume_pct'] = df['volume'].pct_change()

        # 计算均线
        df['ma20'] = CalIndicators.ema(df, self._params['bb_period'], 'close')
        df['volume_ma20'] = CalIndicators.ema(df, self._params['volume_ma'], 'volume')

        # 计算RSI
        df['rsi'] = CalIndicators.rsi(df, self._params['rsi_period'])

        # 计算MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = CalIndicators.macd(df)

        # 计算布林带
        df['mid_band'], df['upper_band'], df['lower_band'] = CalIndicators.bollinger_bands(df, ma_period=self._params[
            'bb_period'], bollinger_k=self._params['bb_std'])

        return df

    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """分析成交量异动"""
        try:
            # 计算最近5天的成交量相对于20日均量的比值
            recent_volume_ratio = (df['volume'].iloc[-self._params['recent_days']:] /
                                   df['volume_ma20'].iloc[-self._params['recent_days']:]).mean()

            # 计算量价配合度
            price_volume_coord = np.corrcoef(
                df['pct_chg'].iloc[-self._params['recent_days']:],
                df['volume_pct'].iloc[-self._params['recent_days']:]
            )[0, 1]
            
            # 处理 nan 和 inf 值
            if np.isnan(price_volume_coord) or np.isinf(price_volume_coord):
                price_volume_coord = 0
            
            if np.isnan(recent_volume_ratio) or np.isinf(recent_volume_ratio):
                recent_volume_ratio = 1

            # 归一化处理
            volume_score = (
                    min(recent_volume_ratio / 3, 1) * 0.6 +
                    (price_volume_coord + 1) / 2 * 0.4
            )

            return float(min(max(volume_score, 0), 1))  # 确保返回值在 0-1 之间
        except:
            return 0.0

    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """分析动量指标"""
        try:
            # RSI指标评分
            rsi = df['rsi'].iloc[-1]
            if np.isnan(rsi) or np.isinf(rsi):
                rsi = 50
            rsi_score = 1 - abs(rsi - 55) / 45  # RSI在55附近最优

            # MACD金叉判断
            macd_cross = (df['macd_hist'].iloc[-1] > 0 and df['macd_hist'].iloc[-2] < 0)

            # 近期收益率
            recent_returns = df['pct_chg'].iloc[-self._params['recent_days']:].mean()
            if np.isnan(recent_returns) or np.isinf(recent_returns):
                recent_returns = 0

            momentum_score = (
                    max(min(rsi_score, 1), 0) * 0.4 +
                    (1 if macd_cross else 0) * 0.3 +
                    min(max(recent_returns * 20, 0), 1) * 0.3
            )

            return float(min(max(momentum_score, 0), 1))
        except:
            return 0.0

    def _analyze_pattern(self, df: pd.DataFrame) -> float:
        """分析价格形态"""
        try:
            # 突破布林带上轨
            resistance_break = (df['close'].iloc[-1] > df['upper_band'].iloc[-1])

            # 底部企稳（价格站上20日均线）
            low_stable = (df['close'].iloc[-1] > df['ma20'].iloc[-1] and
                          df['close'].iloc[-self._params['recent_days']:].min() >
                          df['ma20'].iloc[-self._params['recent_days']:].min())

            pattern_score = (
                    (1 if resistance_break else 0) * 0.5 +
                    (1 if low_stable else 0) * 0.5
            )

            return float(min(max(pattern_score, 0), 1))
        except:
            return 0.0

    @staticmethod
    def _analyze_volatility(df: pd.DataFrame) -> float:
        """分析波动性"""
        try:
            # 布林带宽度变化
            bb_width = (df['upper_band'] - df['lower_band']) / df['ma20']
            bb_width_ratio = bb_width.iloc[-1] / bb_width.iloc[-10:].mean()
            
            if np.isnan(bb_width_ratio) or np.isinf(bb_width_ratio):
                bb_width_ratio = 1

            # 价格回撤程度
            drawdown = (df['close'].iloc[-1] - df['close'].iloc[-20:].max()) / df['close'].iloc[-20:].max()
            if np.isnan(drawdown) or np.isinf(drawdown):
                drawdown = 0

            volatility_score = (
                    min(bb_width_ratio, 1) * 0.5 +
                    min(max(1 + drawdown * 2, 0), 1) * 0.5
            )

            return float(min(max(volatility_score, 0), 1))
        except:
            return 0.0
