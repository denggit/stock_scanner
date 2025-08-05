#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:32 PM
@File       : breakout.py
@Description: 突破策略 - 识别价格突破关键阻力位的交易机会
"""

import pandas as pd

from backend.business.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class BreakoutStrategy(BaseStrategy):
    """
    突破策略
    
    策略逻辑：
    1. 识别关键阻力位和支撑位
    2. 检测价格突破阻力位
    3. 结合成交量确认突破有效性
    4. 计算突破强度和信号评分
    """

    def __init__(self):
        super().__init__(name="突破策略", description="识别价格突破关键阻力位的交易机会")
        self._init_params()

    def _init_params(self):
        """初始化策略参数"""
        self._params = {
            "lookback_period": 20,  # 回溯周期，用于识别阻力位
            "breakout_threshold": 0.02,  # 突破阈值，价格超过阻力位的百分比
            "volume_ratio": 1.5,  # 成交量放大倍数
            "confirmation_days": 2,  # 突破确认天数
            "resistance_levels": 3,  # 识别阻力位数量
            "min_price": 5.0,  # 最小股价
            "max_price": 200.0,  # 最大股价
            "weights": {  # 信号强度评分权重
                "breakout_strength": 0.4,  # 突破强度权重
                "volume_confirmation": 0.3,  # 成交量确认权重
                "trend_strength": 0.3  # 趋势强度权重
            }
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            data: pd.DataFrame, 股票数据
            
        Returns:
            pd.Series: 交易信号
        """
        if not self.validate_data(data):
            return pd.Series({
                'signal': 0,
                'signal_strength': 0,
                'trade_date': None,
                'price': 0,
                'breakout_strength': 0,
                'resistance_price': 0,
                'volume_ratio': 0,
                'volume_confirmation': 0,
                'trend_strength': 0,
                'rsi': 0,
                'macd': 0,
                'ma20': 0,
                'bb_position': 0
            })

        try:
            # 确保数据按时间升序排序
            data = data.sort_index(ascending=True).copy()

            # 确保价格数据为float类型
            data["close"] = data["close"].astype(float)
            data["high"] = data["high"].astype(float)
            data["low"] = data["low"].astype(float)
            data["volume"] = data["volume"].astype(float)

            # 过滤价格范围
            price_filter = (data['close'] >= self._params['min_price']) & (data['close'] <= self._params['max_price'])
            if not price_filter.any():
                return pd.Series({
                    'signal': 0,
                    'signal_strength': 0,
                    'trade_date': data.index[-1] if len(data) > 0 else None,
                    'price': data['close'].iloc[-1] if len(data) > 0 else 0,
                    'breakout_strength': 0,
                    'resistance_price': 0,
                    'volume_ratio': 0,
                    'volume_confirmation': 0,
                    'trend_strength': 0,
                    'rsi': 0,
                    'macd': 0,
                    'ma20': 0,
                    'bb_position': 0
                })

            data = data[price_filter]

            if len(data) < self._params['lookback_period'] + 5:
                return pd.Series({
                    'signal': 0,
                    'signal_strength': 0,
                    'trade_date': data.index[-1] if len(data) > 0 else None,
                    'price': data['close'].iloc[-1] if len(data) > 0 else 0,
                    'breakout_strength': 0,
                    'resistance_price': 0,
                    'volume_ratio': 0,
                    'volume_confirmation': 0,
                    'trend_strength': 0,
                    'rsi': 0,
                    'macd': 0,
                    'ma20': 0,
                    'bb_position': 0
                })

            # 计算技术指标
            data = self._calculate_indicators(data)

            # 识别阻力位
            resistance_levels = self._identify_resistance_levels(data)

            # 检测突破
            breakout_signals = self._detect_breakouts(data, resistance_levels)

            # 生成最终信号
            signal = self._generate_final_signal(data, breakout_signals)

            return signal

        except Exception as e:
            # 记录错误并返回无信号
            return pd.Series({
                'signal': 0,
                'signal_strength': 0,
                'trade_date': data.index[-1] if len(data) > 0 else None,
                'price': data['close'].iloc[-1] if len(data) > 0 else 0,
                'breakout_strength': 0,
                'resistance_price': 0,
                'volume_ratio': 0,
                'volume_confirmation': 0,
                'trend_strength': 0,
                'rsi': 0,
                'macd': 0,
                'ma20': 0,
                'bb_position': 0
            })

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算移动平均线
        data['ma5'] = CalIndicators.sma(data, period=5, cal_value='close')
        data['ma10'] = CalIndicators.sma(data, period=10, cal_value='close')
        data['ma20'] = CalIndicators.sma(data, period=20, cal_value='close')

        # 计算成交量均线
        data['volume_ma5'] = CalIndicators.sma(data, period=5, cal_value='volume')
        data['volume_ma20'] = CalIndicators.sma(data, period=20, cal_value='volume')

        # 计算RSI
        data['rsi'] = CalIndicators.rsi(data, period=14)

        # 计算MACD
        data['macd'], data['macd_signal'], data['macd_hist'] = CalIndicators.macd(
            data, fast_period=12, slow_period=26, signal_period=9
        )

        # 计算布林带
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = CalIndicators.bollinger_bands(
            data, period=20, std_dev=2
        )

        # 计算价格变化率
        data['pct_chg'] = data['close'].pct_change()
        data['volume_ratio'] = data['volume'] / data['volume_ma20']

        return data

    def _identify_resistance_levels(self, data: pd.DataFrame) -> list:
        """识别阻力位"""
        resistance_levels = []
        lookback = self._params['lookback_period']

        for i in range(lookback, len(data) - 1):
            # 检查是否是局部高点
            if (data['high'].iloc[i] > data['high'].iloc[i - 1] and
                    data['high'].iloc[i] > data['high'].iloc[i + 1]):

                # 检查是否在后续期间被多次测试
                test_count = 0
                resistance_price = data['high'].iloc[i]

                for j in range(i + 1, min(i + lookback, len(data))):
                    if (data['high'].iloc[j] >= resistance_price * 0.98 and
                            data['high'].iloc[j] <= resistance_price * 1.02):
                        test_count += 1

                if test_count >= 2:  # 至少被测试2次
                    resistance_levels.append({
                        'price': resistance_price,
                        'date': data.index[i],
                        'strength': test_count
                    })

        # 按强度排序并去重
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)

        # 去重：如果两个阻力位价格相近，保留强度更高的
        unique_levels = []
        for level in resistance_levels:
            is_duplicate = False
            for existing in unique_levels:
                if abs(level['price'] - existing['price']) / existing['price'] < 0.02:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_levels.append(level)

        return unique_levels[:self._params['resistance_levels']]

    def _detect_breakouts(self, data: pd.DataFrame, resistance_levels: list) -> pd.DataFrame:
        """检测突破"""
        signals = pd.DataFrame(index=data.index)
        signals['breakout_signal'] = 0
        signals['breakout_strength'] = 0.0
        signals['resistance_price'] = 0.0
        signals['volume_confirmation'] = 0.0
        signals['trend_strength'] = 0.0

        if not resistance_levels:
            return signals

        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_volume_ratio = data['volume_ratio'].iloc[i]

            # 检查是否突破任何阻力位
            for resistance in resistance_levels:
                resistance_price = resistance['price']

                # 检查突破条件
                if (current_price > resistance_price * (1 + self._params['breakout_threshold']) and
                        current_volume_ratio >= self._params['volume_ratio']):
                    # 计算突破强度
                    breakout_strength = (current_price - resistance_price) / resistance_price

                    # 计算成交量确认度
                    volume_confirmation = min(current_volume_ratio / self._params['volume_ratio'], 3.0)

                    # 计算趋势强度
                    trend_strength = self._calculate_trend_strength(data, i)

                    # 更新信号
                    signals.loc[data.index[i], 'breakout_signal'] = 1
                    signals.loc[data.index[i], 'breakout_strength'] = breakout_strength
                    signals.loc[data.index[i], 'resistance_price'] = resistance_price
                    signals.loc[data.index[i], 'volume_confirmation'] = volume_confirmation
                    signals.loc[data.index[i], 'trend_strength'] = trend_strength

                    break

        return signals

    def _calculate_trend_strength(self, data: pd.DataFrame, current_idx: int) -> float:
        """计算趋势强度"""
        if current_idx < 5:
            return 0.0

        # 计算短期趋势
        recent_prices = data['close'].iloc[current_idx - 5:current_idx + 1]
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

        # 计算均线趋势
        ma_trend = 0.0
        if current_idx >= 20:
            ma20_current = data['ma20'].iloc[current_idx]
            ma20_prev = data['ma20'].iloc[current_idx - 5]
            if ma20_prev > 0:
                ma_trend = (ma20_current - ma20_prev) / ma20_prev

        # 计算MACD趋势
        macd_trend = 0.0
        if current_idx >= 1:
            macd_current = data['macd'].iloc[current_idx]
            macd_prev = data['macd'].iloc[current_idx - 1]
            if macd_prev != 0:
                macd_trend = (macd_current - macd_prev) / abs(macd_prev)

        # 综合趋势强度
        trend_strength = (price_trend * 0.5 + ma_trend * 0.3 + macd_trend * 0.2)
        return max(0.0, min(1.0, trend_strength))

    def _generate_final_signal(self, data: pd.DataFrame, breakout_signals: pd.DataFrame) -> pd.Series:
        """生成最终信号"""
        # 获取最新数据
        latest_idx = data.index[-1]
        latest_data = data.iloc[-1]
        latest_signals = breakout_signals.loc[latest_idx]

        # 检查是否有突破信号
        if latest_signals['breakout_signal'] == 0:
            return pd.Series({'signal': 0})

        # 计算综合得分
        weights = self._params['weights']
        total_score = (
                latest_signals['breakout_strength'] * weights['breakout_strength'] +
                latest_signals['volume_confirmation'] * weights['volume_confirmation'] +
                latest_signals['trend_strength'] * weights['trend_strength']
        )

        # 生成信号
        signal = pd.Series({
            'signal': 1,
            'signal_strength': round(total_score * 100, 2),
            'trade_date': latest_data.name,
            'price': round(float(latest_data['close']), 2),
            'breakout_strength': round(float(latest_signals['breakout_strength'] * 100), 2),
            'resistance_price': round(float(latest_signals['resistance_price']), 2),
            'volume_ratio': round(float(latest_data['volume_ratio']), 2),
            'volume_confirmation': round(float(latest_signals['volume_confirmation']), 2),
            'trend_strength': round(float(latest_signals['trend_strength'] * 100), 2),
            'rsi': round(float(latest_data['rsi']), 2),
            'macd': round(float(latest_data['macd']), 4),
            'ma20': round(float(latest_data['ma20']), 2),
            'bb_position': round(float((latest_data['close'] - latest_data['bb_lower']) /
                                       (latest_data['bb_upper'] - latest_data['bb_lower'])), 2)
        })

        return signal
