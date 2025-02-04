#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:32 PM
@File       : ma_pullback.py
@Description: 
"""

import pandas as pd

from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class MAPullbackStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="均线回踩策略", description="股价回踩均线且符合多重技术条件的交易策略")
        self._params = {
            "ma_period": 20,  # 均线周期
            "lookback_period": 10,  # 回溯周期
            "period_margin": 0.01,  # 价格误差范围
            "volume_ratio": 1.5,  # 成交量比率
            "min_pullback_count": 2,  # 最小回踩次数
            "weights": {  # 信号强度评分权重
                "price": 0.4,  # 价格权重
                "volume": 0.3,  # 成交量权重
                "frequency": 0.3  # 回踩频率权重
            }
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        参数:
        data: pd.DataFrame, 股票数据

        返回:
        pd.Series, 交易信号
        """
        if not self.validate_data(data):
            raise ValueError("数据格式错误")

        # 确保数据按时间升序排序
        data = data.sort_index(ascending=True)

        # 确保价格数据为float类型
        data["close"] = data["close"].astype(float)
        data["volume"] = data["volume"].astype(float)

        # 计算均线
        ma = CalIndicators.ema(data, period=self._params["ma_period"], cal_value="close")
        volume_ma = CalIndicators.sma(data, period=5, cal_value="volume")

        # 处理 MA 为空的情况
        valid_ma = ma.notna()
        valid_volume_ma = volume_ma.notna()

        # 计算价格与均线的偏离度（只在有效的MA值上计算）
        price_distance = pd.Series(index=data.index, dtype=float)
        price_distance.loc[valid_ma] = abs(
            (data.loc[valid_ma, 'close'].astype(float) - ma[valid_ma]) / ma[valid_ma]
        )

        # 计算成交量与均量的偏离度（只在有效的MA值上计算）
        volume_ratio = pd.Series(index=data.index, dtype=float)
        volume_ratio.loc[valid_volume_ma] = data.loc[valid_volume_ma, 'volume'].astype(float) / volume_ma[
            valid_volume_ma]

        # 计算回踩次数
        pullback_count = self._calculate_pullback_count(data, ma)

        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['code'] = data['code']
        signals['trade_date'] = data['trade_date']
        signals['price'] = data['close']
        signals['pct_chg'] = data['pct_chg']
        signals['ma_price'] = ma
        signals['price_to_ma'] = pd.Series(index=data.index, dtype=float)
        signals.loc[valid_ma, 'price_to_ma'] = (
                (data.loc[valid_ma, 'close'] - ma[valid_ma]) / ma[valid_ma] * 100).round(2)  # 价格与均线的偏离度（%）
        signals['volume_ratio'] = volume_ratio.round(2)  # 成交量与过去五日交易均量比例
        signals['signal'] = 0
        signals['signal_strength'] = 0

        # 基本回踩条件
        basic_condition = (
                (price_distance <= self._params['price_margin']) &
                (volume_ratio >= self._params['volume_ratio']) &
                (pullback_count >= self._params['min_pullback_count'])
        )

        # 计算符合条件股票的信号强度
        signals.loc[basic_condition, 'signal'] = 1

        # 计算各个分项得分
        price_score = self._calculate_price_score(price_distance)
        volume_score = self._calculate_volume_score(volume_ratio)
        frequency_score = self._calculate_frequency_score(pullback_count)

        # 计算综合得分
        w = self._params['weights']
        total_score = (
                price_score * w['price'] +
                volume_score * w['volume'] +
                frequency_score * w['frequency']
        )
        signals['total_score'] = total_score

        # 将得分转换为1-5的信号强度
        signals.loc[basic_condition, 'signal_strength'] = ((total_score * 0.4) + 1).clip(1, 5).round()

        signals = signals.astype({
            'code': str,
            'trade_date': str,
            'price': float,
            'pct_chg': float,
            'ma_price': float,
            'price_to_ma': float,
            'volume_ratio': float,
            'signal': int,
            'signal_strength': int,
            'total_score': float
        })

        # 只返回最新一天的价格
        last_day = signals.iloc[-1]

        return last_day

    def _calculate_pullback_count(self, data: pd.DataFrame, ma: pd.Series) -> pd.Series:
        """
        计算回踩次数
        """
        lookback_period = self._params["lookback_period"]
        price_margin = self._params["period_margin"]

        # 计算每天是否发生回踩
        is_pullback = (
                (abs((data["close"] - ma) / ma) <= price_margin) &
                (data["close"] < ma)
        )

        # 计算回踩次数
        pullback_count = is_pullback.rolling(window=lookback_period).sum()

        return pullback_count

    def _calculate_price_score(self, price_distance: pd.Series) -> pd.Series:
        """计算价格偏离度得分（0-10分）
        距离越小，得分越高
        - 偏离 <= 0.2%: 10分
        - 偏离 1%: 5分
        - 偏离 2% 或以上: 0分
        """
        # 创建一个与输入相同大小的零分数组，使用float类型
        price_score = pd.Series(0.0, index=price_distance.index)

        # 偏离 <= 0.2%: 10分
        mask_1 = price_distance <= 0.002
        price_score[mask_1] = 10.0

        # 偏离在 0.2% - 1% 之间：10-5分，线性递减
        mask_2 = (price_distance > 0.002) & (price_distance <= 0.01)
        price_score[mask_2] = 10.0 - (price_distance[mask_2] - 0.002) * (5.0 / 0.008)

        # 偏离在 1% - 2% 之间：5-0分，线性递减
        mask_3 = (price_distance > 0.01) & (price_distance <= 0.02)
        price_score[mask_3] = 5.0 - (price_distance[mask_3] - 0.01) * (5.0 / 0.01)

        return price_score

    def _calculate_volume_score(self, volume_ratio: pd.Series) -> pd.Series:
        """计算成交量偏离度得分（0-10分）
        偏离越大，得分越高
        - 成交量放大 1.5 倍: 5分
        - 成交量放大 2 倍：7分
        - 成交量放大 3 倍或以上：10分
        """
        # 创建一个与输入相同大小的零分数组，使用float类型
        volume_score = pd.Series(0.0, index=volume_ratio.index)

        # 分段计算得分
        # 1.5倍 - 2倍之间：5-7分，线性增长
        mask_1 = (volume_ratio >= 1.5) & (volume_ratio < 2)
        volume_score[mask_1] = 5.0 + (volume_ratio[mask_1] - 1.5) * 4.0

        # 2倍 - 3倍之间：7-10分，线性增长
        mask_2 = (volume_ratio >= 2) & (volume_ratio < 3)
        volume_score[mask_2] = 7.0 + (volume_ratio[mask_2] - 2) * 3.0

        # 3倍及以上：10分
        mask_3 = volume_ratio >= 3
        volume_score[mask_3] = 10.0

        return volume_score

    def _calculate_frequency_score(self, pullback_count: pd.Series) -> pd.Series:
        """计算回踩频率得分（0-10分）
        回踩次数越多，得分越高
        - 回踩1此：2分
        - 回踩2次：4分
        - 回踩3次：6分
        - 回踩4次：8分
        - 回踩5次或以上：10分
        """
        return (2 * pullback_count).clip(0, 10)
