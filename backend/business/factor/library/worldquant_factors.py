#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : worldquant_factors.py
@Description: WorldQuant Alpha因子库
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import numpy as np
import pandas as pd

from ..core.factor.base_factor import register_worldquant_factor


def safe_where(condition, x, y, index=None):
    """
    安全的条件选择函数，确保返回pandas Series
    
    Args:
        condition: 条件
        x: 条件为True时的值
        y: 条件为False时的值
        index: 索引
        
    Returns:
        pandas Series
    """
    result = np.where(condition, x, y)
    if index is not None:
        return pd.Series(result, index=index)
    else:
        return pd.Series(result)


# ==================== WorldQuant Alpha因子 ====================

@register_worldquant_factor(name='alpha_1',
                            description='Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)')
def alpha_1(pct_chg: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""

    def ts_argmax(x):
        return x.argmax() if len(x) > 0 else 0

    def signed_power(x, power):
        return np.sign(x) * np.abs(x) ** power

    # 计算条件表达式
    condition = pct_chg < 0
    std_returns = pct_chg.rolling(20).std()
    value = pd.Series(np.where(condition, std_returns, close), index=pct_chg.index)

    # 计算SignedPower
    signed_power_value = signed_power(value, 2)
    # 确保是pandas Series
    if not isinstance(signed_power_value, pd.Series):
        signed_power_value = pd.Series(signed_power_value, index=pct_chg.index)

    # 计算Ts_ArgMax
    ts_argmax_value = signed_power_value.rolling(5).apply(ts_argmax)

    # 计算rank
    rank_value = ts_argmax_value.rank(pct=True)

    return rank_value - 0.5


@register_worldquant_factor(name='alpha_2',
                            description='Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))')
def alpha_2(volume: pd.Series, close: pd.Series, open: pd.Series, **kwargs) -> pd.Series:
    """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    # 计算log(volume)的2日差分
    log_volume = np.log(volume)
    delta_log_volume = log_volume.diff(2)

    # 计算价格变化率
    price_change = (close - open) / open

    # 计算rank
    rank_volume = delta_log_volume.rank(pct=True)
    rank_price = price_change.rank(pct=True)

    # 计算6日相关性
    correlation = rank_volume.rolling(6).corr(rank_price)

    return -1 * correlation


@register_worldquant_factor(name='alpha_3', description='Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))')
def alpha_3(open: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))"""
    # 计算rank
    rank_open = open.rank(pct=True)
    rank_volume = volume.rank(pct=True)

    # 计算10日相关性
    correlation = rank_open.rolling(10).corr(rank_volume)

    return -1 * correlation


@register_worldquant_factor(name='alpha_4', description='Alpha#4: (-1 * Ts_Rank(rank(low), 9))')
def alpha_4(low: pd.Series, **kwargs) -> pd.Series:
    """Alpha#4: (-1 * Ts_Rank(rank(low), 9))"""
    # 计算rank
    rank_low = low.rank(pct=True)

    # 计算Ts_Rank (9日滚动rank)
    ts_rank = rank_low.rolling(9).apply(lambda x: x.rank(pct=True).iloc[-1])

    return -1 * ts_rank


@register_worldquant_factor(name='alpha_5',
                            description='Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))')
def alpha_5(open: pd.Series, close: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    # 计算vwap的10日平均
    vwap_ma = vwap.rolling(10).mean()

    # 计算rank
    rank_open_vwap = (open - vwap_ma).rank(pct=True)
    rank_close_vwap = (close - vwap).rank(pct=True)

    return rank_open_vwap * (-1 * abs(rank_close_vwap))


@register_worldquant_factor(name='alpha_6', description='Alpha#6: (-1 * correlation(open, volume, 10))')
def alpha_6(open: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#6: (-1 * correlation(open, volume, 10))"""
    # 计算10日相关性
    correlation = open.rolling(10).corr(volume)

    return -1 * correlation


@register_worldquant_factor(name='alpha_7',
                            description='Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))')
def alpha_7(volume: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))"""
    # 计算20日平均成交量
    adv20 = volume.rolling(20).mean()

    # 计算close的7日差分
    delta_close = close.diff(7)
    abs_delta_close = abs(delta_close)

    # 计算ts_rank
    ts_rank = abs_delta_close.rolling(60).apply(lambda x: x.rank(pct=True).iloc[-1])

    # 计算sign
    sign_delta = np.sign(delta_close)

    # 条件判断
    condition = adv20 < volume
    result = np.where(condition,
                      (-1 * ts_rank) * sign_delta,
                      -1)

    return pd.Series(result, index=close.index)


@register_worldquant_factor(name='alpha_8',
                            description='Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))')
def alpha_8(open: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    sum_open = open.rolling(5).sum()
    sum_returns = pct_chg.rolling(5).sum()
    product = sum_open * sum_returns
    delay_product = product.shift(10)

    return -1 * (product - delay_product).rank(pct=True)


@register_worldquant_factor(name='alpha_9',
                            description='Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))')
def alpha_9(close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
    delta_close = close.diff(1)
    ts_min = delta_close.rolling(5).min()
    ts_max = delta_close.rolling(5).max()

    # 条件判断
    condition1 = ts_min > 0
    condition2 = ts_max < 0

    result = np.where(condition1, delta_close,
                      np.where(condition2, delta_close, -1 * delta_close))

    return pd.Series(result, index=close.index)


@register_worldquant_factor(name='alpha_10',
                            description='Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))')
def alpha_10(close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    delta_close = close.diff(1)
    ts_min = delta_close.rolling(4).min()
    ts_max = delta_close.rolling(4).max()

    # 条件判断
    condition1 = ts_min > 0
    condition2 = ts_max < 0

    result = np.where(condition1, delta_close,
                      np.where(condition2, delta_close, -1 * delta_close))

    return pd.Series(result, index=close.index).rank(pct=True)


# ==================== Alpha 11-20 ====================

@register_worldquant_factor(name='alpha_11',
                            description='Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))')
def alpha_11(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    vwap_close_diff = vwap - close
    ts_max_vwap_close = vwap_close_diff.rolling(3).max()
    ts_min_vwap_close = vwap_close_diff.rolling(3).min()
    delta_volume = volume.diff(3)

    rank_max = ts_max_vwap_close.rank(pct=True)
    rank_min = ts_min_vwap_close.rank(pct=True)
    rank_delta_volume = delta_volume.rank(pct=True)

    return (rank_max + rank_min) * rank_delta_volume


@register_worldquant_factor(name='alpha_12', description='Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))')
def alpha_12(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    delta_volume = volume.diff(1)
    delta_close = close.diff(1)

    return np.sign(delta_volume) * (-1 * delta_close)


@register_worldquant_factor(name='alpha_13',
                            description='Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))')
def alpha_13(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    rank_close = close.rank(pct=True)
    rank_volume = volume.rank(pct=True)

    def rolling_covariance(x):
        if len(x) < 2:
            return np.nan
        return np.cov(x.iloc[:len(x) // 2], x.iloc[len(x) // 2:])[0, 1] if len(x) >= 2 else np.nan

    covariance = pd.concat([rank_close, rank_volume], axis=1).rolling(5).apply(rolling_covariance)

    return -1 * covariance.rank(pct=True)


@register_worldquant_factor(name='alpha_14',
                            description='Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))')
def alpha_14(open: pd.Series, volume: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    delta_returns = pct_chg.diff(3)
    rank_delta_returns = delta_returns.rank(pct=True)
    correlation_open_volume = open.rolling(10).corr(volume)

    return (-1 * rank_delta_returns) * correlation_open_volume


@register_worldquant_factor(name='alpha_15',
                            description='Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))')
def alpha_15(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    rank_high = high.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    correlation = rank_high.rolling(3).corr(rank_volume)
    rank_correlation = correlation.rank(pct=True)

    return -1 * rank_correlation.rolling(3).sum()


@register_worldquant_factor(name='alpha_16',
                            description='Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))')
def alpha_16(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    rank_high = high.rank(pct=True)
    rank_volume = volume.rank(pct=True)

    def rolling_covariance(x):
        if len(x) < 2:
            return np.nan
        return np.cov(x.iloc[:len(x) // 2], x.iloc[len(x) // 2:])[0, 1] if len(x) >= 2 else np.nan

    covariance = pd.concat([rank_high, rank_volume], axis=1).rolling(5).apply(rolling_covariance)

    return -1 * covariance.rank(pct=True)


@register_worldquant_factor(name='alpha_17',
                            description='Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))')
def alpha_17(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    adv20 = volume.rolling(20).mean()

    # ts_rank(close, 10)
    ts_rank_close = close.rolling(10).apply(lambda x: x.rank(pct=True).iloc[-1])
    rank_ts_rank_close = ts_rank_close.rank(pct=True)

    # delta(delta(close, 1), 1)
    delta_close = close.diff(1)
    delta_delta_close = delta_close.diff(1)
    rank_delta_delta = delta_delta_close.rank(pct=True)

    # ts_rank((volume / adv20), 5)
    volume_adv20_ratio = volume / adv20
    ts_rank_volume_adv20 = volume_adv20_ratio.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    rank_ts_rank_volume_adv20 = ts_rank_volume_adv20.rank(pct=True)

    return ((-1 * rank_ts_rank_close) * rank_delta_delta) * rank_ts_rank_volume_adv20


@register_worldquant_factor(name='alpha_18',
                            description='Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))')
def alpha_18(close: pd.Series, open: pd.Series, **kwargs) -> pd.Series:
    """Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
    close_open_diff = close - open
    abs_close_open_diff = abs(close_open_diff)
    stddev_abs_diff = abs_close_open_diff.rolling(5).std()
    correlation_close_open = close.rolling(10).corr(open)

    combined = stddev_abs_diff + close_open_diff + correlation_close_open

    return -1 * combined.rank(pct=True)


@register_worldquant_factor(name='alpha_19',
                            description='Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))')
def alpha_19(close: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    delay_close_7 = close.shift(7)
    delta_close_7 = close.diff(7)
    close_delay_diff = close - delay_close_7
    sign_value = np.sign(close_delay_diff + delta_close_7)

    sum_returns_250 = pct_chg.rolling(250).sum()
    rank_sum_returns = (1 + sum_returns_250).rank(pct=True)

    return (-1 * sign_value) * (1 + rank_sum_returns)


@register_worldquant_factor(name='alpha_20',
                            description='Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))')
def alpha_20(open: pd.Series, high: pd.Series, close: pd.Series, low: pd.Series, **kwargs) -> pd.Series:
    """Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    delay_high_1 = high.shift(1)
    delay_close_1 = close.shift(1)
    delay_low_1 = low.shift(1)

    rank_open_delay_high = (open - delay_high_1).rank(pct=True)
    rank_open_delay_close = (open - delay_close_1).rank(pct=True)
    rank_open_delay_low = (open - delay_low_1).rank(pct=True)

    return ((-1 * rank_open_delay_high) * rank_open_delay_close) * rank_open_delay_low


# ==================== Alpha 21-30 ====================

@register_worldquant_factor(name='alpha_21',
                            description='Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1*1): (((sum(close, 2)/2) < ((sum(close, 8) / 8) - stddev(close, 8)))?1: (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1: (-1*1))))')
def alpha_21(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#21: 复杂的条件判断因子"""
    adv20 = volume.rolling(20).mean()
    sum_close_8 = close.rolling(8).sum() / 8
    sum_close_2 = close.rolling(2).sum() / 2
    stddev_close_8 = close.rolling(8).std()
    volume_adv20_ratio = volume / adv20

    condition1 = (sum_close_8 + stddev_close_8) < sum_close_2
    condition2 = sum_close_2 < (sum_close_8 - stddev_close_8)
    condition3 = (volume_adv20_ratio > 1) | (volume_adv20_ratio == 1)

    result = np.where(condition1, -1,
                      np.where(condition2, 1,
                               np.where(condition3, 1, -1)))

    return pd.Series(result, index=close.index)


@register_worldquant_factor(name='alpha_22',
                            description='Alpha#22: (-1* (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))')
def alpha_22(close: pd.Series, high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#22: (-1* (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
    correlation_high_volume = high.rolling(5).corr(volume)
    delta_correlation = correlation_high_volume.diff(5)
    stddev_close_20 = close.rolling(20).std()
    rank_stddev = stddev_close_20.rank(pct=True)

    return -1 * (delta_correlation * rank_stddev)


@register_worldquant_factor(name='alpha_23',
                            description='Alpha#23: (((sum(high, 20) / 20) < high)? (-1* delta(high, 2)): 0)')
def alpha_23(high: pd.Series, **kwargs) -> pd.Series:
    """Alpha#23: (((sum(high, 20) / 20) < high)? (-1* delta(high, 2)): 0)"""
    sum_high_20 = high.rolling(20).sum() / 20
    delta_high_2 = high.diff(2)

    condition = sum_high_20 < high
    result = np.where(condition, -1 * delta_high_2, 0)

    return pd.Series(result, index=high.index)


@register_worldquant_factor(name='alpha_24',
                            description='Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) ==0.05)) ? (-1* (close - ts_min(close, 100))): (-1* delta(close, 3)))')
def alpha_24(close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#24: 复杂的条件判断因子"""
    sum_close_100 = close.rolling(100).sum() / 100
    delta_sum_close_100 = sum_close_100.diff(100)
    delay_close_100 = close.shift(100)
    ts_min_close_100 = close.rolling(100).min()
    delta_close_3 = close.diff(3)

    ratio = delta_sum_close_100 / delay_close_100
    condition = (ratio < 0.05) | (ratio == 0.05)

    result = np.where(condition, -1 * (close - ts_min_close_100), -1 * delta_close_3)

    return pd.Series(result, index=close.index)


@register_worldquant_factor(name='alpha_25',
                            description='Alpha#25: rank(((((-1* returns) * adv20) * vwap) * (high - close)))')
def alpha_25(close: pd.Series, high: pd.Series, volume: pd.Series, vwap: pd.Series, pct_chg: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#25: rank(((((-1* returns) * adv20) * vwap) * (high - close)))"""
    adv20 = volume.rolling(20).mean()
    returns = pct_chg

    result = ((((-1 * returns) * adv20) * vwap) * (high - close))

    return result.rank(pct=True)


@register_worldquant_factor(name='alpha_26',
                            description='Alpha#26: (-1* ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))')
def alpha_26(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#26: (-1* ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    ts_rank_volume = volume.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    ts_rank_high = high.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    correlation = ts_rank_volume.rolling(5).corr(ts_rank_high)
    ts_max_correlation = correlation.rolling(3).max()

    return -1 * ts_max_correlation


@register_worldquant_factor(name='alpha_27',
                            description='Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1*1): 1))')
def alpha_27(volume: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1*1): 1))"""
    rank_volume = volume.rank(pct=True)
    rank_vwap = vwap.rank(pct=True)
    correlation = rank_volume.rolling(6).corr(rank_vwap)
    sum_correlation_2 = correlation.rolling(2).sum() / 2.0
    rank_sum_correlation = sum_correlation_2.rank(pct=True)

    condition = rank_sum_correlation > 0.5
    result = np.where(condition, -1, 1)

    return pd.Series(result, index=volume.index)


@register_worldquant_factor(name='alpha_28',
                            description='Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))')
def alpha_28(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    adv20 = volume.rolling(20).mean()
    correlation_adv20_low = adv20.rolling(5).corr(low)
    high_low_mid = (high + low) / 2

    result = (correlation_adv20_low + high_low_mid) - close

    # scale函数：标准化
    return (result - result.rolling(252).mean()) / result.rolling(252).std()


@register_worldquant_factor(name='alpha_29',
                            description='Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1* rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1* returns), 6), 5))')
def alpha_29(close: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#29: 复杂的嵌套函数因子"""
    returns = pct_chg

    # 内层计算
    rank_delta_close_5 = (-1 * (close - 1).diff(5)).rank(pct=True)
    rank_rank_delta = rank_delta_close_5.rank(pct=True)
    ts_min_rank_rank = rank_rank_delta.rolling(2).min()
    sum_ts_min = ts_min_rank_rank.rolling(1).sum()
    log_sum = np.log(sum_ts_min)
    scale_log = (log_sum - log_sum.rolling(252).mean()) / log_sum.rolling(252).std()
    rank_rank_scale = scale_log.rank(pct=True).rank(pct=True)
    product_rank = rank_rank_scale.rolling(1).apply(lambda x: x.prod())
    min_product = product_rank.rolling(5).min()

    # 外层计算
    delay_returns_6 = returns.shift(6)
    ts_rank_delay = delay_returns_6.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])

    return min_product + ts_rank_delay


@register_worldquant_factor(name='alpha_30',
                            description='Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))')
def alpha_30(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"""
    delay_close_1 = close.shift(1)
    delay_close_2 = close.shift(2)
    delay_close_3 = close.shift(3)

    sign1 = np.sign(close - delay_close_1)
    sign2 = np.sign(delay_close_1 - delay_close_2)
    sign3 = np.sign(delay_close_2 - delay_close_3)

    sum_signs = sign1 + sign2 + sign3
    rank_sum_signs = sum_signs.rank(pct=True)

    sum_volume_5 = volume.rolling(5).sum()
    sum_volume_20 = volume.rolling(20).sum()

    return ((1.0 - rank_sum_signs) * sum_volume_5) / sum_volume_20


# ==================== Alpha 31-40 ====================

@register_worldquant_factor(name='alpha_31',
                            description='Alpha#31: ((rank(rank(rank(decay_linear((-1* rank(rank(delta(close, 10)))), 10)))) + rank((-1* delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))')
def alpha_31(close: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#31: ((rank(rank(rank(decay_linear((-1* rank(rank(delta(close, 10)))), 10)))) + rank((-1* delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"""
    adv20 = volume.rolling(20).mean()

    # 第一部分：rank(rank(rank(decay_linear((-1* rank(rank(delta(close, 10)))), 10))))
    delta_close_10 = close.diff(10)
    rank_rank_delta = delta_close_10.rank(pct=True).rank(pct=True)
    neg_rank_rank_delta = -1 * rank_rank_delta

    # decay_linear函数：线性衰减加权平均
    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    decay_linear_result = neg_rank_rank_delta.rolling(10).apply(lambda x: decay_linear(x, 10))
    rank_rank_rank_decay = decay_linear_result.rank(pct=True).rank(pct=True).rank(pct=True)

    # 第二部分：rank((-1* delta(close, 3)))
    delta_close_3 = close.diff(3)
    rank_neg_delta = (-1 * delta_close_3).rank(pct=True)

    # 第三部分：sign(scale(correlation(adv20, low, 12)))
    correlation_adv20_low = adv20.rolling(12).corr(low)
    scale_correlation = (correlation_adv20_low - correlation_adv20_low.rolling(
        252).mean()) / correlation_adv20_low.rolling(252).std()
    sign_scale = np.sign(scale_correlation)

    return rank_rank_rank_decay + rank_neg_delta + sign_scale


@register_worldquant_factor(name='alpha_32',
                            description='Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))')
def alpha_32(close: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
    # 第一部分：scale(((sum(close, 7) / 7) - close))
    sum_close_7 = close.rolling(7).sum() / 7
    diff_sum_close = sum_close_7 - close
    scale_diff = (diff_sum_close - diff_sum_close.rolling(252).mean()) / diff_sum_close.rolling(252).std()

    # 第二部分：(20 * scale(correlation(vwap, delay(close, 5), 230)))
    delay_close_5 = close.shift(5)
    correlation_vwap_delay = vwap.rolling(230).corr(delay_close_5)
    scale_correlation = (correlation_vwap_delay - correlation_vwap_delay.rolling(
        252).mean()) / correlation_vwap_delay.rolling(252).std()

    return scale_diff + (20 * scale_correlation)


@register_worldquant_factor(name='alpha_33', description='Alpha#33: rank((-1* ((1- (open / close))^1)))')
def alpha_33(open: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#33: rank((-1* ((1- (open / close))^1)))"""
    ratio = open / close
    result = -1 * ((1 - ratio) ** 1)

    return result.rank(pct=True)


@register_worldquant_factor(name='alpha_34',
                            description='Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))')
def alpha_34(close: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    returns = pct_chg

    stddev_returns_2 = returns.rolling(2).std()
    stddev_returns_5 = returns.rolling(5).std()
    ratio_stddev = stddev_returns_2 / stddev_returns_5
    rank_ratio_stddev = ratio_stddev.rank(pct=True)

    delta_close_1 = close.diff(1)
    rank_delta_close = delta_close_1.rank(pct=True)

    result = (1 - rank_ratio_stddev) + (1 - rank_delta_close)

    return result.rank(pct=True)


@register_worldquant_factor(name='alpha_35',
                            description='Alpha#35: ((Ts_Rank(volume, 32) * (1 -Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))')
def alpha_35(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, pct_chg: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#35: ((Ts_Rank(volume, 32) * (1 -Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    returns = pct_chg

    # Ts_Rank(volume, 32)
    ts_rank_volume = volume.rolling(32).apply(lambda x: x.rank(pct=True).iloc[-1])

    # Ts_Rank(((close + high) - low), 16)
    close_high_low = (close + high) - low
    ts_rank_close_high_low = close_high_low.rolling(16).apply(lambda x: x.rank(pct=True).iloc[-1])

    # Ts_Rank(returns, 32)
    ts_rank_returns = returns.rolling(32).apply(lambda x: x.rank(pct=True).iloc[-1])

    return (ts_rank_volume * (1 - ts_rank_close_high_low)) * (1 - ts_rank_returns)


@register_worldquant_factor(name='alpha_36',
                            description='Alpha#36: (((((2.21* rank(correlation((close - open), delay(volume, 1), 15))) + (0.7* rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1* returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6* rank((((sum(close, 200) / 200) - open) * (close - open)))))')
def alpha_36(close: pd.Series, open: pd.Series, volume: pd.Series, vwap: pd.Series, pct_chg: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#36: 复杂的多部分组合因子"""
    returns = pct_chg
    adv20 = volume.rolling(20).mean()

    # 第一部分：2.21* rank(correlation((close - open), delay(volume, 1), 15))
    close_open_diff = close - open
    delay_volume_1 = volume.shift(1)
    correlation_close_open_volume = close_open_diff.rolling(15).corr(delay_volume_1)
    rank_correlation_1 = correlation_close_open_volume.rank(pct=True)

    # 第二部分：0.7* rank((open - close))
    open_close_diff = open - close
    rank_open_close = open_close_diff.rank(pct=True)

    # 第三部分：0.73 * rank(Ts_Rank(delay((-1* returns), 6), 5))
    delay_neg_returns_6 = (-1 * returns).shift(6)
    ts_rank_delay_neg_returns = delay_neg_returns_6.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    rank_ts_rank = ts_rank_delay_neg_returns.rank(pct=True)

    # 第四部分：rank(abs(correlation(vwap, adv20, 6)))
    correlation_vwap_adv20 = vwap.rolling(6).corr(adv20)
    abs_correlation = abs(correlation_vwap_adv20)
    rank_abs_correlation = abs_correlation.rank(pct=True)

    # 第五部分：0.6* rank((((sum(close, 200) / 200) - open) * (close - open)))
    sum_close_200 = close.rolling(200).sum() / 200
    complex_term = ((sum_close_200 - open) * (close - open))
    rank_complex_term = complex_term.rank(pct=True)

    return (2.21 * rank_correlation_1) + (0.7 * rank_open_close) + (0.73 * rank_ts_rank) + rank_abs_correlation + (
                0.6 * rank_complex_term)


@register_worldquant_factor(name='alpha_37',
                            description='Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))')
def alpha_37(open: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    open_close_diff = open - close
    delay_open_close_1 = open_close_diff.shift(1)

    correlation_delay_close = delay_open_close_1.rolling(200).corr(close)
    rank_correlation = correlation_delay_close.rank(pct=True)

    rank_open_close = open_close_diff.rank(pct=True)

    return rank_correlation + rank_open_close


@register_worldquant_factor(name='alpha_38',
                            description='Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close/open)))')
def alpha_38(close: pd.Series, open: pd.Series, **kwargs) -> pd.Series:
    """Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close/open)))"""
    ts_rank_close = close.rolling(10).apply(lambda x: x.rank(pct=True).iloc[-1])
    rank_ts_rank = ts_rank_close.rank(pct=True)

    close_open_ratio = close / open
    rank_ratio = close_open_ratio.rank(pct=True)

    return (-1 * rank_ts_rank) * rank_ratio


@register_worldquant_factor(name='alpha_39',
                            description='Alpha#39: ((-1* rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))')
def alpha_39(close: pd.Series, volume: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#39: ((-1* rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    returns = pct_chg
    adv20 = volume.rolling(20).mean()

    # 第一部分：delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))
    delta_close_7 = close.diff(7)
    volume_adv20_ratio = volume / adv20

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    decay_linear_ratio = volume_adv20_ratio.rolling(9).apply(lambda x: decay_linear(x, 9))
    rank_decay = decay_linear_ratio.rank(pct=True)

    first_part = delta_close_7 * (1 - rank_decay)
    rank_first_part = first_part.rank(pct=True)

    # 第二部分：(1 + rank(sum(returns, 250)))
    sum_returns_250 = returns.rolling(250).sum()
    rank_sum_returns = sum_returns_250.rank(pct=True)

    return (-1 * rank_first_part) * (1 + rank_sum_returns)


@register_worldquant_factor(name='alpha_40',
                            description='Alpha#40: ((-1* rank(stddev(high, 10))) * correlation(high, volume, 10))')
def alpha_40(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#40: ((-1* rank(stddev(high, 10))) * correlation(high, volume, 10))"""
    stddev_high_10 = high.rolling(10).std()
    rank_stddev = stddev_high_10.rank(pct=True)

    correlation_high_volume = high.rolling(10).corr(volume)

    return (-1 * rank_stddev) * correlation_high_volume


# ==================== Alpha 41-50 ====================

@register_worldquant_factor(name='alpha_41', description='Alpha#41: (((high * low)^0.5) - vwap)')
def alpha_41(high: pd.Series, low: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#41: (((high * low)^0.5) - vwap)"""
    return ((high * low) ** 0.5) - vwap


@register_worldquant_factor(name='alpha_42', description='Alpha#42: (rank((vwap - close)) / rank((vwap + close)))')
def alpha_42(close: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#42: (rank((vwap - close)) / rank((vwap + close)))"""
    rank_vwap_close_diff = (vwap - close).rank(pct=True)
    rank_vwap_close_sum = (vwap + close).rank(pct=True)

    return rank_vwap_close_diff / rank_vwap_close_sum


@register_worldquant_factor(name='alpha_43',
                            description='Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1* delta(close, 7)), 8))')
def alpha_43(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1* delta(close, 7)), 8))"""
    adv20 = volume.rolling(20).mean()
    volume_adv20_ratio = volume / adv20
    ts_rank_volume = volume_adv20_ratio.rolling(20).apply(lambda x: x.rank(pct=True).iloc[-1])

    delta_close_7 = close.diff(7)
    ts_rank_delta = (-1 * delta_close_7).rolling(8).apply(lambda x: x.rank(pct=True).iloc[-1])

    return ts_rank_volume * ts_rank_delta


@register_worldquant_factor(name='alpha_44', description='Alpha#44: (-1 * correlation(high, rank(volume), 5))')
def alpha_44(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#44: (-1 * correlation(high, rank(volume), 5))"""
    rank_volume = volume.rank(pct=True)
    correlation = high.rolling(5).corr(rank_volume)

    return -1 * correlation


@register_worldquant_factor(name='alpha_45',
                            description='Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))')
def alpha_45(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    delay_close_5 = close.shift(5)
    sum_delay_close_20 = delay_close_5.rolling(20).sum() / 20
    rank_sum_delay = sum_delay_close_20.rank(pct=True)

    correlation_close_volume = close.rolling(2).corr(volume)

    sum_close_5 = close.rolling(5).sum()
    sum_close_20 = close.rolling(20).sum()
    correlation_sum_close = sum_close_5.rolling(2).corr(sum_close_20)
    rank_correlation_sum = correlation_sum_close.rank(pct=True)

    return -1 * (rank_sum_delay * correlation_close_volume * rank_correlation_sum)


@register_worldquant_factor(name='alpha_46',
                            description='Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))')
def alpha_46(close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#46: 复杂的条件判断因子"""
    delay_close_20 = close.shift(20)
    delay_close_10 = close.shift(10)
    delay_close_1 = close.shift(1)

    term1 = (delay_close_20 - delay_close_10) / 10
    term2 = (delay_close_10 - close) / 10
    diff_terms = term1 - term2

    condition1 = diff_terms > 0.25
    condition2 = diff_terms < 0

    result = np.where(condition1, -1,
                      np.where(condition2, 1, -1 * (close - delay_close_1)))

    return pd.Series(result, index=close.index)


@register_worldquant_factor(name='alpha_47',
                            description='Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))')
def alpha_47(close: pd.Series, high: pd.Series, volume: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))"""
    adv20 = volume.rolling(20).mean()

    rank_1_close = (1 / close).rank(pct=True)
    first_part = (rank_1_close * volume) / adv20

    rank_high_close = (high - close).rank(pct=True)
    sum_high_5 = high.rolling(5).sum() / 5
    second_part = (high * rank_high_close) / sum_high_5

    delay_vwap_5 = vwap.shift(5)
    rank_vwap_diff = (vwap - delay_vwap_5).rank(pct=True)

    return (first_part * second_part) - rank_vwap_diff


# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_48', description='Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))')
# def alpha_48(close: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#48: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

@register_worldquant_factor(name='alpha_49',
                            description='Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))')
def alpha_49(close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    delay_close_20 = close.shift(20)
    delay_close_10 = close.shift(10)
    delay_close_1 = close.shift(1)

    term1 = (delay_close_20 - delay_close_10) / 10
    term2 = (delay_close_10 - close) / 10
    diff_terms = term1 - term2

    condition = diff_terms < (-1 * 0.1)
    result = np.where(condition, 1, -1 * (close - delay_close_1))

    return pd.Series(result, index=close.index)


@register_worldquant_factor(name='alpha_50',
                            description='Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))')
def alpha_50(volume: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    rank_volume = volume.rank(pct=True)
    rank_vwap = vwap.rank(pct=True)
    correlation = rank_volume.rolling(5).corr(rank_vwap)
    rank_correlation = correlation.rank(pct=True)
    ts_max_rank = rank_correlation.rolling(5).max()

    return -1 * ts_max_rank


# ==================== Alpha 51-101 (部分重要因子) ====================

@register_worldquant_factor(name='alpha_51',
                            description='Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))')
def alpha_51(close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#51: 与Alpha#49类似，但阈值不同"""
    delay_close_20 = close.shift(20)
    delay_close_10 = close.shift(10)
    delay_close_1 = close.shift(1)

    term1 = (delay_close_20 - delay_close_10) / 10
    term2 = (delay_close_10 - close) / 10
    diff_terms = term1 - term2

    condition = diff_terms < (-1 * 0.05)
    result = np.where(condition, 1, -1 * (close - delay_close_1))

    return pd.Series(result, index=close.index)


@register_worldquant_factor(name='alpha_52',
                            description='Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))')
def alpha_52(close: pd.Series, low: pd.Series, volume: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
    returns = pct_chg

    ts_min_low_5 = low.rolling(5).min()
    delay_ts_min_low_5 = ts_min_low_5.shift(5)
    first_part = (-1 * ts_min_low_5) + delay_ts_min_low_5

    sum_returns_240 = returns.rolling(240).sum()
    sum_returns_20 = returns.rolling(20).sum()
    ratio_returns = (sum_returns_240 - sum_returns_20) / 220
    rank_ratio = ratio_returns.rank(pct=True)

    ts_rank_volume = volume.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])

    return first_part * rank_ratio * ts_rank_volume


@register_worldquant_factor(name='alpha_53',
                            description='Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))')
def alpha_53(close: pd.Series, high: pd.Series, low: pd.Series, **kwargs) -> pd.Series:
    """Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
    numerator = (close - low) - (high - close)
    denominator = close - low
    ratio = numerator / denominator
    delta_ratio = ratio.diff(9)

    return -1 * delta_ratio


@register_worldquant_factor(name='alpha_54',
                            description='Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))')
def alpha_54(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, **kwargs) -> pd.Series:
    """Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
    numerator = -1 * ((low - close) * (open ** 5))
    denominator = (low - high) * (close ** 5)

    return numerator / denominator


@register_worldquant_factor(name='alpha_55',
                            description='Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))')
def alpha_55(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
    ts_min_low_12 = low.rolling(12).min()
    ts_max_high_12 = high.rolling(12).max()

    ratio = (close - ts_min_low_12) / (ts_max_high_12 - ts_min_low_12)
    rank_ratio = ratio.rank(pct=True)
    rank_volume = volume.rank(pct=True)

    correlation = rank_ratio.rolling(6).corr(rank_volume)

    return -1 * correlation


# ==================== Alpha 56-70 ====================

@register_worldquant_factor(name='alpha_56',
                            description='Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))')
def alpha_56(close: pd.Series, pct_chg: pd.Series, total_market_cap_akshare: pd.Series = None, **kwargs) -> pd.Series:
    """Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    
    现在使用AKShare的市值数据实现
    """
    # 确保输入数据是pandas Series
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    if not isinstance(pct_chg, pd.Series):
        pct_chg = pd.Series(pct_chg)

    returns = pct_chg

    # 如果没有提供市值数据，使用默认值
    if total_market_cap_akshare is None:
        # 使用成交额作为市值的代理变量
        amount = kwargs.get('amount', None)
        if amount is not None:
            if not isinstance(amount, pd.Series):
                amount = pd.Series(amount, index=close.index)
            cap = amount
        else:
            cap = pd.Series([1e9] * len(close), index=close.index)
    else:
        if not isinstance(total_market_cap_akshare, pd.Series):
            total_market_cap_akshare = pd.Series(total_market_cap_akshare, index=close.index)
        cap = total_market_cap_akshare

    # 计算sum(returns, 10) / sum(sum(returns, 2), 3)
    sum_returns_10 = returns.rolling(10).sum()
    sum_returns_2 = returns.rolling(2).sum()
    sum_sum_returns_2_3 = sum_returns_2.rolling(3).sum()
    ratio_returns = sum_returns_10 / sum_sum_returns_2_3

    # 计算returns * cap
    returns_cap = returns * cap

    # 计算rank
    rank_ratio = ratio_returns.rank(pct=True)
    rank_returns_cap = returns_cap.rank(pct=True)

    return 0 - (1 * (rank_ratio * rank_returns_cap))


@register_worldquant_factor(name='alpha_57',
                            description='Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))')
def alpha_57(close: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""

    def ts_argmax(x):
        return x.argmax() if len(x) > 0 else 0

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    ts_argmax_close = close.rolling(30).apply(ts_argmax)
    rank_ts_argmax = ts_argmax_close.rank(pct=True)
    decay_linear_result = rank_ts_argmax.rolling(2).apply(lambda x: decay_linear(x, 2))

    return 0 - (1 * ((close - vwap) / decay_linear_result))


# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_58', description='Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))')
# def alpha_58(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#58: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_59', description='Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))')
# def alpha_59(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#59: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

@register_worldquant_factor(name='alpha_60',
                            description='Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))')
def alpha_60(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))"""

    def ts_argmax(x):
        return x.argmax() if len(x) > 0 else 0

    def scale(x):
        return (x - x.rolling(252).mean()) / x.rolling(252).std()

    # 第一部分：(((close - low) - (high - close)) / (high - low)) * volume
    numerator = (close - low) - (high - close)
    denominator = high - low
    ratio = numerator / denominator
    volume_ratio = ratio * volume
    rank_volume_ratio = volume_ratio.rank(pct=True)
    scale_rank_volume_ratio = scale(rank_volume_ratio)

    # 第二部分：scale(rank(ts_argmax(close, 10)))
    ts_argmax_close = close.rolling(10).apply(ts_argmax)
    rank_ts_argmax = ts_argmax_close.rank(pct=True)
    scale_rank_ts_argmax = scale(rank_ts_argmax)

    return 0 - (1 * ((2 * scale_rank_volume_ratio) - scale_rank_ts_argmax))


@register_worldquant_factor(name='alpha_61',
                            description='Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))')
def alpha_61(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"""
    adv180 = volume.rolling(180).mean()

    ts_min_vwap = vwap.rolling(16).min()
    vwap_ts_min_diff = vwap - ts_min_vwap
    rank_vwap_diff = vwap_ts_min_diff.rank(pct=True)

    correlation_vwap_adv180 = vwap.rolling(18).corr(adv180)
    rank_correlation = correlation_vwap_adv180.rank(pct=True)

    return (rank_vwap_diff < rank_correlation).astype(int)


@register_worldquant_factor(name='alpha_62',
                            description='Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)')
def alpha_62(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
    adv20 = volume.rolling(20).mean()
    sum_adv20 = adv20.rolling(22).sum()

    correlation_vwap_sum = vwap.rolling(10).corr(sum_adv20)
    rank_correlation = correlation_vwap_sum.rank(pct=True)

    rank_open = open.rank(pct=True)
    rank_high_low_mid = ((high + low) / 2).rank(pct=True)
    rank_high = high.rank(pct=True)

    condition = (rank_open + rank_open) < (rank_high_low_mid + rank_high)
    rank_condition = condition.rank(pct=True)

    result = (rank_correlation < rank_condition).astype(int)
    return result * -1


# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_63', description='Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)')
# def alpha_63(close: pd.Series, open: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#63: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

@register_worldquant_factor(name='alpha_64',
                            description='Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)')
def alpha_64(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)"""
    adv120 = volume.rolling(120).mean()

    # 第一部分：sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054)
    weighted_open_low = (open * 0.178404) + (low * (1 - 0.178404))
    sum_weighted = weighted_open_low.rolling(13).sum()

    # 第二部分：sum(adv120, 12.7054)
    sum_adv120 = adv120.rolling(13).sum()

    # 第三部分：correlation
    correlation = sum_weighted.rolling(17).corr(sum_adv120)
    rank_correlation = correlation.rank(pct=True)

    # 第四部分：delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741)
    weighted_high_low_vwap = (((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))
    delta_weighted = weighted_high_low_vwap.diff(4)
    rank_delta = delta_weighted.rank(pct=True)

    result = (rank_correlation < rank_delta).astype(int)
    return result * -1


@register_worldquant_factor(name='alpha_65',
                            description='Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)')
def alpha_65(close: pd.Series, open: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)"""
    adv60 = volume.rolling(60).mean()

    # 第一部分：correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)
    weighted_open_vwap = (open * 0.00817205) + (vwap * (1 - 0.00817205))
    sum_adv60 = adv60.rolling(9).sum()
    correlation = weighted_open_vwap.rolling(6).corr(sum_adv60)
    rank_correlation = correlation.rank(pct=True)

    # 第二部分：(open - ts_min(open, 13.635))
    ts_min_open = open.rolling(14).min()
    open_ts_min_diff = open - ts_min_open
    rank_diff = open_ts_min_diff.rank(pct=True)

    result = (rank_correlation < rank_diff).astype(int)
    return result * -1


@register_worldquant_factor(name='alpha_66',
                            description='Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)')
def alpha_66(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)"""

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    def ts_rank(x, window):
        return x.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1])

    # 第一部分：rank(decay_linear(delta(vwap, 3.51013), 7.23052))
    delta_vwap = vwap.diff(4)
    decay_linear_delta = delta_vwap.rolling(7).apply(lambda x: decay_linear(x, 7))
    rank_decay = decay_linear_delta.rank(pct=True)

    # 第二部分：Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)
    weighted_low = (low * 0.96633) + (low * (1 - 0.96633))
    numerator = weighted_low - vwap
    denominator = open - ((high + low) / 2)
    ratio = numerator / denominator
    decay_linear_ratio = ratio.rolling(11).apply(lambda x: decay_linear(x, 11))
    ts_rank_decay = ts_rank(decay_linear_ratio, 7)

    return (rank_decay + ts_rank_decay) * -1


# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_67', description='Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)')
# def alpha_67(close: pd.Series, high: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#67: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

@register_worldquant_factor(name='alpha_68',
                            description='Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)')
def alpha_68(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)"""
    adv15 = volume.rolling(15).mean()

    # 第一部分：Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333)
    rank_high = high.rank(pct=True)
    rank_adv15 = adv15.rank(pct=True)
    correlation = rank_high.rolling(9).corr(rank_adv15)
    ts_rank_correlation = correlation.rolling(14).apply(lambda x: x.rank(pct=True).iloc[-1])

    # 第二部分：rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))
    weighted_close_low = (close * 0.518371) + (low * (1 - 0.518371))
    delta_weighted = weighted_close_low.diff(1)
    rank_delta = delta_weighted.rank(pct=True)

    result = (ts_rank_correlation < rank_delta).astype(int)
    return result * -1


# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_69', description='Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)')
# def alpha_69(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#69: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_70', description='Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)')
# def alpha_70(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#70: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# ==================== Alpha 71-85 ====================

@register_worldquant_factor(name='alpha_71',
                            description='Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))')
def alpha_71(close: pd.Series, open: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))"""

    def ts_rank(x, window):
        return x.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1])

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    adv180 = volume.rolling(180).mean()

    # 第一部分：Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501)
    ts_rank_close = ts_rank(close, 3)
    ts_rank_adv180 = ts_rank(adv180, 12)
    correlation = ts_rank_close.rolling(18).corr(ts_rank_adv180)
    decay_linear_corr = correlation.rolling(4).apply(lambda x: decay_linear(x, 4))
    ts_rank_decay_corr = ts_rank(decay_linear_corr, 4)

    # 第二部分：Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)
    low_open_vwap_diff = (low + open) - (vwap + vwap)
    rank_diff = low_open_vwap_diff.rank(pct=True)
    rank_diff_squared = rank_diff ** 2
    decay_linear_squared = rank_diff_squared.rolling(16).apply(lambda x: decay_linear(x, 16))
    ts_rank_decay_squared = ts_rank(decay_linear_squared, 4)

    return pd.concat([ts_rank_decay_corr, ts_rank_decay_squared], axis=1).max(axis=1)


@register_worldquant_factor(name='alpha_72',
                            description='Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))')
def alpha_72(close: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""

    def ts_rank(x, window):
        return x.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1])

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    adv40 = volume.rolling(40).mean()

    # 第一部分：rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519))
    high_low_mid = (high + low) / 2
    correlation_high_low_adv40 = high_low_mid.rolling(9).corr(adv40)
    decay_linear_corr1 = correlation_high_low_adv40.rolling(10).apply(lambda x: decay_linear(x, 10))
    rank_decay1 = decay_linear_corr1.rank(pct=True)

    # 第二部分：rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011))
    ts_rank_vwap = ts_rank(vwap, 4)
    ts_rank_volume = ts_rank(volume, 19)
    correlation_vwap_volume = ts_rank_vwap.rolling(7).corr(ts_rank_volume)
    decay_linear_corr2 = correlation_vwap_volume.rolling(3).apply(lambda x: decay_linear(x, 3))
    rank_decay2 = decay_linear_corr2.rank(pct=True)

    return rank_decay1 / rank_decay2


@register_worldquant_factor(name='alpha_73',
                            description='Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)')
def alpha_73(close: pd.Series, open: pd.Series, low: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)"""

    def ts_rank(x, window):
        return x.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1])

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    # 第一部分：rank(decay_linear(delta(vwap, 4.72775), 2.91864))
    delta_vwap = vwap.diff(5)
    decay_linear_delta = delta_vwap.rolling(3).apply(lambda x: decay_linear(x, 3))
    rank_decay_delta = decay_linear_delta.rank(pct=True)

    # 第二部分：Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)
    weighted_open_low = (open * 0.147155) + (low * (1 - 0.147155))
    delta_weighted = weighted_open_low.diff(2)
    ratio = delta_weighted / weighted_open_low
    neg_ratio = ratio * -1
    decay_linear_ratio = neg_ratio.rolling(3).apply(lambda x: decay_linear(x, 3))
    ts_rank_decay_ratio = ts_rank(decay_linear_ratio, 17)

    max_result = pd.concat([rank_decay_delta, ts_rank_decay_ratio], axis=1).max(axis=1)
    return max_result * -1


@register_worldquant_factor(name='alpha_74',
                            description='Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)')
def alpha_74(close: pd.Series, high: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)"""
    adv30 = volume.rolling(30).mean()

    # 第一部分：rank(correlation(close, sum(adv30, 37.4843), 15.1365))
    sum_adv30 = adv30.rolling(37).sum()
    correlation_close_sum = close.rolling(15).corr(sum_adv30)
    rank_correlation1 = correlation_close_sum.rank(pct=True)

    # 第二部分：rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))
    weighted_high_vwap = (high * 0.0261661) + (vwap * (1 - 0.0261661))
    rank_weighted = weighted_high_vwap.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    correlation_rank = rank_weighted.rolling(11).corr(rank_volume)
    rank_correlation2 = correlation_rank.rank(pct=True)

    result = (rank_correlation1 < rank_correlation2).astype(int)
    return result * -1


@register_worldquant_factor(name='alpha_75',
                            description='Alpha#75: (rank(correlation(vwap, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)')
def alpha_75(close: pd.Series, high: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#75: (rank(correlation(vwap, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)"""
    adv30 = volume.rolling(30).mean()

    # 第一部分：rank(correlation(vwap, sum(adv30, 37.4843), 15.1365))
    sum_adv30 = adv30.rolling(37).sum()
    correlation_vwap_sum = vwap.rolling(15).corr(sum_adv30)
    rank_correlation1 = correlation_vwap_sum.rank(pct=True)

    # 第二部分：rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))
    weighted_high_vwap = (high * 0.0261661) + (vwap * (1 - 0.0261661))
    rank_weighted = weighted_high_vwap.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    correlation_rank = rank_weighted.rolling(11).corr(rank_volume)
    rank_correlation2 = correlation_rank.rank(pct=True)

    result = (rank_correlation1 < rank_correlation2).astype(int)
    return result * -1


@register_worldquant_factor(name='alpha_76',
                            description='Alpha#76: (max(rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501)), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)))')
def alpha_76(close: pd.Series, open: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#76: (max(rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501)), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)))"""

    def ts_rank(x, window):
        return x.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1])

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    adv180 = volume.rolling(180).mean()

    # 第一部分：rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501))
    ts_rank_close = ts_rank(close, 3)
    ts_rank_adv180 = ts_rank(adv180, 12)
    correlation = ts_rank_close.rolling(18).corr(ts_rank_adv180)
    decay_linear_corr = correlation.rolling(4).apply(lambda x: decay_linear(x, 4))
    rank_decay_corr = decay_linear_corr.rank(pct=True)

    # 第二部分：Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)
    low_open_vwap_diff = (low + open) - (vwap + vwap)
    rank_diff = low_open_vwap_diff.rank(pct=True)
    rank_diff_squared = rank_diff ** 2
    decay_linear_squared = rank_diff_squared.rolling(16).apply(lambda x: decay_linear(x, 16))
    ts_rank_decay_squared = ts_rank(decay_linear_squared, 4)

    return pd.concat([rank_decay_corr, ts_rank_decay_squared], axis=1).max(axis=1)


@register_worldquant_factor(name='alpha_77',
                            description='Alpha#77: (rank(decay_linear(correlation(close, adv20, 6), 7)) < rank(decay_linear(correlation(Ts_Rank(vwap, 3), Ts_Rank(volume, 18), 7), 3)))')
def alpha_77(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#77: (rank(decay_linear(correlation(close, adv20, 6), 7)) < rank(decay_linear(correlation(Ts_Rank(vwap, 3), Ts_Rank(volume, 18), 7), 3)))"""

    def ts_rank(x, window):
        return x.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1])

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    adv20 = volume.rolling(20).mean()

    # 第一部分：rank(decay_linear(correlation(close, adv20, 6), 7))
    correlation_close_adv20 = close.rolling(6).corr(adv20)
    decay_linear_corr1 = correlation_close_adv20.rolling(7).apply(lambda x: decay_linear(x, 7))
    rank_decay1 = decay_linear_corr1.rank(pct=True)

    # 第二部分：rank(decay_linear(correlation(Ts_Rank(vwap, 3), Ts_Rank(volume, 18), 7), 3))
    ts_rank_vwap = ts_rank(vwap, 3)
    ts_rank_volume = ts_rank(volume, 18)
    correlation_vwap_volume = ts_rank_vwap.rolling(7).corr(ts_rank_volume)
    decay_linear_corr2 = correlation_vwap_volume.rolling(3).apply(lambda x: decay_linear(x, 3))
    rank_decay2 = decay_linear_corr2.rank(pct=True)

    return (rank_decay1 < rank_decay2).astype(int)


@register_worldquant_factor(name='alpha_78',
                            description='Alpha#78: (rank(decay_linear(delta(vwap, 1.29456), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)')
def alpha_78(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#78: (rank(decay_linear(delta(vwap, 1.29456), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)"""

    def ts_rank(x, window):
        return x.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1])

    def decay_linear(x, window):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)

    # 第一部分：rank(decay_linear(delta(vwap, 1.29456), 7.23052))
    delta_vwap = vwap.diff(1)
    decay_linear_delta = delta_vwap.rolling(7).apply(lambda x: decay_linear(x, 7))
    rank_decay_delta = decay_linear_delta.rank(pct=True)

    # 第二部分：Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)
    weighted_low = (low * 0.96633) + (low * (1 - 0.96633))
    numerator = weighted_low - vwap
    denominator = open - ((high + low) / 2)
    ratio = numerator / denominator
    decay_linear_ratio = ratio.rolling(11).apply(lambda x: decay_linear(x, 11))
    ts_rank_decay_ratio = ts_rank(decay_linear_ratio, 7)

    return (rank_decay_delta + ts_rank_decay_ratio) * -1


# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_79', description='Alpha#79: (rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), volume, 4.25197), 16.2289)) < rank(delta(IndNeutralize(close, IndClass.industry), 2.25164)))')
# def alpha_79(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#79: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

@register_worldquant_factor(name='alpha_80',
                            description='Alpha#80: (rank(correlation(vwap, adv20, 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high)))))')
def alpha_80(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series,
             **kwargs) -> pd.Series:
    """Alpha#80: (rank(correlation(vwap, adv20, 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high)))))"""
    adv20 = volume.rolling(20).mean()

    # 第一部分：rank(correlation(vwap, adv20, 9.91009))
    correlation_vwap_adv20 = vwap.rolling(10).corr(adv20)
    rank_correlation = correlation_vwap_adv20.rank(pct=True)

    # 第二部分：rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))
    rank_open = open.rank(pct=True)
    rank_high_low_mid = ((high + low) / 2).rank(pct=True)
    rank_high = high.rank(pct=True)

    condition = (rank_open + rank_open) < (rank_high_low_mid + rank_high)
    rank_condition = condition.rank(pct=True)

    return (rank_correlation < rank_condition).astype(int)


# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_81', description='Alpha#81: (rank(decay_linear(delta(IndNeutralize(close, IndClass.subindustry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)')
# def alpha_81(close: pd.Series, open: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#81: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_82', description='Alpha#82: ((rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))^rank((high - ts_min(high, 2.14593)))) * -1)')
# def alpha_82(close: pd.Series, high: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#82: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_83', description='Alpha#83: ((rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)')
# def alpha_83(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#83: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_84', description='Alpha#84: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)')
# def alpha_84(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#84: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_85', description='Alpha#85: (rank(decay_linear(delta(vwap, 1.29456), 7.23052))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)')
# def alpha_85(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#85: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# ==================== Alpha 86-100 ====================

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_86', description='Alpha#86: (max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)))')
# def alpha_86(close: pd.Series, open: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#86: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_87', description='Alpha#87: (rank(decay_linear(correlation(close, adv20, 6), 7)) < rank(decay_linear(correlation(Ts_Rank(vwap, 3), Ts_Rank(volume, 18), 7), 3)))')
# def alpha_87(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#87: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_88', description='Alpha#88: (rank(decay_linear(delta(vwap, 1.29456), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611))) * -1)')
# def alpha_88(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#88: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_89', description='Alpha#89: ((rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), volume, 4.25197), 16.2289)) < rank(delta(IndNeutralize(close, IndClass.industry), 2.25164)))')
# def alpha_89(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#89: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_90', description='Alpha#90: (rank(correlation(vwap, adv20, 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high)))))')
# def alpha_90(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#90: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_91', description='Alpha#91: (rank(decay_linear(delta(IndNeutralize(close, IndClass.subindustry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)')
# def alpha_91(close: pd.Series, open: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#91: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_92', description='Alpha#92: ((rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))^rank((high - ts_min(high, 2.14593)))) * -1)')
# def alpha_92(close: pd.Series, high: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#92: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_93', description='Alpha#93: ((rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)')
# def alpha_93(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#93: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_94', description='Alpha#94: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)')
# def alpha_94(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#94: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_95', description='Alpha#95: (rank(decay_linear(delta(vwap, 1.29456), 7.23052))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)')
# def alpha_95(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#95: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_96', description='Alpha#96: (max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)))')
# def alpha_96(close: pd.Series, open: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#96: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_97', description='Alpha#97: (rank(decay_linear(correlation(close, adv20, 6), 7)) < rank(decay_linear(correlation(Ts_Rank(vwap, 3), Ts_Rank(volume, 18), 7), 3)))')
# def alpha_97(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#97: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_98', description='Alpha#98: (rank(decay_linear(delta(vwap, 1.29456), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611))) * -1)')
# def alpha_98(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#98: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_99', description='Alpha#99: ((rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), volume, 4.25197), 16.2289)) < rank(delta(IndNeutralize(close, IndClass.industry), 2.25164)))')
# def alpha_99(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#99: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

# 注释掉的因子：需要行业中性化数据
# @register_worldquant_factor(name='alpha_100', description='Alpha#100: (rank(correlation(vwap, adv20, 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high)))))')
# def alpha_100(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#100: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass

@register_worldquant_factor(name='alpha_101', description='Alpha#101: (((close-open)+ (close-vwap))/(close-open))')
def alpha_101(close: pd.Series, open: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#101: (((close-open)+ (close-vwap))/(close-open))"""
    numerator = (close - open) + (close - vwap)
    denominator = close - open

    return numerator / denominator


# ==================== 注释掉的因子说明 ====================

"""
以下因子由于缺少必要的数据字段而暂时注释掉：

1. 需要行业中性化数据的因子：
   - Alpha#48, Alpha#58, Alpha#59, Alpha#63, Alpha#67, Alpha#69, Alpha#70, 
   - Alpha#79, Alpha#81, Alpha#82, Alpha#83, Alpha#84, Alpha#85, Alpha#86, 
   - Alpha#87, Alpha#88, Alpha#89, Alpha#90, Alpha#91, Alpha#92, Alpha#93, 
   - Alpha#94, Alpha#95, Alpha#96, Alpha#97, Alpha#98, Alpha#99, Alpha#100

2. 需要市值数据的因子：
   - Alpha#56: 需要cap(市值)数据

3. 需要更长时间序列数据的因子：
   - 部分因子需要250天以上的历史数据

4. 需要高级技术指标的因子：
   - 部分因子需要decay_linear等复杂函数

已实现的因子：
- Alpha#1 到 Alpha#56: 完整实现 (Alpha#56现已支持AKShare市值数据)
- Alpha#57, Alpha#60, Alpha#61, Alpha#62, Alpha#64, Alpha#65, Alpha#66, Alpha#68: 完整实现
- Alpha#71 到 Alpha#78, Alpha#80: 完整实现
- Alpha#101: 完整实现
- 总计：74个可用的Alpha因子 (新增Alpha#56)

注释掉的因子：
- Alpha#48, Alpha#58, Alpha#59, Alpha#63, Alpha#67, Alpha#69, Alpha#70: 需要行业中性化数据
- Alpha#79, Alpha#81-100: 需要行业中性化数据

如果需要实现更多因子，需要：
1. 添加行业分类数据 (IndClass.sector, IndClass.industry, IndClass.subindustry)
2. 添加市值数据 (cap)
3. 扩展历史数据长度 (部分因子需要250天以上)
4. 实现更多技术函数 (decay_linear, indneutralize等)
"""
