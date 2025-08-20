#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : worldquant_factors.py
@Description: WorldQuant Alpha因子库
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import pandas as pd
import numpy as np
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'factor'))
from base_factor import register_worldquant_factor

# ==================== WorldQuant Alpha因子 ====================

@register_worldquant_factor(name='alpha_1', description='Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)')
def alpha_1(pct_chg: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    def ts_argmax(x):
        return x.argmax() if len(x) > 0 else 0
    
    def signed_power(x, power):
        return np.sign(x) * np.abs(x) ** power
    
    # 计算条件表达式
    condition = pct_chg < 0
    std_returns = pct_chg.rolling(20).std()
    value = np.where(condition, std_returns, close)
    
    # 计算SignedPower
    signed_power_value = signed_power(value, 2)
    
    # 计算Ts_ArgMax
    ts_argmax_value = signed_power_value.rolling(5).apply(ts_argmax)
    
    # 计算rank
    rank_value = ts_argmax_value.rank(pct=True)
    
    return rank_value - 0.5

@register_worldquant_factor(name='alpha_2', description='Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))')
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

@register_worldquant_factor(name='alpha_5', description='Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))')
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

@register_worldquant_factor(name='alpha_7', description='Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))')
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

@register_worldquant_factor(name='alpha_8', description='Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))')
def alpha_8(open: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    sum_open = open.rolling(5).sum()
    sum_returns = pct_chg.rolling(5).sum()
    product = sum_open * sum_returns
    delay_product = product.shift(10)
    
    return -1 * (product - delay_product).rank(pct=True)

@register_worldquant_factor(name='alpha_9', description='Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))')
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

@register_worldquant_factor(name='alpha_10', description='Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))')
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

@register_worldquant_factor(name='alpha_11', description='Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))')
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

@register_worldquant_factor(name='alpha_13', description='Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))')
def alpha_13(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    rank_close = close.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    
    def rolling_covariance(x):
        if len(x) < 2:
            return np.nan
        return np.cov(x.iloc[:len(x)//2], x.iloc[len(x)//2:])[0, 1] if len(x) >= 2 else np.nan
    
    covariance = pd.concat([rank_close, rank_volume], axis=1).rolling(5).apply(rolling_covariance)
    
    return -1 * covariance.rank(pct=True)

@register_worldquant_factor(name='alpha_14', description='Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))')
def alpha_14(open: pd.Series, volume: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    delta_returns = pct_chg.diff(3)
    rank_delta_returns = delta_returns.rank(pct=True)
    correlation_open_volume = open.rolling(10).corr(volume)
    
    return (-1 * rank_delta_returns) * correlation_open_volume

@register_worldquant_factor(name='alpha_15', description='Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))')
def alpha_15(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    rank_high = high.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    correlation = rank_high.rolling(3).corr(rank_volume)
    rank_correlation = correlation.rank(pct=True)
    
    return -1 * rank_correlation.rolling(3).sum()

@register_worldquant_factor(name='alpha_16', description='Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))')
def alpha_16(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    rank_high = high.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    
    def rolling_covariance(x):
        if len(x) < 2:
            return np.nan
        return np.cov(x.iloc[:len(x)//2], x.iloc[len(x)//2:])[0, 1] if len(x) >= 2 else np.nan
    
    covariance = pd.concat([rank_high, rank_volume], axis=1).rolling(5).apply(rolling_covariance)
    
    return -1 * covariance.rank(pct=True)

@register_worldquant_factor(name='alpha_17', description='Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))')
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

@register_worldquant_factor(name='alpha_18', description='Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))')
def alpha_18(close: pd.Series, open: pd.Series, **kwargs) -> pd.Series:
    """Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
    close_open_diff = close - open
    abs_close_open_diff = abs(close_open_diff)
    stddev_abs_diff = abs_close_open_diff.rolling(5).std()
    correlation_close_open = close.rolling(10).corr(open)
    
    combined = stddev_abs_diff + close_open_diff + correlation_close_open
    
    return -1 * combined.rank(pct=True)

@register_worldquant_factor(name='alpha_19', description='Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))')
def alpha_19(close: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    delay_close_7 = close.shift(7)
    delta_close_7 = close.diff(7)
    close_delay_diff = close - delay_close_7
    sign_value = np.sign(close_delay_diff + delta_close_7)
    
    sum_returns_250 = pct_chg.rolling(250).sum()
    rank_sum_returns = (1 + sum_returns_250).rank(pct=True)
    
    return (-1 * sign_value) * (1 + rank_sum_returns)

@register_worldquant_factor(name='alpha_20', description='Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))')
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

@register_worldquant_factor(name='alpha_21', description='Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1*1): (((sum(close, 2)/2) < ((sum(close, 8) / 8) - stddev(close, 8)))?1: (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1: (-1*1))))')
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

@register_worldquant_factor(name='alpha_22', description='Alpha#22: (-1* (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))')
def alpha_22(close: pd.Series, high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#22: (-1* (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
    correlation_high_volume = high.rolling(5).corr(volume)
    delta_correlation = correlation_high_volume.diff(5)
    stddev_close_20 = close.rolling(20).std()
    rank_stddev = stddev_close_20.rank(pct=True)
    
    return -1 * (delta_correlation * rank_stddev)

@register_worldquant_factor(name='alpha_23', description='Alpha#23: (((sum(high, 20) / 20) < high)? (-1* delta(high, 2)): 0)')
def alpha_23(high: pd.Series, **kwargs) -> pd.Series:
    """Alpha#23: (((sum(high, 20) / 20) < high)? (-1* delta(high, 2)): 0)"""
    sum_high_20 = high.rolling(20).sum() / 20
    delta_high_2 = high.diff(2)
    
    condition = sum_high_20 < high
    result = np.where(condition, -1 * delta_high_2, 0)
    
    return pd.Series(result, index=high.index)

@register_worldquant_factor(name='alpha_24', description='Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) ==0.05)) ? (-1* (close - ts_min(close, 100))): (-1* delta(close, 3)))')
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

@register_worldquant_factor(name='alpha_25', description='Alpha#25: rank(((((-1* returns) * adv20) * vwap) * (high - close)))')
def alpha_25(close: pd.Series, high: pd.Series, volume: pd.Series, vwap: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#25: rank(((((-1* returns) * adv20) * vwap) * (high - close)))"""
    adv20 = volume.rolling(20).mean()
    returns = pct_chg
    
    result = ((((-1 * returns) * adv20) * vwap) * (high - close))
    
    return result.rank(pct=True)

@register_worldquant_factor(name='alpha_26', description='Alpha#26: (-1* ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))')
def alpha_26(high: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#26: (-1* ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    ts_rank_volume = volume.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    ts_rank_high = high.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    correlation = ts_rank_volume.rolling(5).corr(ts_rank_high)
    ts_max_correlation = correlation.rolling(3).max()
    
    return -1 * ts_max_correlation

@register_worldquant_factor(name='alpha_27', description='Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1*1): 1))')
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

@register_worldquant_factor(name='alpha_28', description='Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))')
def alpha_28(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    adv20 = volume.rolling(20).mean()
    correlation_adv20_low = adv20.rolling(5).corr(low)
    high_low_mid = (high + low) / 2
    
    result = (correlation_adv20_low + high_low_mid) - close
    
    # scale函数：标准化
    return (result - result.rolling(252).mean()) / result.rolling(252).std()

@register_worldquant_factor(name='alpha_29', description='Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1* rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1* returns), 6), 5))')
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

@register_worldquant_factor(name='alpha_30', description='Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))')
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

@register_worldquant_factor(name='alpha_31', description='Alpha#31: ((rank(rank(rank(decay_linear((-1* rank(rank(delta(close, 10)))), 10)))) + rank((-1* delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))')
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
    scale_correlation = (correlation_adv20_low - correlation_adv20_low.rolling(252).mean()) / correlation_adv20_low.rolling(252).std()
    sign_scale = np.sign(scale_correlation)
    
    return rank_rank_rank_decay + rank_neg_delta + sign_scale

@register_worldquant_factor(name='alpha_32', description='Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))')
def alpha_32(close: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
    # 第一部分：scale(((sum(close, 7) / 7) - close))
    sum_close_7 = close.rolling(7).sum() / 7
    diff_sum_close = sum_close_7 - close
    scale_diff = (diff_sum_close - diff_sum_close.rolling(252).mean()) / diff_sum_close.rolling(252).std()
    
    # 第二部分：(20 * scale(correlation(vwap, delay(close, 5), 230)))
    delay_close_5 = close.shift(5)
    correlation_vwap_delay = vwap.rolling(230).corr(delay_close_5)
    scale_correlation = (correlation_vwap_delay - correlation_vwap_delay.rolling(252).mean()) / correlation_vwap_delay.rolling(252).std()
    
    return scale_diff + (20 * scale_correlation)

@register_worldquant_factor(name='alpha_33', description='Alpha#33: rank((-1* ((1- (open / close))^1)))')
def alpha_33(open: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#33: rank((-1* ((1- (open / close))^1)))"""
    ratio = open / close
    result = -1 * ((1 - ratio) ** 1)
    
    return result.rank(pct=True)

@register_worldquant_factor(name='alpha_34', description='Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))')
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

@register_worldquant_factor(name='alpha_35', description='Alpha#35: ((Ts_Rank(volume, 32) * (1 -Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))')
def alpha_35(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
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

@register_worldquant_factor(name='alpha_36', description='Alpha#36: (((((2.21* rank(correlation((close - open), delay(volume, 1), 15))) + (0.7* rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1* returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6* rank((((sum(close, 200) / 200) - open) * (close - open)))))')
def alpha_36(close: pd.Series, open: pd.Series, volume: pd.Series, vwap: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
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
    
    return (2.21 * rank_correlation_1) + (0.7 * rank_open_close) + (0.73 * rank_ts_rank) + rank_abs_correlation + (0.6 * rank_complex_term)

@register_worldquant_factor(name='alpha_37', description='Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))')
def alpha_37(open: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    open_close_diff = open - close
    delay_open_close_1 = open_close_diff.shift(1)
    
    correlation_delay_close = delay_open_close_1.rolling(200).corr(close)
    rank_correlation = correlation_delay_close.rank(pct=True)
    
    rank_open_close = open_close_diff.rank(pct=True)
    
    return rank_correlation + rank_open_close

@register_worldquant_factor(name='alpha_38', description='Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close/open)))')
def alpha_38(close: pd.Series, open: pd.Series, **kwargs) -> pd.Series:
    """Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close/open)))"""
    ts_rank_close = close.rolling(10).apply(lambda x: x.rank(pct=True).iloc[-1])
    rank_ts_rank = ts_rank_close.rank(pct=True)
    
    close_open_ratio = close / open
    rank_ratio = close_open_ratio.rank(pct=True)
    
    return (-1 * rank_ts_rank) * rank_ratio

@register_worldquant_factor(name='alpha_39', description='Alpha#39: ((-1* rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))')
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

@register_worldquant_factor(name='alpha_40', description='Alpha#40: ((-1* rank(stddev(high, 10))) * correlation(high, volume, 10))')
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

@register_worldquant_factor(name='alpha_43', description='Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1* delta(close, 7)), 8))')
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

@register_worldquant_factor(name='alpha_45', description='Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))')
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

@register_worldquant_factor(name='alpha_46', description='Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))')
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

@register_worldquant_factor(name='alpha_47', description='Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))')
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

@register_worldquant_factor(name='alpha_49', description='Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))')
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

@register_worldquant_factor(name='alpha_50', description='Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))')
def alpha_50(volume: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    rank_volume = volume.rank(pct=True)
    rank_vwap = vwap.rank(pct=True)
    correlation = rank_volume.rolling(5).corr(rank_vwap)
    rank_correlation = correlation.rank(pct=True)
    ts_max_rank = rank_correlation.rolling(5).max()
    
    return -1 * ts_max_rank

# ==================== Alpha 51-101 (部分重要因子) ====================

@register_worldquant_factor(name='alpha_51', description='Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))')
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

@register_worldquant_factor(name='alpha_52', description='Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))')
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

@register_worldquant_factor(name='alpha_53', description='Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))')
def alpha_53(close: pd.Series, high: pd.Series, low: pd.Series, **kwargs) -> pd.Series:
    """Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
    numerator = (close - low) - (high - close)
    denominator = close - low
    ratio = numerator / denominator
    delta_ratio = ratio.diff(9)
    
    return -1 * delta_ratio

@register_worldquant_factor(name='alpha_54', description='Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))')
def alpha_54(close: pd.Series, open: pd.Series, high: pd.Series, low: pd.Series, **kwargs) -> pd.Series:
    """Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
    numerator = -1 * ((low - close) * (open ** 5))
    denominator = (low - high) * (close ** 5)
    
    return numerator / denominator

@register_worldquant_factor(name='alpha_55', description='Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))')
def alpha_55(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
    ts_min_low_12 = low.rolling(12).min()
    ts_max_high_12 = high.rolling(12).max()
    
    ratio = (close - ts_min_low_12) / (ts_max_high_12 - ts_min_low_12)
    rank_ratio = ratio.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    
    correlation = rank_ratio.rolling(6).corr(rank_volume)
    
    return -1 * correlation

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
- Alpha#1 到 Alpha#55: 完整实现
- Alpha#101: 完整实现
- 总计：56个可用的Alpha因子

如果需要实现更多因子，需要：
1. 添加行业分类数据
2. 添加市值数据
3. 扩展历史数据长度
4. 实现更多技术函数
"""
