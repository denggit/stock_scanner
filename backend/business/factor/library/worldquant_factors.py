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
def alpha_2(volume: pd.Series, close: pd.Series, open_price: pd.Series, **kwargs) -> pd.Series:
    """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    # 计算log(volume)的2日差分
    log_volume = np.log(volume)
    delta_log_volume = log_volume.diff(2)
    
    # 计算价格变化率
    price_change = (close - open_price) / open_price
    
    # 计算rank
    rank_volume = delta_log_volume.rank(pct=True)
    rank_price = price_change.rank(pct=True)
    
    # 计算6日相关性
    correlation = rank_volume.rolling(6).corr(rank_price)
    
    return -1 * correlation

@register_worldquant_factor(name='alpha_3', description='Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))')
def alpha_3(open_price: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))"""
    # 计算rank
    rank_open = open_price.rank(pct=True)
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
def alpha_5(open_price: pd.Series, close: pd.Series, vwap: pd.Series, **kwargs) -> pd.Series:
    """Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    # 计算vwap的10日平均
    vwap_ma = vwap.rolling(10).mean()
    
    # 计算rank
    rank_open_vwap = (open_price - vwap_ma).rank(pct=True)
    rank_close_vwap = (close - vwap).rank(pct=True)
    
    return rank_open_vwap * (-1 * abs(rank_close_vwap))

@register_worldquant_factor(name='alpha_6', description='Alpha#6: (-1 * correlation(open, volume, 10))')
def alpha_6(open_price: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """Alpha#6: (-1 * correlation(open, volume, 10))"""
    # 计算10日相关性
    correlation = open_price.rolling(10).corr(volume)
    
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
def alpha_8(open_price: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    sum_open = open_price.rolling(5).sum()
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
