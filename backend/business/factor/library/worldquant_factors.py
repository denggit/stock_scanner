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
def alpha_1(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 pct_chg, close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pct_chg = data['pct_chg']
    close = data['close']

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
def alpha_2(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 volume, close, open 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    volume = data['volume']
    close = data['close']
    open_price = data['open']
    
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
def alpha_3(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open_price = data['open']
    volume = data['volume']
    
    # 计算rank
    rank_open = open_price.rank(pct=True)
    rank_volume = volume.rank(pct=True)

    # 计算10日相关性
    correlation = rank_open.rolling(10).corr(rank_volume)

    return -1 * correlation





@register_worldquant_factor(name='alpha_4', description='Alpha#4: (-1 * Ts_Rank(rank(low), 9))')
def alpha_4(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 low 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    low = data['low']
    
    # 计算rank
    rank_low = low.rank(pct=True)

    # 计算Ts_Rank (9日滚动rank)
    ts_rank = rank_low.rolling(9).apply(lambda x: x.rank(pct=True).iloc[-1])

    return -1 * ts_rank





@register_worldquant_factor(name='alpha_5',
                            description='Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))')
def alpha_5(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, close, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open_price = data['open']
    close = data['close']
    vwap = data['vwap']
    
    # 计算vwap的10日平均
    vwap_ma = vwap.rolling(10).mean()

    # 计算rank
    rank_open_vwap = (open_price - vwap_ma).rank(pct=True)
    rank_close_vwap = (close - vwap).rank(pct=True)

    return rank_open_vwap * (-1 * abs(rank_close_vwap))





@register_worldquant_factor(name='alpha_6', description='Alpha#6: (-1 * correlation(open, volume, 10))')
def alpha_6(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#6: (-1 * correlation(open, volume, 10))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open_price = data['open']
    volume = data['volume']
    
    # 计算10日相关性
    correlation = open_price.rolling(10).corr(volume)

    return -1 * correlation





@register_worldquant_factor(name='alpha_7',
                            description='Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))')
def alpha_7(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 volume, close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    volume = data['volume']
    close = data['close']
    
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
def alpha_8(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open_price = data['open']
    pct_chg = data['pct_chg']
    
    sum_open = open_price.rolling(5).sum()
    sum_returns = pct_chg.rolling(5).sum()
    product = sum_open * sum_returns
    delay_product = product.shift(10)

    return -1 * (product - delay_product).rank(pct=True)





@register_worldquant_factor(name='alpha_9',
                            description='Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))')
def alpha_9(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    
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
def alpha_10(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    
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
def alpha_11(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    vwap = data['vwap']
    volume = data['volume']
    
    vwap_close_diff = vwap - close
    ts_max_vwap_close = vwap_close_diff.rolling(3).max()
    ts_min_vwap_close = vwap_close_diff.rolling(3).min()
    delta_volume = volume.diff(3)

    rank_max = ts_max_vwap_close.rank(pct=True)
    rank_min = ts_min_vwap_close.rank(pct=True)
    rank_delta_volume = delta_volume.rank(pct=True)

    return (rank_max + rank_min) * rank_delta_volume





@register_worldquant_factor(name='alpha_12', description='Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))')
def alpha_12(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
    
    delta_volume = volume.diff(1)
    delta_close = close.diff(1)

    return np.sign(delta_volume) * (-1 * delta_close)





@register_worldquant_factor(name='alpha_13',
                            description='Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))')
def alpha_13(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
    
    rank_close = close.rank(pct=True)
    rank_volume = volume.rank(pct=True)

    def rolling_covariance(x):
        if len(x) < 2:
            return np.nan
        return np.cov(x.iloc[:len(x) // 2], x.iloc[len(x) // 2:])[0, 1] if len(x) >= 2 else np.nan

    covariance = pd.concat([rank_close, rank_volume], axis=1).rolling(5).apply(rolling_covariance)

    return -1 * covariance.rank(pct=True)





@register_worldquant_factor(name='alpha_14',
                            description='alpha_14 因子')
def alpha_14(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_14 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, volume, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open = data['open']
    volume = data['volume']
    pct_chg = data['pct_chg']
    delta_returns = pct_chg.diff(3)
    rank_delta_returns = delta_returns.rank(pct=True)
    correlation_open_volume = open.rolling(10).corr(volume)

    return (-1 * rank_delta_returns) * correlation_open_volume

@register_worldquant_factor(name='alpha_15',
                            description='alpha_15 因子')
def alpha_15(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_15 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    volume = data['volume']
    rank_high = high.rank(pct=True)
    rank_volume = volume.rank(pct=True)
    correlation = rank_high.rolling(3).corr(rank_volume)
    rank_correlation = correlation.rank(pct=True)

    return -1 * rank_correlation.rolling(3).sum()

@register_worldquant_factor(name='alpha_16',
                            description='alpha_16 因子')
def alpha_16(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_16 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    volume = data['volume']
    rank_high = high.rank(pct=True)
    rank_volume = volume.rank(pct=True)

    def rolling_covariance(x):
        if len(x) < 2:
            return np.nan
        return np.cov(x.iloc[:len(x) // 2], x.iloc[len(x) // 2:])[0, 1] if len(x) >= 2 else np.nan

    covariance = pd.concat([rank_high, rank_volume], axis=1).rolling(5).apply(rolling_covariance)

    return -1 * covariance.rank(pct=True)

@register_worldquant_factor(name='alpha_17',
                            description='alpha_17 因子')
def alpha_17(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_17 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
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
                            description='alpha_18 因子')
def alpha_18(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_18 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    close_open_diff = close - open
    abs_close_open_diff = abs(close_open_diff)
    stddev_abs_diff = abs_close_open_diff.rolling(5).std()
    correlation_close_open = close.rolling(10).corr(open)

    combined = stddev_abs_diff + close_open_diff + correlation_close_open

    return -1 * combined.rank(pct=True)

@register_worldquant_factor(name='alpha_19',
                            description='alpha_19 因子')
def alpha_19(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_19 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    pct_chg = data['pct_chg']
    delay_close_7 = close.shift(7)
    delta_close_7 = close.diff(7)
    close_delay_diff = close - delay_close_7
    sign_value = np.sign(close_delay_diff + delta_close_7)

    sum_returns_250 = pct_chg.rolling(250).sum()
    rank_sum_returns = (1 + sum_returns_250).rank(pct=True)

    return (-1 * sign_value) * (1 + rank_sum_returns)

@register_worldquant_factor(name='alpha_20',
                            description='alpha_20 因子')
def alpha_20(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_20 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, high, close, low 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open = data['open']
    high = data['high']
    close = data['close']
    low = data['low']
    delay_high_1 = high.shift(1)
    delay_close_1 = close.shift(1)
    delay_low_1 = low.shift(1)

    rank_open_delay_high = (open - delay_high_1).rank(pct=True)
    rank_open_delay_close = (open - delay_close_1).rank(pct=True)
    rank_open_delay_low = (open - delay_low_1).rank(pct=True)

    return ((-1 * rank_open_delay_high) * rank_open_delay_close) * rank_open_delay_low


# ==================== Alpha 21-30 ====================



@register_worldquant_factor(name='alpha_21',
                            description='alpha_21 因子')
def alpha_21(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_21 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
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
                            description='alpha_22 因子')
def alpha_22(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_22 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    volume = data['volume']
    correlation_high_volume = high.rolling(5).corr(volume)
    delta_correlation = correlation_high_volume.diff(5)
    stddev_close_20 = close.rolling(20).std()
    rank_stddev = stddev_close_20.rank(pct=True)

    return -1 * (delta_correlation * rank_stddev)

@register_worldquant_factor(name='alpha_23',
                            description='alpha_23 因子')
def alpha_23(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_23 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    sum_high_20 = high.rolling(20).sum() / 20
    delta_high_2 = high.diff(2)

    condition = sum_high_20 < high
    result = np.where(condition, -1 * delta_high_2, 0)

    return pd.Series(result, index=close.index)



@register_worldquant_factor(name='alpha_24',
                            description='alpha_24 因子')
def alpha_24(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_24 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
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
                            description='alpha_25 因子')
def alpha_25(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_25 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, volume, vwap, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    volume = data['volume']
    vwap = data['vwap']
    pct_chg = data['pct_chg']
    adv20 = volume.rolling(20).mean()
    returns = pct_chg

    result = ((((-1 * returns) * adv20) * vwap) * (high - close))

    return result.rank(pct=True)

@register_worldquant_factor(name='alpha_26',
                            description='alpha_26 因子')
def alpha_26(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_26 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    volume = data['volume']
    ts_rank_volume = volume.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    ts_rank_high = high.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    correlation = ts_rank_volume.rolling(5).corr(ts_rank_high)
    ts_max_correlation = correlation.rolling(3).max()

    return -1 * ts_max_correlation

@register_worldquant_factor(name='alpha_27',
                            description='alpha_27 因子')
def alpha_27(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_27 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 volume, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    volume = data['volume']
    vwap = data['vwap']
    rank_volume = volume.rank(pct=True)
    rank_vwap = vwap.rank(pct=True)
    correlation = rank_volume.rolling(6).corr(rank_vwap)
    sum_correlation_2 = correlation.rolling(2).sum() / 2.0
    rank_sum_correlation = sum_correlation_2.rank(pct=True)

    condition = rank_sum_correlation > 0.5
    result = np.where(condition, -1, 1)

    return pd.Series(result, index=close.index)



@register_worldquant_factor(name='alpha_28',
                            description='alpha_28 因子')
def alpha_28(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_28 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, low, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    adv20 = volume.rolling(20).mean()
    correlation_adv20_low = adv20.rolling(5).corr(low)
    high_low_mid = (high + low) / 2

    result = (correlation_adv20_low + high_low_mid) - close

    # scale函数：标准化
    return (result - result.rolling(252).mean()) / result.rolling(252).std()

@register_worldquant_factor(name='alpha_29',
                            description='alpha_29 因子')
def alpha_29(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_29 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    pct_chg = data['pct_chg']
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
                            description='alpha_30 因子')
def alpha_30(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_30 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
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
                            description='alpha_31 因子')
def alpha_31(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_31 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, low, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    low = data['low']
    volume = data['volume']
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
                            description='alpha_32 因子')
def alpha_32(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_32 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    vwap = data['vwap']
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

@register_worldquant_factor(name='alpha_33',
                            description='alpha_33 因子')
def alpha_33(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_33 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open = data['open']
    close = data['close']
    ratio = open / close
    result = -1 * ((1 - ratio) ** 1)

    return result.rank(pct=True)

@register_worldquant_factor(name='alpha_34',
                            description='alpha_34 因子')
def alpha_34(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_34 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    pct_chg = data['pct_chg']
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
                            description='alpha_35 因子')
def alpha_35(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_35 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, low, volume, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    pct_chg = data['pct_chg']
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
                            description='alpha_36 因子')
def alpha_36(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_36 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, volume, vwap, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    volume = data['volume']
    vwap = data['vwap']
    pct_chg = data['pct_chg']
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
                            description='alpha_37 因子')
def alpha_37(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_37 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 open, close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    open = data['open']
    close = data['close']
    open_close_diff = open - close
    delay_open_close_1 = open_close_diff.shift(1)

    correlation_delay_close = delay_open_close_1.rolling(200).corr(close)
    rank_correlation = correlation_delay_close.rank(pct=True)

    rank_open_close = open_close_diff.rank(pct=True)

    return rank_correlation + rank_open_close

@register_worldquant_factor(name='alpha_38',
                            description='alpha_38 因子')
def alpha_38(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_38 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    ts_rank_close = close.rolling(10).apply(lambda x: x.rank(pct=True).iloc[-1])
    rank_ts_rank = ts_rank_close.rank(pct=True)

    close_open_ratio = close / open
    rank_ratio = close_open_ratio.rank(pct=True)

    return (-1 * rank_ts_rank) * rank_ratio

@register_worldquant_factor(name='alpha_39',
                            description='alpha_39 因子')
def alpha_39(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_39 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
    pct_chg = data['pct_chg']
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
                            description='alpha_40 因子')
def alpha_40(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_40 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    volume = data['volume']
    stddev_high_10 = high.rolling(10).std()
    rank_stddev = stddev_high_10.rank(pct=True)

    correlation_high_volume = high.rolling(10).corr(volume)

    return (-1 * rank_stddev) * correlation_high_volume


# ==================== Alpha 41-50 ====================



@register_worldquant_factor(name='alpha_41',
                            description='alpha_41 因子')
def alpha_41(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_41 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, low, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    return ((high * low) ** 0.5) - vwap

@register_worldquant_factor(name='alpha_42',
                            description='alpha_42 因子')
def alpha_42(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_42 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    vwap = data['vwap']
    rank_vwap_close_diff = (vwap - close).rank(pct=True)
    rank_vwap_close_sum = (vwap + close).rank(pct=True)

    return rank_vwap_close_diff / rank_vwap_close_sum

@register_worldquant_factor(name='alpha_43',
                            description='alpha_43 因子')
def alpha_43(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_43 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
    adv20 = volume.rolling(20).mean()
    volume_adv20_ratio = volume / adv20
    ts_rank_volume = volume_adv20_ratio.rolling(20).apply(lambda x: x.rank(pct=True).iloc[-1])

    delta_close_7 = close.diff(7)
    ts_rank_delta = (-1 * delta_close_7).rolling(8).apply(lambda x: x.rank(pct=True).iloc[-1])

    return ts_rank_volume * ts_rank_delta

@register_worldquant_factor(name='alpha_44',
                            description='alpha_44 因子')
def alpha_44(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_44 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 high, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    high = data['high']
    volume = data['volume']
    rank_volume = volume.rank(pct=True)
    correlation = high.rolling(5).corr(rank_volume)

    return -1 * correlation

@register_worldquant_factor(name='alpha_45',
                            description='alpha_45 因子')
def alpha_45(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_45 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
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
                            description='alpha_46 因子')
def alpha_46(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_46 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
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
                            description='alpha_47 因子')
def alpha_47(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_47 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, volume, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    volume = data['volume']
    vwap = data['vwap']
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
#

@register_worldquant_factor(name='alpha_48', description='Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))')
# def alpha_48(close: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#48: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass


@register_worldquant_factor(name='alpha_49',
                            description='alpha_49 因子')
def alpha_49(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_49 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
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
                            description='alpha_50 因子')
def alpha_50(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_50 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 volume, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    volume = data['volume']
    vwap = data['vwap']
    rank_volume = volume.rank(pct=True)
    rank_vwap = vwap.rank(pct=True)
    correlation = rank_volume.rolling(5).corr(rank_vwap)
    rank_correlation = correlation.rank(pct=True)
    ts_max_rank = rank_correlation.rolling(5).max()

    return -1 * ts_max_rank


# ==================== Alpha 51-101 (部分重要因子) ====================



@register_worldquant_factor(name='alpha_51',
                            description='alpha_51 因子')
def alpha_51(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_51 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
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
                            description='alpha_52 因子')
def alpha_52(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_52 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, low, volume, pct_chg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    low = data['low']
    volume = data['volume']
    pct_chg = data['pct_chg']
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
                            description='alpha_53 因子')
def alpha_53(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_53 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, low 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    low = data['low']
    numerator = (close - low) - (high - close)
    denominator = close - low
    ratio = numerator / denominator
    delta_ratio = ratio.diff(9)

    return -1 * delta_ratio

@register_worldquant_factor(name='alpha_54',
                            description='alpha_54 因子')
def alpha_54(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_54 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, high, low 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    high = data['high']
    low = data['low']
    numerator = -1 * ((low - close) * (open ** 5))
    denominator = (low - high) * (close ** 5)

    return numerator / denominator

@register_worldquant_factor(name='alpha_55',
                            description='alpha_55 因子')
def alpha_55(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_55 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, low, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
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
def alpha_56(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    
    现在使用AKShare的市值数据实现
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, pct_chg, total_market_cap_akshare 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    pct_chg = data['pct_chg']
    returns = pct_chg

    # 获取市值数据
    total_market_cap_akshare = data.get('total_market_cap_akshare', None)
    
    # 如果没有提供市值数据，使用默认值
    if total_market_cap_akshare is None or total_market_cap_akshare.isna().all():
        # 使用成交额作为市值的代理变量
        amount = data.get('amount', None)
        if amount is not None and not amount.isna().all():
            cap = amount
        else:
            cap = pd.Series([1e9] * len(close), index=close.index)
    else:
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
                            description='alpha_57 因子')
def alpha_57(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_57 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    vwap = data['vwap']

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
#

@register_worldquant_factor(name='alpha_58', description='Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))')
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
                            description='alpha_60 因子')
def alpha_60(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_60 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, low, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']

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
                            description='alpha_61 因子')
def alpha_61(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_61 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    vwap = data['vwap']
    volume = data['volume']
    adv180 = volume.rolling(180).mean()

    ts_min_vwap = vwap.rolling(16).min()
    vwap_ts_min_diff = vwap - ts_min_vwap
    rank_vwap_diff = vwap_ts_min_diff.rank(pct=True)

    correlation_vwap_adv180 = vwap.rolling(18).corr(adv180)
    rank_correlation = correlation_vwap_adv180.rank(pct=True)

    return (rank_vwap_diff < rank_correlation).astype(int)

@register_worldquant_factor(name='alpha_62',
                            description='alpha_62 因子')
def alpha_62(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_62 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, high, low, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']
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
#

@register_worldquant_factor(name='alpha_63', description='Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)')
# def alpha_63(close: pd.Series, open: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#63: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass


@register_worldquant_factor(name='alpha_64',
                            description='alpha_64 因子')
def alpha_64(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_64 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, high, low, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']
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
                            description='alpha_65 因子')
def alpha_65(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_65 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    vwap = data['vwap']
    volume = data['volume']
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
                            description='alpha_66 因子')
def alpha_66(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_66 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, high, low, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']

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
#

@register_worldquant_factor(name='alpha_67', description='Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)')
# def alpha_67(close: pd.Series, high: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#67: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass


@register_worldquant_factor(name='alpha_68',
                            description='alpha_68 因子')
def alpha_68(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_68 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, low, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
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
#

@register_worldquant_factor(name='alpha_69', description='Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)')
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
                            description='alpha_71 因子')
def alpha_71(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_71 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, low, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']

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
                            description='alpha_72 因子')
def alpha_72(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_72 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, low, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']

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
                            description='alpha_73 因子')
def alpha_73(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_73 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, low, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    low = data['low']
    vwap = data['vwap']

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
                            description='alpha_74 因子')
def alpha_74(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_74 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    vwap = data['vwap']
    volume = data['volume']
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
                            description='alpha_75 因子')
def alpha_75(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_75 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, high, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    high = data['high']
    vwap = data['vwap']
    volume = data['volume']
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
                            description='alpha_76 因子')
def alpha_76(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_76 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, low, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']

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
                            description='alpha_77 因子')
def alpha_77(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_77 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    vwap = data['vwap']
    volume = data['volume']

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
                            description='alpha_78 因子')
def alpha_78(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_78 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, high, low, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']

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
#

@register_worldquant_factor(name='alpha_79', description='Alpha#79: (rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), volume, 4.25197), 16.2289)) < rank(delta(IndNeutralize(close, IndClass.industry), 2.25164)))')
# def alpha_79(close: pd.Series, vwap: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
#     """Alpha#79: 需要行业中性化数据，暂时注释"""
#     # 需要行业分类数据，暂时无法实现
#     pass


@register_worldquant_factor(name='alpha_80',
                            description='alpha_80 因子')
def alpha_80(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_80 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, high, low, vwap, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    high = data['high']
    low = data['low']
    vwap = data['vwap']
    volume = data['volume']
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
#

@register_worldquant_factor(name='alpha_81', description='Alpha#81: (rank(decay_linear(delta(IndNeutralize(close, IndClass.subindustry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)')
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


@register_worldquant_factor(name='alpha_101',
                            description='alpha_101 因子')
def alpha_101(data: pd.DataFrame, **kwargs) -> pd.Series:
    """alpha_101 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, open, vwap 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    open = data['open']
    vwap = data['vwap']
    
    # 简单的示例因子实现
    return close.pct_change(1)
