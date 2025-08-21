#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : akshare_factors.py
@Description: 基于AKShare数据的新因子库
@Author     : Zijun Deng
@Date       : 2025-08-21
"""

import numpy as np
import pandas as pd

from ..core.factor.base_factor import register_technical_factor, register_fundamental_factor


# ✅ 向量化重构完成说明：
# 本文件已完成向量化重构，所有因子函数都已转换为新的接口格式
# 已完成：所有 37个 AKShare 因子
# 转换模式：从多个 pd.Series 参数改为单个 pd.DataFrame 参数，并在函数开头提取所需列

# ==================== 机构参与度因子 ====================


@register_technical_factor(name='institution_participation_factor', description='机构参与度因子')
def institution_participation_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """机构参与度因子：机构参与交易的程度
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 institution_participation 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['institution_participation']



@register_technical_factor(name='institution_buy_sell_ratio', description='机构买卖比率因子')
def institution_buy_sell_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
    """机构买卖比率因子：机构买入占比与卖出占比的比值
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 institution_buy_ratio, institution_sell_ratio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    institution_buy_ratio = data['institution_buy_ratio']
    institution_sell_ratio = data['institution_sell_ratio']
    return institution_buy_ratio / (institution_sell_ratio + 1e-8)



@register_technical_factor(name='institution_net_buy_ratio',
                            description='institution_net_buy_ratio 因子')
def institution_net_buy_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
    """institution_net_buy_ratio 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 institution_buy_ratio, institution_sell_ratio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    institution_buy_ratio = data['institution_buy_ratio']
    institution_sell_ratio = data['institution_sell_ratio']
    return institution_buy_ratio - institution_sell_ratio


# ==================== 综合评分因子 ====================


@register_fundamental_factor(name='comprehensive_score_factor',
                            description='comprehensive_score_factor 因子')
def comprehensive_score_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """comprehensive_score_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 comprehensive_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    comprehensive_score = data['comprehensive_score']
    return comprehensive_score



@register_technical_factor(name='technical_score_factor',
                            description='technical_score_factor 因子')
def technical_score_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """technical_score_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 technical_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    technical_score = data['technical_score']
    return technical_score



@register_fundamental_factor(name='fundamental_score_factor',
                            description='fundamental_score_factor 因子')
def fundamental_score_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """fundamental_score_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 fundamental_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    fundamental_score = data['fundamental_score']
    return fundamental_score



@register_technical_factor(name='market_score_factor',
                            description='market_score_factor 因子')
def market_score_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """market_score_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 market_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    market_score = data['market_score']
    return market_score



@register_technical_factor(name='fund_score_factor',
                            description='fund_score_factor 因子')
def fund_score_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """fund_score_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 fund_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    fund_score = data['fund_score']
    return fund_score



@register_technical_factor(name='news_score_factor',
                            description='news_score_factor 因子')
def news_score_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """news_score_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 news_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    news_score = data['news_score']
    return news_score


# ==================== 市场关注度因子 ====================


@register_technical_factor(name='market_attention_factor',
                            description='market_attention_factor 因子')
def market_attention_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """market_attention_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 market_attention 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    market_attention = data['market_attention']
    return market_attention



@register_technical_factor(name='attention_change_factor',
                            description='attention_change_factor 因子')
def attention_change_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """attention_change_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 attention_change 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    attention_change = data['attention_change']
    return attention_change@register_technical_factor(name='attention_momentum_factor',
                            description='attention_momentum_factor 因子')
def attention_momentum_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """attention_momentum_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 market_attention, window 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    market_attention = data['market_attention']
    window = data['window']
    return market_attention.pct_change(window)


# ==================== 筹码分布因子 ====================

@register_technical_factor(name='chip_concentration_factor',
                            description='chip_concentration_factor 因子')
def chip_concentration_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """chip_concentration_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 chip_concentration 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    chip_concentration = data['chip_concentration']
    return chip_concentration@register_technical_factor(name='profit_ratio_factor',
                            description='profit_ratio_factor 因子')
def profit_ratio_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """profit_ratio_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 profit_ratio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    profit_ratio = data['profit_ratio']
    return profit_ratio@register_technical_factor(name='chip_cost_factor',
                            description='chip_cost_factor 因子')
def chip_cost_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """chip_cost_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 chip_cost, close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    chip_cost = data['chip_cost']
    close = data['close']
    return (close - chip_cost) / chip_cost@register_technical_factor(name='chip_distribution_factor',
                            description='chip_distribution_factor 因子')
def chip_distribution_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """chip_distribution_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 chip_ratio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    chip_ratio = data['chip_ratio']
    return chip_ratio


# ==================== 估值因子（基于AKShare数据） ====================

@register_fundamental_factor(name='pe_akshare_factor',
                            description='pe_akshare_factor 因子')
def pe_akshare_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """pe_akshare_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 pe_akshare 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pe_akshare = data['pe_akshare']
    return 1 / (pe_akshare + 1e-8)@register_fundamental_factor(name='pb_akshare_factor',
                            description='pb_akshare_factor 因子')
def pb_akshare_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """pb_akshare_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 pb_akshare 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pb_akshare = data['pb_akshare']
    return 1 / (pb_akshare + 1e-8)@register_fundamental_factor(name='dividend_yield_factor',
                            description='dividend_yield_factor 因子')
def dividend_yield_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """dividend_yield_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 dividend_yield_akshare 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    dividend_yield_akshare = data['dividend_yield_akshare']
    return dividend_yield_akshare


# ==================== 市值因子 ====================

@register_fundamental_factor(name='market_cap_factor',
                            description='market_cap_factor 因子')
def market_cap_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """market_cap_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 total_market_cap_akshare 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    total_market_cap_akshare = data['total_market_cap_akshare']
    return np.log(total_market_cap_akshare + 1e-8)@register_fundamental_factor(name='circulating_market_cap_factor',
                            description='circulating_market_cap_factor 因子')
def circulating_market_cap_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """circulating_market_cap_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 circulating_market_cap_akshare 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    circulating_market_cap_akshare = data['circulating_market_cap_akshare']
    return np.log(circulating_market_cap_akshare + 1e-8)@register_fundamental_factor(name='market_cap_ratio_factor',
                            description='market_cap_ratio_factor 因子')
def market_cap_ratio_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """market_cap_ratio_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 circulating_market_cap_akshare, total_market_cap_akshare 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    circulating_market_cap_akshare = data['circulating_market_cap_akshare']
    total_market_cap_akshare = data['total_market_cap_akshare']
    return circulating_market_cap_akshare / (total_market_cap_akshare + 1e-8)


# ==================== 复合因子 ====================

@register_technical_factor(name='institution_attention_factor',
                            description='institution_attention_factor 因子')
def institution_attention_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """institution_attention_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 institution_participation, market_attention 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    institution_participation = data['institution_participation']
    market_attention = data['market_attention']
    return institution_participation * market_attention@register_technical_factor(name='score_momentum_factor',
                            description='score_momentum_factor 因子')
def score_momentum_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """score_momentum_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 comprehensive_score, window 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    comprehensive_score = data['comprehensive_score']
    window = data['window']
    return comprehensive_score.pct_change(window)@register_technical_factor(name='chip_attention_factor',
                            description='chip_attention_factor 因子')
def chip_attention_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """chip_attention_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 chip_concentration, market_attention 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    chip_concentration = data['chip_concentration']
    market_attention = data['market_attention']
    return chip_concentration * market_attention@register_fundamental_factor(name='valuation_quality_factor',
                            description='valuation_quality_factor 因子')
def valuation_quality_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """valuation_quality_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 pe_akshare, pb_akshare, comprehensive_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pe_akshare = data['pe_akshare']
    pb_akshare = data['pb_akshare']
    comprehensive_score = data['comprehensive_score']
    pe_factor = 1 / (pe_akshare + 1e-8)
    pb_factor = 1 / (pb_akshare + 1e-8)
    return (pe_factor + pb_factor) * comprehensive_score


# ==================== 技术突破因子 ====================

@register_technical_factor(name='breakout_strength_factor',
                            description='breakout_strength_factor 因子')
def breakout_strength_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """breakout_strength_factor 因子
    
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
    # 计算突破强度
    high_20d = high.rolling(20).max()
    low_20d = low.rolling(20).min()

    # 向上突破强度
    up_breakout = (close > high_20d.shift(1)).astype(int) * (close - high_20d.shift(1)) / high_20d.shift(1)

    # 向下突破强度
    down_breakout = (close < low_20d.shift(1)).astype(int) * (low_20d.shift(1) - close) / low_20d.shift(1)

    # 成交量加权
    volume_weight = volume / volume.rolling(20).mean()

    return (up_breakout - down_breakout) * volume_weight@register_technical_factor(name='volume_price_breakout_factor',
                            description='volume_price_breakout_factor 因子')
def volume_price_breakout_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """volume_price_breakout_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, volume 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    volume = data['volume']
    # 价格突破
    price_breakout = close.pct_change(5)

    # 成交量突破
    volume_breakout = volume / volume.rolling(20).mean()

    return price_breakout * volume_breakout


# ==================== 市场情绪因子 ====================

@register_technical_factor(name='sentiment_momentum_factor',
                            description='sentiment_momentum_factor 因子')
def sentiment_momentum_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """sentiment_momentum_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 market_attention, attention_change 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    market_attention = data['market_attention']
    attention_change = data['attention_change']
    attention_momentum = market_attention.pct_change(5)
    change_momentum = attention_change.pct_change(5)

    return attention_momentum + change_momentum@register_technical_factor(name='institution_sentiment_factor',
                            description='institution_sentiment_factor 因子')
def institution_sentiment_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """institution_sentiment_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 institution_participation, institution_buy_ratio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    institution_participation = data['institution_participation']
    institution_buy_ratio = data['institution_buy_ratio']
    participation_momentum = institution_participation.pct_change(5)
    buy_momentum = institution_buy_ratio.pct_change(5)

    return participation_momentum * buy_momentum


# ==================== 质量因子 ====================

@register_fundamental_factor(name='quality_score_factor',
                            description='quality_score_factor 因子')
def quality_score_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """quality_score_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 fundamental_score, comprehensive_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    fundamental_score = data['fundamental_score']
    comprehensive_score = data['comprehensive_score']
    return (fundamental_score + comprehensive_score) / 2@register_fundamental_factor(name='growth_quality_factor',
                            description='growth_quality_factor 因子')
def growth_quality_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """growth_quality_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 comprehensive_score, market_cap_factor 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    comprehensive_score = data['comprehensive_score']
    market_cap_factor = data['market_cap_factor']
    return comprehensive_score / (market_cap_factor + 1e-8)@register_technical_factor(name='momentum_quality_factor',
                            description='momentum_quality_factor 因子')
def momentum_quality_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """momentum_quality_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, comprehensive_score, window 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    comprehensive_score = data['comprehensive_score']
    window = data['window']
    price_momentum = close.pct_change(window)
    return price_momentum * comprehensive_score


# ==================== 风险因子 ====================

@register_technical_factor(name='volatility_attention_factor',
                            description='volatility_attention_factor 因子')
def volatility_attention_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """volatility_attention_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, market_attention, window 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    market_attention = data['market_attention']
    window = data['window']
    volatility = close.pct_change().rolling(window).std()
    return volatility * market_attention@register_technical_factor(name='risk_sentiment_factor',
                            description='risk_sentiment_factor 因子')
def risk_sentiment_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """risk_sentiment_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 profit_ratio, chip_concentration 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    profit_ratio = data['profit_ratio']
    chip_concentration = data['chip_concentration']
    # 获利盘比例越高，风险越大
    risk_level = profit_ratio / (chip_concentration + 1e-8)
    return risk_level


# ==================== 综合因子 ====================

@register_technical_factor(name='akshare_composite_factor',
                            description='akshare_composite_factor 因子')
def akshare_composite_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """akshare_composite_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 institution_participation, market_attention, comprehensive_score, chip_concentration 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    institution_participation = data['institution_participation']
    market_attention = data['market_attention']
    comprehensive_score = data['comprehensive_score']
    chip_concentration = data['chip_concentration']
    # 标准化各个因子
    inst_norm = (institution_participation - institution_participation.rolling(252).mean()) / (
                institution_participation.rolling(252).std() + 1e-8)
    attention_norm = (market_attention - market_attention.rolling(252).mean()) / (
                market_attention.rolling(252).std() + 1e-8)
    score_norm = (comprehensive_score - comprehensive_score.rolling(252).mean()) / (
                comprehensive_score.rolling(252).std() + 1e-8)
    chip_norm = (chip_concentration - chip_concentration.rolling(252).mean()) / (
                chip_concentration.rolling(252).std() + 1e-8)

    # 综合计算
    composite = (inst_norm + attention_norm + score_norm + chip_norm) / 4

    return composite

@register_fundamental_factor(name='akshare_fundamental_factor',
                            description='akshare_fundamental_factor 因子')
def akshare_fundamental_factor(data: pd.DataFrame, **kwargs) -> pd.Series:
    """akshare_fundamental_factor 因子
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 pe_akshare, pb_akshare, dividend_yield_akshare, fundamental_score 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pe_akshare = data['pe_akshare']
    pb_akshare = data['pb_akshare']
    dividend_yield_akshare = data['dividend_yield_akshare']
    fundamental_score = data['fundamental_score']
    # 估值因子
    pe_factor = 1 / (pe_akshare + 1e-8)
    pb_factor = 1 / (pb_akshare + 1e-8)

    # 标准化
    pe_norm = (pe_factor - pe_factor.rolling(252).mean()) / (pe_factor.rolling(252).std() + 1e-8)
    pb_norm = (pb_factor - pb_factor.rolling(252).mean()) / (pb_factor.rolling(252).std() + 1e-8)
    dividend_norm = (dividend_yield_akshare - dividend_yield_akshare.rolling(252).mean()) / (
                dividend_yield_akshare.rolling(252).std() + 1e-8)
    score_norm = (fundamental_score - fundamental_score.rolling(252).mean()) / (
                fundamental_score.rolling(252).std() + 1e-8)

    # 综合计算
    fundamental = (pe_norm + pb_norm + dividend_norm + score_norm) / 4

    return fundamental
