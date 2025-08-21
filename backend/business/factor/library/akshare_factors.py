#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : akshare_factors.py
@Description: 基于AKShare数据的新因子库
@Author     : Zijun Deng
@Date       : 2025-08-21
"""

import pandas as pd
import numpy as np
from typing import Optional
from ..core.factor.base_factor import register_technical_factor, register_fundamental_factor

# ==================== 机构参与度因子 ====================

@register_technical_factor(name='institution_participation_factor', description='机构参与度因子')
def institution_participation_factor(institution_participation: pd.Series, **kwargs) -> pd.Series:
    """机构参与度因子：机构参与交易的程度"""
    return institution_participation

@register_technical_factor(name='institution_buy_sell_ratio', description='机构买卖比率因子')
def institution_buy_sell_ratio(institution_buy_ratio: pd.Series, institution_sell_ratio: pd.Series, **kwargs) -> pd.Series:
    """机构买卖比率因子：机构买入占比与卖出占比的比值"""
    return institution_buy_ratio / (institution_sell_ratio + 1e-8)

@register_technical_factor(name='institution_net_buy_ratio', description='机构净买入比率因子')
def institution_net_buy_ratio(institution_buy_ratio: pd.Series, institution_sell_ratio: pd.Series, **kwargs) -> pd.Series:
    """机构净买入比率因子：机构净买入占比"""
    return institution_buy_ratio - institution_sell_ratio

# ==================== 综合评分因子 ====================

@register_fundamental_factor(name='comprehensive_score_factor', description='综合评分因子')
def comprehensive_score_factor(comprehensive_score: pd.Series, **kwargs) -> pd.Series:
    """综合评分因子：股票综合评分"""
    return comprehensive_score

@register_technical_factor(name='technical_score_factor', description='技术评分因子')
def technical_score_factor(technical_score: pd.Series, **kwargs) -> pd.Series:
    """技术评分因子：技术面评分"""
    return technical_score

@register_fundamental_factor(name='fundamental_score_factor', description='基本面评分因子')
def fundamental_score_factor(fundamental_score: pd.Series, **kwargs) -> pd.Series:
    """基本面评分因子：基本面评分"""
    return fundamental_score

@register_technical_factor(name='market_score_factor', description='市场评分因子')
def market_score_factor(market_score: pd.Series, **kwargs) -> pd.Series:
    """市场评分因子：市场表现评分"""
    return market_score

@register_technical_factor(name='fund_score_factor', description='资金评分因子')
def fund_score_factor(fund_score: pd.Series, **kwargs) -> pd.Series:
    """资金评分因子：资金面评分"""
    return fund_score

@register_technical_factor(name='news_score_factor', description='消息面评分因子')
def news_score_factor(news_score: pd.Series, **kwargs) -> pd.Series:
    """消息面评分因子：消息面评分"""
    return news_score

# ==================== 市场关注度因子 ====================

@register_technical_factor(name='market_attention_factor', description='市场关注度因子')
def market_attention_factor(market_attention: pd.Series, **kwargs) -> pd.Series:
    """市场关注度因子：用户关注程度"""
    return market_attention

@register_technical_factor(name='attention_change_factor', description='关注度变化因子')
def attention_change_factor(attention_change: pd.Series, **kwargs) -> pd.Series:
    """关注度变化因子：关注度变化情况"""
    return attention_change

@register_technical_factor(name='attention_momentum_factor', description='关注度动量因子')
def attention_momentum_factor(market_attention: pd.Series, window: int = 5, **kwargs) -> pd.Series:
    """关注度动量因子：关注度的变化趋势"""
    return market_attention.pct_change(window)

# ==================== 筹码分布因子 ====================

@register_technical_factor(name='chip_concentration_factor', description='筹码集中度因子')
def chip_concentration_factor(chip_concentration: pd.Series, **kwargs) -> pd.Series:
    """筹码集中度因子：筹码集中程度"""
    return chip_concentration

@register_technical_factor(name='profit_ratio_factor', description='获利盘比例因子')
def profit_ratio_factor(profit_ratio: pd.Series, **kwargs) -> pd.Series:
    """获利盘比例因子：获利盘比例"""
    return profit_ratio

@register_technical_factor(name='chip_cost_factor', description='筹码成本因子')
def chip_cost_factor(chip_cost: pd.Series, close: pd.Series, **kwargs) -> pd.Series:
    """筹码成本因子：筹码成本相对当前价格的位置"""
    return (close - chip_cost) / chip_cost

@register_technical_factor(name='chip_distribution_factor', description='筹码分布因子')
def chip_distribution_factor(chip_ratio: pd.Series, **kwargs) -> pd.Series:
    """筹码分布因子：筹码分布情况"""
    return chip_ratio

# ==================== 估值因子（基于AKShare数据） ====================

@register_fundamental_factor(name='pe_akshare_factor', description='AKShare市盈率因子')
def pe_akshare_factor(pe_akshare: pd.Series, **kwargs) -> pd.Series:
    """AKShare市盈率因子：市盈率倒数"""
    return 1 / (pe_akshare + 1e-8)

@register_fundamental_factor(name='pb_akshare_factor', description='AKShare市净率因子')
def pb_akshare_factor(pb_akshare: pd.Series, **kwargs) -> pd.Series:
    """AKShare市净率因子：市净率倒数"""
    return 1 / (pb_akshare + 1e-8)

@register_fundamental_factor(name='dividend_yield_factor', description='股息率因子')
def dividend_yield_factor(dividend_yield_akshare: pd.Series, **kwargs) -> pd.Series:
    """股息率因子：股息率"""
    return dividend_yield_akshare

# ==================== 市值因子 ====================

@register_fundamental_factor(name='market_cap_factor', description='总市值因子')
def market_cap_factor(total_market_cap_akshare: pd.Series, **kwargs) -> pd.Series:
    """总市值因子：总市值的对数"""
    return np.log(total_market_cap_akshare + 1e-8)

@register_fundamental_factor(name='circulating_market_cap_factor', description='流通市值因子')
def circulating_market_cap_factor(circulating_market_cap_akshare: pd.Series, **kwargs) -> pd.Series:
    """流通市值因子：流通市值的对数"""
    return np.log(circulating_market_cap_akshare + 1e-8)

@register_fundamental_factor(name='market_cap_ratio_factor', description='流通市值比率因子')
def market_cap_ratio_factor(circulating_market_cap_akshare: pd.Series, total_market_cap_akshare: pd.Series, **kwargs) -> pd.Series:
    """流通市值比率因子：流通市值占总市值的比例"""
    return circulating_market_cap_akshare / (total_market_cap_akshare + 1e-8)

# ==================== 复合因子 ====================

@register_technical_factor(name='institution_attention_factor', description='机构关注度复合因子')
def institution_attention_factor(institution_participation: pd.Series, market_attention: pd.Series, **kwargs) -> pd.Series:
    """机构关注度复合因子：机构参与度与市场关注度的结合"""
    return institution_participation * market_attention

@register_technical_factor(name='score_momentum_factor', description='评分动量因子')
def score_momentum_factor(comprehensive_score: pd.Series, window: int = 5, **kwargs) -> pd.Series:
    """评分动量因子：综合评分的变化趋势"""
    return comprehensive_score.pct_change(window)

@register_technical_factor(name='chip_attention_factor', description='筹码关注度因子')
def chip_attention_factor(chip_concentration: pd.Series, market_attention: pd.Series, **kwargs) -> pd.Series:
    """筹码关注度因子：筹码集中度与市场关注度的结合"""
    return chip_concentration * market_attention

@register_fundamental_factor(name='valuation_quality_factor', description='估值质量因子')
def valuation_quality_factor(pe_akshare: pd.Series, pb_akshare: pd.Series, comprehensive_score: pd.Series, **kwargs) -> pd.Series:
    """估值质量因子：估值指标与综合评分的结合"""
    pe_factor = 1 / (pe_akshare + 1e-8)
    pb_factor = 1 / (pb_akshare + 1e-8)
    return (pe_factor + pb_factor) * comprehensive_score

# ==================== 技术突破因子 ====================

@register_technical_factor(name='breakout_strength_factor', description='突破强度因子')
def breakout_strength_factor(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """突破强度因子：价格突破的强度"""
    # 计算突破强度
    high_20d = high.rolling(20).max()
    low_20d = low.rolling(20).min()
    
    # 向上突破强度
    up_breakout = (close > high_20d.shift(1)).astype(int) * (close - high_20d.shift(1)) / high_20d.shift(1)
    
    # 向下突破强度
    down_breakout = (close < low_20d.shift(1)).astype(int) * (low_20d.shift(1) - close) / low_20d.shift(1)
    
    # 成交量加权
    volume_weight = volume / volume.rolling(20).mean()
    
    return (up_breakout - down_breakout) * volume_weight

@register_technical_factor(name='volume_price_breakout_factor', description='量价突破因子')
def volume_price_breakout_factor(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    """量价突破因子：价格和成交量的突破组合"""
    # 价格突破
    price_breakout = close.pct_change(5)
    
    # 成交量突破
    volume_breakout = volume / volume.rolling(20).mean()
    
    return price_breakout * volume_breakout

# ==================== 市场情绪因子 ====================

@register_technical_factor(name='sentiment_momentum_factor', description='情绪动量因子')
def sentiment_momentum_factor(market_attention: pd.Series, attention_change: pd.Series, **kwargs) -> pd.Series:
    """情绪动量因子：市场情绪的变化趋势"""
    attention_momentum = market_attention.pct_change(5)
    change_momentum = attention_change.pct_change(5)
    
    return attention_momentum + change_momentum

@register_technical_factor(name='institution_sentiment_factor', description='机构情绪因子')
def institution_sentiment_factor(institution_participation: pd.Series, institution_buy_ratio: pd.Series, **kwargs) -> pd.Series:
    """机构情绪因子：机构参与度和买入意愿的结合"""
    participation_momentum = institution_participation.pct_change(5)
    buy_momentum = institution_buy_ratio.pct_change(5)
    
    return participation_momentum * buy_momentum

# ==================== 质量因子 ====================

@register_fundamental_factor(name='quality_score_factor', description='质量评分因子')
def quality_score_factor(fundamental_score: pd.Series, comprehensive_score: pd.Series, **kwargs) -> pd.Series:
    """质量评分因子：基本面评分与综合评分的结合"""
    return (fundamental_score + comprehensive_score) / 2

@register_fundamental_factor(name='growth_quality_factor', description='成长质量因子')
def growth_quality_factor(comprehensive_score: pd.Series, market_cap_factor: pd.Series, **kwargs) -> pd.Series:
    """成长质量因子：综合评分与市值的结合（小市值高质量）"""
    return comprehensive_score / (market_cap_factor + 1e-8)

@register_technical_factor(name='momentum_quality_factor', description='动量质量因子')
def momentum_quality_factor(close: pd.Series, comprehensive_score: pd.Series, window: int = 20, **kwargs) -> pd.Series:
    """动量质量因子：价格动量与质量的结合"""
    price_momentum = close.pct_change(window)
    return price_momentum * comprehensive_score

# ==================== 风险因子 ====================

@register_technical_factor(name='volatility_attention_factor', description='波动率关注度因子')
def volatility_attention_factor(close: pd.Series, market_attention: pd.Series, window: int = 20, **kwargs) -> pd.Series:
    """波动率关注度因子：价格波动率与关注度的结合"""
    volatility = close.pct_change().rolling(window).std()
    return volatility * market_attention

@register_technical_factor(name='risk_sentiment_factor', description='风险情绪因子')
def risk_sentiment_factor(profit_ratio: pd.Series, chip_concentration: pd.Series, **kwargs) -> pd.Series:
    """风险情绪因子：获利盘比例与筹码集中度的结合"""
    # 获利盘比例越高，风险越大
    risk_level = profit_ratio / (chip_concentration + 1e-8)
    return risk_level

# ==================== 综合因子 ====================

@register_technical_factor(name='akshare_composite_factor', description='AKShare综合因子')
def akshare_composite_factor(institution_participation: pd.Series, 
                           market_attention: pd.Series, 
                           comprehensive_score: pd.Series,
                           chip_concentration: pd.Series,
                           **kwargs) -> pd.Series:
    """AKShare综合因子：多个AKShare指标的综合"""
    # 标准化各个因子
    inst_norm = (institution_participation - institution_participation.rolling(252).mean()) / (institution_participation.rolling(252).std() + 1e-8)
    attention_norm = (market_attention - market_attention.rolling(252).mean()) / (market_attention.rolling(252).std() + 1e-8)
    score_norm = (comprehensive_score - comprehensive_score.rolling(252).mean()) / (comprehensive_score.rolling(252).std() + 1e-8)
    chip_norm = (chip_concentration - chip_concentration.rolling(252).mean()) / (chip_concentration.rolling(252).std() + 1e-8)
    
    # 综合计算
    composite = (inst_norm + attention_norm + score_norm + chip_norm) / 4
    
    return composite

@register_fundamental_factor(name='akshare_fundamental_factor', description='AKShare基本面因子')
def akshare_fundamental_factor(pe_akshare: pd.Series,
                             pb_akshare: pd.Series,
                             dividend_yield_akshare: pd.Series,
                             fundamental_score: pd.Series,
                             **kwargs) -> pd.Series:
    """AKShare基本面因子：基于AKShare数据的基本面综合因子"""
    # 估值因子
    pe_factor = 1 / (pe_akshare + 1e-8)
    pb_factor = 1 / (pb_akshare + 1e-8)
    
    # 标准化
    pe_norm = (pe_factor - pe_factor.rolling(252).mean()) / (pe_factor.rolling(252).std() + 1e-8)
    pb_norm = (pb_factor - pb_factor.rolling(252).mean()) / (pb_factor.rolling(252).std() + 1e-8)
    dividend_norm = (dividend_yield_akshare - dividend_yield_akshare.rolling(252).mean()) / (dividend_yield_akshare.rolling(252).std() + 1e-8)
    score_norm = (fundamental_score - fundamental_score.rolling(252).mean()) / (fundamental_score.rolling(252).std() + 1e-8)
    
    # 综合计算
    fundamental = (pe_norm + pb_norm + dividend_norm + score_norm) / 4
    
    return fundamental
