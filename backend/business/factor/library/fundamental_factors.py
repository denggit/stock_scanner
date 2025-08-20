#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : fundamental_factors.py
@Description: 基本面因子库
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import pandas as pd
import numpy as np
from typing import Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'factor'))
from base_factor import register_fundamental_factor

# ==================== 估值类因子 ====================

@register_fundamental_factor(name='pe_ratio', description='市盈率因子')
def pe_ratio(close: pd.Series, pe_ttm: pd.Series, **kwargs) -> pd.Series:
    """市盈率因子：市盈率倒数"""
    return 1 / pe_ttm

@register_fundamental_factor(name='pb_ratio', description='市净率因子')
def pb_ratio(close: pd.Series, pb_mrq: pd.Series, **kwargs) -> pd.Series:
    """市净率因子：市净率倒数"""
    return 1 / pb_mrq

@register_fundamental_factor(name='ps_ratio', description='市销率因子')
def ps_ratio(close: pd.Series, ps_ttm: pd.Series, **kwargs) -> pd.Series:
    """市销率因子：市销率倒数"""
    return 1 / ps_ttm

@register_fundamental_factor(name='pcf_ratio', description='市现率因子')
def pcf_ratio(close: pd.Series, pcf_ncf_ttm: pd.Series, **kwargs) -> pd.Series:
    """市现率因子：市现率倒数"""
    return 1 / pcf_ncf_ttm

# ==================== 质量类因子 ====================

@register_fundamental_factor(name='roe', description='ROE因子')
def roe(close: pd.Series, roeAvg: pd.Series, **kwargs) -> pd.Series:
    """ROE因子：净资产收益率"""
    return roeAvg

@register_fundamental_factor(name='roa', description='ROA因子')
def roa(close: pd.Series, **kwargs) -> pd.Series:
    """ROA因子：总资产收益率（需要财务数据）"""
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([0.05] * len(close), index=close.index)

# ==================== 成长类因子 ====================

@register_fundamental_factor(name='revenue_growth', description='营收增长因子')
def revenue_growth(close: pd.Series, YOYAsset: pd.Series, **kwargs) -> pd.Series:
    """营收增长因子：资产同比增长率"""
    return YOYAsset

@register_fundamental_factor(name='profit_growth', description='利润增长因子')
def profit_growth(close: pd.Series, YOYNI: pd.Series, **kwargs) -> pd.Series:
    """利润增长因子：净利润同比增长率"""
    return YOYNI

# ==================== 杠杆类因子 ====================

@register_fundamental_factor(name='debt_to_equity', description='资产负债率因子')
def debt_to_equity(close: pd.Series, liabilityToAsset: pd.Series, **kwargs) -> pd.Series:
    """资产负债率因子：资产负债率"""
    return liabilityToAsset

@register_fundamental_factor(name='current_ratio', description='流动比率因子')
def current_ratio(close: pd.Series, currentRatio: pd.Series, **kwargs) -> pd.Series:
    """流动比率因子：流动比率"""
    return currentRatio

# ==================== 盈利能力因子 ====================

@register_fundamental_factor(name='net_profit_margin', description='净利润率因子')
def net_profit_margin(close: pd.Series, npMargin: pd.Series, **kwargs) -> pd.Series:
    """净利润率因子：净利润率"""
    return npMargin

@register_fundamental_factor(name='gross_profit_margin', description='毛利率因子')
def gross_profit_margin(close: pd.Series, gpMargin: pd.Series, **kwargs) -> pd.Series:
    """毛利率因子：毛利率"""
    return gpMargin

@register_fundamental_factor(name='eps_ttm', description='每股收益因子')
def eps_ttm(close: pd.Series, epsTTM: pd.Series, **kwargs) -> pd.Series:
    """每股收益因子：每股收益(TTM)"""
    return epsTTM

# ==================== 营运能力因子 ====================

@register_fundamental_factor(name='asset_turnover', description='资产周转率因子')
def asset_turnover(close: pd.Series, AssetTurnRatio: pd.Series, **kwargs) -> pd.Series:
    """资产周转率因子：总资产周转率"""
    return AssetTurnRatio

@register_fundamental_factor(name='inventory_turnover', description='存货周转率因子')
def inventory_turnover(close: pd.Series, INVTurnRatio: pd.Series, **kwargs) -> pd.Series:
    """存货周转率因子：存货周转率"""
    return INVTurnRatio

# ==================== 现金流量因子 ====================

@register_fundamental_factor(name='cfo_to_revenue', description='经营现金流因子')
def cfo_to_revenue(close: pd.Series, CFOToOR: pd.Series, **kwargs) -> pd.Series:
    """经营现金流因子：经营现金流/营业收入"""
    return CFOToOR

@register_fundamental_factor(name='cfo_to_profit', description='现金流利润比因子')
def cfo_to_profit(close: pd.Series, CFOToNP: pd.Series, **kwargs) -> pd.Series:
    """现金流利润比因子：经营现金流/净利润"""
    return CFOToNP
