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
def roe(close: pd.Series, **kwargs) -> pd.Series:
    """ROE因子：净资产收益率（需要财务数据）"""
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([0.1] * len(close), index=close.index)

@register_fundamental_factor(name='roa', description='ROA因子')
def roa(close: pd.Series, **kwargs) -> pd.Series:
    """ROA因子：总资产收益率（需要财务数据）"""
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([0.05] * len(close), index=close.index)

# ==================== 成长类因子 ====================

@register_fundamental_factor(name='revenue_growth', description='营收增长因子')
def revenue_growth(close: pd.Series, **kwargs) -> pd.Series:
    """营收增长因子（需要财务数据）"""
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([0.15] * len(close), index=close.index)

@register_fundamental_factor(name='profit_growth', description='利润增长因子')
def profit_growth(close: pd.Series, **kwargs) -> pd.Series:
    """利润增长因子（需要财务数据）"""
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([0.12] * len(close), index=close.index)

# ==================== 杠杆类因子 ====================

@register_fundamental_factor(name='debt_to_equity', description='资产负债率因子')
def debt_to_equity(close: pd.Series, **kwargs) -> pd.Series:
    """资产负债率因子（需要财务数据）"""
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([0.4] * len(close), index=close.index)

@register_fundamental_factor(name='current_ratio', description='流动比率因子')
def current_ratio(close: pd.Series, **kwargs) -> pd.Series:
    """流动比率因子（需要财务数据）"""
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([1.5] * len(close), index=close.index)
