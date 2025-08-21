#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : fundamental_factors.py
@Description: 基本面因子库
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import pandas as pd

from ..core.factor.base_factor import register_fundamental_factor


# ==================== 估值类因子 ====================

@register_fundamental_factor(name='pe_ratio', description='市盈率因子')
def pe_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
    """市盈率因子：市盈率倒数
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, pe_ttm 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pe_ttm = data['pe_ttm']
    return 1 / pe_ttm


@register_fundamental_factor(name='pb_ratio', description='市净率因子')
def pb_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
    """市净率因子：市净率倒数
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, pb_mrq 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pb_mrq = data['pb_mrq']
    return 1 / pb_mrq


@register_fundamental_factor(name='ps_ratio', description='市销率因子')
def ps_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
    """市销率因子：市销率倒数
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, ps_ttm 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    ps_ttm = data['ps_ttm']
    return 1 / ps_ttm


@register_fundamental_factor(name='pcf_ratio', description='市现率因子')
def pcf_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
    """市现率因子：市现率倒数
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, pcf_ncf_ttm 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    pcf_ncf_ttm = data['pcf_ncf_ttm']
    return 1 / pcf_ncf_ttm


# ==================== 质量类因子 ====================

@register_fundamental_factor(name='roe', description='ROE因子')
def roe(data: pd.DataFrame, **kwargs) -> pd.Series:
    """ROE因子：净资产收益率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close, roeAvg 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    roeAvg = data['roeAvg']
    return roeAvg


@register_fundamental_factor(name='roa', description='ROA因子')
def roa(data: pd.DataFrame, **kwargs) -> pd.Series:
    """ROA因子：总资产收益率（需要财务数据）
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 close 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    close = data['close']
    # 这里需要根据实际财务数据字段调整
    # 暂时返回一个示例值
    return pd.Series([0.05] * len(close), index=close.index)


# ==================== 成长类因子 ====================

# 🔄 向量化重构说明：以下因子需要按照上述模式进行转换
# 转换模式：
# 1. 函数参数从 (close: pd.Series, field: pd.Series, **kwargs) 改为 (data: pd.DataFrame, **kwargs)
# 2. 在函数开头提取所需字段：field = data['field']
# 3. 添加完整的 docstring

@register_fundamental_factor(name='revenue_growth', description='营收增长因子')
def revenue_growth(data: pd.DataFrame, **kwargs) -> pd.Series:
    """营收增长因子：资产同比增长率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 YOYAsset 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['YOYAsset']


@register_fundamental_factor(name='profit_growth', description='利润增长因子')
def profit_growth(data: pd.DataFrame, **kwargs) -> pd.Series:
    """利润增长因子：净利润同比增长率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 YOYNI 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['YOYNI']


# ==================== 杠杆类因子 ====================

@register_fundamental_factor(name='debt_to_equity', description='资产负债率因子')
def debt_to_equity(data: pd.DataFrame, **kwargs) -> pd.Series:
    """资产负债率因子：资产负债率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 liabilityToAsset 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['liabilityToAsset']


@register_fundamental_factor(name='current_ratio', description='流动比率因子')
def current_ratio(data: pd.DataFrame, **kwargs) -> pd.Series:
    """流动比率因子：流动比率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 currentRatio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['currentRatio']


# ==================== 盈利能力因子 ====================

@register_fundamental_factor(name='net_profit_margin', description='净利润率因子')
def net_profit_margin(data: pd.DataFrame, **kwargs) -> pd.Series:
    """净利润率因子：净利润率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 npMargin 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['npMargin']


@register_fundamental_factor(name='gross_profit_margin', description='毛利率因子')
def gross_profit_margin(data: pd.DataFrame, **kwargs) -> pd.Series:
    """毛利率因子：毛利率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 gpMargin 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['gpMargin']


@register_fundamental_factor(name='eps_ttm', description='每股收益因子')
def eps_ttm(data: pd.DataFrame, **kwargs) -> pd.Series:
    """每股收益因子：每股收益(TTM)
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 epsTTM 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['epsTTM']


# ==================== 营运能力因子 ====================

@register_fundamental_factor(name='asset_turnover', description='资产周转率因子')
def asset_turnover(data: pd.DataFrame, **kwargs) -> pd.Series:
    """资产周转率因子：总资产周转率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 AssetTurnRatio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['AssetTurnRatio']


@register_fundamental_factor(name='inventory_turnover', description='存货周转率因子')
def inventory_turnover(data: pd.DataFrame, **kwargs) -> pd.Series:
    """存货周转率因子：存货周转率
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 INVTurnRatio 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['INVTurnRatio']


# ==================== 现金流量因子 ====================

@register_fundamental_factor(name='cfo_to_revenue', description='经营现金流因子')
def cfo_to_revenue(data: pd.DataFrame, **kwargs) -> pd.Series:
    """经营现金流因子：经营现金流/营业收入
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 CFOToOR 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['CFOToOR']


@register_fundamental_factor(name='cfo_to_profit', description='现金流利润比因子')
def cfo_to_profit(data: pd.DataFrame, **kwargs) -> pd.Series:
    """现金流利润比因子：经营现金流/净利润
    
    Args:
        data: 单只股票的历史数据 DataFrame，包含 CFOToNP 等列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    return data['CFOToNP']
