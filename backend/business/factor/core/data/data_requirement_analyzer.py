#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : data_requirement_analyzer.py
@Description: 数据需求分析器 - 根据因子类型决定数据获取策略
@Author     : Zijun Deng
@Date       : 2025-08-21
"""

import ast
import inspect
from typing import Dict, List, Set, Callable, Any

from backend.utils.logger import setup_logger

logger = setup_logger("backtest_factor")


class DataRequirementAnalyzer:
    """
    数据需求分析器
    
    功能：
    1. 根据因子类型决定数据获取策略
    2. 提供简单直接的数据字段映射
    3. 避免过度复杂的字段分析
    """

    def __init__(self):
        """初始化数据需求分析器"""
        # 定义因子类型到数据字段的映射
        self.factor_data_mapping = {
            # 技术因子 - 只需要行情数据
            'technical': {
                'market_fields': ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'vwap'],
                'valuation_fields': ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm'],
                'financial_types': [],
                'technical_fields': []
            },
            
            # WorldQuant因子 - 需要行情数据
            'worldquant': {
                'market_fields': ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'vwap'],
            'valuation_fields': ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm'],
                'financial_types': [],
                'technical_fields': []
            },
            
            # 基本面因子 - 需要行情数据和财务数据
            'fundamental': {
                'market_fields': ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'vwap'],
                'valuation_fields': ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm'],
                'financial_types': ['profit', 'balance', 'cashflow', 'growth', 'operation', 'dupont', 'dividend'],
                'technical_fields': []
            },
            
            # 通道因子 - 只需要行情数据
            'channel': {
                'market_fields': ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'vwap'],
                'valuation_fields': ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm'],
                'financial_types': [],
                'technical_fields': []
            },
            
            # AKShare因子 - 需要行情数据
            'akshare': {
                'market_fields': ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'vwap'],
                'valuation_fields': ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm'],
                'financial_types': [],
                'technical_fields': []
            }
        }
        
        # 因子名称到类型的映射
        self.factor_type_mapping = {
            # 技术因子
            'momentum_5d': 'technical',
            'momentum_10d': 'technical',
            'momentum_20d': 'technical',
            'rsi': 'technical',
            'macd': 'technical',
            'bollinger': 'technical',
            'ma_cross': 'technical',
            'volume_ratio': 'technical',
            'volatility': 'technical',
            'atr': 'technical',
            
            # WorldQuant因子
            'alpha_1': 'worldquant',
            'alpha_2': 'worldquant',
            'alpha_3': 'worldquant',
            'alpha_4': 'worldquant',
            'alpha_5': 'worldquant',
            'alpha_6': 'worldquant',
            'alpha_7': 'worldquant',
            'alpha_8': 'worldquant',
            'alpha_9': 'worldquant',
            'alpha_10': 'worldquant',
            'alpha_11': 'worldquant',
            'alpha_12': 'worldquant',
            'alpha_13': 'worldquant',
            'alpha_14': 'worldquant',
            'alpha_15': 'worldquant',
            'alpha_16': 'worldquant',
            'alpha_17': 'worldquant',
            'alpha_18': 'worldquant',
            'alpha_19': 'worldquant',
            'alpha_20': 'worldquant',
            'alpha_21': 'worldquant',
            'alpha_22': 'worldquant',
            'alpha_23': 'worldquant',
            'alpha_24': 'worldquant',
            'alpha_25': 'worldquant',
            'alpha_26': 'worldquant',
            'alpha_27': 'worldquant',
            'alpha_28': 'worldquant',
            'alpha_29': 'worldquant',
            'alpha_30': 'worldquant',
            'alpha_31': 'worldquant',
            'alpha_32': 'worldquant',
            'alpha_33': 'worldquant',
            'alpha_34': 'worldquant',
            'alpha_35': 'worldquant',
            'alpha_36': 'worldquant',
            'alpha_37': 'worldquant',
            'alpha_38': 'worldquant',
            'alpha_39': 'worldquant',
            'alpha_40': 'worldquant',
            'alpha_41': 'worldquant',
            'alpha_42': 'worldquant',
            'alpha_43': 'worldquant',
            'alpha_44': 'worldquant',
            'alpha_45': 'worldquant',
            'alpha_46': 'worldquant',
            'alpha_47': 'worldquant',
            'alpha_48': 'worldquant',
            'alpha_49': 'worldquant',
            'alpha_50': 'worldquant',
            'alpha_51': 'worldquant',
            'alpha_52': 'worldquant',
            'alpha_53': 'worldquant',
            'alpha_54': 'worldquant',
            'alpha_55': 'worldquant',
            'alpha_56': 'worldquant',
            'alpha_57': 'worldquant',
            'alpha_58': 'worldquant',
            'alpha_59': 'worldquant',
            'alpha_60': 'worldquant',
            'alpha_61': 'worldquant',
            'alpha_62': 'worldquant',
            'alpha_63': 'worldquant',
            'alpha_64': 'worldquant',
            'alpha_65': 'worldquant',
            'alpha_66': 'worldquant',
            'alpha_67': 'worldquant',
            'alpha_68': 'worldquant',
            'alpha_69': 'worldquant',
            'alpha_70': 'worldquant',
            'alpha_71': 'worldquant',
            'alpha_72': 'worldquant',
            'alpha_73': 'worldquant',
            'alpha_74': 'worldquant',
            'alpha_75': 'worldquant',
            'alpha_76': 'worldquant',
            'alpha_77': 'worldquant',
            'alpha_78': 'worldquant',
            'alpha_79': 'worldquant',
            'alpha_80': 'worldquant',
            'alpha_81': 'worldquant',
            'alpha_82': 'worldquant',
            'alpha_83': 'worldquant',
            'alpha_84': 'worldquant',
            'alpha_85': 'worldquant',
            'alpha_86': 'worldquant',
            'alpha_87': 'worldquant',
            'alpha_88': 'worldquant',
            'alpha_89': 'worldquant',
            'alpha_90': 'worldquant',
            'alpha_91': 'worldquant',
            'alpha_92': 'worldquant',
            'alpha_93': 'worldquant',
            'alpha_94': 'worldquant',
            'alpha_95': 'worldquant',
            'alpha_96': 'worldquant',
            'alpha_97': 'worldquant',
            'alpha_98': 'worldquant',
            'alpha_99': 'worldquant',
            'alpha_100': 'worldquant',
            'alpha_101': 'worldquant',
            
            # 基本面因子
            'roe': 'fundamental',
            'eps': 'fundamental',
            'revenue_growth': 'fundamental',
            'profit_growth': 'fundamental',
            'debt_ratio': 'fundamental',
            'current_ratio': 'fundamental',
            'quick_ratio': 'fundamental',
            'cash_ratio': 'fundamental',
            'asset_turnover': 'fundamental',
            'inventory_turnover': 'fundamental',
            'receivables_turnover': 'fundamental',
            'gross_margin': 'fundamental',
            'net_margin': 'fundamental',
            'operating_margin': 'fundamental',
            'ebitda_margin': 'fundamental',
            'free_cash_flow': 'fundamental',
            'operating_cash_flow': 'fundamental',
            'investing_cash_flow': 'fundamental',
            'financing_cash_flow': 'fundamental',
            'dividend_yield': 'fundamental',
            'payout_ratio': 'fundamental',
            'book_value': 'fundamental',
            'tangible_book_value': 'fundamental',
            'return_on_equity': 'fundamental',
            'return_on_assets': 'fundamental',
            'return_on_capital': 'fundamental',
            'return_on_invested_capital': 'fundamental',
            'economic_value_added': 'fundamental',
            'market_value_added': 'fundamental',
            'enterprise_value': 'fundamental',
            'enterprise_value_ebitda': 'fundamental',
            'enterprise_value_ebit': 'fundamental',
            'enterprise_value_revenue': 'fundamental',
            'enterprise_value_assets': 'fundamental',
            'price_to_book': 'fundamental',
            'price_to_sales': 'fundamental',
            'price_to_cash_flow': 'fundamental',
            'price_to_earnings': 'fundamental',
            'price_to_earnings_growth': 'fundamental',
            'ev_to_ebitda': 'fundamental',
            'ev_to_ebit': 'fundamental',
            'ev_to_revenue': 'fundamental',
            'ev_to_assets': 'fundamental',
            'peg_ratio': 'fundamental',
            'forward_pe': 'fundamental',
            'trailing_pe': 'fundamental',
            'forward_ev_ebitda': 'fundamental',
            'trailing_ev_ebitda': 'fundamental',
            'forward_ev_ebit': 'fundamental',
            'trailing_ev_ebit': 'fundamental',
            'forward_ev_revenue': 'fundamental',
            'trailing_ev_revenue': 'fundamental',
            'forward_ev_assets': 'fundamental',
            'trailing_ev_assets': 'fundamental',
            'forward_pe_growth': 'fundamental',
            'trailing_pe_growth': 'fundamental',
            'forward_peg': 'fundamental',
            'trailing_peg': 'fundamental',
            'forward_pb': 'fundamental',
            'trailing_pb': 'fundamental',
            'forward_ps': 'fundamental',
            'trailing_ps': 'fundamental',
            'forward_pcf': 'fundamental',
            'trailing_pcf': 'fundamental',
            'forward_dividend_yield': 'fundamental',
            'trailing_dividend_yield': 'fundamental',
            'forward_payout_ratio': 'fundamental',
            'trailing_payout_ratio': 'fundamental',
            'forward_roe': 'fundamental',
            'trailing_roe': 'fundamental',
            'forward_roa': 'fundamental',
            'trailing_roa': 'fundamental',
            'forward_roc': 'fundamental',
            'trailing_roc': 'fundamental',
            'forward_roic': 'fundamental',
            'trailing_roic': 'fundamental',
            'forward_eva': 'fundamental',
            'trailing_eva': 'fundamental',
            'forward_mva': 'fundamental',
            'trailing_mva': 'fundamental',
            'forward_ev': 'fundamental',
            'trailing_ev': 'fundamental',
            'forward_ev_ebitda_ratio': 'fundamental',
            'trailing_ev_ebitda_ratio': 'fundamental',
            'forward_ev_ebit_ratio': 'fundamental',
            'trailing_ev_ebit_ratio': 'fundamental',
            'forward_ev_revenue_ratio': 'fundamental',
            'trailing_ev_revenue_ratio': 'fundamental',
            'forward_ev_assets_ratio': 'fundamental',
            'trailing_ev_assets_ratio': 'fundamental',
            'forward_peg_ratio': 'fundamental',
            'trailing_peg_ratio': 'fundamental',
            
            # 通道因子
            'rising_channel': 'channel',
            'falling_channel': 'channel',
            'sideways_channel': 'channel',
            'channel_breakout': 'channel',
            'channel_support': 'channel',
            'channel_resistance': 'channel',
            'channel_width': 'channel',
            'channel_slope': 'channel',
            'channel_volume': 'channel',
            'channel_momentum': 'channel',
            
            # AKShare因子
            'akshare_momentum': 'akshare',
            'akshare_volatility': 'akshare',
            'akshare_volume': 'akshare',
            'akshare_price': 'akshare',
            'akshare_technical': 'akshare',
            'akshare_fundamental': 'akshare',
            'akshare_sentiment': 'akshare',
            'akshare_risk': 'akshare',
            'akshare_quality': 'akshare',
            'akshare_value': 'akshare'
        }

    def get_factor_type(self, factor_name: str) -> str:
        """
        获取因子类型
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子类型
        """
        return self.factor_type_mapping.get(factor_name, 'technical')  # 默认为技术因子

    def analyze_factor_function_batch(self, factor_names: List[str]) -> Dict[str, Any]:
        """
        批量分析因子函数的数据需求
        
        Args:
            factor_names: 因子名称列表
            
        Returns:
            数据需求分析结果
        """
        logger.info(f"开始分析 {len(factor_names)} 个因子的数据需求")

        # 统计各类型因子的数量
        factor_types = {}
        for factor_name in factor_names:
            factor_type = self.get_factor_type(factor_name)
            factor_types[factor_type] = factor_types.get(factor_type, 0) + 1
        
        logger.info(f"因子类型分布: {factor_types}")
        
        # 确定需要获取的数据类型
        data_requirements = {
            'market_fields': [],
            'valuation_fields': [],
            'financial_types': [],
            'technical_fields': []
        }
        
        # 如果有任何类型的因子，都需要获取行情数据
        if factor_types:
            data_requirements['market_fields'] = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'vwap']
            data_requirements['valuation_fields'] = ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm']
        
        # 如果有基本面因子，需要获取财务数据
        if 'fundamental' in factor_types:
            data_requirements['financial_types'] = ['profit', 'balance', 'cashflow', 'growth', 'operation', 'dupont', 'dividend']
        
        # 如果有技术因子，可能需要技术指标数据
        if 'technical' in factor_types:
            data_requirements['technical_fields'] = ['rsi', 'macd', 'bollinger', 'ma', 'ema', 'atr', 'adx', 'cci', 'williams_r', 'kama']
        
        logger.info(f"数据需求: {data_requirements}")
        
        return {
            'factor_types': factor_types,
            'data_requirements': data_requirements,
            'optimized_config': data_requirements
        }

    def generate_data_fetch_plan(self, factor_names: List[str]) -> Dict[str, Any]:
        """
        生成数据获取计划
        
        Args:
            factor_names: 因子名称列表
            
        Returns:
            数据获取计划
        """
        analysis_result = self.analyze_factor_function_batch(factor_names)
        optimized_config = analysis_result['optimized_config']
        
        # 计算性能提升
        total_fields = len(optimized_config['market_fields']) + len(optimized_config['valuation_fields']) + len(optimized_config['financial_types']) + len(optimized_config['technical_fields'])
        max_possible_fields = 50  # 假设最大可能字段数
        memory_saving_ratio = 1 - (total_fields / max_possible_fields)
        performance_improvement = memory_saving_ratio * 0.5  # 假设性能提升与内存节省成正比
        
        # 确定获取策略
        fetch_strategy = {
            'market_data': len(optimized_config['market_fields']) > 0,
            'financial_data': len(optimized_config['financial_types']) > 0,
            'technical_data': len(optimized_config['technical_fields']) > 0,
            'financial_types': optimized_config['financial_types']
        }

        return {
            'optimized_config': optimized_config,
            'memory_saving_ratio': memory_saving_ratio,
            'estimated_performance_improvement': performance_improvement,
            'fetch_strategy': fetch_strategy
        }
