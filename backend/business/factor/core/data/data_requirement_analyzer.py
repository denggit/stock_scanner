#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : data_requirement_analyzer.py
@Description: 数据需求分析器 - 分析因子函数需要的数据字段
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
    1. 分析因子函数的源代码，识别需要的数据字段
    2. 提供数据字段映射和依赖关系
    3. 优化数据获取策略
    """

    def __init__(self):
        """初始化数据需求分析器"""
        # 定义数据字段映射
        self.field_mappings = {
            # 基础价格数据
            'price_fields': ['open', 'high', 'low', 'close', 'preclose', 'vwap'],
            'volume_fields': ['volume', 'amount', 'turn'],
            'market_fields': ['tradestatus', 'is_st', 'pct_chg'],

            # 估值指标
            'valuation_fields': ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm'],

            # 财务数据字段
            'financial_fields': {
                'profit': ['roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare',
                           'liqaShare'],
                'balance': ['currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability', 'liabilityToAsset',
                            'assetToEquity'],
                'cashflow': ['CAToAsset', 'NCAToAsset', 'tangibleAssetToAsset', 'ebitToInterest', 'CFOToOR', 'CFOToNP',
                             'CFOToGr'],
                'growth': ['YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI'],
                'operation': ['NRTurnRatio', 'NRTurnDays', 'INVTurnRatio', 'INVTurnDays', 'CATurnRatio',
                              'AssetTurnRatio'],
                'dupont': ['dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr',
                           'dupontTaxBurden', 'dupontIntburden', 'dupontEbittogr'],
                'dividend': ['dividCashPsBeforeTax', 'dividCashPsAfterTax', 'dividStocksPs', 'dividCashStock',
                             'dividReserveToStockPs']
            },

            # 技术指标字段
            'technical_fields': ['rsi', 'macd', 'bollinger', 'ma', 'ema', 'atr', 'adx', 'cci', 'williams_r', 'kama'],

            # 必需字段（总是需要的）
            'required_fields': ['code', 'trade_date']
        }

        # 字段别名映射
        self.field_aliases = {
            'price': ['open', 'high', 'low', 'close', 'preclose', 'vwap'],
            'ohlc': ['open', 'high', 'low', 'close'],
            'ohlcv': ['open', 'high', 'low', 'close', 'volume'],
            'returns': ['pct_chg'],
            'vol': ['volume'],
            'amt': ['amount'],
            'pe': ['pe_ttm'],
            'pb': ['pb_mrq'],
            'ps': ['ps_ttm'],
            'pcf': ['pcf_ncf_ttm'],
            'roe': ['roeAvg'],
            'eps': ['epsTTM'],
            'revenue': ['MBRevenue'],
            'profit': ['netProfit'],
            'margin': ['npMargin', 'gpMargin'],
            'ratio': ['currentRatio', 'quickRatio', 'cashRatio', 'liabilityToAsset', 'assetToEquity'],
            'turnover': ['NRTurnRatio', 'INVTurnRatio', 'CATurnRatio', 'AssetTurnRatio'],
            'growth': ['YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI'],
            'cashflow': ['CFOToOR', 'CFOToNP', 'CFOToGr'],
            'dupont': ['dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr',
                       'dupontTaxBurden', 'dupontIntburden', 'dupontEbittogr'],
            'dividend': ['dividCashPsBeforeTax', 'dividCashPsAfterTax', 'dividStocksPs', 'dividCashStock',
                         'dividReserveToStockPs']
        }

    def analyze_factor_function(self, factor_func: Callable) -> Dict[str, Any]:
        """
        分析因子函数的数据需求
        
        Args:
            factor_func: 因子函数
            
        Returns:
            数据需求分析结果
        """
        try:
            # 获取函数源代码
            source_code = inspect.getsource(factor_func)

            # 分析数据需求
            required_fields = self._extract_required_fields(source_code)
            data_types = self._determine_data_types(required_fields)
            optimization_suggestions = self._generate_optimization_suggestions(required_fields, data_types)

            return {
                'required_fields': required_fields,
                'data_types': data_types,
                'optimization_suggestions': optimization_suggestions,
                'estimated_memory_saving': self._estimate_memory_saving(required_fields),
                'source_code_length': len(source_code)
            }

        except Exception as e:
            logger.error(f"分析因子函数失败: {e}")
            return {
                'required_fields': self.field_mappings['required_fields'],
                'data_types': ['market'],
                'optimization_suggestions': ['无法分析，使用默认字段'],
                'estimated_memory_saving': 0,
                'source_code_length': 0
            }

    def _extract_required_fields(self, source_code: str) -> Set[str]:
        """
        从源代码中提取需要的数据字段
        
        Args:
            source_code: 函数源代码
            
        Returns:
            需要的数据字段集合
        """
        required_fields = set(self.field_mappings['required_fields'])

        # 使用AST进行更精确的分析（优先）
        try:
            tree = ast.parse(source_code)
            ast_fields = self._extract_fields_from_ast(tree)
            required_fields.update(ast_fields)
        except:
            pass

        # 使用字符串分析作为补充（确保不遗漏任何字段）
        string_fields = self._extract_fields_from_string(source_code)
        required_fields.update(string_fields)

        # 转换为小写进行分析（作为补充）
        code_lower = source_code.lower()

        # 分析各种字段类型（使用更精确的匹配）
        for field_type, fields in self.field_mappings.items():
            if field_type == 'required_fields':
                continue
            elif field_type == 'financial_fields':
                # 处理财务数据字段
                for data_type, type_fields in fields.items():
                    for field in type_fields:
                        # 使用更精确的匹配：确保字段名是独立的词
                        if self._is_field_mentioned(field, code_lower):
                            required_fields.add(field)
                            # 添加对应的财务数据类型
                            required_fields.add(f'financial_{data_type}')
            else:
                # 处理其他字段类型
                for field in fields:
                    # 使用更精确的匹配：确保字段名是独立的词
                    if self._is_field_mentioned(field, code_lower):
                        required_fields.add(field)

        # 处理字段别名
        for alias, fields in self.field_aliases.items():
            # 使用更精确的匹配：确保别名是独立的词
            if self._is_field_mentioned(alias, code_lower):
                required_fields.update(fields)

        return required_fields

    def _extract_fields_from_ast(self, tree: ast.AST) -> Set[str]:
        """
        从AST中提取字段引用
        
        Args:
            tree: AST树
            
        Returns:
            字段引用集合
        """
        fields = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                # 处理字典访问，如 data['close']
                if isinstance(node.slice, ast.Constant):
                    field_name = node.slice.value
                    if isinstance(field_name, str):
                        fields.add(field_name)
                elif isinstance(node.slice, ast.Str):  # Python 3.7及以下版本
                    field_name = node.slice.s
                    if isinstance(field_name, str):
                        fields.add(field_name)
            elif isinstance(node, ast.Name):
                # 处理变量名，可能是字段别名
                name = node.id.lower()
                if name in self.field_aliases:
                    fields.update(self.field_aliases[name])

        return fields

    def _extract_fields_from_string(self, source_code: str) -> Set[str]:
        """
        从源代码字符串中提取字段引用（作为AST分析的补充）
        
        Args:
            source_code: 源代码字符串
            
        Returns:
            字段引用集合
        """
        fields = set()
        code_lower = source_code.lower()
        
        # 检查所有可能的字段
        all_possible_fields = set()
        for field_type, field_list in self.field_mappings.items():
            if field_type != 'required_fields':
                if field_type == 'financial_fields':
                    for type_fields in field_list.values():
                        all_possible_fields.update(type_fields)
                else:
                    all_possible_fields.update(field_list)
        
        # 检查每个字段是否在代码中被提及
        for field in all_possible_fields:
            if self._is_field_mentioned(field, code_lower):
                fields.add(field)
        
        return fields

    def _is_field_mentioned(self, field: str, code_lower: str) -> bool:
        """
        检查字段是否在代码中被提及（使用精确匹配）
        
        Args:
            field: 字段名
            code_lower: 小写的代码字符串
            
        Returns:
            是否提及
        """
        field_lower = field.lower()
        
        # 检查精确匹配：字段名前后应该是非字母数字字符
        import re
        
        # 构建正则表达式模式，确保字段名是独立的词
        pattern = r'\b' + re.escape(field_lower) + r'\b'
        
        return bool(re.search(pattern, code_lower))

    def _determine_data_types(self, required_fields: Set[str]) -> List[str]:
        """
        确定需要的数据类型
        
        Args:
            required_fields: 需要的字段集合
            
        Returns:
            数据类型列表
        """
        data_types = ['market']  # 市场数据总是需要的

        # 检查是否需要财务数据
        financial_types = set()
        for field in required_fields:
            if field.startswith('financial_'):
                financial_types.add(field.replace('financial_', ''))
            elif any(field in fields for fields in self.field_mappings['financial_fields'].values()):
                # 根据字段确定财务数据类型
                for data_type, fields in self.field_mappings['financial_fields'].items():
                    if field in fields:
                        financial_types.add(data_type)

        if financial_types:
            data_types.append('financial')

        # 检查是否需要技术指标
        if any(field in required_fields for field in self.field_mappings['technical_fields']):
            data_types.append('technical')

        return data_types

    def _generate_optimization_suggestions(self, required_fields: Set[str], data_types: List[str]) -> List[str]:
        """
        生成优化建议
        
        Args:
            required_fields: 需要的字段集合
            data_types: 数据类型列表
            
        Returns:
            优化建议列表
        """
        suggestions = []

        # 检查字段使用情况
        all_possible_fields = set()
        for field_type, fields in self.field_mappings.items():
            if field_type != 'required_fields':
                if field_type == 'financial_fields':
                    for type_fields in fields.values():
                        all_possible_fields.update(type_fields)
                else:
                    all_possible_fields.update(fields)

        unused_fields = all_possible_fields - required_fields
        if len(unused_fields) > 10:
            suggestions.append(f"可以排除 {len(unused_fields)} 个未使用的字段以节省内存")

        # 检查数据类型优化
        if 'financial' in data_types:
            suggestions.append("需要财务数据，建议使用增量更新策略")

        if len(required_fields) < 10:
            suggestions.append("字段需求较少，可以使用精简数据模式")

        return suggestions

    def _estimate_memory_saving(self, required_fields: Set[str]) -> float:
        """
        估算内存节省比例
        
        Args:
            required_fields: 需要的字段集合
            
        Returns:
            内存节省比例（0-1）
        """
        # 计算所有可能的字段
        all_fields = set()
        for field_type, fields in self.field_mappings.items():
            if field_type != 'required_fields':
                if field_type == 'financial_fields':
                    for type_fields in fields.values():
                        all_fields.update(type_fields)
                else:
                    all_fields.update(fields)

        # 添加必需字段
        all_fields.update(self.field_mappings['required_fields'])

        if len(all_fields) == 0:
            return 0.0

        # 计算节省比例
        saved_fields = len(all_fields) - len(required_fields)
        saving_ratio = saved_fields / len(all_fields)

        return max(0.0, min(1.0, saving_ratio))

    def get_optimized_fields(self, factor_names: List[str], factor_registry) -> Dict[str, List[str]]:
        """
        获取优化后的字段列表
        
        Args:
            factor_names: 因子名称列表
            factor_registry: 因子注册器
            
        Returns:
            优化后的字段配置
        """
        all_required_fields = set(self.field_mappings['required_fields'])
        data_types = set(['market'])

        for factor_name in factor_names:
            factor_func = factor_registry.get_factor(factor_name)
            if factor_func:
                analysis = self.analyze_factor_function(factor_func)
                all_required_fields.update(analysis['required_fields'])
                data_types.update(analysis['data_types'])

        # 构建优化配置
        optimized_config = {
            'market_fields': list(all_required_fields & set(
                self.field_mappings['price_fields'] + self.field_mappings['volume_fields'] + self.field_mappings[
                    'market_fields'])),
            'valuation_fields': list(all_required_fields & set(self.field_mappings['valuation_fields'])),
            'financial_types': [],
            'technical_fields': list(all_required_fields & set(self.field_mappings['technical_fields']))
        }

        # 确定财务数据类型
        if 'financial' in data_types:
            for field in all_required_fields:
                for data_type, fields in self.field_mappings['financial_fields'].items():
                    if field in fields and data_type not in optimized_config['financial_types']:
                        optimized_config['financial_types'].append(data_type)

        return optimized_config

    def analyze_factor_function_batch(self, factor_names: List[str], factor_registry) -> Dict[str, Any]:
        """
        批量分析因子函数的数据需求
        
        Args:
            factor_names: 因子名称列表
            factor_registry: 因子注册器
            
        Returns:
            批量分析结果
        """
        all_required_fields = set(self.field_mappings['required_fields'])
        all_data_types = set(['market'])

        for factor_name in factor_names:
            factor_func = factor_registry.get_factor(factor_name)
            if factor_func:
                analysis = self.analyze_factor_function(factor_func)
                all_required_fields.update(analysis['required_fields'])
                all_data_types.update(analysis['data_types'])

        return {
            'required_fields': all_required_fields,
            'data_types': list(all_data_types),
            'total_factors': len(factor_names),
            'analyzed_factors': len([f for f in factor_names if factor_registry.get_factor(f)])
        }

    def generate_data_fetch_plan(self, factor_names: List[str], factor_registry) -> Dict[str, Any]:
        """
        生成数据获取计划
        
        Args:
            factor_names: 因子名称列表
            factor_registry: 因子注册器
            
        Returns:
            数据获取计划
        """
        optimized_config = self.get_optimized_fields(factor_names, factor_registry)

        # 计算内存节省
        total_possible_fields = len(self.field_mappings['price_fields'] +
                                    self.field_mappings['volume_fields'] +
                                    self.field_mappings['market_fields'] +
                                    self.field_mappings['valuation_fields'])

        total_required_fields = len(optimized_config['market_fields']) + len(optimized_config['valuation_fields'])
        memory_saving = (
                                    total_possible_fields - total_required_fields) / total_possible_fields if total_possible_fields > 0 else 0

        return {
            'optimized_config': optimized_config,
            'memory_saving_ratio': memory_saving,
            'estimated_performance_improvement': memory_saving * 0.3,  # 估算性能提升
            'fetch_strategy': {
                'market_data': True,
                'financial_data': len(optimized_config['financial_types']) > 0,
                'technical_data': len(optimized_config['technical_fields']) > 0,
                'financial_types': optimized_config['financial_types']
            }
        }


# 全局分析器实例
data_requirement_analyzer = DataRequirementAnalyzer()
