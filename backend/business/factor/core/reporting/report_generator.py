#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : report_generator.py
@Description: 因子报告生成器 - 基于Jinja2模板的交互式HTML报告系统
@Author     : Zijun Deng
@Date       : 2025-08-23
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from jinja2 import Environment, FileSystemLoader, Template

# 尝试导入logger，如果失败则使用简单的print
try:
    from backend.utils.logger import setup_logger
    logger = setup_logger("factor_report_generator")
except ImportError:
    # 简单的logger替代
    class SimpleLogger:
        def __init__(self, name):
            self.name = name
        
        def info(self, msg):
            print(f"[INFO] {self.name}: {msg}")
        
        def error(self, msg):
            print(f"[ERROR] {self.name}: {msg}")
        
        def warning(self, msg):
            print(f"[WARNING] {self.name}: {msg}")
    
    logger = SimpleLogger("factor_report_generator")


class FactorReportGenerator:
    """
    因子报告生成器
    
    基于Jinja2模板系统，生成专业的交互式HTML因子分析报告
    支持批量报告和合并报告两种模式
    """
    
    def __init__(self, template_path: str = None):
        """
        初始化报告生成器
        
        Args:
            template_path: 模板文件路径，默认为当前目录下的templates文件夹
        """
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), 'templates')
        
        self.template_path = template_path
        self.env = Environment(
            loader=FileSystemLoader(template_path),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # 添加自定义过滤器
        self.env.filters['format_percentage'] = self._format_percentage
        self.env.filters['format_number'] = self._format_number
        self.env.filters['safe_json'] = self._safe_json
        
        logger.info(f"报告生成器初始化完成，模板路径: {template_path}")
    
    def _get_metric(self, stats: dict, keys: list, default: float = 0.0) -> float:
        """
        健壮地从stats字典中获取指标值。
        它会尝试多个可能的键，并正确处理不同数据类型（包括numpy和字符串）。
        """
        if not isinstance(stats, dict):
            logger.warning(f"输入的stats不是一个字典，而是一个 {type(stats)}")
            return default

        value = None
        found_key = None
        for key in keys:
            if key in stats:
                value = stats[key]
                found_key = key
                break

        if value is None:
            logger.warning(f"在keys {keys} 中未找到任何有效键")
            return default

        # 如果值是字符串 (例如 '52.5%' 或 '-10.2%'), 尝试转换
        if isinstance(value, str):
            try:
                # 去除 '%' 并转换为浮点数，然后除以100
                value = float(value.strip().replace('%', '')) / 100.0
            except (ValueError, TypeError):
                logger.warning(f"无法将字符串 '{value}' 从键 '{found_key}' 转换为浮点数")
                return default
        
        # 如果键名包含 [%] 但值不是字符串，说明已经是百分比数值，需要除以100转换为小数
        elif found_key and '[%]' in found_key and isinstance(value, (int, float, np.number)):
            value = float(value) / 100.0

        if not isinstance(value, (int, float, np.number)):
            logger.warning(f"键 '{found_key}' 的值 '{value}' (类型: {type(value)}) 不是有效的数值")
            return default

        return float(value)
    
    def _is_numeric(self, value: Any) -> bool:
        """
        检查值是否为数值类型
        
        支持Python内置数值类型和NumPy数值类型
        
        Args:
            value: 要检查的值
            
        Returns:
            是否为数值类型
        """
        # 检查Python内置数值类型
        if isinstance(value, (int, float)):
            return True
        
        # 检查NumPy数值类型
        if isinstance(value, np.number):
            return True
        
        # 检查pandas数值类型
        if pd.api.types.is_numeric_dtype(type(value)):
            return True
        
        # 检查是否可以转换为float
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _safe_convert_to_float(self, value: Any) -> float:
        """
        安全地将值转换为float
        
        Args:
            value: 要转换的值
            
        Returns:
            转换后的float值，如果转换失败则返回0.0
        """
        if self._is_numeric(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"无法转换 {value} 为float")
                return 0.0
        return 0.0

    def generate_batch_report(self, 
                             batch_name: str, 
                             report_data: Dict[str, Any], 
                             output_path: str,
                             **kwargs) -> str:
        """
        生成单个批次的HTML报告
        
        Args:
            batch_name: 批次名称
            report_data: 报告数据字典
            output_path: 输出文件路径
            **kwargs: 其他参数
            
        Returns:
            生成的报告文件路径
        """
        logger.info(f"开始生成批次报告: {batch_name}")
        
        try:
            # 准备模板上下文
            context = self._prepare_batch_context(batch_name, report_data, **kwargs)
            
            # 渲染模板
            template = self.env.get_template('base_template.html')
            html_content = template.render(**context)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"批次报告生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成批次报告失败: {e}")
            raise
    
    def generate_merged_report(self, 
                              all_batches_data: List[Dict[str, Any]], 
                              output_path: str,
                              **kwargs) -> str:
        """
        生成合并的综合报告
        
        Args:
            all_batches_data: 所有批次的数据列表
            output_path: 输出文件路径
            **kwargs: 其他参数
            
        Returns:
            生成的报告文件路径
        """
        logger.info("开始生成合并报告")
        
        try:
            # 合并所有批次数据
            merged_data = self._merge_batch_data(all_batches_data)
            
            # 执行高级分析
            analysis_results = self._perform_advanced_analysis(merged_data)
            
            # 准备模板上下文
            context = self._prepare_merged_context(merged_data, analysis_results, **kwargs)
            
            # 渲染模板
            template = self.env.get_template('base_template.html')
            html_content = template.render(**context)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"合并报告生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成合并报告失败: {e}")
            raise
    
    def _prepare_batch_context(self, 
                              batch_name: str, 
                              report_data: Dict[str, Any], 
                              **kwargs) -> Dict[str, Any]:
        """
        准备批次报告的模板上下文
        
        Args:
            batch_name: 批次名称
            report_data: 报告数据
            **kwargs: 其他参数
            
        Returns:
            模板上下文字典
        """
        factor_names = report_data.get('factor_names', [])
        
        # 提取性能指标
        performance_data = self._extract_performance_data(report_data)
        
        # 提取IC指标
        ic_data = self._extract_ic_data(report_data)
        
        # 提取时间序列数据
        chart_data = self._extract_chart_data(report_data)
        
        # 提取详细分析数据
        detailed_data = self._extract_detailed_data(report_data)
        
        # 计算统计指标
        stats = self._calculate_batch_statistics(performance_data, ic_data)
        
        context = {
            # 报告基本信息
            'report_title': f'因子分析报告 - {batch_name}',
            'report_subtitle': f'批次分析报告，包含 {len(factor_names)} 个因子',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'factor_count': len(factor_names),
            'is_merged_report': False,
            
            # 因子列表
            'factor_names': factor_names,
            
            # 数据
            'performance_data': performance_data,
            'ic_data': ic_data,
            'chart_data': chart_data,
            'detailed_data': detailed_data,
            
            # 统计信息
            'total_factors': stats['total_factors'],
            'positive_factors': stats['positive_factors'],
            'avg_sharpe': stats['avg_sharpe'],
            'avg_ic': stats['avg_ic'],
            'best_sharpe': stats['best_sharpe'],
            'worst_sharpe': stats['worst_sharpe'],
            
            # 分析配置
            'start_date': kwargs.get('start_date', 'N/A'),
            'end_date': kwargs.get('end_date', 'N/A'),
            'stock_pool': kwargs.get('stock_pool', 'no_st'),
            'top_n': kwargs.get('top_n', 10),
            'n_groups': kwargs.get('n_groups', 5),
            
            # 推荐因子（批次报告为空）
            'top_factors': [],
            'bottom_factors': [],
        }
        
        return context
    
    def _prepare_merged_context(self, 
                               merged_data: Dict[str, Any], 
                               analysis_results: Dict[str, Any],
                               **kwargs) -> Dict[str, Any]:
        """
        准备合并报告的模板上下文
        
        Args:
            merged_data: 合并后的数据
            analysis_results: 分析结果
            **kwargs: 其他参数
            
        Returns:
            模板上下文字典
        """
        factor_names = merged_data.get('factor_names', [])
        
        # 提取各种数据
        performance_data = merged_data.get('performance_data', {})
        ic_data = merged_data.get('ic_data', {})
        chart_data = merged_data.get('chart_data', {})
        detailed_data = merged_data.get('detailed_data', {})
        
        # 计算统计指标
        stats = self._calculate_merged_statistics(performance_data, ic_data)
        
        context = {
            # 报告基本信息
            'report_title': '因子分析综合报告',
            'report_subtitle': f'综合分析报告，包含 {len(factor_names)} 个因子',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'factor_count': len(factor_names),
            'is_merged_report': True,
            
            # 因子列表
            'factor_names': factor_names,
            
            # 数据
            'performance_data': performance_data,
            'ic_data': ic_data,
            'chart_data': chart_data,
            'detailed_data': detailed_data,
            
            # 统计信息
            'total_factors': stats['total_factors'],
            'positive_factors': stats['positive_factors'],
            'avg_sharpe': stats['avg_sharpe'],
            'avg_ic': stats['avg_ic'],
            'best_sharpe': stats['best_sharpe'],
            'worst_sharpe': stats['worst_sharpe'],
            
            # 分析配置
            'start_date': kwargs.get('start_date', 'N/A'),
            'end_date': kwargs.get('end_date', 'N/A'),
            'stock_pool': kwargs.get('stock_pool', 'no_st'),
            'top_n': kwargs.get('top_n', 10),
            'n_groups': kwargs.get('n_groups', 5),
            
            # 推荐因子
            'top_factors': analysis_results.get('top_factors', []),
            'bottom_factors': analysis_results.get('bottom_factors', []),
        }
        
        return context
    
    def _merge_batch_data(self, all_batches_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并所有批次的数据
        
        Args:
            all_batches_data: 所有批次的数据列表
            
        Returns:
            合并后的数据字典
        """
        merged_data = {
            'factor_names': [],
            'performance_data': {},
            'ic_data': {},
            'chart_data': {},
            'detailed_data': {}
        }
        
        for batch_data in all_batches_data:
            # 合并因子名称
            factor_names = batch_data.get('factor_names', [])
            merged_data['factor_names'].extend(factor_names)
            
            # 合并性能数据
            performance_data = batch_data.get('performance_metrics', {})
            merged_data['performance_data'].update(performance_data)
            
            # 合并IC数据
            ic_data = batch_data.get('ic_metrics', {})
            merged_data['ic_data'].update(ic_data)
            
            # 合并图表数据
            time_series_returns = batch_data.get('time_series_returns', {})
            merged_data['chart_data'].update(time_series_returns)
            
            # 合并详细数据
            detailed_analysis = batch_data.get('detailed_analysis', {})
            merged_data['detailed_data'].update(detailed_analysis)
        
        # 去重因子名称
        merged_data['factor_names'] = list(set(merged_data['factor_names']))
        
        # 处理嵌套结构 - 提取性能数据和IC数据
        merged_data['performance_data'] = self._extract_performance_data({
            'performance_metrics': merged_data['performance_data']
        })
        merged_data['ic_data'] = self._extract_ic_data({
            'ic_metrics': merged_data['ic_data']
        })
        
        return merged_data
    
    def _perform_advanced_analysis(self, merged_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行高级分析，生成推荐因子和问题因子
        
        Args:
            merged_data: 合并后的数据
            
        Returns:
            分析结果字典
        """
        performance_data = merged_data.get('performance_data', {})
        ic_data = merged_data.get('ic_data', {})
        
        # 创建因子评分表
        factor_scores = []
        
        for factor_name in merged_data.get('factor_names', []):
            performance = performance_data.get(factor_name, {})
            ic = ic_data.get(factor_name, {})
            
            # 直接使用提取后的数据（已经处理了嵌套结构）
            sharpe_ratio = self._get_metric(
                performance, 
                ['sharpe_ratio', 'smart_sharpe', 'sharpe', 'Sharpe', 'sharpe_ratio_annual', 'Sharpe Ratio'], 
                default=0.0
            )
            
            annual_return = self._get_metric(
                performance, 
                ['annual_return', 'annual_ret', 'return_annual', 'Annual Return [%]', 'Annual Return', 'annual_ret_pct'], 
                default=0.0
            )
            
            # 提取IC IR - 直接使用提取后的数据（已经处理了嵌套结构）
            ic_ir = self._get_metric(
                ic, 
                ['ic_ir', 'ic_information_ratio', 'information_ratio', 'ir'], 
                default=0.0
            )
            
            # 综合评分 = 夏普比率 * 0.6 + IC IR * 0.4
            composite_score = sharpe_ratio * 0.6 + ic_ir * 0.4
            
            factor_scores.append({
                'name': factor_name,
                'sharpe_ratio': sharpe_ratio,
                'ic_ir': ic_ir,
                'annual_return': annual_return,
                'composite_score': composite_score
            })
        
        # 按综合评分排序
        factor_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # 提取前5名和后5名
        top_factors = []
        bottom_factors = []
        
        for i, factor in enumerate(factor_scores[:5]):
            top_factors.append({
                'rank': i + 1,
                'name': factor['name'],
                'sharpe_ratio': factor['sharpe_ratio'],
                'annual_return': factor['annual_return'],
                'composite_score': factor['composite_score']
            })
        
        for i, factor in enumerate(factor_scores[-5:]):
            bottom_factors.append({
                'rank': len(factor_scores) - 4 + i,
                'name': factor['name'],
                'sharpe_ratio': factor['sharpe_ratio'],
                'annual_return': factor['annual_return'],
                'composite_score': factor['composite_score']
            })
        
        return {
            'top_factors': top_factors,
            'bottom_factors': bottom_factors,
            'factor_scores': factor_scores
        }
    
    def _extract_performance_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从报告数据中提取性能指标
        
        Args:
            report_data: 报告数据
            
        Returns:
            性能指标字典
        """
        performance_metrics = report_data.get('performance_metrics', {})
        
        try:
            # 如果performance_metrics是DataFrame，转换为字典
            if isinstance(performance_metrics, pd.DataFrame):
                logger.info(f"性能指标数据是DataFrame，形状: {performance_metrics.shape}")
                
                # 确保索引是字符串类型
                if not performance_metrics.index.dtype == 'object':
                    performance_metrics.index = performance_metrics.index.astype(str)
                
                # 转换为字典，处理NaN值
                performance_dict = {}
                for factor_name, row in performance_metrics.iterrows():
                    # 将NaN值替换为0
                    clean_row = row.fillna(0.0)
                    performance_dict[str(factor_name)] = clean_row.to_dict()
                
                logger.info(f"成功转换DataFrame为字典，包含 {len(performance_dict)} 个因子")
                return performance_dict
            
            elif isinstance(performance_metrics, dict):
                logger.info(f"性能指标数据是字典，包含 {len(performance_metrics)} 个因子")
                
                # 检查是否为嵌套结构，如果是，提取stats或portfolio_stats
                processed_performance_dict = {}
                for factor_name, performance_data in performance_metrics.items():
                    if isinstance(performance_data, dict):
                        # 检查是否有嵌套的stats结构
                        if 'stats' in performance_data and isinstance(performance_data['stats'], dict):
                            # 提取嵌套的stats
                            processed_performance_dict[factor_name] = performance_data['stats']
                            logger.debug(f"提取因子 {factor_name} 的嵌套stats")
                        elif 'portfolio_stats' in performance_data and isinstance(performance_data['portfolio_stats'], dict):
                            # 提取嵌套的portfolio_stats
                            processed_performance_dict[factor_name] = performance_data['portfolio_stats']
                            logger.debug(f"提取因子 {factor_name} 的嵌套portfolio_stats")
                        else:
                            # 使用原始数据
                            processed_performance_dict[factor_name] = performance_data
                    else:
                        # 非字典类型，保持原样
                        processed_performance_dict[factor_name] = performance_data
                
                logger.info(f"处理后的性能指标数据包含 {len(processed_performance_dict)} 个因子")
                return processed_performance_dict
            else:
                logger.warning(f"性能指标数据类型不支持: {type(performance_metrics)}")
                return {}
                
        except Exception as e:
            logger.error(f"提取性能指标数据时出错: {e}")
            return {}
    
    def _extract_ic_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从报告数据中提取IC指标
        
        Args:
            report_data: 报告数据
            
        Returns:
            IC指标字典
        """
        ic_metrics = report_data.get('ic_metrics', {})
        
        try:
            # 如果ic_metrics是DataFrame，转换为字典
            if isinstance(ic_metrics, pd.DataFrame):
                logger.info(f"IC指标数据是DataFrame，形状: {ic_metrics.shape}")
                
                # 确保索引是字符串类型
                if not ic_metrics.index.dtype == 'object':
                    ic_metrics.index = ic_metrics.index.astype(str)
                
                # 转换为字典，处理NaN值
                ic_dict = {}
                for factor_name, row in ic_metrics.iterrows():
                    # 将NaN值替换为0
                    clean_row = row.fillna(0.0)
                    ic_dict[str(factor_name)] = clean_row.to_dict()
                
                logger.info(f"成功转换IC DataFrame为字典，包含 {len(ic_dict)} 个因子")
                return ic_dict
            
            elif isinstance(ic_metrics, dict):
                logger.info(f"IC指标数据是字典，包含 {len(ic_metrics)} 个因子")
                
                # 检查是否为嵌套结构，如果是，提取ic_stats
                processed_ic_dict = {}
                for factor_name, ic_data in ic_metrics.items():
                    if isinstance(ic_data, dict):
                        # 检查是否有嵌套的ic_analysis.ic_stats结构
                        if 'ic_analysis' in ic_data and isinstance(ic_data['ic_analysis'], dict):
                            ic_analysis = ic_data['ic_analysis']
                            if 'ic_stats' in ic_analysis and isinstance(ic_analysis['ic_stats'], dict):
                                # 提取嵌套的ic_stats
                                processed_ic_dict[factor_name] = ic_analysis['ic_stats']
                                logger.debug(f"提取因子 {factor_name} 的嵌套IC统计")
                            else:
                                # 使用原始数据
                                processed_ic_dict[factor_name] = ic_data
                        else:
                            # 使用原始数据
                            processed_ic_dict[factor_name] = ic_data
                    else:
                        # 非字典类型，保持原样
                        processed_ic_dict[factor_name] = ic_data
                
                logger.info(f"处理后的IC指标数据包含 {len(processed_ic_dict)} 个因子")
                return processed_ic_dict
            else:
                logger.warning(f"IC指标数据类型不支持: {type(ic_metrics)}")
                return {}
                
        except Exception as e:
            logger.error(f"提取IC指标数据时出错: {e}")
            return {}
    
    def _extract_chart_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从报告数据中提取图表数据
        
        Args:
            report_data: 报告数据
            
        Returns:
            图表数据字典
        """
        time_series_returns = report_data.get('time_series_returns', {})
        chart_data = {}
        
        for factor_name, returns_series in time_series_returns.items():
            try:
                if isinstance(returns_series, pd.Series):
                    # 确保索引是datetime类型
                    if not isinstance(returns_series.index, pd.DatetimeIndex):
                        logger.warning(f"因子 {factor_name} 的时间序列索引不是DatetimeIndex")
                        continue
                    
                    # 确保数据是数值类型
                    if not pd.api.types.is_numeric_dtype(returns_series):
                        logger.warning(f"因子 {factor_name} 的时间序列数据不是数值类型")
                        continue
                    
                    # 移除NaN值
                    clean_series = returns_series.dropna()
                    
                    if len(clean_series) == 0:
                        logger.warning(f"因子 {factor_name} 的时间序列数据为空")
                        continue
                    
                    # 计算累计收益率
                    cumulative_returns = (1 + clean_series).cumprod()
                    
                    chart_data[factor_name] = {
                        'dates': clean_series.index.strftime('%Y-%m-%d').tolist(),
                        'values': cumulative_returns.tolist()
                    }
                    
                    logger.info(f"成功提取因子 {factor_name} 的图表数据，数据点数量: {len(clean_series)}")
                else:
                    logger.warning(f"因子 {factor_name} 的时间序列数据不是pandas.Series类型: {type(returns_series)}")
            except Exception as e:
                logger.error(f"处理因子 {factor_name} 的图表数据时出错: {e}")
                continue
        
        logger.info(f"成功提取 {len(chart_data)} 个因子的图表数据")
        return chart_data
    
    def _extract_detailed_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从报告数据中提取详细分析数据
        
        Args:
            report_data: 报告数据
            
        Returns:
            详细数据字典
        """
        detailed_analysis = report_data.get('detailed_analysis', {})
        processed_data = {}
        
        for factor_name, factor_data in detailed_analysis.items():
            try:
                if not isinstance(factor_data, dict):
                    logger.warning(f"因子 {factor_name} 的详细数据不是字典类型: {type(factor_data)}")
                    continue
                
                # 处理指标数据
                metrics = factor_data.get('metrics', {})
                if not isinstance(metrics, dict):
                    logger.warning(f"因子 {factor_name} 的指标数据不是字典类型")
                    metrics = {}
                
                # 处理分组结果
                group_results = factor_data.get('group_results', {})
                if not isinstance(group_results, dict):
                    logger.warning(f"因子 {factor_name} 的分组结果不是字典类型")
                    group_results = {}
                
                # 处理IC统计
                ic_stats = factor_data.get('ic_stats', {})
                if not isinstance(ic_stats, dict):
                    logger.warning(f"因子 {factor_name} 的IC统计不是字典类型")
                    ic_stats = {}
                
                # 处理时间序列数据
                returns_series = factor_data.get('returns_series')
                if returns_series is not None and not isinstance(returns_series, pd.Series):
                    logger.warning(f"因子 {factor_name} 的收益率序列不是pandas.Series类型")
                    returns_series = None
                
                drawdown_series = factor_data.get('drawdown_series')
                if drawdown_series is not None and not isinstance(drawdown_series, pd.Series):
                    logger.warning(f"因子 {factor_name} 的回撤序列不是pandas.Series类型")
                    drawdown_series = None
                
                ic_series = factor_data.get('ic_series')
                if ic_series is not None and not isinstance(ic_series, pd.Series):
                    logger.warning(f"因子 {factor_name} 的IC序列不是pandas.Series类型")
                    ic_series = None
                
                # 处理月度收益数据
                monthly_returns = factor_data.get('monthly_returns', [])
                if not isinstance(monthly_returns, list):
                    logger.warning(f"因子 {factor_name} 的月度收益数据不是列表类型")
                    monthly_returns = []
                
                # 处理风险指标
                risk_metrics = factor_data.get('risk_metrics', {})
                if not isinstance(risk_metrics, dict):
                    logger.warning(f"因子 {factor_name} 的风险指标不是字典类型")
                    risk_metrics = {}
                
                processed_data[factor_name] = {
                    'metrics': metrics,
                    'group_results': group_results,
                    'ic_stats': ic_stats,
                    'returns_series': returns_series,
                    'drawdown_series': drawdown_series,
                    'ic_series': ic_series,
                    'monthly_returns': monthly_returns,
                    'risk_metrics': risk_metrics
                }
                
                logger.info(f"成功处理因子 {factor_name} 的详细数据")
                
            except Exception as e:
                logger.error(f"处理因子 {factor_name} 的详细数据时出错: {e}")
                continue
        
        logger.info(f"成功处理 {len(processed_data)} 个因子的详细数据")
        return processed_data
    
    def _calculate_batch_statistics(self, 
                                   performance_data: Dict[str, Any], 
                                   ic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算批次统计指标
        
        Args:
            performance_data: 性能数据
            ic_data: IC数据
            
        Returns:
            统计指标字典
        """
        if not performance_data:
            return {
                'total_factors': 0,
                'positive_factors': 0,
                'avg_sharpe': 0,
                'avg_ic': 0,
                'best_sharpe': 0,
                'worst_sharpe': 0
            }
        
        # 使用健壮的指标提取方法
        sharpe_ratios = []
        annual_returns = []
        ic_means = []
        
        for factor_name, performance in performance_data.items():
            # 直接使用提取后的数据（已经处理了嵌套结构）
            sharpe_ratio = self._get_metric(
                performance, 
                ['sharpe_ratio', 'sharpe', 'Sharpe', 'sharpe_ratio_annual', 'Sharpe Ratio'], 
                default=0.0
            )
            
            annual_return = self._get_metric(
                performance, 
                ['annual_return', 'annual_ret', 'return_annual', 'Annual Return [%]', 'Annual Return'], 
                default=0.0
            )
            
            sharpe_ratios.append(sharpe_ratio)
            annual_returns.append(annual_return)
        
        # 处理IC数据 - 直接使用提取后的数据（已经处理了嵌套结构）
        for factor_name, ic_stats in ic_data.items():
            ic_mean = self._get_metric(
                ic_stats, 
                ['mean_ic', 'ic_mean', 'ic'], 
                default=0.0
            )
            ic_means.append(ic_mean)
        
        return {
            'total_factors': len(performance_data),
            'positive_factors': sum(1 for r in annual_returns if r > 0),
            'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_ic': np.mean(ic_means) if ic_means else 0,
            'best_sharpe': max(sharpe_ratios) if sharpe_ratios else 0,
            'worst_sharpe': min(sharpe_ratios) if sharpe_ratios else 0
        }
    
    def _calculate_merged_statistics(self, 
                                    performance_data: Dict[str, Any], 
                                    ic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算合并报告统计指标
        
        Args:
            performance_data: 性能数据
            ic_data: IC数据
            
        Returns:
            统计指标字典
        """
        return self._calculate_batch_statistics(performance_data, ic_data)
    
    def _format_percentage(self, value: float) -> str:
        """
        格式化百分比
        
        Args:
            value: 数值
            
        Returns:
            格式化后的百分比字符串
        """
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value:.2%}"
    
    def _format_number(self, value: float, decimals: int = 4) -> str:
        """
        格式化数字
        
        Args:
            value: 数值
            decimals: 小数位数
            
        Returns:
            格式化后的数字字符串
        """
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value:.{decimals}f}"
    
    def _safe_json(self, obj: Any) -> str:
        """
        安全地将对象转换为JSON字符串
        
        Args:
            obj: 要转换的对象
            
        Returns:
            JSON字符串
        """
        try:
            return json.dumps(obj, default=str, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON序列化失败: {e}")
            return "{}"
    
    def get_template_list(self) -> List[str]:
        """
        获取可用模板列表
        
        Returns:
            模板名称列表
        """
        return self.env.list_templates()
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        渲染指定模板
        
        Args:
            template_name: 模板名称
            context: 模板上下文
            
        Returns:
            渲染后的HTML内容
        """
        template = self.env.get_template(template_name)
        return template.render(**context)
    
    def _generate_topn_results_html(self, factor_names: List[str], backtest_results: Dict[str, Any]) -> str:
        """生成TopN回测结果HTML (使用正确的QuantStats键名)"""
        # ... (函数的第一部分保持不变: a lot of html) ...
        html = """
        <table>
            <thead>
                <tr>
                    <th>因子名称</th>
                    <th>总收益率</th>
                    <th>年化收益率</th>
                    <th>年化波动率</th>
                    <th>夏普比率</th>
                    <th>最大回撤</th>
                    <th>交易天数</th>
                </tr>
            </thead>
            <tbody>
        """
        for factor_name in factor_names:
            topn_key = next((k for k in backtest_results if k.startswith(f'topn_{factor_name}')), None)
            if topn_key and topn_key in backtest_results:
                result = backtest_results[topn_key]
                stats = result.get('stats', {})

                # 使用完全正确的键名列表
                total_return = self._get_metric(stats, ['Cumulative Return', 'total_return', 'Total Return [%]'])
                annual_return = self._get_metric(stats, ['annual_return', 'annual_ret', 'return_annual', 'Annual Return [%]', 'Annual Return'])
                annual_volatility = self._get_metric(stats, ['Annual Volatility', 'volatility', 'Annual Volatility [%]'])
                sharpe_ratio = self._get_metric(stats, ['sharpe_ratio', 'sharpe', 'Sharpe', 'sharpe_ratio_annual', 'Sharpe Ratio'])
                max_drawdown = self._get_metric(stats, ['Max Drawdown', 'max_drawdown', 'Max Drawdown [%]'])
                # 'period' 的值可能是 '100 days' 这样的字符串，需要特殊处理
                trading_days_str = stats.get('Period', '0 days')
                trading_days = int(''.join(filter(str.isdigit, trading_days_str)))


                html += f"""
                <tr>
                    <td>{factor_name}</td>
                    <td class="{'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</td>
                    <td class="{'positive' if annual_return > 0 else 'negative'}">{annual_return:.2%}</td>
                    <td>{annual_volatility:.2%}</td>
                    <td class="{'positive' if sharpe_ratio > 0.5 else 'negative'}">{sharpe_ratio:.2f}</td>
                    <td class="negative">{max_drawdown:.2%}</td>
                    <td>{trading_days}</td>
                </tr>
                """
            else:
                html += f"""
                <tr>
                    <td>{factor_name}</td>
                    <td colspan="6" style="text-align:center;">N/A</td>
                </tr>
                """

        html += "</tbody></table>"
        return html
