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
from typing import Dict, List, Any, Optional
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
            
            # 计算综合评分（基于夏普比率和IC IR）
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            ic_ir = ic.get('ic_ir', 0)
            
            # 综合评分 = 夏普比率 * 0.6 + IC IR * 0.4
            composite_score = sharpe_ratio * 0.6 + ic_ir * 0.4
            
            factor_scores.append({
                'name': factor_name,
                'sharpe_ratio': sharpe_ratio,
                'ic_ir': ic_ir,
                'annual_return': performance.get('annual_return', 0),
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
        
        # 如果performance_metrics是DataFrame，转换为字典
        if isinstance(performance_metrics, pd.DataFrame):
            return performance_metrics.to_dict('index')
        
        return performance_metrics
    
    def _extract_ic_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从报告数据中提取IC指标
        
        Args:
            report_data: 报告数据
            
        Returns:
            IC指标字典
        """
        ic_metrics = report_data.get('ic_metrics', {})
        
        # 如果ic_metrics是DataFrame，转换为字典
        if isinstance(ic_metrics, pd.DataFrame):
            return ic_metrics.to_dict('index')
        
        return ic_metrics
    
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
            if isinstance(returns_series, pd.Series):
                # 计算累计收益率
                cumulative_returns = (1 + returns_series).cumprod()
                
                chart_data[factor_name] = {
                    'dates': returns_series.index.strftime('%Y-%m-%d').tolist(),
                    'values': cumulative_returns.tolist()
                }
        
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
            processed_data[factor_name] = {
                'metrics': factor_data.get('metrics', {}),
                'group_results': factor_data.get('group_results', {}),
                'ic_stats': factor_data.get('ic_stats', {}),
                'returns_series': factor_data.get('returns_series'),
                'drawdown_series': factor_data.get('drawdown_series'),
                'ic_series': factor_data.get('ic_series'),
                'monthly_returns': factor_data.get('monthly_returns', []),
                'risk_metrics': factor_data.get('risk_metrics', {})
            }
        
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
        
        # 计算统计指标
        sharpe_ratios = [p.get('sharpe_ratio', 0) for p in performance_data.values()]
        annual_returns = [p.get('annual_return', 0) for p in performance_data.values()]
        ic_means = [ic.get('mean_ic', 0) for ic in ic_data.values()]
        
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
