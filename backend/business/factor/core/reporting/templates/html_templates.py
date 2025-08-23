#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : html_templates.py
@Description: HTML模板管理器
@Author     : Zijun Deng
@Date       : 2025-08-23
"""

from typing import Dict, Any, List
from datetime import datetime

from .base_template import BaseTemplate


class HTMLTemplateManager:
    """
    HTML模板管理器
    
    负责管理和渲染各种HTML报告模板
    """
    
    def __init__(self):
        """初始化模板管理器"""
        self.templates = {}
        self._init_templates()
    
    def _init_templates(self):
        """初始化所有模板"""
        self.templates = {
            'simple_html_report': SimpleHTMLReportTemplate(),
            'simple_group_report': SimpleGroupReportTemplate(),
            'interactive_summary': InteractiveSummaryTemplate(),
            'single_factor_interactive': SingleFactorInteractiveTemplate(),
            'error_report': ErrorReportTemplate()
        }
    
    def get_template(self, template_name: str) -> BaseTemplate:
        """
        获取模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板对象
            
        Raises:
            KeyError: 模板不存在
        """
        if template_name not in self.templates:
            raise KeyError(f"模板 '{template_name}' 不存在")
        return self.templates[template_name]
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        渲染模板
        
        Args:
            template_name: 模板名称
            **kwargs: 模板变量
            
        Returns:
            渲染后的HTML内容
        """
        template = self.get_template(template_name)
        return template.render(**kwargs)
    
    def list_templates(self) -> List[str]:
        """
        列出所有可用模板
        
        Returns:
            模板名称列表
        """
        return list(self.templates.keys())


class SimpleHTMLReportTemplate(BaseTemplate):
    """简单HTML报告模板"""
    
    def render(self, **kwargs) -> str:
        """
        渲染简单HTML报告
        
        Args:
            result_key: 结果键名
            total_return: 总收益率
            annual_return: 年化收益率
            volatility: 年化波动率
            sharpe_ratio: 夏普比率
            max_drawdown: 最大回撤
            trading_days: 交易天数
            returns_series: 收益率序列
            
        Returns:
            HTML内容
        """
        result_key = kwargs.get('result_key', 'Unknown')
        total_return = kwargs.get('total_return', 0.0)
        annual_return = kwargs.get('annual_return', 0.0)
        volatility = kwargs.get('volatility', 0.0)
        sharpe_ratio = kwargs.get('sharpe_ratio', 0.0)
        max_drawdown = kwargs.get('max_drawdown', 0.0)
        trading_days = kwargs.get('trading_days', 0)
        returns_series = kwargs.get('returns_series', [])
        
        # 生成图表数据
        dates = list(range(len(returns_series))) if returns_series else []
        cumulative = list((1 + pd.Series(returns_series)).cumprod()) if returns_series else []
        
        return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>因子回测报告 - {result_key}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
                    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                    .stat-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
                    .stat-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                    .stat-label {{ color: #6c757d; margin-top: 5px; }}
                    .chart-container {{ margin-top: 30px; }}
                    .info {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>因子回测报告</h1>
                        <h2>{result_key}</h2>
                        <p>生成时间: {self.format_datetime()}</p>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{self.format_percentage(total_return)}</div>
                            <div class="stat-label">总收益率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{self.format_percentage(annual_return)}</div>
                            <div class="stat-label">年化收益率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{self.format_percentage(volatility)}</div>
                            <div class="stat-label">年化波动率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{self.format_number(sharpe_ratio)}</div>
                            <div class="stat-label">夏普比率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{self.format_percentage(max_drawdown)}</div>
                            <div class="stat-label">最大回撤</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{trading_days}</div>
                            <div class="stat-label">交易天数</div>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>累计收益率走势</h3>
                        <div id="cumulative-chart"></div>
                    </div>
                    
                    <div class="info">
                        <h3>报告说明</h3>
                        <p>这是一个简化的回测报告，包含基本的回测统计指标和收益率走势图。</p>
                    </div>
                </div>
                
                <script>
                    // 生成累计收益率图表
                    const dates = {dates};
                    const cumulative = {cumulative};
                    
                    const trace = {{
                        x: dates,
                        y: cumulative,
                        type: 'scatter',
                        mode: 'lines',
                        name: '累计收益率',
                        line: {{ color: '#007bff', width: 2 }}
                    }};
                    
                    const layout = {{
                        title: '累计收益率走势',
                        xaxis: {{ title: '日期' }},
                        yaxis: {{ title: '累计收益率' }},
                        hovermode: 'x unified'
                    }};
                    
                    Plotly.newPlot('cumulative-chart', [trace], layout);
                </script>
            </body>
            </html>
            """


class SimpleGroupReportTemplate(BaseTemplate):
    """简单分组报告模板"""
    
    def render(self, **kwargs) -> str:
        """
        渲染简单分组报告
        
        Args:
            result_key: 结果键名
            group_stats: 分组统计信息列表
            
        Returns:
            HTML内容
        """
        result_key = kwargs.get('result_key', 'Unknown')
        group_stats = kwargs.get('group_stats', [])
        
        # 生成统计表格HTML
        stats_html = ""
        for stat in group_stats:
            stats_html += f"""
                <tr>
                    <td>{stat.get('分组名称', 'N/A')}</td>
                    <td>{stat.get('总收益率', 'N/A')}</td>
                    <td>{stat.get('年化收益率', 'N/A')}</td>
                    <td>{stat.get('年化波动率', 'N/A')}</td>
                    <td>{stat.get('夏普比率', 'N/A')}</td>
                    <td>{stat.get('最大回撤', 'N/A')}</td>
                    <td>{stat.get('交易天数', 'N/A')}</td>
                </tr>
                """
        
        return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>分组回测报告 - {result_key}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #007bff; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .info {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>分组回测报告</h1>
                        <h2>{result_key}</h2>
                        <p>生成时间: {self.format_datetime()}</p>
                    </div>
                    
                    <h3>分组表现对比</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>分组名称</th>
                                <th>总收益率</th>
                                <th>年化收益率</th>
                                <th>年化波动率</th>
                                <th>夏普比率</th>
                                <th>最大回撤</th>
                                <th>交易天数</th>
                            </tr>
                        </thead>
                        <tbody>
                            {stats_html}
                        </tbody>
                    </table>
                    
                    <div class="info">
                        <h3>报告说明</h3>
                        <p>这是一个简化的分组回测报告，展示各分组的表现对比。</p>
                    </div>
                </div>
            </body>
            </html>
            """


class InteractiveSummaryTemplate(BaseTemplate):
    """交互式汇总报告模板"""
    
    def render(self, **kwargs) -> str:
        """
        渲染交互式汇总报告
        
        Args:
            factor_names: 因子名称列表
            analysis_summary: 分析总结字典
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            top_n: 选股数量
            n_groups: 分组数量
            
        Returns:
            HTML内容
        """
        factor_names = kwargs.get('factor_names', [])
        analysis_summary = kwargs.get('analysis_summary', {})
        start_date = kwargs.get('start_date', 'N/A')
        end_date = kwargs.get('end_date', 'N/A')
        stock_pool = kwargs.get('stock_pool', 'no_st')
        top_n = kwargs.get('top_n', 10)
        n_groups = kwargs.get('n_groups', 5)
        
        factor_names_str = ', '.join(factor_names[:5])
        if len(factor_names) > 5:
            factor_names_str += '...'
        
        return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>因子分析交互式报告</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }}
                    .header h1 {{
                        font-size: 2.5em;
                        margin-bottom: 10px;
                        font-weight: 300;
                    }}
                    .header p {{
                        font-size: 1.1em;
                        opacity: 0.9;
                    }}
                    .content {{
                        padding: 40px;
                    }}
                    .section {{
                        margin-bottom: 40px;
                        background: #f8f9fa;
                        border-radius: 15px;
                        padding: 30px;
                        border-left: 5px solid #667eea;
                        transition: transform 0.3s ease;
                    }}
                    .section:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    }}
                    .section h2 {{
                        color: #2c3e50;
                        margin-bottom: 25px;
                        font-size: 1.8em;
                        border-bottom: 2px solid #e9ecef;
                        padding-bottom: 15px;
                    }}
                    .metric-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .metric-card {{
                        background: white;
                        padding: 25px;
                        border-radius: 12px;
                        text-align: center;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                        transition: transform 0.3s ease;
                    }}
                    .metric-card:hover {{
                        transform: translateY(-3px);
                    }}
                    .metric-value {{
                        font-size: 2em;
                        font-weight: bold;
                        margin-bottom: 8px;
                    }}
                    .metric-label {{
                        color: #6c757d;
                        font-size: 0.9em;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }}
                    .positive {{ color: #28a745; }}
                    .negative {{ color: #dc3545; }}
                    .neutral {{ color: #6c757d; }}
                    .info-box {{
                        background-color: #e7f3ff;
                        padding: 15px;
                        border-radius: 5px;
                        margin-top: 15px;
                        border-left: 4px solid #007bff;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 因子分析交互式报告</h1>
                        <p>分析因子: {factor_names_str} | 生成时间: {self.format_datetime()}</p>
                    </div>
                    
                    <div class="content">
                        <div class="section">
                            <h2>📈 分析总览</h2>
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <div class="metric-value positive">{len(factor_names)}</div>
                                    <div class="metric-label">总因子数</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value positive">{len(analysis_summary.get('successful_factors', []))}</div>
                                    <div class="metric-label">成功因子</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value positive">{self.format_percentage(analysis_summary.get('success_rate', 0))}</div>
                                    <div class="metric-label">成功率</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value positive">{analysis_summary.get('best_factor', 'N/A')}</div>
                                    <div class="metric-label">最佳因子</div>
                                </div>
                            </div>
                            
                            <div class="info-box">
                                <h3>分析概况</h3>
                                <p><strong>分析因子数量:</strong> {len(factor_names)}</p>
                                <p><strong>分析因子:</strong> {', '.join(factor_names)}</p>
                                <p><strong>分析期间:</strong> {start_date} 至 {end_date}</p>
                                <p><strong>股票池:</strong> {stock_pool}</p>
                                <p><strong>选股数量:</strong> {top_n}</p>
                                <p><strong>分组数量:</strong> {n_groups}</p>
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>📋 报告说明</h2>
                            <div class="info-box">
                                <p>这是一个简化的交互式因子分析报告。主要功能包括：</p>
                                <ul>
                                    <li>因子表现对比分析</li>
                                    <li>回测结果可视化</li>
                                    <li>IC分析结果展示</li>
                                    <li>分组回测结果</li>
                                </ul>
                                <p>如需完整功能，请查看对应的单个因子回测报告。</p>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """


class SingleFactorInteractiveTemplate(BaseTemplate):
    """单个因子交互式报告模板"""
    
    def render(self, **kwargs) -> str:
        """
        渲染单个因子交互式报告
        
        Args:
            factor_name: 因子名称
            
        Returns:
            HTML内容
        """
        factor_name = kwargs.get('factor_name', 'Unknown')
        
        return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{factor_name} 因子分析报告</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }}
                    .header h1 {{
                        font-size: 2.5em;
                        margin-bottom: 10px;
                        font-weight: 300;
                    }}
                    .header p {{
                        font-size: 1.1em;
                        opacity: 0.9;
                    }}
                    .content {{
                        padding: 40px;
                    }}
                    .section {{
                        margin-bottom: 40px;
                        background: #f8f9fa;
                        border-radius: 15px;
                        padding: 30px;
                        border-left: 5px solid #667eea;
                        transition: transform 0.3s ease;
                    }}
                    .section:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    }}
                    .section h2 {{
                        color: #2c3e50;
                        margin-bottom: 25px;
                        font-size: 1.8em;
                        border-bottom: 2px solid #e9ecef;
                        padding-bottom: 15px;
                    }}
                    .info-box {{
                        background-color: #e7f3ff;
                        padding: 15px;
                        border-radius: 5px;
                        margin-top: 15px;
                        border-left: 4px solid #007bff;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 {factor_name} 因子分析报告</h1>
                        <p>详细的因子表现分析和可视化图表</p>
                    </div>
                    
                    <div class="content">
                        <div class="section">
                            <h2>📈 因子概览</h2>
                            <div class="info-box">
                                <h3>报告说明</h3>
                                <p>这是因子 {factor_name} 的简化分析报告。主要功能包括：</p>
                                <ul>
                                    <li>因子表现指标展示</li>
                                    <li>收益率走势图</li>
                                    <li>回撤分析</li>
                                    <li>风险指标统计</li>
                                </ul>
                                <p>如需完整功能，请查看对应的回测报告。</p>
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>📋 报告信息</h2>
                            <div class="info-box">
                                <p><strong>因子名称:</strong> {factor_name}</p>
                                <p><strong>生成时间:</strong> {self.format_datetime()}</p>
                                <p><strong>报告类型:</strong> 简化版交互式报告</p>
                                <p><strong>说明:</strong> 此报告为重构后的简化版本，保留了核心功能。</p>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """


class ErrorReportTemplate(BaseTemplate):
    """错误报告模板"""
    
    def render(self, **kwargs) -> str:
        """
        渲染错误报告
        
        Args:
            result_key: 结果键名
            error_message: 错误信息
            report_type: 报告类型
            
        Returns:
            HTML内容
        """
        result_key = kwargs.get('result_key', 'Unknown')
        error_message = kwargs.get('error_message', '未知错误')
        report_type = kwargs.get('report_type', '回测报告')
        
        return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report_type}生成失败 - {result_key}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .error {{ color: red; background-color: #ffe6e6; padding: 20px; border-radius: 5px; }}
                    .info {{ background-color: #e6f3ff; padding: 20px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h1>{report_type} - {result_key}</h1>
                <div class="error">
                    <h2>报告生成失败</h2>
                    <p><strong>错误信息:</strong> {self.escape_html(error_message)}</p>
                    <p><strong>时间:</strong> {self.format_datetime()}</p>
                </div>
                <div class="info">
                    <h3>可能的原因:</h3>
                    <ul>
                        <li>数据格式问题</li>
                        <li>数据为空或全为NaN</li>
                        <li>库兼容性问题</li>
                        <li>系统资源不足</li>
                    </ul>
                </div>
            </body>
            </html>
            """


# 导入pandas用于数据处理
import pandas as pd
