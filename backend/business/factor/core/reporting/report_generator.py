#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : report_generator.py
@Description: 报告生成器，负责生成完整的因子研究报告
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import quantstats as qs

from backend.business.factor.core.analysis.factor_analyzer import FactorAnalyzer
from backend.business.factor.core.backtest.backtest_engine import FactorBacktestEngine
from backend.business.factor.core.data.data_manager import FactorDataManager
from backend.business.factor.core.factor.factor_engine import FactorEngine
from backend.utils.logger import setup_logger

logger = setup_logger("backtest_factor")

# 设置QuantStats
qs.extend_pandas()


class FactorReportGenerator:
    """
    报告生成器，负责生成完整的因子研究报告
    
    功能：
    1. 生成因子研究报告
    2. 生成回测报告
    3. 生成综合分析报告
    4. 导出Excel和PDF报告
    """

    def __init__(self,
                 factor_engine: FactorEngine,
                 backtest_engine: FactorBacktestEngine,
                 analyzer: FactorAnalyzer,
                 data_manager: FactorDataManager):
        """
        初始化报告生成器
        
        Args:
            factor_engine: 因子引擎实例
            backtest_engine: 回测引擎实例
            analyzer: 分析器实例
            data_manager: 数据管理器实例
        """
        self.factor_engine = factor_engine
        self.backtest_engine = backtest_engine
        self.analyzer = analyzer
        self.data_manager = data_manager
        self._reports = {}

    def generate_factor_report(self,
                               factor_name: str,
                               output_dir: str = "reports",
                               **kwargs) -> str:
        """
        生成单个因子研究报告（已废弃，只生成QuantStats HTML报告）
        
        Args:
            factor_name: 因子名称
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            报告文件路径
        """
        logger.info(f"因子报告生成已废弃，请使用QuantStats HTML报告")
        return ""

    def generate_backtest_report(self,
                                 result_key: str,
                                 output_dir: str = "reports",
                                 **kwargs) -> str:
        """
        生成回测报告
        
        Args:
            result_key: 回测结果键名
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            报告文件路径
        """
        logger.info(f"开始生成回测报告: {result_key}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取回测结果
        backtest_result = self.backtest_engine.get_backtest_results(result_key)
        if backtest_result is None:
            # 尝试从框架结果中获取
            if hasattr(self, '_framework_results') and self._framework_results:
                backtest_results = self._framework_results.get('backtest_results', {})
                backtest_result = backtest_results.get(result_key)

        if backtest_result is None:
            raise ValueError(f"回测结果 {result_key} 不存在")

        # 生成QuantStats HTML报告
        report_path = self._generate_quantstats_report(backtest_result, result_key, output_dir, **kwargs)

        logger.info(f"回测报告已生成: {report_path}")
        return report_path

    def generate_comprehensive_report(self,
                                      factor_names: List[str],
                                      output_dir: str = "reports",
                                      backtest_results: Optional[Dict[str, Any]] = None,
                                      **kwargs) -> str:
        """
        生成综合分析报告
        
        Args:
            factor_names: 因子名称列表
            output_dir: 输出目录
            backtest_results: 回测结果字典
            **kwargs: 其他参数
            
        Returns:
            报告文件路径
        """
        logger.info(f"开始生成综合分析报告: {factor_names}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"comprehensive_report_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)

        # 保存回测结果供后续使用
        if backtest_results:
            self._framework_results = {'backtest_results': backtest_results}

        # 生成汇总报告
        try:
            self._generate_summary_report(factor_names, report_path, **kwargs)
            return report_path
        except Exception as e:
            logger.error(f"汇总分析报告生成失败: {e}")
            raise

    def generate_merged_comprehensive_report(self,
                                             factor_names: List[str],
                                             merged_results: Dict[str, Any],
                                             analysis_summary: Dict[str, Any],
                                             report_path: str,
                                             start_date: str,
                                             end_date: str,
                                             stock_pool: str = "no_st",
                                             top_n: int = 10,
                                             n_groups: int = 5) -> None:
        """
        生成合并的综合报告（包含分析总结）
        
        Args:
            factor_names: 因子名称列表
            merged_results: 合并的结果字典
            analysis_summary: 分析总结字典
            report_path: 报告文件路径
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            top_n: 选股数量
            n_groups: 分组数量
        """
        logger.info(f"开始生成合并的综合报告，包含 {len(factor_names)} 个因子")

        try:
            # 保存合并结果供后续使用
            self._framework_results = merged_results

            # 生成HTML报告
            html_content = self._generate_merged_summary_html(
                factor_names=factor_names,
                merged_results=merged_results,
                analysis_summary=analysis_summary,
                start_date=start_date,
                end_date=end_date,
                stock_pool=stock_pool,
                top_n=top_n,
                n_groups=n_groups
            )

            # 确保目录存在
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"合并综合报告已生成: {report_path}")

        except Exception as e:
            logger.error(f"生成合并综合报告失败: {e}")
            raise

    def _generate_summary_report(self,
                                 factor_names: List[str],
                                 report_path: str,
                                 **kwargs) -> None:
        """
        生成汇总分析报告，包含所有分析结果
        
        Args:
            factor_names: 因子名称列表
            report_path: 报告文件路径
            **kwargs: 其他参数
        """
        try:
            # 收集所有分析数据
            summary_data = self._collect_summary_data(factor_names)

            # 生成HTML报告
            html_content = self._generate_summary_html(summary_data, factor_names)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"汇总报告已生成: {report_path}")

        except Exception as e:
            logger.error(f"生成汇总报告失败: {e}")
            raise

    def _collect_summary_data(self, factor_names: List[str]) -> Dict[str, Any]:
        """
        收集所有分析数据
        
        Args:
            factor_names: 因子名称列表
            
        Returns:
            汇总数据字典
        """
        summary_data = {
            'factor_names': factor_names,
            'topn_results': {},
            'group_results': {},
            'multifactor_results': {},
            'ic_results': {},
            'effectiveness_results': {}
        }

        # 收集TopN回测结果
        for factor_name in factor_names:
            try:
                # 尝试多种可能的result_key格式
                possible_keys = [
                    f'topn_{factor_name}',  # 原始格式
                    f'topn_{factor_name}_10',  # 带n参数的格式
                    f'topn_{factor_name}_5',   # 可能的其他n值
                    f'topn_{factor_name}_20'   # 可能的其他n值
                ]
                
                backtest_result = None
                used_key = None
                
                for result_key in possible_keys:
                    backtest_result = self.backtest_engine.get_backtest_results(result_key)
                    if backtest_result is None and hasattr(self, '_framework_results'):
                        backtest_results = self._framework_results.get('backtest_results', {})
                        backtest_result = backtest_results.get(result_key)
                    
                    if backtest_result is not None:
                        used_key = result_key
                        break
                
                if backtest_result and 'stats' in backtest_result:
                    # 直接使用回测引擎已经计算好的stats
                    stats = backtest_result['stats']
                    total_return = stats.get('total_return', 0.0)
                    max_drawdown = stats.get('max_drawdown', 0.0)
                    
                    # 确保total_return是标量值
                    if hasattr(total_return, 'iloc'):
                        total_return = total_return.iloc[0] if len(total_return) > 0 else 0.0
                    elif hasattr(total_return, 'item'):
                        total_return = total_return.item()
                    else:
                        total_return = float(total_return) if total_return is not None else 0.0
                    
                    # 确保max_drawdown是标量值
                    if hasattr(max_drawdown, 'iloc'):
                        max_drawdown = max_drawdown.iloc[0] if len(max_drawdown) > 0 else 0.0
                    elif hasattr(max_drawdown, 'item'):
                        max_drawdown = max_drawdown.item()
                    else:
                        max_drawdown = float(max_drawdown) if max_drawdown is not None else 0.0
                    
                    # 计算其他指标（如果需要）
                    if 'portfolio' in backtest_result:
                        portfolio = backtest_result['portfolio']
                        returns = portfolio.returns()
                        if returns is not None and not returns.empty:
                            # 处理returns数据
                            if isinstance(returns, pd.Series):
                                returns_series = returns.dropna()
                            elif isinstance(returns, pd.DataFrame):
                                if returns.shape[1] == 1:
                                    returns_series = returns.iloc[:, 0].dropna()
                                else:
                                    returns_series = returns.mean(axis=1).dropna()
                            elif isinstance(returns, np.ndarray):
                                if returns.ndim == 1:
                                    returns_series = pd.Series(returns).dropna()
                                else:
                                    returns_series = pd.Series(returns.mean(axis=1)).dropna()
                            else:
                                returns_series = pd.Series(returns).dropna()

                            if len(returns_series) > 0:
                                annual_return = total_return * 252 / len(returns_series) if len(returns_series) > 0 else 0
                                volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
                                sharpe_ratio = annual_return / volatility if volatility > 0 else 0

                                summary_data['topn_results'][factor_name] = {
                                    'total_return': total_return,
                                    'annual_return': annual_return,
                                    'volatility': volatility,
                                    'sharpe_ratio': sharpe_ratio,
                                    'max_drawdown': max_drawdown,
                                    'trading_days': len(returns_series),
                                    'returns': returns_series
                                }
                                logger.info(f"成功收集因子 {factor_name} TopN结果，使用键: {used_key}")
                            else:
                                logger.warning(f"因子 {factor_name} TopN结果returns_series为空")
                        else:
                            logger.warning(f"因子 {factor_name} TopN结果portfolio.returns()为空或None")
                    else:
                        # 如果没有portfolio，只使用stats中的基本指标
                        summary_data['topn_results'][factor_name] = {
                            'total_return': total_return,
                            'annual_return': 0.0,
                            'volatility': 0.0,
                            'sharpe_ratio': 0.0,
                            'max_drawdown': max_drawdown,
                            'trading_days': 0,
                            'returns': pd.Series()
                        }
                        logger.info(f"成功收集因子 {factor_name} TopN结果（仅stats），使用键: {used_key}")
                elif backtest_result and 'portfolio' in backtest_result:
                    # 如果没有stats，使用原来的计算方法
                    portfolio = backtest_result['portfolio']
                    returns = portfolio.returns()
                    if returns is not None and not returns.empty:
                        # 处理returns数据
                        if isinstance(returns, pd.Series):
                            returns_series = returns.dropna()
                        elif isinstance(returns, pd.DataFrame):
                            if returns.shape[1] == 1:
                                returns_series = returns.iloc[:, 0].dropna()
                            else:
                                returns_series = returns.mean(axis=1).dropna()
                        elif isinstance(returns, np.ndarray):
                            if returns.ndim == 1:
                                returns_series = pd.Series(returns).dropna()
                            else:
                                returns_series = pd.Series(returns.mean(axis=1)).dropna()
                        else:
                            returns_series = pd.Series(returns).dropna()

                        if len(returns_series) > 0:
                            # 计算统计指标
                            total_return = (1 + returns_series).prod() - 1
                            annual_return = total_return * 252 / len(returns_series) if len(returns_series) > 0 else 0
                            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
                            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                            cumulative = (1 + returns_series).cumprod()
                            running_max = cumulative.expanding().max()
                            drawdown = (cumulative - running_max) / running_max
                            max_drawdown = drawdown.min()

                            summary_data['topn_results'][factor_name] = {
                                'total_return': total_return,
                                'annual_return': annual_return,
                                'volatility': volatility,
                                'sharpe_ratio': sharpe_ratio,
                                'max_drawdown': max_drawdown,
                                'trading_days': len(returns_series),
                                'returns': returns_series
                            }
                            logger.info(f"成功收集因子 {factor_name} TopN结果（重新计算），使用键: {used_key}")
                        else:
                            logger.warning(f"因子 {factor_name} TopN结果returns_series为空")
                    else:
                        logger.warning(f"因子 {factor_name} TopN结果portfolio.returns()为空或None")
                else:
                    logger.warning(f"因子 {factor_name} TopN结果未找到，尝试的键: {possible_keys}")
            except Exception as e:
                logger.warning(f"收集因子 {factor_name} TopN结果失败: {e}")

        # 收集分组回测结果
        for factor_name in factor_names:
            try:
                # 尝试多种可能的result_key格式
                possible_keys = [
                    f'group_{factor_name}',  # 原始格式
                    f'group_{factor_name}_5',  # 带n_groups参数的格式
                    f'group_{factor_name}_10', # 可能的其他n_groups值
                    f'group_{factor_name}_3'   # 可能的其他n_groups值
                ]
                
                backtest_result = None
                used_key = None
                
                for result_key in possible_keys:
                    backtest_result = self.backtest_engine.get_backtest_results(result_key)
                    if backtest_result is None and hasattr(self, '_framework_results'):
                        backtest_results = self._framework_results.get('backtest_results', {})
                        backtest_result = backtest_results.get(result_key)
                    
                    if backtest_result is not None:
                        used_key = result_key
                        break
                
                if backtest_result and 'stats' in backtest_result:
                    # 直接使用回测引擎已经计算好的stats
                    stats = backtest_result['stats']
                    total_return = stats.get('total_return', 0.0)
                    max_drawdown = stats.get('max_drawdown', 0.0)
                    
                    # 确保total_return是标量值
                    if hasattr(total_return, 'iloc'):
                        total_return = total_return.iloc[0] if len(total_return) > 0 else 0.0
                    elif hasattr(total_return, 'item'):
                        total_return = total_return.item()
                    else:
                        total_return = float(total_return) if total_return is not None else 0.0
                    
                    # 确保max_drawdown是标量值
                    if hasattr(max_drawdown, 'iloc'):
                        max_drawdown = max_drawdown.iloc[0] if len(max_drawdown) > 0 else 0.0
                    elif hasattr(max_drawdown, 'item'):
                        max_drawdown = max_drawdown.item()
                    else:
                        max_drawdown = float(max_drawdown) if max_drawdown is not None else 0.0
                    
                    # 计算其他指标（如果需要）
                    if 'portfolios' in backtest_result:
                        portfolios = backtest_result['portfolios']
                        group_stats = {}

                        for group_name, portfolio in portfolios.items():
                            try:
                                returns = portfolio.returns()
                                if returns is not None and not returns.empty:
                                    # 处理returns数据
                                    if isinstance(returns, pd.Series):
                                        returns_series = returns.dropna()
                                    elif isinstance(returns, pd.DataFrame):
                                        if returns.shape[1] == 1:
                                            returns_series = returns.iloc[:, 0].dropna()
                                        else:
                                            returns_series = returns.mean(axis=1).dropna()
                                    elif isinstance(returns, np.ndarray):
                                        if returns.ndim == 1:
                                            returns_series = pd.Series(returns).dropna()
                                        else:
                                            returns_series = pd.Series(returns.mean(axis=1)).dropna()
                                    else:
                                        returns_series = pd.Series(returns).dropna()

                                    if len(returns_series) > 0:
                                        annual_return = total_return * 252 / len(returns_series) if len(
                                            returns_series) > 0 else 0
                                        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
                                        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

                                        group_stats[group_name] = {
                                            'total_return': total_return,
                                            'annual_return': annual_return,
                                            'volatility': volatility,
                                            'sharpe_ratio': sharpe_ratio,
                                            'max_drawdown': max_drawdown,
                                            'trading_days': len(returns_series)
                                        }
                            except Exception as e:
                                logger.warning(f"收集分组 {group_name} 结果失败: {e}")

                        if group_stats:
                            summary_data['group_results'][factor_name] = group_stats
                            logger.info(f"成功收集因子 {factor_name} 分组结果，使用键: {used_key}")
                        else:
                            logger.warning(f"因子 {factor_name} 分组结果group_stats为空")
                    else:
                        # 如果没有portfolios，只使用stats中的基本指标
                        summary_data['group_results'][factor_name] = {
                            'group_1': {
                                'total_return': total_return,
                                'annual_return': 0.0,
                                'volatility': 0.0,
                                'sharpe_ratio': 0.0,
                                'max_drawdown': max_drawdown,
                                'trading_days': 0
                            }
                        }
                        logger.info(f"成功收集因子 {factor_name} 分组结果（仅stats），使用键: {used_key}")
                elif backtest_result and 'portfolios' in backtest_result:
                    # 如果没有stats，使用原来的计算方法
                    portfolios = backtest_result['portfolios']
                    group_stats = {}

                    for group_name, portfolio in portfolios.items():
                        try:
                            returns = portfolio.returns()
                            if returns is not None and not returns.empty:
                                # 处理returns数据
                                if isinstance(returns, pd.Series):
                                    returns_series = returns.dropna()
                                elif isinstance(returns, pd.DataFrame):
                                    if returns.shape[1] == 1:
                                        returns_series = returns.iloc[:, 0].dropna()
                                    else:
                                        returns_series = returns.mean(axis=1).dropna()
                                elif isinstance(returns, np.ndarray):
                                    if returns.ndim == 1:
                                        returns_series = pd.Series(returns).dropna()
                                    else:
                                        returns_series = pd.Series(returns.mean(axis=1)).dropna()
                                else:
                                    returns_series = pd.Series(returns).dropna()

                                if len(returns_series) > 0:
                                    total_return = (1 + returns_series).prod() - 1
                                    annual_return = total_return * 252 / len(returns_series) if len(
                                        returns_series) > 0 else 0
                                    volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
                                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                                    cumulative = (1 + returns_series).cumprod()
                                    running_max = cumulative.expanding().max()
                                    drawdown = (cumulative - running_max) / running_max
                                    max_drawdown = drawdown.min()

                                    group_stats[group_name] = {
                                        'total_return': total_return,
                                        'annual_return': annual_return,
                                        'volatility': volatility,
                                        'sharpe_ratio': sharpe_ratio,
                                        'max_drawdown': max_drawdown,
                                        'trading_days': len(returns_series)
                                    }
                        except Exception as e:
                            logger.warning(f"收集分组 {group_name} 结果失败: {e}")

                    if group_stats:
                        summary_data['group_results'][factor_name] = group_stats
                        logger.info(f"成功收集因子 {factor_name} 分组结果（重新计算），使用键: {used_key}")
                    else:
                        logger.warning(f"因子 {factor_name} 分组结果group_stats为空")
                else:
                    logger.warning(f"因子 {factor_name} 分组结果未找到，尝试的键: {possible_keys}")
            except Exception as e:
                logger.warning(f"收集因子 {factor_name} 分组结果失败: {e}")

        # 收集多因子回测结果
        try:
            result_key = 'multifactor'
            backtest_result = self.backtest_engine.get_backtest_results(result_key)
            if backtest_result is None and hasattr(self, '_framework_results'):
                backtest_results = self._framework_results.get('backtest_results', {})
                backtest_result = backtest_results.get(result_key)

            if backtest_result and 'portfolio' in backtest_result:
                portfolio = backtest_result['portfolio']
                returns = portfolio.returns()
                if returns is not None and not returns.empty:
                    # 处理returns数据
                    if isinstance(returns, pd.Series):
                        returns_series = returns.dropna()
                    elif isinstance(returns, pd.DataFrame):
                        if returns.shape[1] == 1:
                            returns_series = returns.iloc[:, 0].dropna()
                        else:
                            returns_series = returns.mean(axis=1).dropna()
                    elif isinstance(returns, np.ndarray):
                        if returns.ndim == 1:
                            returns_series = pd.Series(returns).dropna()
                        else:
                            returns_series = pd.Series(returns.mean(axis=1)).dropna()
                    else:
                        returns_series = pd.Series(returns).dropna()

                    if len(returns_series) > 0:
                        total_return = (1 + returns_series).prod() - 1
                        annual_return = total_return * 252 / len(returns_series) if len(returns_series) > 0 else 0
                        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
                        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                        cumulative = (1 + returns_series).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = drawdown.min()

                        summary_data['multifactor_results'] = {
                            'total_return': total_return,
                            'annual_return': annual_return,
                            'volatility': volatility,
                            'sharpe_ratio': sharpe_ratio,
                            'max_drawdown': max_drawdown,
                            'trading_days': len(returns_series),
                            'returns': returns_series
                        }
        except Exception as e:
            logger.warning(f"收集多因子结果失败: {e}")

        # 收集IC和有效性分析结果
        for factor_name in factor_names:
            try:
                # IC结果
                ic_key = f'ic_{factor_name}_pearson'
                ic_result = self.analyzer.get_analysis_results(ic_key)
                if ic_result is not None:
                    summary_data['ic_results'][factor_name] = {
                        'pearson_ic': ic_result.get('ic_stats', {}).get('mean_ic', 0),
                        'ic_ir': ic_result.get('ic_stats', {}).get('ir', 0),
                        'ic_win_rate': ic_result.get('ic_stats', {}).get('win_rate', 0)
                    }

                # 有效性分析结果
                effectiveness_key = f'effectiveness_{factor_name}'
                effectiveness_result = self.analyzer.get_analysis_results(effectiveness_key)
                if effectiveness_result is not None:
                    summary_data['effectiveness_results'][factor_name] = effectiveness_result
            except Exception as e:
                logger.warning(f"收集因子 {factor_name} IC和有效性结果失败: {e}")

        return summary_data

    def _generate_summary_html(self, summary_data: Dict[str, Any], factor_names: List[str]) -> str:
        """
        生成汇总HTML报告
        
        Args:
            summary_data: 汇总数据
            factor_names: 因子名称列表
            
        Returns:
            HTML内容
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 生成导航菜单
        nav_html = self._generate_navigation_menu(factor_names)

        # 生成各个部分的HTML
        overview_html = self._generate_overview_section(summary_data, factor_names)
        topn_html = self._generate_topn_section(summary_data)
        group_html = self._generate_group_section(summary_data)
        multifactor_html = self._generate_multifactor_section(summary_data)
        ic_html = self._generate_ic_section(summary_data)
        effectiveness_html = self._generate_effectiveness_section(summary_data)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>因子分析汇总报告</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background-color: white; 
                    border-radius: 10px; 
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    text-align: center;
                }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                
                .nav {{ 
                    background-color: #2c3e50; 
                    padding: 0; 
                    position: sticky; 
                    top: 0; 
                    z-index: 100;
                }}
                .nav ul {{ 
                    list-style: none; 
                    margin: 0; 
                    padding: 0; 
                    display: flex; 
                    flex-wrap: wrap;
                }}
                .nav li {{ margin: 0; }}
                .nav a {{ 
                    display: block; 
                    color: white; 
                    text-decoration: none; 
                    padding: 15px 20px; 
                    transition: background-color 0.3s;
                }}
                .nav a:hover {{ background-color: #34495e; }}
                
                .content {{ padding: 30px; }}
                .section {{ 
                    margin-bottom: 40px; 
                    padding: 25px; 
                    border-radius: 8px; 
                    background-color: #f8f9fa; 
                    border-left: 4px solid #007bff;
                }}
                .section h2 {{ 
                    color: #2c3e50; 
                    margin-top: 0; 
                    border-bottom: 2px solid #e9ecef; 
                    padding-bottom: 10px;
                }}
                
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 15px; 
                    background-color: white;
                    border-radius: 5px;
                    overflow: hidden;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    padding: 12px 15px; 
                    text-align: left; 
                    border-bottom: 1px solid #dee2e6; 
                }}
                th {{ 
                    background-color: #007bff; 
                    color: white; 
                    font-weight: bold;
                }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e9ecef; }}
                
                .metric-card {{ 
                    display: inline-block; 
                    background: white; 
                    padding: 15px; 
                    margin: 10px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    min-width: 150px;
                    text-align: center;
                }}
                .metric-value {{ 
                    font-size: 1.5em; 
                    font-weight: bold; 
                    color: #007bff; 
                }}
                .metric-label {{ 
                    color: #6c757d; 
                    font-size: 0.9em; 
                    margin-top: 5px;
                }}
                
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                
                .chart-container {{ 
                    margin-top: 20px; 
                    padding: 20px; 
                    background-color: white; 
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                
                .info-box {{ 
                    background-color: #e7f3ff; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-top: 15px;
                    border-left: 4px solid #007bff;
                }}
                
                @media (max-width: 768px) {{
                    .nav ul {{ flex-direction: column; }}
                    .nav a {{ text-align: center; }}
                    .metric-card {{ min-width: 120px; }}
                }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>因子分析汇总报告</h1>
                    <p>分析因子: {', '.join(factor_names)} | 生成时间: {timestamp}</p>
                </div>
                
                <nav class="nav">
                    {nav_html}
                </nav>
                
                <div class="content">
                    {overview_html}
                    {topn_html}
                    {group_html}
                    {multifactor_html}
                    {ic_html}
                    {effectiveness_html}
                </div>
            </div>
            
            <script>
                // 平滑滚动到锚点
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                    anchor.addEventListener('click', function (e) {{
                        e.preventDefault();
                        const target = document.querySelector(this.getAttribute('href'));
                        if (target) {{
                            target.scrollIntoView({{
                                behavior: 'smooth',
                                block: 'start'
                            }});
                        }}
                    }});
                }});
            </script>
        </body>
        </html>
        """

        return html_content

    def _generate_navigation_menu(self, factor_names: List[str]) -> str:
        """生成导航菜单"""
        nav_items = [
            '<li><a href="#overview">总览</a></li>',
            '<li><a href="#topn">TopN回测</a></li>',
            '<li><a href="#group">分组回测</a></li>',
            '<li><a href="#multifactor">多因子回测</a></li>',
            '<li><a href="#ic">IC分析</a></li>',
            '<li><a href="#effectiveness">有效性分析</a></li>'
        ]
        return '<ul>' + ''.join(nav_items) + '</ul>'

    def _generate_overview_section(self, summary_data: Dict[str, Any], factor_names: List[str]) -> str:
        """生成总览部分"""
        overview_html = f"""
        <div id="overview" class="section">
            <h2>📊 分析总览</h2>
            <div class="info-box">
                <h3>分析概况</h3>
                <p><strong>分析因子数量:</strong> {len(factor_names)}</p>
                <p><strong>分析因子:</strong> {', '.join(factor_names)}</p>
                <p><strong>分析期间:</strong> 根据数据时间范围确定</p>
            </div>
        </div>
        """
        return overview_html

    def _generate_topn_section(self, summary_data: Dict[str, Any]) -> str:
        """生成TopN回测部分"""
        if not summary_data['topn_results']:
            return '<div id="topn" class="section"><h2>📈 TopN回测</h2><p>暂无TopN回测数据</p></div>'

        # 生成统计表格
        table_rows = []
        for factor_name, result in summary_data['topn_results'].items():
            row = f"""
            <tr>
                <td>{factor_name}</td>
                <td class="{'positive' if result['total_return'] > 0 else 'negative'}">{result['total_return']:.2%}</td>
                <td class="{'positive' if result['annual_return'] > 0 else 'negative'}">{result['annual_return']:.2%}</td>
                <td>{result['volatility']:.2%}</td>
                <td class="{'positive' if result['sharpe_ratio'] > 0 else 'negative'}">{result['sharpe_ratio']:.2f}</td>
                <td class="negative">{result['max_drawdown']:.2%}</td>
                <td>{result['trading_days']}</td>
            </tr>
            """
            table_rows.append(row)

        topn_html = f"""
        <div id="topn" class="section">
            <h2>📈 TopN回测结果</h2>
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
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
        """
        return topn_html

    def _generate_group_section(self, summary_data: Dict[str, Any]) -> str:
        """生成分组回测部分"""
        if not summary_data['group_results']:
            return '<div id="group" class="section"><h2>📊 分组回测</h2><p>暂无分组回测数据</p></div>'

        group_html = '<div id="group" class="section"><h2>📊 分组回测结果</h2>'

        for factor_name, group_stats in summary_data['group_results'].items():
            group_html += f'<h3>{factor_name} 分组表现</h3>'

            # 生成分组表格
            table_rows = []
            for group_name, stats in group_stats.items():
                row = f"""
                <tr>
                    <td>{group_name}</td>
                    <td class="{'positive' if stats['total_return'] > 0 else 'negative'}">{stats['total_return']:.2%}</td>
                    <td class="{'positive' if stats['annual_return'] > 0 else 'negative'}">{stats['annual_return']:.2%}</td>
                    <td>{stats['volatility']:.2%}</td>
                    <td class="{'positive' if stats['sharpe_ratio'] > 0 else 'negative'}">{stats['sharpe_ratio']:.2f}</td>
                    <td class="negative">{stats['max_drawdown']:.2%}</td>
                    <td>{stats['trading_days']}</td>
                </tr>
                """
                table_rows.append(row)

            group_html += f"""
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
                    {''.join(table_rows)}
                </tbody>
            </table>
            """

        group_html += '</div>'
        return group_html

    def _generate_multifactor_section(self, summary_data: Dict[str, Any]) -> str:
        """生成多因子回测部分"""
        if not summary_data['multifactor_results']:
            return '<div id="multifactor" class="section"><h2>🔗 多因子回测</h2><p>暂无多因子回测数据</p></div>'

        result = summary_data['multifactor_results']
        multifactor_html = f"""
        <div id="multifactor" class="section">
            <h2>🔗 多因子回测结果</h2>
            <div class="metric-card">
                <div class="metric-value {'positive' if result['total_return'] > 0 else 'negative'}">{result['total_return']:.2%}</div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if result['annual_return'] > 0 else 'negative'}">{result['annual_return']:.2%}</div>
                <div class="metric-label">年化收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result['volatility']:.2%}</div>
                <div class="metric-label">年化波动率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if result['sharpe_ratio'] > 0 else 'negative'}">{result['sharpe_ratio']:.2f}</div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{result['max_drawdown']:.2%}</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result['trading_days']}</div>
                <div class="metric-label">交易天数</div>
            </div>
        </div>
        """
        return multifactor_html

    def _generate_ic_section(self, summary_data: Dict[str, Any]) -> str:
        """生成IC分析部分"""
        if not summary_data['ic_results']:
            return '<div id="ic" class="section"><h2>📊 IC分析</h2><p>暂无IC分析数据</p></div>'

        # 生成IC表格
        table_rows = []
        for factor_name, ic_data in summary_data['ic_results'].items():
            row = f"""
            <tr>
                <td>{factor_name}</td>
                <td class="{'positive' if ic_data['pearson_ic'] > 0 else 'negative'}">{ic_data['pearson_ic']:.4f}</td>
                <td class="{'positive' if ic_data['ic_ir'] > 0 else 'negative'}">{ic_data['ic_ir']:.4f}</td>
                <td class="{'positive' if ic_data['ic_win_rate'] > 0.5 else 'negative'}">{ic_data['ic_win_rate']:.2%}</td>
            </tr>
            """
            table_rows.append(row)

        ic_html = f"""
        <div id="ic" class="section">
            <h2>📊 IC分析结果</h2>
            <table>
                <thead>
                    <tr>
                        <th>因子名称</th>
                        <th>Pearson IC</th>
                        <th>IC IR</th>
                        <th>IC胜率</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
        """
        return ic_html

    def _generate_effectiveness_section(self, summary_data: Dict[str, Any]) -> str:
        """生成有效性分析部分"""
        if not summary_data['effectiveness_results']:
            return '<div id="effectiveness" class="section"><h2>📈 有效性分析</h2><p>暂无有效性分析数据</p></div>'

        effectiveness_html = '<div id="effectiveness" class="section"><h2>📈 有效性分析结果</h2>'

        for factor_name, effectiveness_data in summary_data['effectiveness_results'].items():
            effectiveness_html += f'<h3>{factor_name} 有效性指标</h3>'

            # 解析有效性数据
            ic_analysis = effectiveness_data.get('ic_analysis', {})
            rank_ic_analysis = effectiveness_data.get('rank_ic_analysis', {})
            group_returns = effectiveness_data.get('group_returns', {})
            stability_metrics = effectiveness_data.get('stability_metrics', {})

            # 1. IC分析表格
            if ic_analysis and 'ic_stats' in ic_analysis:
                ic_stats = ic_analysis['ic_stats']
                effectiveness_html += '''
                <h4>📊 IC分析 (Pearson相关系数)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                            <th>说明</th>
                        </tr>
                    </thead>
                    <tbody>
                '''

                ic_metrics = [
                    ('mean_ic', '平均IC', '因子预测能力'),
                    ('std_ic', 'IC标准差', 'IC波动性'),
                    ('ir', '信息比率', '风险调整后收益'),
                    ('positive_ic_rate', '正IC比率', 'IC为正的比例'),
                    ('abs_mean_ic', '绝对平均IC', '预测能力强度'),
                    ('ic_skewness', 'IC偏度', 'IC分布偏斜程度'),
                    ('ic_kurtosis', 'IC峰度', 'IC分布尖峭程度'),
                    ('min_ic', '最小IC', 'IC最小值'),
                    ('max_ic', '最大IC', 'IC最大值'),
                    ('ic_count', 'IC样本数', '有效IC数量')
                ]

                for key, name, desc in ic_metrics:
                    if key in ic_stats:
                        value = ic_stats[key]
                        if key in ['positive_ic_rate']:
                            formatted_value = f"{value:.2%}"
                        elif key in ['mean_ic', 'std_ic', 'ir', 'abs_mean_ic', 'ic_skewness', 'ic_kurtosis', 'min_ic',
                                     'max_ic']:
                            formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = str(value)

                        effectiveness_html += f'''
                        <tr>
                            <td>{name}</td>
                            <td class="{'positive' if key in ['mean_ic', 'ir', 'positive_ic_rate', 'abs_mean_ic'] and value > 0 else 'negative' if key in ['mean_ic', 'ir', 'positive_ic_rate'] and value < 0 else ''}">{formatted_value}</td>
                            <td>{desc}</td>
                        </tr>
                        '''

                effectiveness_html += '</tbody></table>'

            # 2. Rank IC分析表格
            if rank_ic_analysis and 'ic_stats' in rank_ic_analysis:
                rank_ic_stats = rank_ic_analysis['ic_stats']
                effectiveness_html += '''
                <h4>📊 Rank IC分析 (Spearman秩相关系数)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                            <th>说明</th>
                        </tr>
                    </thead>
                    <tbody>
                '''

                for key, name, desc in ic_metrics:
                    if key in rank_ic_stats:
                        value = rank_ic_stats[key]
                        if key in ['positive_ic_rate']:
                            formatted_value = f"{value:.2%}"
                        elif key in ['mean_ic', 'std_ic', 'ir', 'abs_mean_ic', 'ic_skewness', 'ic_kurtosis', 'min_ic',
                                     'max_ic']:
                            formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = str(value)

                        effectiveness_html += f'''
                        <tr>
                            <td>{name}</td>
                            <td class="{'positive' if key in ['mean_ic', 'ir', 'positive_ic_rate', 'abs_mean_ic'] and value > 0 else 'negative' if key in ['mean_ic', 'ir', 'positive_ic_rate'] and value < 0 else ''}">{formatted_value}</td>
                            <td>{desc}</td>
                        </tr>
                        '''

                effectiveness_html += '</tbody></table>'

            # 3. 分组收益分析表格
            if group_returns and 'group_stats' in group_returns:
                group_stats = group_returns['group_stats']
                effectiveness_html += '''
                <h4>📈 分组收益分析</h4>
                <table>
                    <thead>
                        <tr>
                            <th>分组</th>
                            <th>平均收益率</th>
                            <th>收益率标准差</th>
                            <th>夏普比率</th>
                            <th>胜率</th>
                            <th>样本数</th>
                        </tr>
                    </thead>
                    <tbody>
                '''

                for group_name in sorted(group_stats.keys()):
                    stats = group_stats[group_name]
                    effectiveness_html += f'''
                    <tr>
                        <td>{group_name}</td>
                        <td class="{'positive' if stats.get('mean_return', 0) > 0 else 'negative'}">{stats.get('mean_return', 0):.4f}</td>
                        <td>{stats.get('std_return', 0):.4f}</td>
                        <td class="{'positive' if stats.get('sharpe_ratio', 0) > 0 else 'negative'}">{stats.get('sharpe_ratio', 0):.4f}</td>
                        <td class="{'positive' if stats.get('win_rate', 0) > 0.5 else 'negative'}">{stats.get('win_rate', 0):.2%}</td>
                        <td>{stats.get('count', 0)}</td>
                    </tr>
                    '''

                effectiveness_html += '</tbody></table>'

            # 4. 稳定性指标表格
            if stability_metrics:
                effectiveness_html += '''
                <h4>🔒 稳定性指标</h4>
                <table>
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                            <th>说明</th>
                        </tr>
                    </thead>
                    <tbody>
                '''

                stability_metric_names = [
                    ('mean_change', '平均变化', '因子值平均变化幅度'),
                    ('std_change', '变化标准差', '因子值变化波动性'),
                    ('autocorr_1d', '1日自相关', '相邻日期因子值相关性'),
                    ('autocorr_5d', '5日自相关', '5天间隔因子值相关性'),
                    ('autocorr_20d', '20日自相关', '20天间隔因子值相关性')
                ]

                for key, name, desc in stability_metric_names:
                    if key in stability_metrics:
                        value = stability_metrics[key]
                        if pd.isna(value):
                            formatted_value = "N/A"
                        elif key.startswith('autocorr'):
                            formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = f"{value:.4f}"

                        effectiveness_html += f'''
                        <tr>
                            <td>{name}</td>
                            <td>{formatted_value}</td>
                            <td>{desc}</td>
                        </tr>
                        '''

                effectiveness_html += '</tbody></table>'

            # 5. 参数信息
            effectiveness_html += '''
            <h4>⚙️ 分析参数</h4>
            <table>
                <thead>
                    <tr>
                        <th>参数</th>
                        <th>数值</th>
                    </tr>
                </thead>
                <tbody>
            '''

            effectiveness_html += f'''
            <tr>
                <td>因子名称</td>
                <td>{effectiveness_data.get('factor_name', 'N/A')}</td>
            </tr>
            <tr>
                <td>预测期</td>
                <td>{effectiveness_data.get('forward_period', 'N/A')}</td>
            </tr>
            '''

            effectiveness_html += '</tbody></table>'

        effectiveness_html += '</div>'
        return effectiveness_html

    def _generate_quantstats_comprehensive_report(self,
                                                  factor_names: List[str],
                                                  report_path: str,
                                                  **kwargs) -> None:
        """
        生成QuantStats综合分析报告
        
        Args:
            factor_names: 因子名称列表
            report_path: 报告文件路径
            **kwargs: 其他参数
        """
        # 如果只有一个因子，直接生成简单报告
        if len(factor_names) == 1:
            logger.info("只有一个因子，生成简单报告")
            self._generate_simple_comprehensive_report_single(factor_names[0], report_path)
            return

        returns_dict = {}

        for factor_name in factor_names:
            try:
                # 获取TopN回测结果
                result_key = f'topn_{factor_name}'
                backtest_result = self.backtest_engine.get_backtest_results(result_key)

                if backtest_result is None and hasattr(self, '_framework_results'):
                    backtest_results = self._framework_results.get('backtest_results', {})
                    backtest_result = backtest_results.get(result_key)

                if backtest_result and 'portfolio' in backtest_result:
                    portfolio = backtest_result['portfolio']
                    returns = portfolio.returns()
                    if returns is not None and not returns.empty:
                        # 确保索引是datetime类型
                        if not isinstance(returns.index, pd.DatetimeIndex):
                            try:
                                returns.index = pd.to_datetime(returns.index)
                            except:
                                pass

                        # 清理数据
                        returns = returns.dropna()
                        if not returns.empty and len(returns) > 1:  # 确保有足够的数据
                            returns_dict[factor_name] = returns

            except Exception as e:
                logger.warning(f"获取因子 {factor_name} 收益率数据失败: {e}")
                continue

        if not returns_dict:
            raise ValueError("没有可用的收益率数据")

        # 确保所有Series有相同的索引
        try:
            # 找到所有Series的公共索引
            common_index = None
            for factor_name, returns in returns_dict.items():
                if common_index is None:
                    common_index = returns.index
                else:
                    common_index = common_index.intersection(returns.index)

            if common_index is None or len(common_index) == 0:
                raise ValueError("没有共同的日期索引")

            # 重新索引所有Series
            aligned_returns = {}
            for factor_name, returns in returns_dict.items():
                aligned_returns[factor_name] = returns.reindex(common_index).dropna()

            # 检查是否有有效数据
            valid_returns = {}
            for factor_name, returns in aligned_returns.items():
                if not returns.empty and len(returns) > 1:
                    valid_returns[factor_name] = returns

            if not valid_returns:
                raise ValueError("没有有效的收益率数据")

            # 创建收益率DataFrame
            returns_df = pd.DataFrame(valid_returns)

            # 确保数据不为空且有正确的索引
            if returns_df.empty:
                raise ValueError("没有可用的收益率数据")

            # 移除全为NaN的行
            returns_df = returns_df.dropna()

            if returns_df.empty:
                raise ValueError("处理后没有可用的收益率数据")

            # 尝试生成QuantStats对比报告
            try:
                qs.reports.html(returns_df,
                                output=report_path,
                                title="因子综合分析报告",
                                download_filename=os.path.basename(report_path),
                                benchmark=None)
                logger.info(f"QuantStats综合分析报告已生成: {report_path}")
            except Exception as e:
                logger.warning(f"QuantStats综合分析报告生成失败: {e}，生成简单对比报告")
                self._generate_simple_comprehensive_report(returns_df, report_path)

        except Exception as e:
            logger.error(f"处理收益率数据失败: {e}")
            # 生成简单的多因子对比报告
            logger.info("生成简单的多因子对比报告")
            self._generate_simple_comprehensive_report_multi(factor_names, report_path)

    def _generate_simple_comprehensive_report(self, returns_df: pd.DataFrame, report_path: str) -> None:
        """
        生成简单的综合分析报告
        
        Args:
            returns_df: 多因子收益率DataFrame
            report_path: 报告文件路径
        """
        try:
            # 计算各因子的基本统计指标
            stats_data = []
            for factor_name in returns_df.columns:
                returns = returns_df[factor_name].dropna()
                if len(returns) > 0:
                    total_return = (1 + returns).prod() - 1
                    annual_return = total_return * 252 / len(returns)
                    volatility = returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

                    # 计算最大回撤
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()

                    stats_data.append({
                        '因子名称': factor_name,
                        '总收益率': f"{total_return:.2%}",
                        '年化收益率': f"{annual_return:.2%}",
                        '年化波动率': f"{volatility:.2%}",
                        '夏普比率': f"{sharpe_ratio:.2f}",
                        '最大回撤': f"{max_drawdown:.2%}",
                        '交易天数': len(returns)
                    })

            stats_html = ""
            for stat in stats_data:
                stats_html += f"""
                <tr>
                    <td>{stat['因子名称']}</td>
                    <td>{stat['总收益率']}</td>
                    <td>{stat['年化收益率']}</td>
                    <td>{stat['年化波动率']}</td>
                    <td>{stat['夏普比率']}</td>
                    <td>{stat['最大回撤']}</td>
                    <td>{stat['交易天数']}</td>
                </tr>
                """

            # 生成HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>因子综合分析报告</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #007bff; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .chart-container {{ margin-top: 30px; }}
                    .info {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>因子综合分析报告</h1>
                        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                    
                    <h3>因子表现对比</h3>
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
                            {stats_html}
                        </tbody>
                    </table>
                    
                    <div class="chart-container">
                        <h3>累计收益率对比</h3>
                        <div id="comparison-chart"></div>
                    </div>
                    
                    <div class="info">
                        <h3>报告说明</h3>
                        <p>这是一个简化的综合分析报告，由于QuantStats报告生成失败而生成。包含各因子的基本统计指标对比和累计收益率走势图。</p>
                    </div>
                </div>
                
                <script>
                    // 生成累计收益率对比图表
                    const dates = {list(returns_df.index.strftime('%Y-%m-%d'))};
                    const traces = [];
                    
                    {chr(10).join([f'''
                    const cumulative_{factor_name.replace('-', '_')} = {list((1 + returns_df[factor_name]).cumprod())};
                    traces.push({{
                        x: dates,
                        y: cumulative_{factor_name.replace('-', '_')},
                        type: 'scatter',
                        mode: 'lines',
                        name: '{factor_name}',
                        line: {{ width: 2 }}
                    }});
                    ''' for factor_name in returns_df.columns])}
                    
                    const layout = {{
                        title: '因子累计收益率对比',
                        xaxis: {{ title: '日期' }},
                        yaxis: {{ title: '累计收益率' }},
                        hovermode: 'x unified'
                    }};
                    
                    Plotly.newPlot('comparison-chart', traces, layout);
                </script>
            </body>
            </html>
            """

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"简单综合分析报告已生成: {report_path}")

        except Exception as e:
            logger.error(f"简单综合分析报告生成失败: {e}")
            raise

    def _generate_simple_comprehensive_report_single(self, factor_name: str, report_path: str) -> None:
        """
        生成单个因子的简单综合分析报告
        
        Args:
            factor_name: 因子名称
            report_path: 报告文件路径
        """
        try:
            # 获取TopN回测结果
            result_key = f'topn_{factor_name}'
            backtest_result = self.backtest_engine.get_backtest_results(result_key)

            if backtest_result is None and hasattr(self, '_framework_results'):
                backtest_results = self._framework_results.get('backtest_results', {})
                backtest_result = backtest_results.get(result_key)

            if not backtest_result or 'portfolio' not in backtest_result:
                raise ValueError(f"无法获取因子 {factor_name} 的回测结果")

            portfolio = backtest_result['portfolio']
            returns = portfolio.returns()

            if returns is None or returns.empty:
                raise ValueError(f"因子 {factor_name} 没有收益率数据")

            # 生成简单的HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>因子综合分析报告 - {factor_name}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
                    .info {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>因子综合分析报告</h1>
                        <h2>{factor_name}</h2>
                        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                    
                    <div class="info">
                        <h3>报告说明</h3>
                        <p>这是因子 {factor_name} 的简单综合分析报告。由于只有一个因子，无法进行多因子对比分析。</p>
                        <p>请查看对应的单个因子回测报告以获取详细的回测结果。</p>
                    </div>
                </div>
            </body>
            </html>
            """

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"简单综合分析报告已生成: {report_path}")

        except Exception as e:
            logger.error(f"简单综合分析报告生成失败: {e}")
            raise

    def _generate_quantstats_report(self,
                                    backtest_result: Dict[str, Any],
                                    result_key: str,
                                    output_dir: str,
                                    **kwargs) -> str:
        """
        使用QuantStats生成HTML回测报告
        
        Args:
            backtest_result: 回测结果
            result_key: 结果键名
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            HTML报告文件路径
        """
        try:
            # 检查是否是分组回测结果
            if 'portfolios' in backtest_result:
                # 分组回测结果，生成分组对比报告
                return self._generate_group_backtest_report(backtest_result, result_key, output_dir, **kwargs)

            # 获取portfolio对象
            portfolio = backtest_result.get('portfolio')
            if portfolio is None:
                raise ValueError(f"回测结果 {result_key} 中没有portfolio对象")

            # 获取收益率序列
            returns = portfolio.returns()
            if returns is None or returns.empty:
                raise ValueError("收益率数据为空")

            # 确保收益率数据格式正确
            if isinstance(returns, pd.Series):
                # 确保索引是datetime类型
                if not isinstance(returns.index, pd.DatetimeIndex):
                    try:
                        returns.index = pd.to_datetime(returns.index)
                    except:
                        pass

                # 转换为DataFrame，确保列名为'Strategy'
                returns = returns.to_frame('Strategy')

            # 确保数据不为空且有效
            if returns.empty or returns.isnull().all().all():
                raise ValueError("收益率数据为空或全为NaN")

            # 移除全为NaN的行
            returns = returns.dropna()

            if returns.empty:
                raise ValueError("处理后收益率数据为空")

            # 确保至少有一些非零收益率
            if (returns == 0).all().all():
                logger.warning("所有收益率都为0，可能影响报告质量")

            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"backtest_report_{result_key}_{timestamp}.html"
            report_path = os.path.join(output_dir, report_filename)

            # 尝试生成QuantStats报告
            try:
                qs.reports.html(returns,
                                output=report_path,
                                title=f"因子回测报告 - {result_key}",
                                download_filename=report_filename,
                                benchmark=None)
                logger.info(f"QuantStats HTML报告已生成: {report_path}")
                return report_path
            except Exception as qs_error:
                logger.warning(f"QuantStats报告生成失败: {qs_error}，生成简单HTML报告")
                return self._generate_simple_html_report(returns, result_key, report_path)

        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            # 生成一个简单的错误报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"error_report_{result_key}_{timestamp}.html"
            report_path = os.path.join(output_dir, report_filename)

            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>回测报告生成失败 - {result_key}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .error {{ color: red; background-color: #ffe6e6; padding: 20px; border-radius: 5px; }}
                    .info {{ background-color: #e6f3ff; padding: 20px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h1>因子回测报告 - {result_key}</h1>
                <div class="error">
                    <h2>报告生成失败</h2>
                    <p><strong>错误信息:</strong> {str(e)}</p>
                    <p><strong>时间:</strong> {timestamp}</p>
                </div>
                <div class="info">
                    <h3>可能的原因:</h3>
                    <ul>
                        <li>收益率数据格式问题</li>
                        <li>数据为空或全为NaN</li>
                        <li>QuantStats库兼容性问题</li>
                    </ul>
                </div>
            </body>
            </html>
            """

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(error_html)

            logger.info(f"错误报告已生成: {report_path}")
            return report_path

    def _generate_simple_html_report(self, returns: pd.DataFrame, result_key: str, report_path: str) -> str:
        """
        生成简单的HTML回测报告
        
        Args:
            returns: 收益率数据
            result_key: 结果键名
            report_path: 报告文件路径
            
        Returns:
            报告文件路径
        """
        try:
            # 确保returns是DataFrame格式
            if isinstance(returns, pd.Series):
                returns = returns.to_frame('Strategy')

            # 获取第一列数据
            returns_series = returns.iloc[:, 0]

            # 计算基本统计指标
            total_return = (1 + returns_series).prod() - 1
            annual_return = total_return * 252 / len(returns_series) if len(returns_series) > 0 else 0
            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            # 计算最大回撤
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 生成HTML报告
            html_content = f"""
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
                        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{total_return:.2%}</div>
                            <div class="stat-label">总收益率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{annual_return:.2%}</div>
                            <div class="stat-label">年化收益率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{volatility:.2%}</div>
                            <div class="stat-label">年化波动率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{sharpe_ratio:.2f}</div>
                            <div class="stat-label">夏普比率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{max_drawdown:.2%}</div>
                            <div class="stat-label">最大回撤</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{len(returns_series)}</div>
                            <div class="stat-label">交易天数</div>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>累计收益率走势</h3>
                        <div id="cumulative-chart"></div>
                    </div>
                    
                    <div class="info">
                        <h3>报告说明</h3>
                        <p>这是一个简化的回测报告，由于QuantStats报告生成失败而生成。包含基本的回测统计指标和收益率走势图。</p>
                    </div>
                </div>
                
                <script>
                    // 生成累计收益率图表
                    const dates = {list(returns_series.index.strftime('%Y-%m-%d'))};
                    const cumulative = {list((1 + returns_series).cumprod())};
                    
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

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"简单HTML报告已生成: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"简单HTML报告生成失败: {e}")
            raise

    def _generate_simple_comprehensive_report_multi(self, factor_names: List[str], report_path: str) -> None:
        """
        生成简单的多因子综合分析报告
        
        Args:
            factor_names: 因子名称列表
            report_path: 报告文件路径
        """
        try:
            # 收集各因子的基本统计信息
            factor_stats = []

            for factor_name in factor_names:
                try:
                    # 获取TopN回测结果
                    result_key = f'topn_{factor_name}'
                    backtest_result = self.backtest_engine.get_backtest_results(result_key)

                    if backtest_result is None and hasattr(self, '_framework_results'):
                        backtest_results = self._framework_results.get('backtest_results', {})
                        backtest_result = backtest_results.get(result_key)

                    if backtest_result and 'portfolio' in backtest_result:
                        portfolio = backtest_result['portfolio']
                        returns = portfolio.returns()

                        if returns is not None and not returns.empty:
                            # 计算基本统计指标
                            total_return = (1 + returns).prod() - 1
                            annual_return = total_return * 252 / len(returns) if len(returns) > 0 else 0
                            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
                            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

                            # 计算最大回撤
                            cumulative = (1 + returns).cumprod()
                            running_max = cumulative.expanding().max()
                            drawdown = (cumulative - running_max) / running_max
                            max_drawdown = drawdown.min()

                            factor_stats.append({
                                '因子名称': factor_name,
                                '总收益率': f"{total_return:.2%}",
                                '年化收益率': f"{annual_return:.2%}",
                                '年化波动率': f"{volatility:.2%}",
                                '夏普比率': f"{sharpe_ratio:.2f}",
                                '最大回撤': f"{max_drawdown:.2%}",
                                '交易天数': len(returns)
                            })
                        else:
                            factor_stats.append({
                                '因子名称': factor_name,
                                '总收益率': 'N/A',
                                '年化收益率': 'N/A',
                                '年化波动率': 'N/A',
                                '夏普比率': 'N/A',
                                '最大回撤': 'N/A',
                                '交易天数': 0
                            })
                    else:
                        factor_stats.append({
                            '因子名称': factor_name,
                            '总收益率': 'N/A',
                            '年化收益率': 'N/A',
                            '年化波动率': 'N/A',
                            '夏普比率': 'N/A',
                            '最大回撤': 'N/A',
                            '交易天数': 0
                        })

                except Exception as e:
                    logger.warning(f"获取因子 {factor_name} 统计信息失败: {e}")
                    factor_stats.append({
                        '因子名称': factor_name,
                        '总收益率': 'N/A',
                        '年化收益率': 'N/A',
                        '年化波动率': 'N/A',
                        '夏普比率': 'N/A',
                        '最大回撤': 'N/A',
                        '交易天数': 0
                    })

            # 生成统计表格HTML
            stats_html = ""
            for stat in factor_stats:
                stats_html += f"""
                <tr>
                    <td>{stat['因子名称']}</td>
                    <td>{stat['总收益率']}</td>
                    <td>{stat['年化收益率']}</td>
                    <td>{stat['年化波动率']}</td>
                    <td>{stat['夏普比率']}</td>
                    <td>{stat['最大回撤']}</td>
                    <td>{stat['交易天数']}</td>
                </tr>
                """

            # 生成HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>因子综合分析报告</title>
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
                        <h1>因子综合分析报告</h1>
                        <p>分析因子: {', '.join(factor_names)}</p>
                        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                    
                    <h3>因子表现对比</h3>
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
                            {stats_html}
                        </tbody>
                    </table>
                    
                    <div class="info">
                        <h3>报告说明</h3>
                        <p>这是一个简化的多因子综合分析报告，由于数据格式问题无法生成完整的QuantStats对比报告。</p>
                        <p>请查看对应的单个因子回测报告以获取详细的回测结果和图表。</p>
                    </div>
                </div>
            </body>
            </html>
            """

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"简单多因子综合分析报告已生成: {report_path}")

        except Exception as e:
            logger.error(f"简单多因子综合分析报告生成失败: {e}")
            raise

    def get_report_summary(self) -> pd.DataFrame:
        """
        获取报告摘要
        
        Returns:
            报告摘要DataFrame
        """
        summary_data = []

        for report_type, reports in self._reports.items():
            for report_name, report_info in reports.items():
                summary_data.append({
                    '报告类型': report_type,
                    '报告名称': report_name,
                    '生成时间': report_info.get('timestamp', ''),
                    '文件路径': report_info.get('file_path', '')
                })

        return pd.DataFrame(summary_data)

    def _generate_group_backtest_report(self,
                                        backtest_result: Dict[str, Any],
                                        result_key: str,
                                        output_dir: str,
                                        **kwargs) -> str:
        """
        生成分组回测报告
        
        Args:
            backtest_result: 分组回测结果
            result_key: 结果键名
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            报告文件路径
        """
        try:
            portfolios = backtest_result.get('portfolios', {})
            stats = backtest_result.get('stats', pd.DataFrame())

            if not portfolios:
                raise ValueError("分组回测结果中没有portfolio数据")

            # 收集各组的收益率数据
            returns_dict = {}
            for group_name, portfolio in portfolios.items():
                try:
                    returns = portfolio.returns()
                    if returns is not None and not returns.empty:
                        # 确保索引是datetime类型
                        if not isinstance(returns.index, pd.DatetimeIndex):
                            try:
                                returns.index = pd.to_datetime(returns.index)
                            except:
                                pass

                        # 清理数据
                        returns = returns.dropna()
                        if not returns.empty and len(returns) > 1:
                            returns_dict[group_name] = returns
                except Exception as e:
                    logger.warning(f"获取分组 {group_name} 收益率数据失败: {e}")
                    continue

            if not returns_dict:
                raise ValueError("没有可用的分组收益率数据")

            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"group_backtest_report_{result_key}_{timestamp}.html"
            report_path = os.path.join(output_dir, report_filename)

            # 尝试生成QuantStats对比报告
            try:
                # 创建收益率DataFrame
                returns_df = pd.DataFrame(returns_dict)

                # 确保数据不为空
                if returns_df.empty:
                    raise ValueError("分组收益率数据为空")

                # 移除全为NaN的行
                returns_df = returns_df.dropna()

                if returns_df.empty:
                    raise ValueError("处理后分组收益率数据为空")

                qs.reports.html(returns_df,
                                output=report_path,
                                title=f"分组回测报告 - {result_key}",
                                download_filename=report_filename,
                                benchmark=None)
                logger.info(f"分组QuantStats HTML报告已生成: {report_path}")
                return report_path
            except Exception as e:
                logger.warning(f"分组QuantStats报告生成失败: {e}，生成简单分组报告")
                return self._generate_simple_group_report(backtest_result, result_key, report_path)

        except Exception as e:
            logger.error(f"分组报告生成失败: {e}")
            # 生成错误报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"error_report_{result_key}_{timestamp}.html"
            report_path = os.path.join(output_dir, report_filename)

            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>分组回测报告生成失败 - {result_key}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .error {{ color: red; background-color: #ffe6e6; padding: 20px; border-radius: 5px; }}
                    .info {{ background-color: #e6f3ff; padding: 20px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h1>分组回测报告 - {result_key}</h1>
                <div class="error">
                    <h2>报告生成失败</h2>
                    <p><strong>错误信息:</strong> {str(e)}</p>
                    <p><strong>时间:</strong> {timestamp}</p>
                </div>
                <div class="info">
                    <h3>可能的原因:</h3>
                    <ul>
                        <li>分组收益率数据格式问题</li>
                        <li>数据为空或全为NaN</li>
                        <li>QuantStats库兼容性问题</li>
                    </ul>
                </div>
            </body>
            </html>
            """

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(error_html)

            logger.info(f"分组错误报告已生成: {report_path}")
            return report_path

    def _generate_simple_group_report(self,
                                      backtest_result: Dict[str, Any],
                                      result_key: str,
                                      report_path: str) -> str:
        """
        生成简单的分组回测报告
        
        Args:
            backtest_result: 分组回测结果
            result_key: 结果键名
            report_path: 报告文件路径
            
        Returns:
            报告文件路径
        """
        try:
            portfolios = backtest_result.get('portfolios', {})
            stats = backtest_result.get('stats', pd.DataFrame())

            # 收集各组的统计信息
            group_stats = []
            for group_name, portfolio in portfolios.items():
                try:
                    returns = portfolio.returns()
                    if returns is not None and not returns.empty:
                        # 处理returns数据，确保是1D Series
                        if isinstance(returns, pd.Series):
                            returns_series = returns.dropna()
                        elif isinstance(returns, pd.DataFrame):
                            # 如果是DataFrame，取第一列或计算平均值
                            if returns.shape[1] == 1:
                                returns_series = returns.iloc[:, 0].dropna()
                            else:
                                # 多列数据，计算平均值作为组合收益率
                                returns_series = returns.mean(axis=1).dropna()
                        elif isinstance(returns, np.ndarray):
                            # 如果是numpy数组，转换为Series
                            if returns.ndim == 1:
                                returns_series = pd.Series(returns).dropna()
                            else:
                                # 2D数组，计算平均值
                                returns_series = pd.Series(returns.mean(axis=1)).dropna()
                        else:
                            returns_series = pd.Series(returns).dropna()

                        if len(returns_series) > 0:
                            # 计算基本统计指标
                            total_return = (1 + returns_series).prod() - 1
                            annual_return = total_return * 252 / len(returns_series) if len(returns_series) > 0 else 0
                            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
                            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

                            # 计算最大回撤
                            cumulative = (1 + returns_series).cumprod()
                            running_max = cumulative.expanding().max()
                            drawdown = (cumulative - running_max) / running_max
                            max_drawdown = drawdown.min()

                            group_stats.append({
                                '分组名称': group_name,
                                '总收益率': f"{total_return:.2%}",
                                '年化收益率': f"{annual_return:.2%}",
                                '年化波动率': f"{volatility:.2%}",
                                '夏普比率': f"{sharpe_ratio:.2f}",
                                '最大回撤': f"{max_drawdown:.2%}",
                                '交易天数': len(returns_series)
                            })
                        else:
                            group_stats.append({
                                '分组名称': group_name,
                                '总收益率': 'N/A',
                                '年化收益率': 'N/A',
                                '年化波动率': 'N/A',
                                '夏普比率': 'N/A',
                                '最大回撤': 'N/A',
                                '交易天数': 0
                            })
                    else:
                        group_stats.append({
                            '分组名称': group_name,
                            '总收益率': 'N/A',
                            '年化收益率': 'N/A',
                            '年化波动率': 'N/A',
                            '夏普比率': 'N/A',
                            '最大回撤': 'N/A',
                            '交易天数': 0
                        })
                except Exception as e:
                    logger.warning(f"计算分组 {group_name} 统计信息失败: {e}")
                    group_stats.append({
                        '分组名称': group_name,
                        '总收益率': 'N/A',
                        '年化收益率': 'N/A',
                        '年化波动率': 'N/A',
                        '夏普比率': 'N/A',
                        '最大回撤': 'N/A',
                        '交易天数': 0
                    })

            # 生成统计表格HTML
            stats_html = ""
            for stat in group_stats:
                stats_html += f"""
                <tr>
                    <td>{stat['分组名称']}</td>
                    <td>{stat['总收益率']}</td>
                    <td>{stat['年化收益率']}</td>
                    <td>{stat['年化波动率']}</td>
                    <td>{stat['夏普比率']}</td>
                    <td>{stat['最大回撤']}</td>
                    <td>{stat['交易天数']}</td>
                </tr>
                """

            # 生成HTML报告
            html_content = f"""
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
                        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
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
                        <p>这是一个简化的分组回测报告，由于数据格式问题无法生成完整的QuantStats对比报告。</p>
                        <p>请查看对应的单个因子回测报告以获取详细的回测结果和图表。</p>
                    </div>
                </div>
            </body>
            </html>
            """

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"简单分组回测报告已生成: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"简单分组报告生成失败: {e}")
            raise

    def _generate_merged_summary_html(self,
                                      factor_names: List[str],
                                      merged_results: Dict[str, Any],
                                      analysis_summary: Dict[str, Any],
                                      start_date: str,
                                      end_date: str,
                                      stock_pool: str = "no_st",
                                      top_n: int = 10,
                                      n_groups: int = 5) -> str:
        """
        生成合并的综合报告HTML内容
        
        Args:
            factor_names: 因子名称列表
            merged_results: 合并的结果字典
            analysis_summary: 分析总结字典
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            top_n: 选股数量
            n_groups: 分组数量
            
        Returns:
            HTML内容字符串
        """
        logger.info("生成合并综合报告HTML内容...")

        # 收集数据
        backtest_results = merged_results.get('backtest_results', {})
        effectiveness_results = merged_results.get('effectiveness_results', {})

        # 生成分析总结HTML
        analysis_html = self._generate_analysis_summary_html(analysis_summary)

        # 生成TopN回测结果HTML
        topn_html = self._generate_topn_results_html(factor_names, backtest_results)

        # 生成分组回测结果HTML
        group_html = self._generate_group_results_html(factor_names, backtest_results)

        # 生成多因子回测结果HTML
        multifactor_html = self._generate_multifactor_results_html(backtest_results)

        # 生成IC分析结果HTML
        ic_html = self._generate_ic_results_html(factor_names, effectiveness_results)

        # 生成有效性分析结果HTML
        effectiveness_html = self._generate_effectiveness_results_html(factor_names, effectiveness_results)

        # 生成完整的HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>因子分析汇总报告</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background-color: white; 
                    border-radius: 10px; 
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    text-align: center;
                }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                
                .nav {{ 
                    background-color: #2c3e50; 
                    padding: 0; 
                    position: sticky; 
                    top: 0; 
                    z-index: 100;
                }}
                .nav ul {{ 
                    list-style: none; 
                    margin: 0; 
                    padding: 0; 
                    display: flex; 
                    flex-wrap: wrap;
                }}
                .nav li {{ margin: 0; }}
                .nav a {{ 
                    display: block; 
                    color: white; 
                    text-decoration: none; 
                    padding: 15px 20px; 
                    transition: background-color 0.3s;
                }}
                .nav a:hover {{ background-color: #34495e; }}
                
                .content {{ padding: 30px; }}
                .section {{ 
                    margin-bottom: 40px; 
                    padding: 25px; 
                    border-radius: 8px; 
                    background-color: #f8f9fa; 
                    border-left: 4px solid #007bff;
                }}
                .section h2 {{ 
                    color: #2c3e50; 
                    margin-top: 0; 
                    border-bottom: 2px solid #e9ecef; 
                    padding-bottom: 10px;
                }}
                
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 15px; 
                    background-color: white;
                    border-radius: 5px;
                    overflow: hidden;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    padding: 12px 15px; 
                    text-align: left; 
                    border-bottom: 1px solid #dee2e6; 
                }}
                th {{ 
                    background-color: #007bff; 
                    color: white; 
                    font-weight: bold;
                }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e9ecef; }}
                
                .metric-card {{ 
                    display: inline-block; 
                    background: white; 
                    padding: 15px; 
                    margin: 10px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    min-width: 150px;
                    text-align: center;
                }}
                .metric-value {{ 
                    font-size: 1.5em; 
                    font-weight: bold; 
                    color: #007bff; 
                }}
                .metric-label {{ 
                    color: #6c757d; 
                    font-size: 0.9em; 
                    margin-top: 5px;
                }}
                
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                
                .chart-container {{ 
                    margin-top: 20px; 
                    padding: 20px; 
                    background-color: white; 
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                
                .info-box {{ 
                    background-color: #e7f3ff; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-top: 15px;
                    border-left: 4px solid #007bff;
                }}
                
                .analysis-summary {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                
                .recommendation {{
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                
                .warning {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                
                @media (max-width: 768px) {{
                    .nav ul {{ flex-direction: column; }}
                    .nav a {{ text-align: center; }}
                    .metric-card {{ min-width: 120px; }}
                }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>因子分析汇总报告</h1>
                    <p>分析因子: {', '.join(factor_names)} | 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <nav class="nav">
                    <ul>
                        <li><a href="#overview">总览</a></li>
                        <li><a href="#analysis">分析总结</a></li>
                        <li><a href="#topn">TopN回测</a></li>
                        <li><a href="#group">分组回测</a></li>
                        <li><a href="#multifactor">多因子回测</a></li>
                        <li><a href="#ic">IC分析</a></li>
                        <li><a href="#effectiveness">有效性分析</a></li>
                    </ul>
                </nav>
                
                <div class="content">
                    
        <div id="overview" class="section">
            <h2>📊 分析总览</h2>
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
        
        <div id="analysis" class="section">
            <h2>📈 分析总结</h2>
            {analysis_html}
        </div>
        
        <div id="topn" class="section">
            <h2>📈 TopN回测结果</h2>
            {topn_html}
        </div>
        
        <div id="group" class="section">
            <h2>📊 分组回测结果</h2>
            {group_html}
        </div>
        
        <div id="multifactor" class="section">
            <h2>🔗 多因子回测结果</h2>
            {multifactor_html}
        </div>
        
        <div id="ic" class="section">
            <h2>📊 IC分析结果</h2>
            {ic_html}
        </div>
        
        <div id="effectiveness" class="section">
            <h2>📈 有效性分析结果</h2>
            {effectiveness_html}
        </div>
        
                </div>
            </div>
            
            <script>
                // 平滑滚动到锚点
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                    anchor.addEventListener('click', function (e) {{
                        e.preventDefault();
                        const target = document.querySelector(this.getAttribute('href'));
                        if (target) {{
                            target.scrollIntoView({{
                                behavior: 'smooth',
                                block: 'start'
                            }});
                        }}
                    }});
                }});
            </script>
        </body>
        </html>
        """

        return html_content

    def _generate_analysis_summary_html(self, analysis_summary: Dict[str, Any]) -> str:
        """生成分析总结HTML"""
        html = f"""
        <div class="analysis-summary">
            <h3>📊 总体统计</h3>
            <div class="metric-card">
                <div class="metric-value">{analysis_summary['total_factors']}</div>
                <div class="metric-label">总因子数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(analysis_summary['successful_factors'])}</div>
                <div class="metric-label">成功因子</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(analysis_summary['failed_factors'])}</div>
                <div class="metric-label">失败因子</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis_summary['success_rate']:.1%}</div>
                <div class="metric-label">成功率</div>
            </div>
        </div>
        """

        # 优秀因子
        if analysis_summary['top_performers']:
            html += """
            <div class="recommendation">
                <h3>🏆 优秀因子</h3>
                <table>
                    <thead>
                        <tr>
                            <th>因子名称</th>
                            <th>夏普比率</th>
                            <th>总收益率</th>
                            <th>IC均值</th>
                            <th>IC IR</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for factor in analysis_summary['top_performers']:
                html += f"""
                <tr>
                    <td>{factor['factor_name']}</td>
                    <td class="positive">{factor['sharpe_ratio']:.2f}</td>
                    <td class="positive">{factor['total_return']:.2%}</td>
                    <td class="positive">{factor['ic_mean']:.4f}</td>
                    <td class="positive">{factor['ic_ir']:.4f}</td>
                </tr>
                """

            html += """
                    </tbody>
                </table>
            </div>
            """

        # 问题因子
        if analysis_summary['problematic_factors']:
            html += """
            <div class="warning">
                <h3>⚠️ 问题因子</h3>
                <table>
                    <thead>
                        <tr>
                            <th>因子名称</th>
                            <th>问题类型</th>
                            <th>夏普比率</th>
                            <th>IC均值</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for factor in analysis_summary['problematic_factors']:
                html += f"""
                <tr>
                    <td>{factor['factor_name']}</td>
                    <td>{factor['issue']}</td>
                    <td class="negative">{factor['sharpe_ratio']:.2f}</td>
                    <td class="negative">{factor['ic_mean']:.4f}</td>
                </tr>
                """

            html += """
                    </tbody>
                </table>
            </div>
            """

        # 建议
        if analysis_summary['recommendations']:
            html += """
            <div class="info-box">
                <h3>💡 建议</h3>
                <ul>
            """

            for recommendation in analysis_summary['recommendations']:
                html += f"<li>{recommendation}</li>"

            html += """
                </ul>
            </div>
            """

        return html

    def _generate_topn_results_html(self, factor_names: List[str], backtest_results: Dict[str, Any]) -> str:
        """生成TopN回测结果HTML"""
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
            topn_key = f'topn_{factor_name}'
            if topn_key in backtest_results:
                result = backtest_results[topn_key]
                # 兼容不同的数据结构
                stats = None
                if 'portfolio_stats' in result:
                    stats = result['portfolio_stats']
                elif 'stats' in result:
                    stats = result['stats']
                
                if stats is not None:
                    # 兼容不同的字段名称
                    total_return = stats.get('total_return', stats.get('Total Return [%]', 0)) / 100 if isinstance(stats.get('total_return', stats.get('Total Return [%]', 0)), (int, float)) else 0
                    annual_return = stats.get('annual_return', stats.get('Annual Return [%]', 0)) / 100 if isinstance(stats.get('annual_return', stats.get('Annual Return [%]', 0)), (int, float)) else 0
                    annual_volatility = stats.get('annual_volatility', stats.get('Annual Volatility [%]', 0)) / 100 if isinstance(stats.get('annual_volatility', stats.get('Annual Volatility [%]', 0)), (int, float)) else 0
                    sharpe_ratio = stats.get('sharpe_ratio', stats.get('Sharpe Ratio', 0))
                    max_drawdown = stats.get('max_drawdown', stats.get('Max Drawdown [%]', 0)) / 100 if isinstance(stats.get('max_drawdown', stats.get('Max Drawdown [%]', 0)), (int, float)) else 0
                    trading_days = stats.get('trading_days', stats.get('Period', 0))

                    html += f"""
                    <tr>
                        <td>{factor_name}</td>
                        <td class="{'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</td>
                        <td class="{'positive' if annual_return > 0 else 'negative'}">{annual_return:.2%}</td>
                        <td>{annual_volatility:.2%}</td>
                        <td class="{'positive' if sharpe_ratio > 0 else 'negative'}">{sharpe_ratio:.2f}</td>
                        <td class="negative">{max_drawdown:.2%}</td>
                        <td>{trading_days}</td>
                    </tr>
                    """
            else:
                html += f"""
                <tr>
                    <td>{factor_name}</td>
                    <td class="negative">N/A</td>
                    <td class="negative">N/A</td>
                    <td>N/A</td>
                    <td class="negative">N/A</td>
                    <td class="negative">N/A</td>
                    <td>N/A</td>
                </tr>
                """

        html += """
            </tbody>
        </table>
        """

        return html

    def _generate_group_results_html(self, factor_names: List[str], backtest_results: Dict[str, Any]) -> str:
        """生成分组回测结果HTML"""
        html = ""

        for factor_name in factor_names:
            group_key = f'group_{factor_name}'
            if group_key in backtest_results:
                result = backtest_results[group_key]
                # 兼容不同的数据结构
                group_stats = None
                if 'group_stats' in result:
                    group_stats = result['group_stats']
                elif 'stats' in result:
                    group_stats = result['stats']
                
                if group_stats is not None:

                    html += f"""
                    <h3>{factor_name} 分组表现</h3>
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
                    """

                    for _, row in group_stats.iterrows():
                        # 兼容不同的字段名称
                        total_return = row.get('total_return', row.get('Total Return [%]', 0)) / 100 if isinstance(row.get('total_return', row.get('Total Return [%]', 0)), (int, float)) else 0
                        annual_return = row.get('annual_return', row.get('Annual Return [%]', 0)) / 100 if isinstance(row.get('annual_return', row.get('Annual Return [%]', 0)), (int, float)) else 0
                        annual_volatility = row.get('annual_volatility', row.get('Annual Volatility [%]', 0)) / 100 if isinstance(row.get('annual_volatility', row.get('Annual Volatility [%]', 0)), (int, float)) else 0
                        sharpe_ratio = row.get('sharpe_ratio', row.get('Sharpe Ratio', 0))
                        max_drawdown = row.get('max_drawdown', row.get('Max Drawdown [%]', 0)) / 100 if isinstance(row.get('max_drawdown', row.get('Max Drawdown [%]', 0)), (int, float)) else 0
                        trading_days = row.get('trading_days', row.get('Period', 0))

                        html += f"""
                        <tr>
                            <td>{row.get('group_name', 'N/A')}</td>
                            <td class="{'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</td>
                            <td class="{'positive' if annual_return > 0 else 'negative'}">{annual_return:.2%}</td>
                            <td>{annual_volatility:.2%}</td>
                            <td class="{'positive' if sharpe_ratio > 0 else 'negative'}">{sharpe_ratio:.2f}</td>
                            <td class="negative">{max_drawdown:.2%}</td>
                            <td>{trading_days}</td>
                        </tr>
                        """

                    html += """
                        </tbody>
                    </table>
                    """

        return html

    def _generate_multifactor_results_html(self, backtest_results: Dict[str, Any]) -> str:
        """生成多因子回测结果HTML"""
        if 'multifactor' in backtest_results:
            result = backtest_results['multifactor']
            # 兼容不同的数据结构
            stats = None
            if 'portfolio_stats' in result:
                stats = result['portfolio_stats']
            elif 'stats' in result:
                stats = result['stats']
            
            if stats is not None:

                # 兼容不同的字段名称
                total_return = stats.get('total_return', stats.get('Total Return [%]', 0)) / 100 if isinstance(stats.get('total_return', stats.get('Total Return [%]', 0)), (int, float)) else 0
                annual_return = stats.get('annual_return', stats.get('Annual Return [%]', 0)) / 100 if isinstance(stats.get('annual_return', stats.get('Annual Return [%]', 0)), (int, float)) else 0
                annual_volatility = stats.get('annual_volatility', stats.get('Annual Volatility [%]', 0)) / 100 if isinstance(stats.get('annual_volatility', stats.get('Annual Volatility [%]', 0)), (int, float)) else 0
                sharpe_ratio = stats.get('sharpe_ratio', stats.get('Sharpe Ratio', 0))
                max_drawdown = stats.get('max_drawdown', stats.get('Max Drawdown [%]', 0)) / 100 if isinstance(stats.get('max_drawdown', stats.get('Max Drawdown [%]', 0)), (int, float)) else 0
                trading_days = stats.get('trading_days', stats.get('Period', 0))

                html = f"""
                <div class="metric-card">
                    <div class="metric-value {'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if annual_return > 0 else 'negative'}">{annual_return:.2%}</div>
                    <div class="metric-label">年化收益率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{annual_volatility:.2%}</div>
                    <div class="metric-label">年化波动率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if sharpe_ratio > 0 else 'negative'}">{sharpe_ratio:.2f}</div>
                    <div class="metric-label">夏普比率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value negative">{max_drawdown:.2%}</div>
                    <div class="metric-label">最大回撤</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{trading_days}</div>
                    <div class="metric-label">交易天数</div>
                </div>
                """
                return html

        return "<p>暂无多因子回测结果</p>"

    def _generate_ic_results_html(self, factor_names: List[str], effectiveness_results: Dict[str, Any]) -> str:
        """生成IC分析结果HTML"""
        html = """
        <table>
            <thead>
                <tr>
                    <th>因子名称</th>
                    <th>Pearson IC</th>
                    <th>IC IR</th>
                    <th>IC胜率</th>
                </tr>
            </thead>
            <tbody>
        """

        for factor_name in factor_names:
            if factor_name in effectiveness_results:
                result = effectiveness_results[factor_name]
                if 'ic_analysis' in result:
                    ic_analysis = result['ic_analysis']
                    ic_mean = ic_analysis.get('ic_mean', 0)
                    ic_ir = ic_analysis.get('ic_ir', 0)
                    ic_win_rate = ic_analysis.get('ic_win_rate', 0)

                    html += f"""
                    <tr>
                        <td>{factor_name}</td>
                        <td class="{'positive' if ic_mean > 0 else 'negative'}">{ic_mean:.4f}</td>
                        <td class="{'positive' if ic_ir > 0 else 'negative'}">{ic_ir:.4f}</td>
                        <td class="{'positive' if ic_win_rate > 0.5 else 'negative'}">{ic_win_rate:.2%}</td>
                    </tr>
                    """
            else:
                html += f"""
                <tr>
                    <td>{factor_name}</td>
                    <td class="negative">nan</td>
                    <td class="negative">nan</td>
                    <td class="negative">0.00%</td>
                </tr>
                """

        html += """
            </tbody>
        </table>
        """

        return html

    def _generate_effectiveness_results_html(self, factor_names: List[str],
                                             effectiveness_results: Dict[str, Any]) -> str:
        """生成有效性分析结果HTML"""
        html = ""

        for factor_name in factor_names:
            if factor_name in effectiveness_results:
                result = effectiveness_results[factor_name]

                html += f"""
                <h3>{factor_name} 有效性指标</h3>
                """

                # IC分析
                if 'ic_analysis' in result:
                    ic_analysis = result['ic_analysis']
                    html += """
                    <h4>📊 IC分析 (Pearson相关系数)</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                                <th>说明</th>
                            </tr>
                        </thead>
                        <tbody>
                    """

                    for key, value in ic_analysis.items():
                        if isinstance(value, float):
                            display_value = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
                        else:
                            display_value = str(value)

                        html += f"""
                        <tr>
                            <td>{key}</td>
                            <td class="{'positive' if isinstance(value, (int, float)) and value > 0 else ''}">{display_value}</td>
                            <td>IC分析指标</td>
                        </tr>
                        """

                    html += """
                        </tbody>
                    </table>
                    """

                # Rank IC分析
                if 'rank_ic_analysis' in result:
                    rank_ic_analysis = result['rank_ic_analysis']
                    html += """
                    <h4>📊 Rank IC分析 (Spearman秩相关系数)</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                                <th>说明</th>
                            </tr>
                        </thead>
                        <tbody>
                    """

                    for key, value in rank_ic_analysis.items():
                        if isinstance(value, float):
                            display_value = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
                        else:
                            display_value = str(value)

                        html += f"""
                        <tr>
                            <td>{key}</td>
                            <td class="{'positive' if isinstance(value, (int, float)) and value > 0 else ''}">{display_value}</td>
                            <td>Rank IC分析指标</td>
                        </tr>
                        """

                    html += """
                        </tbody>
                    </table>
                    """

                # 分组收益分析
                if 'group_analysis' in result:
                    group_analysis = result['group_analysis']
                    html += """
                    <h4>📈 分组收益分析</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>分组</th>
                                <th>平均收益率</th>
                                <th>收益率标准差</th>
                                <th>夏普比率</th>
                                <th>胜率</th>
                                <th>样本数</th>
                            </tr>
                        </thead>
                        <tbody>
                    """

                    for group_name, group_data in group_analysis.items():
                        html += f"""
                        <tr>
                            <td>{group_name}</td>
                            <td class="{'positive' if group_data.get('mean_return', 0) > 0 else 'negative'}">{group_data.get('mean_return', 0):.4f}</td>
                            <td>{group_data.get('std_return', 0):.4f}</td>
                            <td class="{'positive' if group_data.get('sharpe_ratio', 0) > 0 else 'negative'}">{group_data.get('sharpe_ratio', 0):.4f}</td>
                            <td class="{'positive' if group_data.get('win_rate', 0) > 0.5 else 'negative'}">{group_data.get('win_rate', 0):.2%}</td>
                            <td>{group_data.get('sample_count', 0)}</td>
                        </tr>
                        """

                    html += """
                        </tbody>
                    </table>
                    """

                # 稳定性指标
                if 'stability_analysis' in result:
                    stability_analysis = result['stability_analysis']
                    html += """
                    <h4>🔒 稳定性指标</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                                <th>说明</th>
                            </tr>
                        </thead>
                        <tbody>
                    """

                    for key, value in stability_analysis.items():
                        if isinstance(value, float):
                            display_value = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
                        else:
                            display_value = str(value)

                        html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{display_value}</td>
                            <td>稳定性分析指标</td>
                        </tr>
                        """

                    html += """
                        </tbody>
                    </table>
                    """

                # 分析参数
                html += """
                <h4>⚙️ 分析参数</h4>
                <table>
                    <thead>
                        <tr>
                            <th>参数</th>
                            <th>数值</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                analysis_params = result.get('analysis_params', {})
                for key, value in analysis_params.items():
                    html += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
                    """

                html += """
                    </tbody>
                </table>
                """

        return html
