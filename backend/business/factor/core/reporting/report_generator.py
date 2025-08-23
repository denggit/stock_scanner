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
from backend.business.factor.core.reporting.templates import HTMLTemplateManager
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
        self.template_manager = HTMLTemplateManager()

    def generate_factor_report(self,
                               factor_name: str,
                               output_dir: str = "reports",
                               **kwargs) -> str:
        """
        生成单个因子研究报告（交互式版本）
        
        Args:
            factor_name: 因子名称
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            报告文件路径
        """
        logger.info(f"开始生成单个因子交互式报告: {factor_name}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"factor_report_{factor_name}_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)

        try:
            # 获取回测结果
            result_key = f'topn_{factor_name}'
            backtest_result = self.backtest_engine.get_backtest_results(result_key)

            if backtest_result is None:
                # 尝试从框架结果中获取
                if hasattr(self, '_framework_results') and self._framework_results:
                    backtest_results = self._framework_results.get('backtest_results', {})
                    backtest_result = backtest_results.get(result_key)

            if backtest_result is None:
                logger.warning(f"因子 {factor_name} 的回测结果不存在，生成基础报告")
                # 生成基础交互式报告
                html_content = self.template_manager.render_template('single_factor_interactive', factor_name=factor_name)
            else:
                # 生成完整的交互式报告
                html_content = self.template_manager.render_template('single_factor_interactive', factor_name=factor_name)

            # 保存报告
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"单个因子交互式报告已生成: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"生成单个因子交互式报告失败: {e}")
            raise

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
                                      merged_results: Optional[Dict[str, Any]] = None,
                                      analysis_summary: Optional[Dict[str, Any]] = None,
                                      **kwargs) -> str:
        """
        生成综合分析报告（统一交互式版本）
        
        Args:
            factor_names: 因子名称列表
            output_dir: 输出目录
            backtest_results: 单个因子回测结果字典
            merged_results: 合并的回测结果字典（用于多因子）
            analysis_summary: 分析总结字典（用于多因子）
            **kwargs: 其他参数
            
        Returns:
            报告文件路径
        """
        logger.info(f"开始生成交互式综合分析报告: {factor_names}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成报告文件名 - 多因子报告添加"总"字
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if merged_results and analysis_summary and len(factor_names) > 1:
            # 多因子合并模式，添加"总"字
            report_filename = f"总_comprehensive_report_{timestamp}.html"
        else:
            # 单个因子模式
            report_filename = f"comprehensive_report_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)

        # 准备数据 - 支持单个因子和多因子两种情况
        if merged_results and analysis_summary:
            # 多因子合并模式
            results_data = merged_results
            summary_data = analysis_summary
        else:
            # 单个因子模式
            results_data = {'backtest_results': backtest_results} if backtest_results else {}
            summary_data = {
                'successful_factors': factor_names,
                'success_rate': 1.0,
                'best_factor': factor_names[0] if factor_names else 'N/A'
            }

        # 生成交互式HTML报告
        try:
            html_content = self.template_manager.render_template('interactive_summary',
                factor_names=factor_names,
                merged_results=results_data,
                analysis_summary=summary_data,
                start_date=kwargs.get('start_date', 'N/A'),
                end_date=kwargs.get('end_date', 'N/A'),
                stock_pool=kwargs.get('stock_pool', 'no_st'),
                top_n=kwargs.get('top_n', 10),
                n_groups=kwargs.get('n_groups', 5)
            )

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"交互式综合分析报告已生成: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"生成交互式综合分析报告失败: {e}")
            raise

    def _calculate_metrics_from_returns(self, returns_series: pd.Series) -> Dict[str, Any]:
        """从收益率序列计算核心回测指标"""
        if returns_series is None or returns_series.empty or len(returns_series) < 2:
            return {
                'total_return': 0.0, 'annual_return': 0.0, 'volatility': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'trading_days': len(returns_series) if returns_series is not None else 0
            }

        # 计算累计收益率
        total_return = (1 + returns_series).prod() - 1

        # 计算年化收益率 (修正公式)
        trading_days = len(returns_series)
        annual_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0

        # 计算年化波动率
        volatility = returns_series.std() * np.sqrt(252) if trading_days > 0 else 0

        # 计算夏普比率
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # 计算最大回撤
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trading_days': trading_days
        }

    def _get_returns_from_portfolio(self, portfolio: Any) -> Optional[pd.Series]:
        """从 portfolio 对象中安全地提取和处理收益率序列"""
        if portfolio is None:
            return None

        returns = portfolio.returns()
        if returns is None or returns.empty:
            return None

        # 标准化为 pd.Series
        if isinstance(returns, pd.DataFrame):
            returns_series = returns.iloc[:, 0]
        elif isinstance(returns, np.ndarray):
            returns_series = pd.Series(returns.flatten())
        else:
            returns_series = returns

        # 确保返回的是Series
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)

        return returns_series.dropna()

    def _collect_summary_data(self, factor_names: List[str]) -> Dict[str, Any]:
        """
        收集所有分析数据 (重构和修正版本)
        
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

        backtest_results_pool = self._framework_results.get('backtest_results', {}) if hasattr(self,
                                                                                               '_framework_results') else {}
        if not backtest_results_pool:
            backtest_results_pool = self.backtest_engine._results  # Fallback

        # 收集TopN回测结果
        for factor_name in factor_names:
            try:
                result_key = f'topn_{factor_name}'  # 简化key的查找
                backtest_result = backtest_results_pool.get(result_key)

                if backtest_result and 'portfolio' in backtest_result:
                    returns_series = self._get_returns_from_portfolio(backtest_result['portfolio'])
                    stats = self._calculate_metrics_from_returns(returns_series)
                    summary_data['topn_results'][factor_name] = stats
                    logger.info(f"成功收集因子 {factor_name} TopN结果，使用键: {result_key}")
                else:
                    logger.warning(f"因子 {factor_name} TopN结果中未找到 'portfolio' 对象，无法计算指标。")
            except Exception as e:
                logger.error(f"收集因子 {factor_name} TopN结果失败: {e}")

        # 收集分组回测结果
        for factor_name in factor_names:
            try:
                result_key = f'group_{factor_name}'  # 简化key的查找
                backtest_result = backtest_results_pool.get(result_key)

                if backtest_result and 'portfolios' in backtest_result:
                    group_stats = {}
                    portfolios = backtest_result['portfolios']
                    for group_name, portfolio in portfolios.items():
                        returns_series = self._get_returns_from_portfolio(portfolio)
                        stats = self._calculate_metrics_from_returns(returns_series)
                        group_stats[group_name] = stats

                    summary_data['group_results'][factor_name] = group_stats
                    logger.info(f"成功收集因子 {factor_name} 分组结果，使用键: {result_key}")
                else:
                    logger.warning(f"因子 {factor_name} 分组结果中未找到 'portfolios' 对象，无法计算指标。")
            except Exception as e:
                logger.error(f"收集因子 {factor_name} 分组结果失败: {e}")

        # 收集多因子回测结果
        try:
            result_key = 'multifactor'
            backtest_result = backtest_results_pool.get(result_key)

            if backtest_result and 'portfolio' in backtest_result:
                returns_series = self._get_returns_from_portfolio(backtest_result['portfolio'])
                stats = self._calculate_metrics_from_returns(returns_series)
                summary_data['multifactor_results'] = stats
            else:
                logger.warning("多因子结果中未找到 'portfolio' 对象。")
        except Exception as e:
            logger.error(f"收集多因子结果失败: {e}")

        # 收集IC和有效性分析结果 (保持不变)
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

            error_html = self.template_manager.render_template('error_report',
                result_key=result_key,
                error_message=str(e),
                report_type='回测报告'
            )

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

            # 使用模板生成HTML报告
            html_content = self.template_manager.render_template('simple_html_report',
                result_key=result_key,
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                trading_days=len(returns_series),
                returns_series=returns_series.tolist()
            )

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"简单HTML报告已生成: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"简单HTML报告生成失败: {e}")
            raise

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

            error_html = self.template_manager.render_template('error_report',
                result_key=result_key,
                error_message=str(e),
                report_type='分组回测报告'
            )

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

            # 使用模板生成HTML报告
            html_content = self.template_manager.render_template('simple_group_report',
                result_key=result_key,
                group_stats=group_stats
            )

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"简单分组回测报告已生成: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"简单分组报告生成失败: {e}")
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
