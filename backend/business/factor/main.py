#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : main.py
@Description: 因子研究框架主入口
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from backend.business.factor.core.config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_STOCK_POOL,
    DEFAULT_TOP_N, DEFAULT_N_GROUPS, DEFAULT_BATCH_SIZE,
    DEFAULT_REPORT_OUTPUT_DIR, DEFAULT_USE_PARALLEL, DEFAULT_MAX_WORKERS
)
from backend.utils.logger import setup_logger
from .core.analysis.factor_analyzer import FactorAnalyzer
from .core.backtest.backtest_engine import FactorBacktestEngine
from .core.data.data_manager import FactorDataManager
from .core.factor.base_factor import BaseFactor
from .core.factor.factor_engine import FactorEngine
from .core.reporting.report_generator import FactorReportGenerator

logger = setup_logger("backtest_factor")


class FactorResearchFramework:
    """
    因子研究框架
    
    整合因子开发到回测的全流程，提供一站式因子研究解决方案
    """

    def __init__(self, output_dir: str = DEFAULT_REPORT_OUTPUT_DIR):
        """
        初始化因子研究框架
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)

        # 初始化各个组件
        self.data_manager = FactorDataManager()
        self.factor_engine = FactorEngine(self.data_manager)
        self.backtest_engine = FactorBacktestEngine(self.factor_engine, self.data_manager)
        self.analyzer = FactorAnalyzer(self.factor_engine, self.data_manager)
        # 修复：新的FactorReportGenerator只接受template_path参数
        self.report_generator = FactorReportGenerator()

        logger.info("因子研究框架初始化完成")

    def run_and_report(self, factor_names: List[str], batch_name: str, output_dir: str,
                      start_date: str = DEFAULT_START_DATE,
                      end_date: str = DEFAULT_END_DATE,
                      stock_codes: Optional[List[str]] = None,
                      stock_pool: str = DEFAULT_STOCK_POOL,
                      top_n: int = DEFAULT_TOP_N,
                      n_groups: int = DEFAULT_N_GROUPS,
                      **kwargs):
        """
        Runs the complete factor study and generates a report in a single, linear flow.
        """
        logger.info(f"开始执行完整的因子研究和报告生成流程: {factor_names}")

        try:
            # Step 1: 数据准备
            logger.info("步骤1: 数据准备")
            data = self.data_manager.prepare_factor_data(
                start_date=start_date,
                end_date=end_date,
                stock_codes=stock_codes,
                stock_pool=stock_pool
            )

            # Step 2: 因子计算
            logger.info("步骤2: 因子计算")
            factor_data = self.factor_engine.calculate_factors(factor_names)

            # Step 3: 因子预处理
            logger.info("步骤3: 因子预处理")
            standardized_factors = self.factor_engine.standardize_factors(factor_names)
            winsorized_factors = self.factor_engine.winsorize_factors(factor_names)

            # Step 4: 因子有效性分析
            logger.info("步骤4: 因子有效性分析")
            effectiveness_results = {}
            for factor_name in factor_names:
                effectiveness_results[factor_name] = self.analyzer.analyze_factor_effectiveness(
                    factor_name, forward_period=kwargs.get('forward_period', 1)
                )

            # Step 5: 回测分析
            logger.info("步骤5: 回测分析")
            self.run_backtest_analysis(
                factor_names=factor_names,
                top_n=top_n,
                n_groups=n_groups
            )

            # Step 6: 收集所有结果直接来自实例的引擎
            logger.info("正在从引擎收集所有最终结果...")
            
            # 获取时间序列收益率数据
            time_series_returns = self.backtest_engine.get_time_series_returns()
            logger.info(f"获取到 {len(time_series_returns)} 个因子的时间序列收益率数据")
            
            merged_results = {
                'backtest_results': self.backtest_engine.get_backtest_results(),
                'effectiveness_results': effectiveness_results,
                'time_series_returns': time_series_returns
            }

            # Step 7: 生成报告
            logger.info("开始生成最终报告...")
            report_path = self.generate_report(
                batch_name=batch_name,
                merged_results=merged_results,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                stock_pool=stock_pool,
                top_n=top_n,
                n_groups=n_groups
            )

            logger.info(f"完整的因子研究和报告生成流程完成，报告路径: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"因子研究和报告生成失败: {e}")
            raise

    def generate_report(self, batch_name: str, merged_results: Dict[str, Any], 
                       output_dir: str, start_date: str, end_date: str, 
                       stock_pool: str, top_n: int, n_groups: int) -> str:
        """
        生成报告
        
        Args:
            batch_name: 批次名称
            merged_results: 合并的结果
            output_dir: 输出目录
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            top_n: 选股数量
            n_groups: 分组数量
            
        Returns:
            报告路径
        """
        # 准备报告数据
        factor_names = list(merged_results.get('effectiveness_results', {}).keys())
        
        # 从回测结果中提取性能指标，按因子名称组织
        performance_metrics = {}
        for factor_name in factor_names:
            # 提取TopN回测结果
            topn_key = f'topn_{factor_name}'
            backtest_results = merged_results.get('backtest_results', {})
            if topn_key in backtest_results:
                performance_metrics[factor_name] = backtest_results[topn_key]
        
        # 构建详细分析数据
        detailed_analysis = {}
        backtest_results = merged_results.get('backtest_results', {})
        
        for factor_name in factor_names:
            # 获取TopN回测结果
            topn_key = f'topn_{factor_name}'
            topn_result = backtest_results.get(topn_key, {})
            
            # 获取分组回测结果
            group_key = f'group_{factor_name}'
            group_result = backtest_results.get(group_key, {})
            
            # 获取IC分析结果
            ic_result = merged_results.get('effectiveness_results', {}).get(factor_name, {})
            
            # 构建详细数据
            detailed_analysis[factor_name] = {
                'metrics': topn_result.get('stats', {}),
                'group_results': self._extract_group_results(group_result),
                'ic_stats': ic_result.get('ic_stats', {}),
                'risk_metrics': self._calculate_risk_metrics(topn_result)
            }
        
        report_data = {
            'factor_names': factor_names,
            'performance_metrics': performance_metrics,
            'ic_metrics': merged_results.get('effectiveness_results', {}),
            'time_series_returns': merged_results.get('time_series_returns', {}),
            'detailed_analysis': detailed_analysis
        }
        
        # 生成批次报告
        report_path = self.report_generator.generate_batch_report(
            batch_name=batch_name,
            report_data=report_data,
            output_path=os.path.join(output_dir, f"{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            top_n=top_n,
            n_groups=n_groups
        )
        
        return report_path

    def _extract_group_results(self, group_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从分组回测结果中提取分组数据
        
        Args:
            group_result: 分组回测结果
            
        Returns:
            分组结果字典
        """
        group_results = {}
        
        if 'portfolios' in group_result:
            portfolios = group_result['portfolios']
            stats = group_result.get('stats', pd.DataFrame())
            
            # 如果stats是DataFrame，转换为字典
            if isinstance(stats, pd.DataFrame):
                for i, (group_name, portfolio) in enumerate(portfolios.items()):
                    if i < len(stats):
                        group_stats = stats.iloc[i].to_dict()
                        group_results[group_name] = {
                            'total_return': group_stats.get('Total Return [%]', 0) / 100,
                            'annual_return': group_stats.get('Annual Return [%]', 0) / 100,
                            'sharpe_ratio': group_stats.get('Sharpe Ratio', 0),
                            'max_drawdown': group_stats.get('Max Drawdown [%]', 0) / 100,
                            'win_rate': group_stats.get('Win Rate [%]', 0) / 100
                        }
            else:
                # 如果stats不是DataFrame，尝试从portfolios中提取
                for i, (group_name, portfolio) in enumerate(portfolios.items()):
                    if hasattr(portfolio, 'stats'):
                        portfolio_stats = portfolio.stats
                        group_results[group_name] = {
                            'total_return': getattr(portfolio_stats, 'total_return', 0),
                            'annual_return': getattr(portfolio_stats, 'annual_return', 0),
                            'sharpe_ratio': getattr(portfolio_stats, 'sharpe_ratio', 0),
                            'max_drawdown': getattr(portfolio_stats, 'max_drawdown', 0),
                            'win_rate': getattr(portfolio_stats, 'win_rate', 0)
                        }
        
        # 确保有5个分组的数据，如果没有则填充默认值
        for i in range(5):
            group_key = f'group_{i}'
            if group_key not in group_results:
                group_results[group_key] = {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
        
        return group_results

    def _calculate_risk_metrics(self, topn_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算风险指标
        
        Args:
            topn_result: TopN回测结果
            
        Returns:
            风险指标字典
        """
        stats = topn_result.get('stats', {})
        
        # 从stats中提取风险指标
        risk_metrics = {
            'var_95': abs(stats.get('Max Drawdown [%]', 0) / 100 * 0.5),  # 简化的VaR计算
            'cvar_95': abs(stats.get('Max Drawdown [%]', 0) / 100 * 0.7),  # 简化的CVaR计算
            'beta': stats.get('Beta', 1.0),
            'alpha': stats.get('Annual Return [%]', 0) / 100 - 0.05  # 简化的Alpha计算
        }
        
        return risk_metrics

    def run_factor_research(self,
                            factor_names: List[str],
                            start_date: str = DEFAULT_START_DATE,
                            end_date: str = DEFAULT_END_DATE,
                            stock_codes: Optional[List[str]] = None,
                            stock_pool: str = DEFAULT_STOCK_POOL,
                            **kwargs) -> Dict[str, Any]:
        """
        运行完整的因子研究流程
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
            stock_pool: 股票池
            **kwargs: 其他参数
            
        Returns:
            研究结果字典
        """
        logger.info(f"开始因子研究: {factor_names}")

        try:
            # 1. 数据准备
            logger.info("步骤1: 数据准备")
            data = self.data_manager.prepare_factor_data(
                start_date=start_date,
                end_date=end_date,
                stock_codes=stock_codes,
                stock_pool=stock_pool
            )

            # 2. 因子计算
            logger.info("步骤2: 因子计算")
            factor_data = self.factor_engine.calculate_factors(factor_names)

            # 3. 因子预处理
            logger.info("步骤3: 因子预处理")
            standardized_factors = self.factor_engine.standardize_factors(factor_names)
            winsorized_factors = self.factor_engine.winsorize_factors(factor_names)

            # 4. 因子有效性分析
            logger.info("步骤4: 因子有效性分析")
            effectiveness_results = {}
            for factor_name in factor_names:
                effectiveness_results[factor_name] = self.analyzer.analyze_factor_effectiveness(
                    factor_name, forward_period=kwargs.get('forward_period', 1)
                )

            # 5. 回测分析
            logger.info("步骤5: 回测分析")
            
            # 调用专门的回测分析函数
            self.run_backtest_analysis(
                factor_names=factor_names,
                top_n=kwargs.get('top_n', DEFAULT_TOP_N),
                n_groups=kwargs.get('n_groups', DEFAULT_N_GROUPS)
            )
            
            # 获取回测结果
            backtest_results = self.backtest_engine.get_backtest_results()

            # 6. 生成报告
            logger.info("步骤6: 生成报告")
            
            # 准备报告数据 - 修复数据结构以匹配报告生成器的期望
            # 从回测结果中提取性能指标，按因子名称组织
            performance_metrics = {}
            for factor_name in factor_names:
                # 提取TopN回测结果
                topn_key = f'topn_{factor_name}'
                if topn_key in backtest_results:
                    performance_metrics[factor_name] = backtest_results[topn_key]
            
            report_data = {
                'factor_names': factor_names,
                'performance_metrics': performance_metrics,
                'ic_metrics': effectiveness_results,
                'time_series_returns': {},  # 需要从回测结果中提取
                'detailed_analysis': {}     # 需要从回测结果中提取
            }
            
            # 生成批次报告
            report_path = self.report_generator.generate_batch_report(
                batch_name="因子分析报告",
                report_data=report_data,
                output_path=os.path.join(self.output_dir, f"factor_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
                start_date=start_date,
                end_date=end_date,
                stock_pool=stock_pool,
                top_n=kwargs.get('top_n', DEFAULT_TOP_N),
                n_groups=kwargs.get('n_groups', DEFAULT_N_GROUPS)
            )

            # 7. 汇总结果
            results = {
                'factor_names': factor_names,
                'data_info': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'stock_pool': stock_pool,
                    'data_shape': data.shape
                },
                'factor_data': factor_data,
                'effectiveness_results': effectiveness_results,
                'backtest_results': backtest_results,
                'report_path': report_path
            }

            logger.info("因子研究完成")
            return results

        except Exception as e:
            logger.error(f"因子研究失败: {e}")
            raise

    def run_single_factor_analysis(self,
                                   factor_name: str,
                                   start_date: str = DEFAULT_START_DATE,
                                   end_date: str = DEFAULT_END_DATE,
                                   stock_codes: Optional[List[str]] = None,
                                   stock_pool: str = DEFAULT_STOCK_POOL,
                                   **kwargs) -> Dict[str, Any]:
        """
        运行单因子分析
        
        Args:
            factor_name: 因子名称
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
            stock_pool: 股票池
            **kwargs: 其他参数
            
        Returns:
            分析结果
        """
        return self.run_factor_research(
            [factor_name], start_date, end_date, stock_codes, stock_pool, **kwargs
        )

    def run_backtest_analysis(self, factor_names: List[str], top_n: int = 10, n_groups: int = 5) -> None:
        """
        运行所有类型（TopN、分组、多因子）的回测分析。
        """
        logger.info("开始执行回测分析流程...")

        # 1. 为每个因子运行TopN和分组回测
        for factor_name in factor_names:
            try:
                logger.info(f"--- 正在为因子 '{factor_name}' 运行TopN回测 ---")
                self.backtest_engine.run_topn_backtest(
                    factor_name=factor_name,
                    n=top_n
                )
            except Exception as e:
                logger.error(f"因子 '{factor_name}' TopN回测失败: {e}", exc_info=True)

            try:
                logger.info(f"--- 正在为因子 '{factor_name}' 运行分组回测 ---")
                self.backtest_engine.run_group_backtest(
                    factor_name=factor_name,
                    n_groups=n_groups
                )
            except Exception as e:
                logger.error(f"因子 '{factor_name}' 分组回测失败: {e}", exc_info=True)

        # 2. 运行多因子回测
        if len(factor_names) > 1:
            try:
                logger.info(f"--- 正在运行多因子回测 ---")
                self.backtest_engine.run_multifactor_backtest(
                    factor_names=factor_names,
                    n=top_n
                )
            except Exception as e:
                logger.error(f"多因子回测失败: {e}", exc_info=True)

        logger.info("所有回测分析流程执行完毕。")

    def get_available_factors(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        获取可用因子列表
        
        Args:
            category: 因子类别
            
        Returns:
            因子信息DataFrame
        """
        return BaseFactor.list_factors(category)

    def register_custom_factor(self,
                               name: str,
                               category: str,
                               description: str,
                               factor_func) -> None:
        """
        注册自定义因子
        
        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
            factor_func: 因子计算函数
        """
        # 使用装饰器注册因子
        decorator = BaseFactor.register_factor(name, category, description)
        decorated_func = decorator(factor_func)

        logger.info(f"自定义因子 {name} 注册成功")

    def get_factor_info(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """
        获取因子信息
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子信息
        """
        from .core.factor.factor_registry import factor_registry
        return factor_registry.get_factor_info(factor_name)

    def calculate_single_factor(self, factor_name: str, **kwargs) -> pd.Series:
        """
        计算单个因子
        
        Args:
            factor_name: 因子名称
            **kwargs: 因子计算参数
            
        Returns:
            因子值序列
        """
        return BaseFactor.calculate_factor(factor_name, **kwargs)

    def save_results(self, results: Dict[str, Any], file_path: str) -> None:
        """
        保存研究结果
        
        Args:
            results: 研究结果
            file_path: 保存路径
        """
        import pickle

        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

        logger.info(f"研究结果已保存到: {file_path}")

    def load_results(self, file_path: str) -> Dict[str, Any]:
        """
        加载研究结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            研究结果
        """
        import pickle

        with open(file_path, 'rb') as f:
            results = pickle.load(f)

        logger.info(f"研究结果已从 {file_path} 加载")
        return results

    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        生成汇总报告
        
        Args:
            results: 研究结果
            
        Returns:
            报告路径
        """
        factor_names = results.get('factor_names', [])
        
        # 准备报告数据
        report_data = {
            'factor_names': factor_names,
            'performance_metrics': results.get('backtest_results', {}),
            'ic_metrics': results.get('effectiveness_results', {}),
            'time_series_returns': {},
            'detailed_analysis': {}
        }
        
        return self.report_generator.generate_batch_report(
            batch_name="汇总报告",
            report_data=report_data,
            output_path=os.path.join(self.output_dir, "summary_report.html")
        )

    def generate_comprehensive_report(self,
                                     factor_names: List[str],
                                     merged_results: Optional[Dict[str, Any]] = None,
                                     analysis_summary: Optional[Dict[str, Any]] = None,
                                     start_date: str = DEFAULT_START_DATE,
                                     end_date: str = DEFAULT_END_DATE,
                                     stock_pool: str = DEFAULT_STOCK_POOL,
                                     top_n: int = DEFAULT_TOP_N,
                                     n_groups: int = DEFAULT_N_GROUPS) -> str:
        """
        生成综合分析报告（统一交互式版本）
        
        Args:
            factor_names: 因子名称列表
            merged_results: 合并的回测结果字典（用于多因子）
            analysis_summary: 分析总结字典（用于多因子）
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            top_n: 选股数量
            n_groups: 分组数量
            
        Returns:
            报告路径
        """
        logger.info(f"开始生成综合分析报告: {factor_names}")
        
        # 准备报告数据
        report_data = {
            'factor_names': factor_names,
            'performance_metrics': merged_results.get('backtest_results', {}) if merged_results else {},
            'ic_metrics': merged_results.get('effectiveness_results', {}) if merged_results else {},
            'time_series_returns': {},
            'detailed_analysis': {}
        }
        
        return self.report_generator.generate_batch_report(
            batch_name="综合分析报告",
            report_data=report_data,
            output_path=os.path.join(self.output_dir, f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            top_n=top_n,
            n_groups=n_groups
        )

    def _generate_factor_analysis_summary(self, factor_names: List[str], merged_results: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        生成因子分析总结
        
        Args:
            factor_names: 因子名称列表
            merged_results: 合并的结果字典
            
        Returns:
            分析总结字典
        """
        logger.info("生成因子分析总结...")

        summary = {
            'total_factors': len(factor_names),
            'successful_factors': merged_results.get('successful_factors', []),
            'failed_factors': merged_results.get('failed_factors', []),
            'success_rate': merged_results.get('success_rate', 0),
            'top_performers': [],
            'problematic_factors': [],
            'recommendations': []
        }

        # 分析TopN表现
        backtest_results = merged_results.get('backtest_results', {})
        effectiveness_results = merged_results.get('effectiveness_results', {})

        factor_performance = []

        for factor_name in factor_names:
            performance_data = {
                'factor_name': factor_name,
                'topn_sharpe': 0,
                'topn_return': 0,
                'topn_max_drawdown': 0,
                'ic_mean': 0,
                'ic_ir': 0,
                'ic_win_rate': 0,
                'group_monotonicity': 0,
                'status': 'normal'
            }

            # 获取TopN回测结果 - 修复：使用正确的键名和字段
            topn_key = f'topn_{factor_name}'
            if topn_key in backtest_results:
                topn_result = backtest_results[topn_key]
                if 'stats' in topn_result:
                    stats = topn_result['stats']
                    # 确保stats是字典类型
                    if isinstance(stats, dict):
                        performance_data['topn_return'] = stats.get('total_return', 0)
                        performance_data['topn_max_drawdown'] = stats.get('max_drawdown', 0)
                        # 计算夏普比率（如果没有的话）
                        if 'sharpe_ratio' in stats:
                            performance_data['topn_sharpe'] = stats.get('sharpe_ratio', 0)
                        else:
                            # 简单估算夏普比率
                            total_return = stats.get('total_return', 0)
                            max_drawdown = abs(stats.get('max_drawdown', 0))
                            if max_drawdown > 0:
                                performance_data['topn_sharpe'] = total_return / max_drawdown
                            else:
                                performance_data['topn_sharpe'] = total_return if total_return > 0 else 0

            # 获取有效性分析结果 - 修复：使用正确的字段名
            if factor_name in effectiveness_results:
                eff_result = effectiveness_results[factor_name]
                if 'ic_analysis' in eff_result:
                    ic_analysis = eff_result['ic_analysis']
                    if 'ic_stats' in ic_analysis:
                        ic_stats = ic_analysis['ic_stats']
                        performance_data['ic_mean'] = ic_stats.get('mean_ic', 0)
                        performance_data['ic_ir'] = ic_stats.get('ir', 0)
                        performance_data['ic_win_rate'] = ic_stats.get('positive_ic_rate', 0)

            # 检查分组单调性 - 修复：使用正确的键名 'stats'
            group_key = f'group_{factor_name}'
            if group_key in backtest_results:
                group_result = backtest_results[group_key]
                if 'stats' in group_result:
                    group_stats = group_result['stats']
                    # 确保group_stats是DataFrame
                    if isinstance(group_stats, pd.DataFrame) and len(group_stats) >= 2:
                        # 计算单调性得分（高组收益应该大于低组）
                        high_group_return = group_stats.iloc[-1]['total_return'] if len(group_stats) > 0 else 0
                        low_group_return = group_stats.iloc[0]['total_return'] if len(group_stats) > 0 else 0
                        performance_data['group_monotonicity'] = high_group_return - low_group_return

            # 判断因子状态 - 修复：调整判断逻辑
            if performance_data['topn_sharpe'] == 0 and performance_data['ic_mean'] == 0:
                performance_data['status'] = 'failed'
            elif performance_data['topn_sharpe'] > 1.0 and performance_data['ic_mean'] > 0.005:
                performance_data['status'] = 'excellent'
            elif performance_data['topn_sharpe'] < 0.2 or performance_data['ic_mean'] < 0.002:
                performance_data['status'] = 'poor'

            factor_performance.append(performance_data)

        # 排序并分类
        factor_performance.sort(key=lambda x: x['topn_sharpe'], reverse=True)

        # 优秀因子（Top 20%）
        excellent_count = max(1, len(factor_performance) // 5)
        summary['top_performers'] = [
            {
                'factor_name': fp['factor_name'],
                'sharpe_ratio': fp['topn_sharpe'],
                'total_return': fp['topn_return'],
                'ic_mean': fp['ic_mean'],
                'ic_ir': fp['ic_ir']
            }
            for fp in factor_performance[:excellent_count] if fp['status'] == 'excellent'
        ]

        # 问题因子
        summary['problematic_factors'] = [
            {
                'factor_name': fp['factor_name'],
                'issue': 'failed' if fp['status'] == 'failed' else 'poor_performance',
                'sharpe_ratio': fp['topn_sharpe'],
                'ic_mean': fp['ic_mean']
            }
            for fp in factor_performance if fp['status'] in ['failed', 'poor']
        ]

        # 生成建议
        recommendations = []

        if summary['success_rate'] < 0.8:
            recommendations.append(f"成功率较低({summary['success_rate']:.1%})，建议检查失败因子的实现和数据依赖")

        if summary['top_performers']:
            recommendations.append(f"发现{len(summary['top_performers'])}个优秀因子，建议优先纳入组合")

        if summary['problematic_factors']:
            recommendations.append(f"发现{len(summary['problematic_factors'])}个问题因子，建议修复或排除")

        # 添加具体建议
        if summary['top_performers']:
            top_factor = summary['top_performers'][0]
            recommendations.append(
                f"最佳因子: {top_factor['factor_name']} (夏普比率: {top_factor['sharpe_ratio']:.2f}, IC: {top_factor['ic_mean']:.4f})")

        summary['recommendations'] = recommendations
        summary['factor_performance'] = factor_performance

        logger.info(
            f"分析总结完成: {len(summary['top_performers'])}个优秀因子, {len(summary['problematic_factors'])}个问题因子")

        return summary


# 便捷函数
def create_factor_research_framework(output_dir: str = DEFAULT_REPORT_OUTPUT_DIR) -> FactorResearchFramework:
    """创建因子研究框架实例"""
    return FactorResearchFramework(output_dir)


def run_quick_factor_analysis(factor_name: str,
                              start_date: str = DEFAULT_START_DATE,
                              end_date: str = DEFAULT_END_DATE,
                              stock_pool: str = DEFAULT_STOCK_POOL,
                              **kwargs) -> Dict[str, Any]:
    """
    快速因子分析
    
    Args:
        factor_name: 因子名称
        start_date: 开始日期
        end_date: 结束日期
        stock_pool: 股票池
        **kwargs: 其他参数
        
    Returns:
        分析结果
    """
    framework = create_factor_research_framework()
    return framework.run_single_factor_analysis(
        factor_name, start_date, end_date, stock_pool=stock_pool, **kwargs
    )
