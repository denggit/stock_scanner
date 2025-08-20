#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : factor_framework.py
@Description: 主要的因子回测框架类，整合所有组件，提供一键式因子研究到策略落地流程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import os
from backend.business.backtest_factor.core.base_factor import BaseFactor
from backend.business.backtest_factor.core.data_manager import FactorDataManager
from backend.business.backtest_factor.core.factor_engine import FactorEngine
from backend.business.backtest_factor.core.backtest_engine import FactorBacktestEngine
from backend.business.backtest_factor.core.analyzer import FactorAnalyzer
from backend.business.backtest_factor.core.report_generator import FactorReportGenerator
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FactorFramework:
    """
    因子回测框架主类，提供一键式因子研究到策略落地流程
    
    使用示例:
    framework = FactorFramework()
    
    # 定义因子
    @BaseFactor.register_factor(name='alpha_8', description='Alpha#8因子')
    @staticmethod
    def alpha_8(open_price: pd.Series, pct_chg: pd.Series) -> pd.Series:
        sum_open = open_price.rolling(5).sum()
        sum_returns = pct_chg.rolling(5).sum()
        product = sum_open * sum_returns
        delay_product = product.shift(10)
        return -1 * (product - delay_product).rank(pct=True)
    
    # 一键运行完整流程
    results = framework.run_complete_analysis(
        factor_names=['alpha_8'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        stock_codes=['sz.000001', 'sz.000002', 'sz.000858']
    )
    """
    
    def __init__(self, output_dir: str = "factor_reports"):
        """
        初始化因子框架
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化各个组件
        self.data_manager = FactorDataManager()
        self.factor_engine = FactorEngine(self.data_manager)
        self.backtest_engine = FactorBacktestEngine(self.factor_engine, self.data_manager)
        self.analyzer = FactorAnalyzer(self.factor_engine, self.data_manager)
        self.report_generator = FactorReportGenerator(
            self.factor_engine, self.backtest_engine, self.analyzer, self.data_manager
        )
        
        self._analysis_results = {}
        
    def run_complete_analysis(self,
                            factor_names: List[str],
                            start_date: str,
                            end_date: str,
                            stock_codes: Optional[List[str]] = None,
                            stock_pool: str = "hs300",
                            **kwargs) -> Dict[str, Any]:
        """
        运行完整的因子分析流程
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表，优先使用此参数
            stock_pool: 股票池名称，当stock_codes为None时使用
            **kwargs: 其他参数
            
        Returns:
            完整分析结果字典
        """
        logger.info(f"开始运行完整因子分析流程: {factor_names}")
        
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
            
            # 4. 因子有效性验证
            logger.info("步骤4: 因子有效性验证")
            effectiveness_results = {}
            for factor_name in factor_names:
                effectiveness_results[factor_name] = self.analyzer.analyze_factor_effectiveness(
                    factor_name, forward_period=kwargs.get('forward_period', 1)
                )
            
            # 5. 横截面回测
            logger.info("步骤5: 横截面回测")
            backtest_results = {}
            
            # TopN回测
            for factor_name in factor_names:
                topn_result = self.backtest_engine.run_topn_backtest(
                    factor_name, n=kwargs.get('top_n', 10)
                )
                backtest_results[f'topn_{factor_name}'] = topn_result
            
            # 分组回测
            for factor_name in factor_names:
                group_result = self.backtest_engine.run_group_backtest(
                    factor_name, n_groups=kwargs.get('n_groups', 5)
                )
                backtest_results[f'group_{factor_name}'] = group_result
            
            # 多因子回测
            if len(factor_names) > 1:
                backtest_results['multifactor'] = self.backtest_engine.run_multifactor_backtest(
                    factor_names, weights=kwargs.get('weights'), n=kwargs.get('top_n', 10)
                )
            
            # 6. 生成汇总报告
            logger.info("步骤6: 生成汇总报告")
            
            # 设置报告生成器的框架结果
            self.report_generator._framework_results = {
                'backtest_results': backtest_results,
                'effectiveness_results': effectiveness_results
            }
            
            # 生成汇总报告（包含所有分析结果）
            summary_report_path = self.report_generator.generate_comprehensive_report(
                factor_names, output_dir=self.output_dir
            )
            
            report_paths = {
                'summary': summary_report_path
            }
            
            # 7. 汇总结果
            results = {
                'data_info': self.data_manager.get_data_info(),
                'factor_summary': self.factor_engine.get_factor_summary(),
                'effectiveness_results': effectiveness_results,
                'backtest_results': backtest_results,
                'report_paths': report_paths,
                'analysis_params': {
                    'factor_names': factor_names,
                    'start_date': start_date,
                    'end_date': end_date,
                    'stock_codes': stock_codes,
                    **kwargs
                }
            }
            
            self._analysis_results = results
            logger.info("完整因子分析流程运行完成")
            
            return results
            
        except Exception as e:
            logger.error(f"因子分析流程运行失败: {e}")
            raise
    
    def run_single_factor_analysis(self,
                                 factor_name: str,
                                 start_date: str,
                                 end_date: str,
                                 stock_codes: Optional[List[str]] = None,
                                 stock_pool: str = "hs300",
                                 **kwargs) -> Dict[str, Any]:
        """
        运行单个因子分析
        
        Args:
            factor_name: 因子名称
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表，优先使用此参数
            stock_pool: 股票池名称，当stock_codes为None时使用
            **kwargs: 其他参数
            
        Returns:
            单个因子分析结果
        """
        return self.run_complete_analysis(
            [factor_name], start_date, end_date, stock_codes, stock_pool, **kwargs
        )
    
    def run_factor_comparison(self,
                            factor_names: List[str],
                            start_date: str,
                            end_date: str,
                            stock_codes: Optional[List[str]] = None,
                            stock_pool: str = "hs300",
                            **kwargs) -> Dict[str, Any]:
        """
        运行因子对比分析
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表，优先使用此参数
            stock_pool: 股票池名称，当stock_codes为None时使用
            **kwargs: 其他参数
            
        Returns:
            因子对比分析结果
        """
        logger.info(f"开始因子对比分析: {factor_names}")
        
        # 运行完整分析
        results = self.run_complete_analysis(
            factor_names, start_date, end_date, stock_codes, stock_pool, **kwargs
        )
        
        # 添加对比分析
        comparison_results = {}
        
        # IC对比
        ic_comparison = []
        for factor_name in factor_names:
            if factor_name in results['effectiveness_results']:
                ic_result = results['effectiveness_results'][factor_name]['ic_analysis']
                ic_stats = ic_result['ic_stats'].copy()
                ic_stats['factor_name'] = factor_name
                ic_comparison.append(ic_stats)
        
        if ic_comparison:
            comparison_results['ic_comparison'] = pd.DataFrame(ic_comparison)
        
        # 回测结果对比
        backtest_comparison = []
        for factor_name in factor_names:
            result_key = f'topn_{factor_name}'
            if result_key in results['backtest_results']:
                stats = results['backtest_results'][result_key]['stats'].copy()
                stats['factor_name'] = factor_name
                backtest_comparison.append(stats)
        
        if backtest_comparison:
            comparison_results['backtest_comparison'] = pd.DataFrame(backtest_comparison)
        
        results['comparison_results'] = comparison_results
        
        return results
    
    def get_registered_factors(self) -> pd.DataFrame:
        """
        获取所有注册的因子
        
        Returns:
            注册因子信息DataFrame
        """
        return BaseFactor.list_factors()
    
    def register_custom_factor(self,
                             name: str,
                             description: str,
                             factor_func: Callable) -> None:
        """
        注册自定义因子
        
        Args:
            name: 因子名称
            description: 因子描述
            factor_func: 因子计算函数
        """
        BaseFactor.register_factor(name=name, description=description)(factor_func)
        logger.info(f"自定义因子 {name} 注册成功")
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """
        获取分析结果
        
        Returns:
            分析结果字典
        """
        return self._analysis_results
    
    def save_results(self, file_path: str):
        """
        保存分析结果
        
        Args:
            file_path: 文件路径
        """
        if not self._analysis_results:
            raise ValueError("没有可保存的分析结果")
        
        # 保存到Excel文件
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 保存数据信息
            data_info = self._analysis_results.get('data_info', {})
            if data_info:
                pd.DataFrame([data_info]).to_excel(writer, sheet_name='数据信息', index=False)
            
            # 保存因子摘要
            factor_summary = self._analysis_results.get('factor_summary')
            if factor_summary is not None and not factor_summary.empty:
                factor_summary.to_excel(writer, sheet_name='因子摘要', index=False)
            
            # 保存有效性结果
            effectiveness_results = self._analysis_results.get('effectiveness_results', {})
            for factor_name, result in effectiveness_results.items():
                if 'ic_analysis' in result and 'ic_stats' in result['ic_analysis']:
                    ic_stats = pd.DataFrame([result['ic_analysis']['ic_stats']])
                    ic_stats.to_excel(writer, sheet_name=f'{factor_name}_IC分析', index=False)
            
            # 保存回测结果
            backtest_results = self._analysis_results.get('backtest_results', {})
            for result_key, result in backtest_results.items():
                if 'stats' in result:
                    stats = result['stats']
                    if isinstance(stats, dict):
                        stats_df = pd.DataFrame([stats])
                    else:
                        stats_df = stats
                    stats_df.to_excel(writer, sheet_name=f'{result_key}_回测结果', index=False)
        
        logger.info(f"分析结果已保存到: {file_path}")
    
    def generate_summary_report(self) -> str:
        """
        生成摘要报告
        
        Returns:
            报告文件路径
        """
        if not self._analysis_results:
            raise ValueError("没有可生成报告的分析结果")
        
        # 生成摘要报告
        factor_names = self._analysis_results['analysis_params']['factor_names']
        report_path = self.report_generator.generate_comprehensive_report(
            factor_names, output_dir=self.output_dir
        )
        
        return report_path
    
    def plot_results(self, result_type: str = 'all'):
        """
        绘制结果图表
        
        Args:
            result_type: 结果类型 ('all', 'ic', 'backtest', 'comparison')
        """
        if not self._analysis_results:
            raise ValueError("没有可绘制的分析结果")
        
        if result_type in ['all', 'ic']:
            # 绘制IC分析图
            effectiveness_results = self._analysis_results.get('effectiveness_results', {})
            for factor_name, result in effectiveness_results.items():
                if 'ic_analysis' in result:
                    try:
                        self.analyzer.plot_ic_analysis(f"ic_{factor_name}_pearson")
                    except Exception as e:
                        logger.warning(f"绘制因子 {factor_name} IC图失败: {e}")
        
        if result_type in ['all', 'backtest']:
            # 绘制回测结果图
            backtest_results = self._analysis_results.get('backtest_results', {})
            for result_key, result in backtest_results.items():
                try:
                    self.backtest_engine.plot_results(result_key)
                except Exception as e:
                    logger.warning(f"绘制回测结果 {result_key} 失败: {e}")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        获取性能摘要
        
        Returns:
            性能摘要DataFrame
        """
        if not self._analysis_results:
            raise ValueError("没有可获取性能摘要的分析结果")
        
        summary_data = []
        
        # 收集IC性能
        effectiveness_results = self._analysis_results.get('effectiveness_results', {})
        for factor_name, result in effectiveness_results.items():
            if 'ic_analysis' in result and 'ic_stats' in result['ic_analysis']:
                ic_stats = result['ic_analysis']['ic_stats']
                summary_data.append({
                    'factor_name': factor_name,
                    'metric_type': 'IC',
                    'mean_ic': ic_stats.get('mean_ic', 0),
                    'ir': ic_stats.get('ir', 0),
                    'positive_ic_rate': ic_stats.get('positive_ic_rate', 0)
                })
        
        # 收集回测性能
        backtest_results = self._analysis_results.get('backtest_results', {})
        for result_key, result in backtest_results.items():
            if 'stats' in result:
                stats = result['stats']
                if isinstance(stats, dict):
                    summary_data.append({
                        'factor_name': result_key,
                        'metric_type': 'Backtest',
                        'total_return': stats.get('total_return', 0),
                        'sharpe_ratio': stats.get('sharpe_ratio', 0),
                        'max_drawdown': stats.get('max_drawdown', 0)
                    })
        
        return pd.DataFrame(summary_data)
