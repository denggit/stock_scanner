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

logger = setup_logger(__name__)


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

        logger.info("因子研究框架初始化完成")

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
            backtest_results = {}

            # TopN回测
            for factor_name in factor_names:
                topn_result = self.backtest_engine.run_topn_backtest(
                    factor_name, n=kwargs.get('top_n', DEFAULT_TOP_N)
                )
                backtest_results[f'topn_{factor_name}'] = topn_result

            # 分组回测
            for factor_name in factor_names:
                group_result = self.backtest_engine.run_group_backtest(
                    factor_name, n_groups=kwargs.get('n_groups', DEFAULT_N_GROUPS)
                )
                backtest_results[f'group_{factor_name}'] = group_result

            # 多因子回测
            if len(factor_names) > 1:
                multifactor_result = self.backtest_engine.run_multifactor_backtest(
                    factor_names, weights=kwargs.get('weights')
                )
                backtest_results['multifactor'] = multifactor_result

            # 6. 生成报告
            logger.info("步骤6: 生成报告")
            report_path = self.report_generator.generate_comprehensive_report(
                factor_names, output_dir=self.output_dir, backtest_results=backtest_results
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

    def run_factor_comparison(self,
                              factor_names: List[str],
                              start_date: str = DEFAULT_START_DATE,
                              end_date: str = DEFAULT_END_DATE,
                              stock_codes: Optional[List[str]] = None,
                              stock_pool: str = DEFAULT_STOCK_POOL,
                              **kwargs) -> Dict[str, Any]:
        """
        运行因子对比分析
        
        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
            stock_pool: 股票池
            **kwargs: 其他参数
            
        Returns:
            对比分析结果
        """
        logger.info(f"开始因子对比分析: {factor_names}")

        # 运行完整研究
        results = self.run_factor_research(
            factor_names, start_date, end_date, stock_codes, stock_pool, **kwargs
        )

        # 添加对比分析
        comparison_results = {}

        # 对比因子有效性
        ic_comparison = {}
        for factor_name in factor_names:
            if factor_name in results['effectiveness_results']:
                ic_comparison[factor_name] = results['effectiveness_results'][factor_name].get('ic_mean', 0)

        comparison_results['ic_comparison'] = ic_comparison

        # 对比回测表现
        performance_comparison = {}
        for factor_name in factor_names:
            topn_key = f'topn_{factor_name}'
            if topn_key in results['backtest_results']:
                topn_result = results['backtest_results'][topn_key]
                if 'portfolio_stats' in topn_result:
                    performance_comparison[factor_name] = topn_result['portfolio_stats']

        comparison_results['performance_comparison'] = performance_comparison

        results['comparison_results'] = comparison_results

        logger.info("因子对比分析完成")
        return results

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
        return self.report_generator.generate_comprehensive_report(
            factor_names, output_dir=self.output_dir
        )

    def generate_merged_comprehensive_report(self,
                                             factor_names: List[str],
                                             merged_results: Dict[str, Any],
                                             start_date: str = DEFAULT_START_DATE,
                                             end_date: str = DEFAULT_END_DATE,
                                             stock_pool: str = DEFAULT_STOCK_POOL,
                                             top_n: int = DEFAULT_TOP_N,
                                             n_groups: int = DEFAULT_N_GROUPS) -> str:
        """
        生成合并的综合报告（包含分析总结）
        
        Args:
            factor_names: 因子名称列表
            merged_results: 合并的结果字典
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            top_n: 选股数量
            n_groups: 分组数量
            
        Returns:
            报告路径
        """
        logger.info(f"开始生成合并的综合报告，包含 {len(factor_names)} 个因子")

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"worldquant{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # 生成报告文件名
        report_filename = f"comprehensive_report_{timestamp}.html"
        report_path = os.path.join(report_dir, report_filename)

        # 生成分析总结
        analysis_summary = self._generate_factor_analysis_summary(factor_names, merged_results)

        # 生成合并报告
        try:
            self.report_generator.generate_merged_comprehensive_report(
                factor_names=factor_names,
                merged_results=merged_results,
                analysis_summary=analysis_summary,
                report_path=report_path,
                start_date=start_date,
                end_date=end_date,
                stock_pool=stock_pool,
                top_n=top_n,
                n_groups=n_groups
            )

            logger.info(f"合并综合报告生成成功: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"生成合并报告失败: {e}")
            raise

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

            # 获取TopN回测结果
            topn_key = f'topn_{factor_name}'
            if topn_key in backtest_results:
                topn_result = backtest_results[topn_key]
                if 'portfolio_stats' in topn_result:
                    stats = topn_result['portfolio_stats']
                    performance_data['topn_sharpe'] = stats.get('sharpe_ratio', 0)
                    performance_data['topn_return'] = stats.get('total_return', 0)
                    performance_data['topn_max_drawdown'] = stats.get('max_drawdown', 0)

            # 获取有效性分析结果
            if factor_name in effectiveness_results:
                eff_result = effectiveness_results[factor_name]
                if 'ic_analysis' in eff_result:
                    ic_analysis = eff_result['ic_analysis']
                    performance_data['ic_mean'] = ic_analysis.get('ic_mean', 0)
                    performance_data['ic_ir'] = ic_analysis.get('ic_ir', 0)
                    performance_data['ic_win_rate'] = ic_analysis.get('ic_win_rate', 0)

            # 检查分组单调性
            group_key = f'group_{factor_name}'
            if group_key in backtest_results:
                group_result = backtest_results[group_key]
                if 'group_stats' in group_result:
                    group_stats = group_result['group_stats']
                    # 计算单调性得分（高组收益应该大于低组）
                    if len(group_stats) >= 2:
                        high_group_return = group_stats.iloc[-1]['total_return'] if len(group_stats) > 0 else 0
                        low_group_return = group_stats.iloc[0]['total_return'] if len(group_stats) > 0 else 0
                        performance_data['group_monotonicity'] = high_group_return - low_group_return

            # 判断因子状态
            if performance_data['topn_sharpe'] == 0 and performance_data['ic_mean'] == 0:
                performance_data['status'] = 'failed'
            elif performance_data['topn_sharpe'] > 2.0 and performance_data['ic_mean'] > 0.02:
                performance_data['status'] = 'excellent'
            elif performance_data['topn_sharpe'] < 0.5 or performance_data['ic_mean'] < 0.01:
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
