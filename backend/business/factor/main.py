#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : main.py
@Description: 因子研究框架主入口
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Any
from backend.utils.logger import setup_logger

from .core.factor.factor_engine import FactorEngine
from .core.data.data_manager import FactorDataManager
from .core.backtest.backtest_engine import FactorBacktestEngine
from .core.analysis.factor_analyzer import FactorAnalyzer
from .core.reporting.report_generator import FactorReportGenerator
from .core.factor.base_factor import BaseFactor

logger = setup_logger(__name__)

class FactorResearchFramework:
    """
    因子研究框架
    
    整合因子开发到回测的全流程，提供一站式因子研究解决方案
    """
    
    def __init__(self, output_dir: str = "storage/reports"):
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
                          start_date: str,
                          end_date: str,
                          stock_codes: Optional[List[str]] = None,
                          stock_pool: str = "hs300",
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
                multifactor_result = self.backtest_engine.run_multifactor_backtest(
                    factor_names, weights=kwargs.get('weights')
                )
                backtest_results['multifactor'] = multifactor_result
            
            # 6. 生成报告
            logger.info("步骤6: 生成报告")
            report_path = self.report_generator.generate_comprehensive_report(
                factor_names, output_dir=self.output_dir
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
                                 start_date: str,
                                 end_date: str,
                                 stock_codes: Optional[List[str]] = None,
                                 stock_pool: str = "hs300",
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

# 便捷函数
def create_factor_research_framework(output_dir: str = "storage/reports") -> FactorResearchFramework:
    """创建因子研究框架实例"""
    return FactorResearchFramework(output_dir)

def run_quick_factor_analysis(factor_name: str,
                            start_date: str,
                            end_date: str,
                            stock_pool: str = "hs300",
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
