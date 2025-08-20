#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : analyzer.py
@Description: 因子分析器，负责因子有效性验证和IC/RankIC计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from backend.business.backtest_factor.core.factor_engine import FactorEngine
from backend.business.backtest_factor.core.data_manager import FactorDataManager
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FactorAnalyzer:
    """
    因子分析器，负责因子有效性验证和IC/RankIC计算
    
    功能：
    1. 计算IC和RankIC
    2. 因子有效性分析
    3. 因子稳定性分析
    4. 因子相关性分析
    """
    
    def __init__(self, factor_engine: FactorEngine, data_manager: FactorDataManager):
        """
        初始化因子分析器
        
        Args:
            factor_engine: 因子引擎实例
            data_manager: 数据管理器实例
        """
        self.factor_engine = factor_engine
        self.data_manager = data_manager
        self._analysis_results = {}
        
    def calculate_ic(self,
                    factor_name: str,
                    forward_returns: Optional[pd.Series] = None,
                    forward_period: int = 1,
                    method: str = 'pearson') -> Dict[str, Any]:
        """
        计算因子IC值
        
        Args:
            factor_name: 因子名称
            forward_returns: 未来收益率，None表示自动计算
            forward_period: 未来期数
            method: 相关系数方法 ('pearson', 'spearman')
            
        Returns:
            IC分析结果字典
        """
        logger.info(f"开始计算因子IC: {factor_name}")
        
        # 获取因子数据
        factor_data = self.factor_engine.get_factor_data()
        if factor_data is None:
            raise ValueError("请先调用 factor_engine.calculate_factors 计算因子")
        
        # 获取价格数据
        price_data = self._prepare_price_data()
        
        # 计算未来收益率
        if forward_returns is None:
            forward_returns = self._calculate_forward_returns(price_data, forward_period)
        
        # 准备因子和收益率数据
        factor_matrix = factor_data.pivot(index='trade_date', columns='code', values=factor_name)
        
        # 对齐数据
        common_dates = factor_matrix.index.intersection(forward_returns.index)
        factor_matrix = factor_matrix.loc[common_dates]
        forward_returns = forward_returns.loc[common_dates]
        
        # 计算IC
        ic_series = self._calculate_ic_series(factor_matrix, forward_returns, method)
        
        # 计算统计指标
        ic_stats = self._calculate_ic_stats(ic_series)
        
        # 保存结果
        result_key = f"ic_{factor_name}_{method}"
        self._analysis_results[result_key] = {
            'ic_series': ic_series,
            'ic_stats': ic_stats,
            'factor_matrix': factor_matrix,
            'forward_returns': forward_returns,
            'params': {
                'factor_name': factor_name,
                'forward_period': forward_period,
                'method': method
            }
        }
        
        logger.info(f"IC计算完成: {result_key}")
        return self._analysis_results[result_key]
    
    def calculate_rank_ic(self,
                         factor_name: str,
                         forward_returns: Optional[pd.Series] = None,
                         forward_period: int = 1) -> Dict[str, Any]:
        """
        计算因子RankIC值
        
        Args:
            factor_name: 因子名称
            forward_returns: 未来收益率，None表示自动计算
            forward_period: 未来期数
            
        Returns:
            RankIC分析结果字典
        """
        return self.calculate_ic(factor_name, forward_returns, forward_period, method='spearman')
    
    def analyze_factor_effectiveness(self,
                                   factor_name: str,
                                   forward_period: int = 1) -> Dict[str, Any]:
        """
        分析因子有效性
        
        Args:
            factor_name: 因子名称
            forward_period: 未来期数
            
        Returns:
            因子有效性分析结果
        """
        logger.info(f"开始分析因子有效性: {factor_name}")
        
        # 计算IC和RankIC
        ic_result = self.calculate_ic(factor_name, forward_period=forward_period)
        rank_ic_result = self.calculate_rank_ic(factor_name, forward_period=forward_period)
        
        # 计算分组收益
        group_returns = self._calculate_group_returns(factor_name, forward_period)
        
        # 计算因子稳定性
        stability_metrics = self._calculate_factor_stability(factor_name)
        
        # 合并结果
        effectiveness_result = {
            'ic_analysis': ic_result,
            'rank_ic_analysis': rank_ic_result,
            'group_returns': group_returns,
            'stability_metrics': stability_metrics,
            'factor_name': factor_name,
            'forward_period': forward_period
        }
        
        # 保存结果
        result_key = f"effectiveness_{factor_name}"
        self._analysis_results[result_key] = effectiveness_result
        
        logger.info(f"因子有效性分析完成: {result_key}")
        return effectiveness_result
    
    def analyze_factor_correlation(self,
                                 factor_names: List[str]) -> Dict[str, Any]:
        """
        分析因子间相关性
        
        Args:
            factor_names: 因子名称列表
            
        Returns:
            因子相关性分析结果
        """
        logger.info(f"开始分析因子相关性: {factor_names}")
        
        # 获取因子数据
        factor_data = self.factor_engine.get_factor_data()
        if factor_data is None:
            raise ValueError("请先调用 factor_engine.calculate_factors 计算因子")
        
        # 准备因子矩阵
        factor_matrices = {}
        for factor_name in factor_names:
            if factor_name in factor_data.columns:
                factor_matrix = factor_data.pivot(index='trade_date', columns='code', values=factor_name)
                factor_matrices[factor_name] = factor_matrix
        
        # 计算相关性矩阵
        correlation_matrix = self._calculate_factor_correlation(factor_matrices)
        
        # 计算平均相关性
        avg_correlation = self._calculate_avg_correlation(correlation_matrix)
        
        # 保存结果
        result_key = f"correlation_{'_'.join(factor_names)}"
        self._analysis_results[result_key] = {
            'correlation_matrix': correlation_matrix,
            'avg_correlation': avg_correlation,
            'factor_names': factor_names
        }
        
        logger.info(f"因子相关性分析完成: {result_key}")
        return self._analysis_results[result_key]
    
    def _prepare_price_data(self) -> pd.DataFrame:
        """
        准备价格数据
        
        Returns:
            价格数据DataFrame
        """
        if self.data_manager._processed_data is None:
            raise ValueError("请先调用 data_manager.prepare_factor_data 准备数据")
        
        data = self.data_manager._processed_data.copy()
        price_data = data.pivot(index='trade_date', columns='code', values='close')
        
        return price_data
    
    def _calculate_forward_returns(self,
                                 price_data: pd.DataFrame,
                                 forward_period: int) -> pd.DataFrame:
        """
        计算未来收益率
        
        Args:
            price_data: 价格数据
            forward_period: 未来期数
            
        Returns:
            未来收益率DataFrame
        """
        # 计算未来价格
        future_prices = price_data.shift(-forward_period)
        
        # 计算收益率
        forward_returns = (future_prices - price_data) / price_data
        
        return forward_returns
    
    def _calculate_ic_series(self,
                           factor_matrix: pd.DataFrame,
                           forward_returns: pd.DataFrame,
                           method: str) -> pd.Series:
        """
        计算IC时间序列
        
        Args:
            factor_matrix: 因子矩阵
            forward_returns: 未来收益率矩阵
            method: 相关系数方法
            
        Returns:
            IC时间序列
        """
        ic_series = pd.Series(index=factor_matrix.index, dtype=float)
        
        for date in factor_matrix.index:
            if date in forward_returns.index:
                factor_values = factor_matrix.loc[date].dropna()
                return_values = forward_returns.loc[date].dropna()
                
                # 对齐数据
                common_codes = factor_values.index.intersection(return_values.index)
                if len(common_codes) > 10:  # 至少需要10个观测值
                    factor_values = factor_values[common_codes]
                    return_values = return_values[common_codes]
                    
                    if method == 'pearson':
                        ic = factor_values.corr(return_values)
                    else:  # spearman
                        ic = factor_values.corr(return_values, method='spearman')
                    
                    ic_series[date] = ic
        
        return ic_series.dropna()
    
    def _calculate_ic_stats(self, ic_series: pd.Series) -> Dict[str, float]:
        """
        计算IC统计指标
        
        Args:
            ic_series: IC时间序列
            
        Returns:
            IC统计指标字典
        """
        stats = {
            'mean_ic': ic_series.mean(),
            'std_ic': ic_series.std(),
            'ir': ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0,
            'positive_ic_rate': (ic_series > 0).mean(),
            'abs_mean_ic': ic_series.abs().mean(),
            'ic_skewness': ic_series.skew(),
            'ic_kurtosis': ic_series.kurtosis(),
            'min_ic': ic_series.min(),
            'max_ic': ic_series.max(),
            'ic_count': len(ic_series)
        }
        
        return stats
    
    def _calculate_group_returns(self,
                               factor_name: str,
                               forward_period: int,
                               n_groups: int = 5) -> Dict[str, Any]:
        """
        计算分组收益
        
        Args:
            factor_name: 因子名称
            forward_period: 未来期数
            n_groups: 分组数量
            
        Returns:
            分组收益结果
        """
        # 获取因子数据
        factor_data = self.factor_engine.get_factor_data()
        if factor_data is None:
            raise ValueError("请先调用 factor_engine.calculate_factors 计算因子")
        
        # 获取价格数据
        price_data = self._prepare_price_data()
        forward_returns = self._calculate_forward_returns(price_data, forward_period)
        
        # 准备因子矩阵
        factor_matrix = factor_data.pivot(index='trade_date', columns='code', values=factor_name)
        
        # 对齐数据
        common_dates = factor_matrix.index.intersection(forward_returns.index)
        factor_matrix = factor_matrix.loc[common_dates]
        forward_returns = forward_returns.loc[common_dates]
        
        # 计算分组收益
        group_returns = {}
        
        for date in common_dates:
            factor_values = factor_matrix.loc[date].dropna()
            return_values = forward_returns.loc[date].dropna()
            
            # 对齐数据
            common_codes = factor_values.index.intersection(return_values.index)
            if len(common_codes) > n_groups:
                factor_values = factor_values[common_codes]
                return_values = return_values[common_codes]
                
                # 分组
                try:
                    groups = pd.qcut(factor_values, n_groups, labels=False, duplicates='drop')
                    group_df = pd.DataFrame({
                        'factor': factor_values,
                        'returns': return_values,
                        'group': groups
                    })
                    
                    # 计算各组平均收益
                    for group_id in range(n_groups):
                        group_data = group_df[group_df['group'] == group_id]
                        if len(group_data) > 0:
                            if f'group_{group_id}' not in group_returns:
                                group_returns[f'group_{group_id}'] = []
                            group_returns[f'group_{group_id}'].append(group_data['returns'].mean())
                except:
                    continue
        
        # 计算各组统计指标
        group_stats = {}
        for group_name, returns in group_returns.items():
            returns_series = pd.Series(returns)
            group_stats[group_name] = {
                'mean_return': returns_series.mean(),
                'std_return': returns_series.std(),
                'sharpe_ratio': returns_series.mean() / returns_series.std() if returns_series.std() != 0 else 0,
                'win_rate': (returns_series > 0).mean(),
                'count': len(returns_series)
            }
        
        return {
            'group_returns': group_returns,
            'group_stats': group_stats
        }
    
    def _calculate_factor_stability(self, factor_name: str) -> Dict[str, float]:
        """
        计算因子稳定性指标
        
        Args:
            factor_name: 因子名称
            
        Returns:
            稳定性指标字典
        """
        # 获取因子数据
        factor_data = self.factor_engine.get_factor_data()
        if factor_data is None:
            raise ValueError("请先调用 factor_engine.calculate_factors 计算因子")
        
        # 准备因子矩阵
        factor_matrix = factor_data.pivot(index='trade_date', columns='code', values=factor_name)
        
        # 计算因子值变化
        factor_changes = factor_matrix.pct_change()
        
        # 计算稳定性指标
        stability_metrics = {
            'mean_change': factor_changes.mean().mean(),
            'std_change': factor_changes.std().mean(),
            'autocorr_1d': factor_matrix.corrwith(factor_matrix.shift(1)).mean(),
            'autocorr_5d': factor_matrix.corrwith(factor_matrix.shift(5)).mean(),
            'autocorr_20d': factor_matrix.corrwith(factor_matrix.shift(20)).mean(),
        }
        
        return stability_metrics
    
    def _calculate_factor_correlation(self,
                                    factor_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算因子间相关性矩阵
        
        Args:
            factor_matrices: 因子矩阵字典
            
        Returns:
            相关性矩阵DataFrame
        """
        # 对齐数据
        common_dates = None
        for factor_matrix in factor_matrices.values():
            if common_dates is None:
                common_dates = factor_matrix.index
            else:
                common_dates = common_dates.intersection(factor_matrix.index)
        
        if common_dates is None or len(common_dates) == 0:
            return pd.DataFrame()
        
        # 计算相关性
        correlation_data = {}
        factor_names = list(factor_matrices.keys())
        
        for i, factor1 in enumerate(factor_names):
            correlation_data[factor1] = {}
            for j, factor2 in enumerate(factor_names):
                if i == j:
                    correlation_data[factor1][factor2] = 1.0
                else:
                    # 计算时间序列相关性
                    factor1_series = factor_matrices[factor1].loc[common_dates].mean()
                    factor2_series = factor_matrices[factor2].loc[common_dates].mean()
                    
                    # 对齐数据
                    common_codes = factor1_series.index.intersection(factor2_series.index)
                    if len(common_codes) > 10:
                        corr = factor1_series[common_codes].corr(factor2_series[common_codes])
                        correlation_data[factor1][factor2] = corr
                    else:
                        correlation_data[factor1][factor2] = np.nan
        
        return pd.DataFrame(correlation_data)
    
    def _calculate_avg_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """
        计算平均相关性
        
        Args:
            correlation_matrix: 相关性矩阵
            
        Returns:
            平均相关性
        """
        if correlation_matrix.empty:
            return 0.0
        
        # 获取上三角矩阵（排除对角线）
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # 计算平均相关性
        avg_corr = upper_triangle.stack().mean()
        
        return avg_corr if not pd.isna(avg_corr) else 0.0
    
    def plot_ic_analysis(self, result_key: str, **kwargs):
        """
        绘制IC分析图
        
        Args:
            result_key: 结果键名
            **kwargs: 绘图参数
        """
        if result_key not in self._analysis_results:
            raise ValueError(f"结果键 {result_key} 不存在")
        
        result = self._analysis_results[result_key]
        ic_series = result['ic_series']
        ic_stats = result['ic_stats']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # IC时间序列图
        axes[0, 0].plot(ic_series.index, ic_series.values)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f"IC时间序列 (均值: {ic_stats['mean_ic']:.4f})")
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('IC值')
        
        # IC分布直方图
        axes[0, 1].hist(ic_series.values, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=ic_stats['mean_ic'], color='r', linestyle='--', label=f"均值: {ic_stats['mean_ic']:.4f}")
        axes[0, 1].set_title(f"IC分布 (IR: {ic_stats['ir']:.4f})")
        axes[0, 1].set_xlabel('IC值')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        
        # IC累积分布
        sorted_ic = np.sort(ic_series.values)
        cumulative_prob = np.arange(1, len(sorted_ic) + 1) / len(sorted_ic)
        axes[1, 0].plot(sorted_ic, cumulative_prob)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title("IC累积分布")
        axes[1, 0].set_xlabel('IC值')
        axes[1, 0].set_ylabel('累积概率')
        
        # 统计指标表格
        stats_text = '\n'.join([f"{k}: {v:.4f}" for k, v in ic_stats.items()])
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='center')
        axes[1, 1].set_title("IC统计指标")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_analysis_results(self, result_key: Optional[str] = None) -> Dict[str, Any]:
        """
        获取分析结果
        
        Args:
            result_key: 结果键名，None表示所有结果
            
        Returns:
            分析结果字典
        """
        if result_key is None:
            return self._analysis_results
        else:
            return self._analysis_results.get(result_key)
    
    def save_analysis_results(self, result_key: str, file_path: str):
        """
        保存分析结果
        
        Args:
            result_key: 结果键名
            file_path: 文件路径
        """
        if result_key not in self._analysis_results:
            raise ValueError(f"结果键 {result_key} 不存在")
        
        result = self._analysis_results[result_key]
        
        # 保存到Excel文件
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 保存IC统计指标
            if 'ic_stats' in result:
                pd.DataFrame([result['ic_stats']]).to_excel(writer, sheet_name='IC_Stats', index=False)
            
            # 保存IC时间序列
            if 'ic_series' in result:
                result['ic_series'].to_frame('IC').to_excel(writer, sheet_name='IC_Series')
            
            # 保存分组收益
            if 'group_returns' in result:
                group_stats = result['group_returns']['group_stats']
                pd.DataFrame(group_stats).T.to_excel(writer, sheet_name='Group_Returns')
        
        logger.info(f"分析结果已保存到: {file_path}")
