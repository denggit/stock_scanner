#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : backtest_engine.py
@Description: 因子回测引擎，使用vectorbt进行横截面回测
"""

from typing import Dict, List, Optional, Any

import pandas as pd
import vectorbt as vbt

from backend.business.factor.core.data.data_manager import FactorDataManager
from backend.business.factor.core.factor.factor_engine import FactorEngine
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FactorBacktestEngine:
    """
    因子回测引擎，使用vectorbt进行横截面回测
    
    功能：
    1. TopN策略回测
    2. 分组策略回测
    3. 多因子组合回测
    4. 回测结果分析
    """

    def __init__(self, factor_engine: FactorEngine, data_manager: FactorDataManager):
        """
        初始化回测引擎
        
        Args:
            factor_engine: 因子引擎实例
            data_manager: 数据管理器实例
        """
        self.factor_engine = factor_engine
        self.data_manager = data_manager
        self._backtest_results = {}
        self._portfolio_stats = {}

    def run_topn_backtest(self,
                          factor_name: str,
                          n: int = 10,
                          rebalance_freq: str = '1d',
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        运行TopN策略回测
        
        Args:
            factor_name: 因子名称
            n: 选股数量
            rebalance_freq: 调仓频率
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始TopN回测: {factor_name}, N={n}")

        # 获取因子数据
        factor_data = self.factor_engine.get_factor_data()
        if factor_data is None:
            raise ValueError("请先调用 factor_engine.calculate_factors 计算因子")

        # 获取价格数据
        price_data = self._prepare_price_data(start_date, end_date)

        # 准备因子信号
        factor_signals = self._prepare_factor_signals(factor_data, factor_name, n)

        # 运行回测
        portfolio = vbt.Portfolio.from_signals(
            close=price_data,
            entries=factor_signals['entries'],
            exits=factor_signals['exits'],
            **kwargs
        )

        # 计算统计指标
        stats = self._calculate_portfolio_stats(portfolio)

        # 保存结果
        result_key = f"topn_{factor_name}_{n}"
        self._backtest_results[result_key] = {
            'portfolio': portfolio,
            'stats': stats,
            'signals': factor_signals,
            'params': {
                'factor_name': factor_name,
                'n': n,
                'rebalance_freq': rebalance_freq
            }
        }

        logger.info(f"TopN回测完成: {result_key}")
        return self._backtest_results[result_key]

    def run_group_backtest(self,
                           factor_name: str,
                           n_groups: int = 5,
                           rebalance_freq: str = '1d',
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        运行分组策略回测
        
        Args:
            factor_name: 因子名称
            n_groups: 分组数量
            rebalance_freq: 调仓频率
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始分组回测: {factor_name}, 分组数={n_groups}")

        # 获取因子数据
        factor_data = self.factor_engine.get_factor_data()
        if factor_data is None:
            raise ValueError("请先调用 factor_engine.calculate_factors 计算因子")

        # 获取价格数据
        price_data = self._prepare_price_data(start_date, end_date)

        # 准备分组信号
        group_signals = self._prepare_group_signals(factor_data, factor_name, n_groups)

        # 运行回测
        portfolios = {}
        stats_list = []

        for group_id in range(n_groups):
            group_entries = group_signals[f'group_{group_id}_entries']
            group_exits = group_signals[f'group_{group_id}_exits']

            # 确保信号矩阵与价格数据对齐
            group_entries = group_entries.reindex_like(price_data).fillna(False).infer_objects(copy=False)
            group_exits = group_exits.reindex_like(price_data).fillna(True).infer_objects(copy=False)

            portfolio = vbt.Portfolio.from_signals(
                close=price_data,
                entries=group_entries,
                exits=group_exits,
                **kwargs
            )

            stats = self._calculate_portfolio_stats(portfolio)
            stats['group_id'] = group_id

            portfolios[f'group_{group_id}'] = portfolio
            stats_list.append(stats)

        # 合并统计结果
        combined_stats = pd.DataFrame(stats_list)

        # 保存结果
        result_key = f"group_{factor_name}_{n_groups}"
        self._backtest_results[result_key] = {
            'portfolios': portfolios,
            'stats': combined_stats,
            'signals': group_signals,
            'params': {
                'factor_name': factor_name,
                'n_groups': n_groups,
                'rebalance_freq': rebalance_freq
            }
        }

        logger.info(f"分组回测完成: {result_key}")
        return self._backtest_results[result_key]

    def run_multifactor_backtest(self,
                                 factor_names: List[str],
                                 weights: Optional[List[float]] = None,
                                 n: int = 10,
                                 rebalance_freq: str = '1d',
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        运行多因子组合回测
        
        Args:
            factor_names: 因子名称列表
            weights: 因子权重列表
            n: 选股数量
            rebalance_freq: 调仓频率
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始多因子回测: {factor_names}")

        # 获取因子数据
        factor_data = self.factor_engine.get_factor_data()
        if factor_data is None:
            raise ValueError("请先调用 factor_engine.calculate_factors 计算因子")

        # 获取价格数据
        price_data = self._prepare_price_data(start_date, end_date)

        # 准备多因子信号
        multifactor_signals = self._prepare_multifactor_signals(
            factor_data, factor_names, weights, n
        )

        # 运行回测
        portfolio = vbt.Portfolio.from_signals(
            close=price_data,
            entries=multifactor_signals['entries'],
            exits=multifactor_signals['exits'],
            **kwargs
        )

        # 计算统计指标
        stats = self._calculate_portfolio_stats(portfolio)

        # 保存结果
        result_key = f"multifactor_{'_'.join(factor_names)}_{n}"
        self._backtest_results[result_key] = {
            'portfolio': portfolio,
            'stats': stats,
            'signals': multifactor_signals,
            'params': {
                'factor_names': factor_names,
                'weights': weights,
                'n': n,
                'rebalance_freq': rebalance_freq
            }
        }

        logger.info(f"多因子回测完成: {result_key}")
        return self._backtest_results[result_key]

    def _prepare_price_data(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        准备价格数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            价格数据DataFrame
        """
        # 获取原始数据
        if self.data_manager._processed_data is None:
            raise ValueError("请先调用 data_manager.prepare_factor_data 准备数据")

        data = self.data_manager._processed_data.copy()

        # 过滤日期范围
        if start_date:
            data = data[data['trade_date'] >= start_date]
        if end_date:
            data = data[data['trade_date'] <= end_date]

        # 转换为价格矩阵
        price_data = data.pivot(index='trade_date', columns='code', values='close')

        return price_data

    def _prepare_factor_signals(self,
                                factor_data: pd.DataFrame,
                                factor_name: str,
                                n: int) -> Dict[str, pd.DataFrame]:
        """
        准备因子信号
        
        Args:
            factor_data: 因子数据
            factor_name: 因子名称
            n: 选股数量
            
        Returns:
            信号字典
        """
        # 转换为因子矩阵
        factor_matrix = factor_data.pivot(index='trade_date', columns='code', values=factor_name)

        # 计算排名
        rank_matrix = factor_matrix.rank(axis=1, ascending=False)

        # 生成信号
        entries = rank_matrix <= n
        exits = rank_matrix > n

        return {
            'entries': entries,
            'exits': exits,
            'factor_matrix': factor_matrix,
            'rank_matrix': rank_matrix
        }

    def _prepare_group_signals(self,
                               factor_data: pd.DataFrame,
                               factor_name: str,
                               n_groups: int) -> Dict[str, pd.DataFrame]:
        """
        准备分组信号
        
        Args:
            factor_data: 因子数据
            factor_name: 因子名称
            n_groups: 分组数量
            
        Returns:
            分组信号字典
        """
        # 转换为因子矩阵
        factor_matrix = factor_data.pivot(index='trade_date', columns='code', values=factor_name)

        # 计算分组
        group_matrix = pd.qcut(factor_matrix.stack(), n_groups, labels=False, duplicates='drop')
        group_matrix = group_matrix.unstack()

        # 生成各组信号
        signals = {}

        for group_id in range(n_groups):
            group_mask = group_matrix == group_id
            signals[f'group_{group_id}_entries'] = group_mask
            signals[f'group_{group_id}_exits'] = ~group_mask

        signals['factor_matrix'] = factor_matrix
        signals['group_matrix'] = group_matrix

        return signals

    def _prepare_multifactor_signals(self,
                                     factor_data: pd.DataFrame,
                                     factor_names: List[str],
                                     weights: Optional[List[float]] = None,
                                     n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        准备多因子信号
        
        Args:
            factor_data: 因子数据
            factor_names: 因子名称列表
            weights: 因子权重列表
            n: 选股数量
            
        Returns:
            多因子信号字典
        """
        if weights is None:
            weights = [1.0 / len(factor_names)] * len(factor_names)

        # 计算综合因子
        combined_factor = pd.DataFrame()

        for i, factor_name in enumerate(factor_names):
            factor_matrix = factor_data.pivot(index='trade_date', columns='code', values=factor_name)
            # 标准化
            factor_matrix = (factor_matrix - factor_matrix.mean()) / factor_matrix.std()
            combined_factor = combined_factor.add(factor_matrix * weights[i], fill_value=0)

        # 计算排名
        rank_matrix = combined_factor.rank(axis=1, ascending=False)

        # 生成信号
        entries = rank_matrix <= n
        exits = rank_matrix > n

        return {
            'entries': entries,
            'exits': exits,
            'combined_factor': combined_factor,
            'rank_matrix': rank_matrix
        }

    def _calculate_portfolio_stats(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """
        计算组合统计指标
        
        Args:
            portfolio: vectorbt组合对象
            
        Returns:
            统计指标字典
        """
        try:
            # 尝试使用vectorbt的统计指标，但避免单位问题
            total_return = portfolio.total_return()

            # 对于最大回撤，使用自定义计算避免单位问题
            returns = portfolio.returns()
            max_drawdown = self._calculate_simple_max_drawdown(
                returns) if returns is not None and not returns.empty else 0.0

            stats = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
            }
        except Exception as e:
            logger.warning(f"计算portfolio统计指标失败: {e}，使用简化指标")
            # 使用简化的统计指标
            try:
                returns = portfolio.returns()
                if returns is not None and not returns.empty:
                    stats = {
                        'total_return': (1 + returns).prod() - 1,
                        'max_drawdown': self._calculate_simple_max_drawdown(returns),
                    }
                else:
                    stats = {
                        'total_return': 0.0,
                        'max_drawdown': 0.0,
                    }
            except Exception as e2:
                logger.error(f"计算简化统计指标也失败: {e2}")
                stats = {
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                }

        return stats

    def _calculate_simple_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算简单的最大回撤
        
        Args:
            returns: 收益率序列
            
        Returns:
            最大回撤
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0

    def get_backtest_results(self, result_key: Optional[str] = None) -> Dict[str, Any]:
        """
        获取回测结果
        
        Args:
            result_key: 结果键名，None表示所有结果
            
        Returns:
            回测结果字典
        """
        if result_key is None:
            return self._backtest_results
        else:
            return self._backtest_results.get(result_key)

    def compare_strategies(self, result_keys: List[str]) -> pd.DataFrame:
        """
        比较多个策略
        
        Args:
            result_keys: 结果键名列表
            
        Returns:
            策略比较DataFrame
        """
        comparison_data = []

        for key in result_keys:
            if key in self._backtest_results:
                result = self._backtest_results[key]
                stats = result['stats']

                if isinstance(stats, dict):
                    stats['strategy'] = key
                    comparison_data.append(stats)
                elif isinstance(stats, pd.DataFrame):
                    # 对于分组策略，取平均值
                    avg_stats = stats.mean().to_dict()
                    avg_stats['strategy'] = key
                    comparison_data.append(avg_stats)

        return pd.DataFrame(comparison_data)

    def plot_results(self, result_key: str, **kwargs):
        """
        绘制回测结果
        
        Args:
            result_key: 结果键名
            **kwargs: 绘图参数
        """
        if result_key not in self._backtest_results:
            raise ValueError(f"结果键 {result_key} 不存在")

        result = self._backtest_results[result_key]

        if 'portfolio' in result:
            # 单个组合
            portfolio = result['portfolio']
            portfolio.plot(**kwargs)
        elif 'portfolios' in result:
            # 多个组合（分组策略）
            portfolios = result['portfolios']
            for name, portfolio in portfolios.items():
                portfolio.plot(title=f"{name} - {result_key}", **kwargs)

    def save_results(self, result_key: str, file_path: str):
        """
        保存回测结果
        
        Args:
            result_key: 结果键名
            file_path: 文件路径
        """
        if result_key not in self._backtest_results:
            raise ValueError(f"结果键 {result_key} 不存在")

        result = self._backtest_results[result_key]

        # 保存统计结果
        stats = result['stats']
        if isinstance(stats, dict):
            pd.DataFrame([stats]).to_csv(file_path, index=False)
        else:
            stats.to_csv(file_path, index=False)

        logger.info(f"回测结果已保存到: {file_path}")
