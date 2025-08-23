#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : backtest_engine.py
@Description: 因子回测引擎，使用vectorbt进行横截面回测
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from backend.business.factor.core.data.data_manager import FactorDataManager
from backend.business.factor.core.factor.factor_engine import FactorEngine
from backend.utils.logger import setup_logger

logger = setup_logger("backtest_factor")


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
        logger.debug(f"开始TopN回测: {factor_name}, N={n}")

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

        logger.debug(f"TopN回测完成: {result_key}")
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
        logger.debug(f"开始分组回测: {factor_name}, 分组数={n_groups}")

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
            group_entries = group_entries.reindex_like(price_data).fillna(False)
            group_exits = group_exits.reindex_like(price_data).fillna(True)
            
            # 兼容不同版本的pandas
            try:
                group_entries = group_entries.infer_objects(copy=False)
                group_exits = group_exits.infer_objects(copy=False)
            except TypeError:
                # 较新版本的pandas不支持copy参数
                group_entries = group_entries.infer_objects()
                group_exits = group_exits.infer_objects()

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

        logger.debug(f"分组回测完成: {result_key}")
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

        # 填充NaN值，确保信号矩阵完整
        factor_matrix = factor_matrix.fillna(0)

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
            # 获取组合的整体收益率序列
            returns = portfolio.returns()
            if returns is None or returns.empty:
                logger.warning("portfolio.returns()返回None或空")
                return self._get_default_stats()
            
            # 计算组合的整体收益率（所有股票的平均收益率）
            if hasattr(returns, 'mean'):
                # 如果returns是DataFrame，计算每行的平均值
                portfolio_returns = returns.mean(axis=1)
                logger.debug(f"使用DataFrame.mean(axis=1)，portfolio_returns长度: {len(portfolio_returns)}")
            else:
                # 如果returns是Series，直接使用
                portfolio_returns = returns
                logger.debug(f"直接使用Series，portfolio_returns长度: {len(portfolio_returns)}")
            
            # 计算总收益率
            total_return = (1 + portfolio_returns).prod() - 1
            logger.debug(f"计算得到total_return: {total_return}")
            
            # 确保total_return是标量值
            if hasattr(total_return, 'iloc'):
                total_return = total_return.iloc[0] if len(total_return) > 0 else 0.0
                logger.debug(f"使用.iloc[0]后total_return: {total_return}")
            elif hasattr(total_return, 'item'):
                total_return = total_return.item()
                logger.debug(f"使用.item()后total_return: {total_return}")
            else:
                total_return = float(total_return) if total_return is not None else 0.0
                logger.debug(f"使用float()后total_return: {total_return}")

            # 计算最大回撤
            max_drawdown = self._calculate_simple_max_drawdown(portfolio_returns)
            logger.debug(f"计算得到max_drawdown: {max_drawdown}")

            stats = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
            }
            logger.debug(f"返回stats: {stats}")
        except Exception as e:
            logger.warning(f"计算portfolio统计指标失败: {e}，使用手动计算的详细指标")
            # 使用手动计算的详细统计指标
            try:
                returns = portfolio.returns()
                if returns is not None and not returns.empty:
                    # 计算组合的整体收益率
                    if hasattr(returns, 'mean'):
                        portfolio_returns = returns.mean(axis=1)
                    else:
                        portfolio_returns = returns
                    stats = self._calculate_manual_portfolio_stats(portfolio_returns)
                else:
                    stats = self._get_default_stats()
            except Exception as e2:
                logger.error(f"计算手动统计指标也失败: {e2}")
                stats = self._get_default_stats()

        return stats

    def _calculate_manual_portfolio_stats(self, returns: pd.Series) -> Dict[str, float]:
        """
        手动计算详细的组合统计指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            统计指标字典，指标名称与vectorbt的stats()方法保持一致
        """
        try:
            # 移除NaN值
            returns_clean = returns.dropna()
            
            if len(returns_clean) == 0:
                return self._get_default_stats()
            
            # 计算基本指标
            total_return = (1 + returns_clean).prod() - 1
            annual_return = total_return * 252 / len(returns_clean) if len(returns_clean) > 0 else 0.0
            volatility = returns_clean.std() * np.sqrt(252) if len(returns_clean) > 0 else 0.0
            
            # 计算最大回撤
            max_drawdown = self._calculate_simple_max_drawdown(returns_clean)
            
            # 计算夏普比率（假设无风险利率为0）
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0
            
            # 计算卡玛比率
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            
            # 计算索提诺比率（只考虑下行风险）
            downside_returns = returns_clean[returns_clean < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
            sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0.0
            
            # 计算VaR（95%置信水平）
            var_95 = returns_clean.quantile(0.05) if len(returns_clean) > 0 else 0.0
            
            # 计算CVaR（条件VaR）
            cvar_95 = returns_clean[returns_clean <= var_95].mean() if len(returns_clean[returns_clean <= var_95]) > 0 else 0.0
            
            # 计算胜率
            win_rate = (returns_clean > 0).mean() * 100 if len(returns_clean) > 0 else 0.0
            
            # 计算盈亏比
            winning_returns = returns_clean[returns_clean > 0]
            losing_returns = returns_clean[returns_clean < 0]
            avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0.0
            avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0.0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # 计算Omega比率
            threshold = 0.0  # 使用0作为阈值
            positive_returns = returns_clean[returns_clean > threshold]
            negative_returns = returns_clean[returns_clean < threshold]
            omega_ratio = (positive_returns.sum() / abs(negative_returns.sum()) 
                          if negative_returns.sum() != 0 else float('inf'))
            
            # 构建统计指标字典，使用与vectorbt一致的名称
            stats = {
                'Total Return [%]': total_return * 100,
                'Max Drawdown [%]': max_drawdown * 100,
                'Sharpe Ratio': sharpe_ratio,
                'Calmar Ratio': calmar_ratio,
                'Sortino Ratio': sortino_ratio,
                'Win Rate [%]': win_rate,
                'Profit Factor': profit_factor,
                'Omega Ratio': omega_ratio,
                'Value at Risk [%]': var_95 * 100,
                'Conditional VaR [%]': cvar_95 * 100,
                'Annual Return [%]': annual_return * 100,
                'Annual Volatility [%]': volatility * 100,
                'Best Trade [%]': returns_clean.max() * 100,
                'Worst Trade [%]': returns_clean.min() * 100,
                'Avg Winning Trade [%]': avg_win * 100,
                'Avg Losing Trade [%]': avg_loss * 100,
                'Total Trades': len(returns_clean),
                'Total Closed Trades': len(returns_clean),
                'Total Open Trades': 0.0,
                'Open Trade PnL': 0.0,
                'Expectancy': (avg_win * (win_rate/100) + avg_loss * (1-win_rate/100)) * 100,
                'Max Drawdown Duration': self._calculate_max_drawdown_duration(returns_clean),
                'Avg Winning Trade Duration': 1.0,  # 简化处理
                'Avg Losing Trade Duration': 1.0,   # 简化处理
                'Start': returns.index[0] if len(returns) > 0 else None,
                'End': returns.index[-1] if len(returns) > 0 else None,
                'Period': len(returns),
                'Start Value': 10000.0,  # 假设初始资金
                'End Value': 10000.0 * (1 + total_return),
                'Benchmark Return [%]': 0.0,  # 简化处理
                'Max Gross Exposure [%]': 100.0,  # 简化处理
                'Total Fees Paid': 0.0  # 简化处理
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"手动计算统计指标失败: {e}")
            return self._get_default_stats()
    
    def _get_default_stats(self) -> Dict[str, float]:
        """
        获取默认的统计指标
        
        Returns:
            默认统计指标字典
        """
        return {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'omega_ratio': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_winning_trade': 0.0,
            'avg_losing_trade': 0.0,
            'total_trades': 0.0,
            'total_closed_trades': 0.0,
            'total_open_trades': 0.0,
            'open_trade_pnl': 0.0,
            'expectancy': 0.0,
            'max_drawdown_duration': 0.0,
            'avg_winning_trade_duration': 0.0,
            'avg_losing_trade_duration': 0.0,
            'start': None,
            'end': None,
            'period': 0.0,
            'start_value': 10000.0,
            'end_value': 10000.0,
            'benchmark_return': 0.0,
            'max_gross_exposure': 0.0,
            'total_fees_paid': 0.0
        }
    
    def _calculate_max_drawdown_duration(self, returns: pd.Series) -> float:
        """
        计算最大回撤持续时间
        
        Args:
            returns: 收益率序列
            
        Returns:
            最大回撤持续时间（天数）
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            # 找到最大回撤的结束点
            max_dd_idx = drawdown.idxmin()
            
            # 找到最大回撤的开始点（在最大回撤结束点之前的最高点）
            peak_idx = cumulative.loc[:max_dd_idx].idxmax()
            
            # 计算持续时间
            duration = (max_dd_idx - peak_idx).days if hasattr(max_dd_idx - peak_idx, 'days') else 0
            
            return float(duration)
        except:
            return 0.0

    def _calculate_simple_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算简单的最大回撤
        
        Args:
            returns: 收益率序列
            
        Returns:
            最大回撤（标量值）
        """
        if returns is None or returns.empty:
            return 0.0
        
        # 计算累积收益
        cumulative = (1 + returns).cumprod()
        
        # 计算滚动最大值
        rolling_max = cumulative.expanding().max()
        
        # 计算回撤
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # 计算最大回撤
        max_drawdown = drawdown.min()
        
        # 确保返回标量值
        if hasattr(max_drawdown, 'iloc'):
            max_drawdown = max_drawdown.iloc[0] if len(max_drawdown) > 0 else 0.0
        elif hasattr(max_drawdown, 'item'):
            max_drawdown = max_drawdown.item()
        else:
            max_drawdown = float(max_drawdown) if max_drawdown is not None else 0.0
        
        return max_drawdown

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
