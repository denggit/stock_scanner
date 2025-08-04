#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 3/1/25 10:26 PM
@File       : performance_analyzer.py
@Description: 单因子有效性测试，计算每个因子的IC值、胜率等指标

通过统计的方式，快速验证因子有效性（无需模拟交易）
输出：IC值、分位数收益、月度胜率
e.g.:
    输入：PE因子值、未来N日收益率
    过程：
        1. 计算因子IC值（因子值与未来收益的横截面相关性）
        2. 按因子分位数分组统计各组平均收益
        3. 生成月度胜率报告

    输出示例：
        PE因子近5年IC均值 = 0.12
        前10%低PE组5日平均收益 = 1.2%
        胜率 = 58%


IC均值范围	因子预测能力	实际案例
>0.05	强有效	动量因子（IC均值0.08-0.12）
0.02-0.05	中等有效	低波动率因子（IC均值0.03-0.05）
<0.02	弱有效或无效	市值因子（IC均值接近0，已失效）

IR范围	稳定性评价	应用建议
IR > 1.5	极优	核心因子，赋予高权重（如60%-80%）
IR 1.0-1.5	优秀	主力因子，可长期使用
IR 0.5-1.0	合格	需搭配其他因子分散风险
IR < 0.5	不稳定	谨慎使用，需动态监控或淘汰
"""

import logging
from typing import Dict, List, Union

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 查找系统中支持中文的字体
chinese_fonts = [f.name for f in fm.fontManager.ttflist if '黑体' in f.name or 'Heiti' in f.name or 'SimHei' in f.name]
if chinese_fonts:
    plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
else:
    # 如果没有找到中文字体，可以使用其他通用字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class FactorAnalyzer:
    """因子有效性分析器，执行IC计算、分组收益分析等"""

    def __init__(self, factor_data: Union[pd.DataFrame, pd.Series],
                 price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 is_panel_data: bool = False):
        """
        初始化因子分析器
        
        Args:
            factor_data: 因子值数据，可以是:
                         - 单只股票的Series (is_panel_data=False)
                         - 多只股票的DataFrame，索引为日期，列为股票代码 (is_panel_data=True)
            price_data: 价格数据，可以是:
                        - 单只股票的DataFrame，含close列 (is_panel_data=False)
                        - 多只股票的字典，key为'close'等，value为面板数据 (is_panel_data=True)
            is_panel_data: 是否为面板数据格式（多只股票数据）
        """
        self.is_panel_data = is_panel_data
        self.factor_data = factor_data
        self.price_data = price_data
        self.returns_data = None
        self.ic_series = None
        self.daily_ic = None
        self.group_returns = None

    def calculate_forward_returns(self, periods: List[int] = [1, 5, 10, 20]) -> Union[
        pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        计算未来收益率
        
        Args:
            periods: 未来收益率的周期列表，单位为交易日
            
        Returns:
            未来收益率数据，格式与输入数据格式一致
        """
        if self.is_panel_data:
            # 多只股票数据的处理
            returns = {}

            # 获取收盘价面板
            close_panel = self.price_data['close']

            for period in periods:
                # 对每只股票计算未来收益率，明确指定fill_method=None
                pct_change = close_panel.pct_change(period, fill_method=None).shift(-period)
                returns[f'return_{period}d'] = pct_change
        else:
            # 单只股票数据的处理
            returns = pd.DataFrame(index=self.price_data.index)

            for period in periods:
                pct_change = self.price_data['close'].pct_change(period, fill_method=None).shift(-period)
                returns[f'return_{period}d'] = pct_change

        self.returns_data = returns
        return returns

    def calculate_ic(self, method: str = 'spearman') -> pd.DataFrame:
        """
        计算因子IC值
        
        Args:
            method: 相关系数计算方法，'pearman' 或 'spearman'(秩相关)
            
        Returns:
            IC值数据框
        """
        if self.returns_data is None:
            self.calculate_forward_returns()

        if self.is_panel_data:
            # 多只股票数据的横截面IC计算
            daily_ic = {}
            ic_summary = {}

            # 添加调试信息
            logging.info(f"因子数据形状: {self.factor_data.shape}")
            logging.info(f"收益率数据周期: {list(self.returns_data.keys())}")

            for period, returns_panel in self.returns_data.items():
                # 计算每个交易日的IC值
                ic_series = pd.Series(index=self.factor_data.index)
                valid_dates = 0  # 计数有效日期数
                constant_dates = 0  # 计数因子值为常数的日期数

                for date in self.factor_data.index:
                    if date not in returns_panel.index:
                        continue

                    # 获取当天所有股票的因子值和未来收益
                    factors = self.factor_data.loc[date].dropna()
                    future_returns = returns_panel.loc[date, factors.index].dropna()

                    # 对齐数据
                    common_stocks = factors.index.intersection(future_returns.index)
                    if len(common_stocks) < 10:  # 要求至少有10只股票才计算IC
                        continue

                    aligned_factors = factors[common_stocks]
                    aligned_returns = future_returns[common_stocks]

                    # 检查因子值是否为常数
                    if aligned_factors.nunique() == 1:
                        constant_dates += 1
                        continue

                    # 计算相关系数
                    try:
                        if method == 'spearman':
                            ic = aligned_factors.corr(aligned_returns, method='spearman')
                        else:
                            ic = aligned_factors.corr(aligned_returns, method='pearson')

                        ic_series[date] = ic
                        valid_dates += 1
                    except Exception as e:
                        logging.exception(f"计算{date}的IC值时出错: {e}")

                # 存储每日IC值
                daily_ic[period] = ic_series.dropna()

                # 计算IC统计指标
                if not ic_series.dropna().empty:
                    ic_std = ic_series.std()
                    ic_mean = ic_series.mean()
                    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan  # 处理零标准差情况

                    ic_summary[period] = {
                        'ic': ic_mean,
                        'ic_std': ic_std,
                        'ic_ir': ic_ir,
                        'abs_ic': ic_series.abs().mean(),
                        'pos_rate': (ic_series > 0).mean(),
                        'mean_monthly_ic': self._calculate_monthly_ic_panel(ic_series, method)
                    }
                else:
                    logging.info(f"警告: {period} 周期无有效IC值")
                    ic_summary[period] = {
                        'ic': float('nan'),
                        'ic_std': float('nan'),
                        'ic_ir': float('nan'),
                        'abs_ic': float('nan'),
                        'pos_rate': float('nan'),
                        'mean_monthly_ic': float('nan')
                    }

            self.daily_ic = daily_ic
            self.ic_series = pd.DataFrame(ic_summary).T if ic_summary else pd.DataFrame()

        else:
            # 单只股票数据的时间序列IC计算（原有逻辑）
            ic_values = {}

            for col in self.returns_data.columns:
                # 对齐因子数据和收益率数据
                aligned_data = pd.DataFrame({
                    'factor': self.factor_data,
                    'return': self.returns_data[col]
                }).dropna()

                if method == 'spearman':
                    ic = aligned_data['factor'].corr(aligned_data['return'], method='spearman')
                else:
                    ic = aligned_data['factor'].corr(aligned_data['return'], method='pearson')

                ic_values[col] = ic

            # 创建一个包含所有周期IC值的Series
            self.ic_series = pd.Series(ic_values)

            # 创建按月统计的IC值数据框
            monthly_ic = self._calculate_monthly_ic(method)

            self.ic_series = pd.DataFrame({
                'ic': self.ic_series,
                'abs_ic': self.ic_series.abs(),
                'pos_rate': (monthly_ic > 0).mean(),
                'mean_monthly_ic': monthly_ic.mean()
            })

        return self.ic_series

    def _calculate_monthly_ic(self, method: str = 'spearman') -> pd.Series:
        """
        计算月度IC值（单只股票）
        
        Args:
            method: 相关系数计算方法
            
        Returns:
            月度IC值序列
        """
        if self.returns_data is None:
            self.calculate_forward_returns()

        # 对齐数据并按月分组
        aligned_data = pd.DataFrame({
            'factor': self.factor_data,
            'return': self.returns_data['return_20d']  # 使用20日收益计算月度IC
        }).dropna()

        # 添加月份信息
        aligned_data['month'] = aligned_data.index.to_period('M')

        # 按月计算IC
        monthly_ic = aligned_data.groupby('month').apply(
            lambda x: x['factor'].corr(x['return'], method=method)
        )

        return monthly_ic

    def _calculate_monthly_ic_panel(self, daily_ic: pd.Series, method: str = 'spearman') -> float:
        """
        计算面板数据的月度平均IC值
        
        Args:
            daily_ic: 每日IC值序列
            method: 相关系数计算方法
            
        Returns:
            月度平均IC值
        """
        # 将日度IC按月分组
        daily_ic_df = daily_ic.to_frame('ic')
        daily_ic_df['month'] = daily_ic_df.index.to_period('M')

        # 计算每月平均IC
        monthly_mean_ic = daily_ic_df.groupby('month')['ic'].mean()

        return monthly_mean_ic.mean() if not monthly_mean_ic.empty else np.nan

    def calculate_quantile_returns(self, num_quantiles: int = 5) -> pd.DataFrame:
        """
        计算分位数分组收益
        
        Args:
            num_quantiles: 分组数量，默认为5（五分位）
            
        Returns:
            各分位数组平均收益数据框
        """
        if self.returns_data is None:
            self.calculate_forward_returns()

        if self.is_panel_data:
            # 多只股票数据的分位数收益计算
            group_returns = {}

            for period, returns_panel in self.returns_data.items():
                daily_group_returns = []

                for date in self.factor_data.index:
                    if date not in returns_panel.index:
                        continue

                    # 获取当天所有股票的因子值和未来收益
                    factors = self.factor_data.loc[date].dropna()
                    future_returns = returns_panel.loc[date, factors.index].dropna()

                    # 对齐数据
                    common_stocks = factors.index.intersection(future_returns.index)
                    if len(common_stocks) < num_quantiles:  # 要求至少有与分组数相同的股票数
                        continue

                    aligned_factors = factors[common_stocks]
                    aligned_returns = future_returns[common_stocks]

                    # 按因子值分组
                    try:
                        quantiles = pd.qcut(aligned_factors, num_quantiles, labels=False) + 1

                        # 计算每组平均收益
                        group_data = pd.DataFrame({
                            'quantile': quantiles,
                            'return': aligned_returns
                        })

                        group_avg = group_data.groupby('quantile')['return'].mean()
                        group_avg.name = date
                        daily_group_returns.append(group_avg)
                    except ValueError:
                        # 处理因子值相同导致无法分组的情况
                        continue

                if daily_group_returns:
                    # 合并所有日期的分组收益
                    period_returns = pd.DataFrame(daily_group_returns)
                    # 计算每组的平均收益
                    avg_returns = period_returns.mean()
                    group_returns[period] = avg_returns

            # 合并所有周期的结果
            self.group_returns = pd.DataFrame(group_returns)

        else:
            # 单只股票数据的分位数收益计算（原有逻辑）
            # 对齐因子数据和收益率数据
            aligned_data = pd.DataFrame({
                'factor': self.factor_data
            })

            for col in self.returns_data.columns:
                aligned_data[col] = self.returns_data[col]

            aligned_data = aligned_data.dropna()

            # 计算分位数
            aligned_data['quantile'] = pd.qcut(
                aligned_data['factor'],
                num_quantiles,
                labels=False
            ) + 1

            # 计算各分位数组的平均收益
            group_returns = {}

            for col in self.returns_data.columns:
                group_returns[col] = aligned_data.groupby('quantile')[col].mean()

            self.group_returns = pd.DataFrame(group_returns)

        # 添加多空组合收益和纯多头收益
        if num_quantiles > 1 and not self.group_returns.empty:
            # 多空组合收益（顶层组减底层组）
            long_short = self.group_returns.loc[num_quantiles] - self.group_returns.loc[1]
            self.group_returns.loc['long_short'] = long_short

            # 纯多头收益（顶层组绝对收益）
            top_group = self.group_returns.loc[num_quantiles]
            self.group_returns.loc['top_group'] = top_group

        return self.group_returns

    def calculate_win_rate(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        计算胜率
        
        Args:
            threshold: 胜率阈值，默认为0（收益为正即视为胜利）
            
        Returns:
            胜率数据框
        """
        if self.returns_data is None:
            self.calculate_forward_returns()

        if self.group_returns is None:
            self.calculate_quantile_returns()

        if self.is_panel_data:
            # 多只股票数据的胜率计算
            win_rates = {}

            for period, returns_panel in self.returns_data.items():
                # 初始化各分位组的胜负记录
                quantile_wins = {q: [] for q in range(1, 6)}

                for date in self.factor_data.index:
                    if date not in returns_panel.index:
                        continue

                    # 获取当天所有股票的因子值和未来收益
                    factors = self.factor_data.loc[date].dropna()
                    future_returns = returns_panel.loc[date, factors.index].dropna()

                    # 对齐数据
                    common_stocks = factors.index.intersection(future_returns.index)
                    if len(common_stocks) < 10:  # 要求至少有10只股票
                        continue

                    aligned_factors = factors[common_stocks]
                    aligned_returns = future_returns[common_stocks]

                    # 按因子值分组
                    try:
                        quantiles = pd.qcut(aligned_factors, 5, labels=False) + 1

                        # 计算每组的胜负情况
                        for q in range(1, 6):
                            stocks_in_quantile = quantiles == q
                            if stocks_in_quantile.sum() > 0:
                                returns_in_quantile = aligned_returns[stocks_in_quantile]
                                win_rate = (returns_in_quantile > threshold).mean()
                                quantile_wins[q].append(win_rate)
                    except ValueError:
                        continue

                # 计算每个分位组的平均胜率
                period_win_rates = {}
                for q in range(1, 6):
                    if quantile_wins[q]:
                        period_win_rates[q] = np.mean(quantile_wins[q])
                    else:
                        period_win_rates[q] = np.nan

                win_rates[period] = period_win_rates

            return pd.DataFrame(win_rates)
        else:
            # 单只股票数据的胜率计算（原有逻辑）
            # 计算每个分位数组的胜率
            win_rates = {}
            monthly_win_rates = {}

            for col in self.returns_data.columns:
                # 对齐因子数据和收益率数据
                aligned_data = pd.DataFrame({
                    'factor': self.factor_data,
                    'return': self.returns_data[col]
                }).dropna()

                # 计算分位数
                aligned_data['quantile'] = pd.qcut(
                    aligned_data['factor'],
                    5,
                    labels=False
                ) + 1

                # 添加月份信息
                aligned_data['month'] = aligned_data.index.to_period('M')

                # 计算整体胜率
                win_rates[col] = {}
                for quantile in range(1, 6):
                    wins = (aligned_data[aligned_data['quantile'] == quantile]['return'] > threshold).sum()
                    total = (aligned_data['quantile'] == quantile).sum()
                    win_rates[col][quantile] = wins / total if total > 0 else np.nan

                # 计算月度胜率
                monthly_data = aligned_data.groupby(['month', 'quantile'])['return'].mean()
                monthly_wins = (monthly_data > threshold).groupby(level=1).mean()
                monthly_win_rates[col] = monthly_wins

            # 整理结果
            overall_win_rates = pd.DataFrame(win_rates)
            monthly_win_rates = pd.DataFrame(monthly_win_rates)

            return pd.concat([
                overall_win_rates.add_prefix('overall_'),
                monthly_win_rates.add_prefix('monthly_')
            ], axis=1)

    def plot_ic_series(self) -> None:
        """绘制IC值序列"""
        if self.is_panel_data:
            if self.daily_ic is None:
                self.calculate_ic()

            plt.figure(figsize=(12, 6))

            for period, ic_series in self.daily_ic.items():
                plt.plot(ic_series.index, ic_series.values, label=period)

            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('因子IC值时间序列 (与未来收益的相关性)')
            plt.ylabel('IC值')
            plt.xlabel('日期')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            if self.ic_series is None:
                self.calculate_ic()

            plt.figure(figsize=(12, 6))
            self.ic_series['ic'].plot(kind='bar')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Factor IC Values (Correlation with Future Returns)')
            plt.ylabel('IC Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def plot_quantile_returns(self) -> None:
        """绘制分位数收益"""
        if self.group_returns is None:
            self.calculate_quantile_returns()

        # 剔除long_short行
        plot_data = self.group_returns.drop(
            'long_short') if 'long_short' in self.group_returns.index else self.group_returns

        plt.figure(figsize=(12, 6))
        sns.heatmap(plot_data, annot=True, cmap='RdYlGn', fmt='.2%')
        plt.title('Average Returns by Quantile Group')
        plt.tight_layout()
        plt.show()

    def plot_cumulative_returns(self, period: str = 'return_20d') -> None:
        """
        绘制累积收益曲线
        
        Args:
            period: 收益周期，默认为20日收益
        """
        if self.returns_data is None:
            self.calculate_forward_returns()

        if self.group_returns is None:
            self.calculate_quantile_returns()

        if self.is_panel_data:
            # 多只股票数据的累积收益曲线
            if period not in self.returns_data:
                logging.info(f"返回周期 {period} 不存在")
                return

            returns_panel = self.returns_data[period]

            # 初始化累积收益数据框
            cum_returns = pd.DataFrame(index=returns_panel.index)

            # 每日重新分组计算累积收益
            for date in sorted(self.factor_data.index):
                if date not in returns_panel.index:
                    continue

                # 获取当天所有股票的因子值和未来收益
                factors = self.factor_data.loc[date].dropna()
                future_returns = returns_panel.loc[date, factors.index].dropna()

                # 对齐数据
                common_stocks = factors.index.intersection(future_returns.index)
                if len(common_stocks) < 5:  # 要求至少有5只股票
                    continue

                aligned_factors = factors[common_stocks]
                aligned_returns = future_returns[common_stocks]

                # 按因子值分组
                try:
                    quantiles = pd.qcut(aligned_factors, 5, labels=False) + 1

                    # 计算每组平均收益
                    for q in range(1, 6):
                        stocks_in_quantile = (quantiles == q)
                        if stocks_in_quantile.sum() > 0:
                            returns_in_quantile = aligned_returns[stocks_in_quantile].mean()
                            cum_returns.loc[date, f'Q{q}'] = returns_in_quantile
                except ValueError:
                    continue

            # 计算累积收益
            cum_returns = cum_returns.dropna(how='all')
            for col in cum_returns.columns:
                cum_returns[col] = (1 + cum_returns[col].fillna(0)).cumprod() - 1

            # 添加多空组合
            if 'Q5' in cum_returns.columns and 'Q1' in cum_returns.columns:
                cum_returns['Q5-Q1'] = cum_returns['Q5'] - cum_returns['Q1']

            plt.figure(figsize=(12, 6))
            cum_returns.plot()
            plt.title(f'Cumulative Returns by Factor Quantile ({period})')
            plt.ylabel('Cumulative Return')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            # 单只股票数据的累积收益曲线（原有逻辑）
            # 对齐因子数据和收益率数据
            aligned_data = pd.DataFrame({
                'factor': self.factor_data,
                'return': self.returns_data[period]
            }).dropna()

            # 计算分位数
            aligned_data['quantile'] = pd.qcut(
                aligned_data['factor'],
                5,
                labels=False
            ) + 1

            # 计算每个分位数组的累积收益
            cumulative_returns = pd.DataFrame(index=aligned_data.index)

            for quantile in range(1, 6):
                returns = aligned_data[aligned_data['quantile'] == quantile]['return']
                cumulative_returns[f'Q{quantile}'] = (1 + returns).cumprod() - 1

            # 添加多空组合
            cumulative_returns['Q5-Q1'] = cumulative_returns['Q5'] - cumulative_returns['Q1']

            plt.figure(figsize=(12, 6))
            cumulative_returns.plot()
            plt.title(f'Cumulative Returns by Factor Quantile ({period})')
            plt.ylabel('Cumulative Return')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def generate_report(self) -> Dict:
        """
        生成完整的因子分析报告
        
        Returns:
            包含所有分析结果的字典
        """
        # 计算各项指标
        if self.ic_series is None:
            self.calculate_ic()

        if self.group_returns is None:
            self.calculate_quantile_returns()

        win_rates = self.calculate_win_rate()

        # 汇总报告
        if self.is_panel_data:
            # 多只股票数据的报告

            # 更灵活地选择代表周期：优先使用20日，否则使用第一个可用周期
            available_periods = self.ic_series.index.tolist()
            representative_period = 'return_20d' if 'return_20d' in available_periods else available_periods[
                0] if available_periods else None

            # 计算所有周期的平均值作为备选
            ic_mean_all_periods = self.ic_series['ic'].mean() if not self.ic_series.empty else np.nan
            ic_ir_all_periods = self.ic_series['ic_ir'].mean() if not self.ic_series.empty else np.nan

            report = {
                'ic_data': self.ic_series,
                'group_returns': self.group_returns,
                'win_rates': win_rates,
                'summary': {
                    # 主要指标：使用代表周期或所有周期均值
                    'ic_mean': self.ic_series.loc[
                        representative_period, 'ic'] if representative_period else ic_mean_all_periods,
                    'ic_std': self.ic_series.loc[representative_period, 'ic_std'] if representative_period else np.nan,
                    'ic_ir': self.ic_series.loc[
                        representative_period, 'ic_ir'] if representative_period else ic_ir_all_periods,
                    'ic_pos_rate': self.ic_series.loc[
                        representative_period, 'pos_rate'] if representative_period else np.nan,

                    # 收益相关指标
                    'top_group_return': self.group_returns.loc[5, representative_period]
                    if 5 in self.group_returns.index and representative_period in self.group_returns.columns
                    else np.nan,
                    'bottom_group_return': self.group_returns.loc[1, representative_period]
                    if 1 in self.group_returns.index and representative_period in self.group_returns.columns
                    else np.nan,
                    'long_short_return': self.group_returns.loc['long_short', representative_period]
                    if 'long_short' in self.group_returns.index and representative_period in self.group_returns.columns
                    else np.nan,
                    'pure_long_return': self.group_returns.loc['top_group', representative_period]
                    if 'top_group' in self.group_returns.index and representative_period in self.group_returns.columns
                    else np.nan,
                    'top_group_win_rate': win_rates.loc[5, representative_period]
                    if 5 in win_rates.index and representative_period in win_rates.columns
                    else np.nan,

                    # 附加所有周期的均值指标
                    'all_periods_ic_mean': ic_mean_all_periods,
                    'all_periods_ic_ir': ic_ir_all_periods,
                    'representative_period': representative_period
                }
            }
        else:
            # 单只股票数据的报告（原有逻辑）
            report = {
                'ic_data': self.ic_series,
                'group_returns': self.group_returns,
                'win_rates': win_rates,
                'summary': {
                    'ic_mean': self.ic_series.loc[
                        'return_20d', 'ic'] if 'return_20d' in self.ic_series.index else np.nan,
                    'ic_std': self.ic_series.loc[
                        'return_20d', 'ic'].std() if 'return_20d' in self.ic_series.index else np.nan,
                    'ic_ir': self.ic_series.loc['return_20d', 'ic'].mean() / self.ic_series.loc[
                        'return_20d', 'ic'].std()
                    if 'return_20d' in self.ic_series.index and self.ic_series.loc[
                        'return_20d', 'ic'].std() != 0 else np.nan,
                    'top_group_return': self.group_returns.loc[
                        5, 'return_20d'] if 5 in self.group_returns.index and 'return_20d' in self.group_returns.columns else np.nan,
                    'bottom_group_return': self.group_returns.loc[
                        1, 'return_20d'] if 1 in self.group_returns.index and 'return_20d' in self.group_returns.columns else np.nan,
                    'long_short_return': self.group_returns.loc[
                        'long_short', 'return_20d'] if 'long_short' in self.group_returns.index and 'return_20d' in self.group_returns.columns else np.nan,
                    'top_group_win_rate': win_rates.loc[
                        5, 'monthly_return_20d'] if 5 in win_rates.index and 'monthly_return_20d' in win_rates.columns else np.nan
                }
            }

        return report

    def print_summary(self, need_plot=True) -> None:
        """打印因子分析摘要"""
        report = self.generate_report()
        summary = report['summary']

        logging.info("=" * 50)
        logging.info("因子有效性分析摘要")
        logging.info("=" * 50)

        try:
            # ====== IC分析结果解读 ======
            logging.info("\n【IC分析结果】(Information Coefficient，因子预测能力)")
            logging.info("-" * 50)

            if self.ic_series is not None and not self.ic_series.empty:
                # 计算各列最大宽度
                periods = self.ic_series.index.tolist()
                max_period_len = max(len(p) for p in periods) + 1
                col_widths = {
                    'ic': 8,
                    'ic_std': 8,
                    'ic_ir': 6,
                    'pos_rate': 6,
                    'abs_ic': 8
                }

                # 构建表头
                header_parts = [
                    f"{'收益周期':<{max_period_len - 2}}",
                    f"{'IC均值':^{col_widths['ic'] - 1}}",
                    f"{'IC标准差':^{col_widths['ic_std'] - 2}}",
                    f"{'IR':^{col_widths['ic_ir']}}",
                    f"{'正比例':^{col_widths['pos_rate'] - 2}}",
                    f"{'绝对IC':^{col_widths['abs_ic']}}"
                ]
                header = "│".join(header_parts)

                # 构建分隔线
                separator = "─" * max_period_len + "┼" + \
                            "─" * col_widths['ic'] + "┼" + \
                            "─" * col_widths['ic_std'] + "┼" + \
                            "─" * col_widths['ic_ir'] + "┼" + \
                            "─" * col_widths['pos_rate'] + "┼" + \
                            "─" * col_widths['abs_ic']

                logging.info("\nIC统计表：")
                logging.info(header)
                logging.info(separator)

                # 输出每行数据
                for period, row in self.ic_series.iterrows():
                    parts = [
                        f"{period:<{max_period_len}}",
                        f"{row['ic']:^{col_widths['ic']}.4f}",
                        f"{row['ic_std']:^{col_widths['ic_std']}.4f}",
                    ]
                    ic_ir_value = row['ic_ir']
                    if pd.isna(ic_ir_value):
                        parts.append(f"{'N/A':^{col_widths['ic_ir']}}")
                    else:
                        parts.append(f"{ic_ir_value:^{col_widths['ic_ir']}.2f}")

                    parts.extend([
                        f"{row['pos_rate']:^{col_widths['pos_rate']}.1%}",
                        f"{row['abs_ic']:^{col_widths['abs_ic']}.4f}"
                    ])
                    logging.info("│".join(parts))

                # 添加单位说明
                logging.info("* 单位说明:")
                logging.info("  - IC均值和标准差保留4位小数")
                logging.info("  - IR(信息比率) = IC均值 / IC标准差")
                logging.info("  - 正比例显示为百分比格式")
            else:
                logging.info("未能计算有效IC值，请检查数据")

            # ====== 分组收益结果解读 ======
            logging.info("\n\n【分组收益分析】(按因子值分组后各组平均收益)")
            logging.info("-" * 50)

            if self.group_returns is not None and not self.group_returns.empty:
                # 打印格式化的分组收益表
                pd.set_option('display.float_format', '{:.2%}'.format)
                logging.info(self.group_returns)
                pd.reset_option('display.float_format')

                # 提取关键信息并解读
                if 'long_short' in self.group_returns.index:
                    logging.info("\n多空组合收益:")
                    for col in self.group_returns.columns:
                        logging.info(f"{col}: {self.group_returns.loc['long_short', col]:.2%}")

                logging.info("\n\n* 分组解读: 若高分位组收益显著高于低分位组，表明因子有区分能力")
                logging.info("* 多空组合: 顶层组减底层组的收益，越大表示区分能力越强")
            else:
                logging.info("未能计算有效分组收益，请检查数据")

            # ====== 胜率分析结果解读 ======
            win_rates = self.calculate_win_rate()
            logging.info("\n\n【胜率分析】(收益为正的比例)")
            logging.info("-" * 50)

            if win_rates is not None and not win_rates.empty:
                pd.set_option('display.float_format', '{:.2%}'.format)
                logging.info(win_rates)
                pd.reset_option('display.float_format')

                logging.info("\n* 胜率解读: 顶层组胜率>55%通常认为因子有效")
            else:
                logging.info("未能计算有效胜率，请检查数据")

        except Exception as e:
            logging.exception(f"打印摘要时出错: {e}")
            import traceback
            traceback.print_exc()

        logging.info("\n" + "=" * 50)

        if need_plot:
            # 绘图
            try:
                self.plot_ic_series()
            except Exception as e:
                logging.exception(f"绘制IC序列时出错: {e}")

            try:
                self.plot_quantile_returns()
            except Exception as e:
                logging.info(f"绘制分位数收益时出错: {e}")

            try:
                self.plot_cumulative_returns()
            except Exception as e:
                logging.info(f"绘制累积收益时出错: {e}")

    def save_report(self, filename: str, factor_name: str) -> None:
        """
        将因子分析结果保存为HTML报告
        
        Args:
            filename: 报告文件名，应以.html结尾
            factor_name: 因子名称，将显示在报告标题中
        
        Returns:
            None，但会在指定目录生成HTML报告文件
        """
        import os
        import jinja2
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        # 确保目标目录存在
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logging.info(f"已创建目录: {directory}")
            except Exception as e:
                logging.exception(f"无法创建目录: {e}")
                # 使用当前目录作为备选方案
                filename = os.path.basename(filename)
                logging.info(f"将在当前目录保存报告: {filename}")

        # 确保生成报告前已计算所有必要指标
        if self.ic_series is None:
            self.calculate_ic()

        if self.group_returns is None:
            self.calculate_quantile_returns()

        win_rates = self.calculate_win_rate()
        report_data = self.generate_report()

        # 纯函数：将DataFrame转换为HTML表格
        def df_to_html(df, float_format='{:.4f}', caption=''):
            if df is None or df.empty:
                return "<p>无有效数据</p>"
            styled_df = df.style.format(float_format)
            if caption:
                styled_df = styled_df.set_caption(caption)
            return styled_df.to_html()

        # 修复图表生成函数，确保每个图表有新的图形对象并正确关闭
        def plot_to_base64(plot_func):
            try:
                # 清除当前图形
                plt.close('all')
                # 创建新的图表
                fig, ax = plt.subplots(figsize=(10, 6))
                # 执行绘图函数，传入轴对象
                plot_func(ax)
                # 调整布局
                plt.tight_layout()
                # 保存到内存缓冲区
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                # 转换为base64
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f'<img src="data:image/png;base64,{img_str}" alt="图表" style="max-width:100%;" />'
            except Exception as e:
                return f'<div class="error">生成图表时出错: {str(e)}</div>'

        # 生成IC图表 - 修改为接受ax参数的版本
        def plot_ic(ax):
            if self.ic_series is not None and not self.ic_series.empty:
                self.ic_series.plot(y='ic', kind='bar', ax=ax, color='skyblue')
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                ax.set_title('IC值分布')
                ax.set_ylabel('IC值')
                ax.set_xlabel('收益周期')
                # 添加水平参考线
                ax.axhline(y=0.05, color='g', linestyle='--', alpha=0.5)
                ax.axhline(y=-0.05, color='g', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)

        # 生成分位数收益图表
        def plot_quantile(ax):
            if self.group_returns is not None and not self.group_returns.empty:
                # 只选择数值分组，排除long_short行
                returns = self.group_returns.loc[[i for i in self.group_returns.index if isinstance(i, (int, float))]]
                returns.T.plot(kind='bar', ax=ax)
                ax.set_title('分位数收益对比')
                ax.set_ylabel('收益率')
                ax.set_xlabel('收益周期')
                ax.grid(True, alpha=0.3)
                ax.legend(title='分位数')

        # 生成累积收益图表
        def plot_cumulative(ax):
            if hasattr(self, 'cumulative_returns') and self.cumulative_returns is not None:
                self.cumulative_returns.plot(ax=ax)
                ax.set_title('累积收益曲线')
                ax.set_ylabel('累积收益')
                ax.set_xlabel('日期')
                ax.grid(True, alpha=0.3)
            else:
                # 如果没有累积收益数据，尝试计算并绘制
                try:
                    if self.group_returns is not None and not self.group_returns.empty:
                        # 简单绘制最高分位和最低分位的对比
                        if 5 in self.group_returns.index and 1 in self.group_returns.index:
                            # 创建一个虚拟的累积收益曲线
                            import pandas as pd
                            import numpy as np
                            columns = self.group_returns.columns
                            dates = pd.date_range(end=pd.Timestamp.today(), periods=len(columns))
                            data = {
                                '最高分位(纯多头)': np.cumprod(1 + self.group_returns.loc[5].values) - 1,
                                '最低分位': np.cumprod(1 + self.group_returns.loc[1].values) - 1,
                                '多空组合': np.cumprod(1 + self.group_returns.loc['long_short'].values) - 1
                            }
                            pd.DataFrame(data, index=columns).plot(ax=ax)
                            ax.set_title('累积收益曲线(模拟)')
                            ax.set_ylabel('累积收益')
                            ax.grid(True, alpha=0.3)
                except Exception as e:
                    ax.text(0.5, 0.5, f"无法生成累积收益图表: {e}",
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        # 生成图表
        ic_plot = plot_to_base64(plot_ic)
        quantile_plot = plot_to_base64(plot_quantile)
        cumulative_plot = plot_to_base64(plot_cumulative)

        # 预先格式化数据，避免在模板中使用Python的格式化语法
        summary = {
            'ic_mean': f"{report_data['summary'].get('ic_mean', float('nan')):.4f}",
            'ic_ir': f"{report_data['summary'].get('ic_ir', float('nan')):.4f}",
            'ic_pos_rate': f"{report_data['summary'].get('ic_pos_rate', float('nan')) * 100:.2f}%",
            'long_short_return': f"{report_data['summary'].get('long_short_return', float('nan')) * 100:.2f}%",
            'pure_long_return': f"{report_data['summary'].get('pure_long_return', float('nan')) * 100:.2f}%",
            'top_group_win_rate': f"{report_data['summary'].get('top_group_win_rate', float('nan')) * 100:.2f}%"
        }

        # 其余代码保持不变
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>因子分析报告 - {{factor_name}}</title>
            <style>
                body { font-family: SimHei, Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; text-align: center; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2980b9; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .summary { background-color: #eaf7ff; padding: 15px; border-radius: 5px; }
                .chart { margin: 20px 0; text-align: center; }
                .footnote { font-size: 0.8em; color: #7f8c8d; margin-top: 20px; }
                .error { color: #e74c3c; padding: 10px; background-color: #fadbd8; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>因子分析报告 - {{factor_name}}</h1>
            
            <div class="summary">
                <h2>摘要</h2>
                <p>IC均值: {{summary.ic_mean}} (>0.05为强有效因子)</p>
                <p>IC IR值: {{summary.ic_ir}} (>1.0为优秀因子, >1.5为极优)</p>
                <p>IC正比例: {{summary.ic_pos_rate}} (>55%表示方向一致)</p>
                <p>多空组合收益: {{summary.long_short_return}} (越大表示区分能力越强)</p>
                <p>纯多头收益: {{summary.pure_long_return}} (顶层组绝对收益)</p>
                <p>顶层组胜率: {{summary.top_group_win_rate}} (>55%通常认为有效)</p>
            </div>
            
            <h2>IC分析</h2>
            <p>Information Coefficient（因子与未来收益的相关性）</p>
            {{ic_table}}
            
            <div class="chart">
                <h3>IC时间序列</h3>
                {{ic_plot}}
            </div>
            
            <h2>分组收益分析</h2>
            <p>按因子值分组后各组平均收益</p>
            {{group_returns_table}}
            
            <div class="chart">
                <h3>分位数收益热力图</h3>
                {{quantile_plot}}
            </div>
            
            <div class="chart">
                <h3>累积收益曲线</h3>
                {{cumulative_plot}}
            </div>
            
            <h2>胜率分析</h2>
            <p>各分组收益为正的比例</p>
            {{win_rates_table}}
            
            <div class="footnote">
                <p>生成时间: {{generation_time}}</p>
                <p>注：IC均值大于0.05通常被视为强有效因子；IR>1.0被视为优秀因子；顶层组胜率>55%通常认为因子有效</p>
            </div>
        </body>
        </html>
        """

        # 格式化表格
        ic_table = df_to_html(self.ic_series, float_format='{:.4f}', caption='IC值统计表')
        group_returns_table = df_to_html(self.group_returns, float_format='{:.2%}', caption='分组收益表')
        win_rates_table = df_to_html(win_rates, float_format='{:.2%}', caption='胜率统计表')

        # 准备报告数据
        from datetime import datetime
        context = {
            'factor_name': factor_name,  # 添加因子名称到上下文
            'summary': summary,
            'ic_table': ic_table,
            'group_returns_table': group_returns_table,
            'win_rates_table': win_rates_table,
            'ic_plot': ic_plot,
            'quantile_plot': quantile_plot,
            'cumulative_plot': cumulative_plot,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 渲染HTML
        template = jinja2.Template(html_template)
        html_content = template.render(**context)

        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logging.info(f"分析报告已保存到: {os.path.abspath(filename)}")


def analyze_single_factor(factor_data: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]],
                          price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                          factor_name: str = '',
                          date_col: str = 'trade_date',
                          need_plot: bool = True) -> FactorAnalyzer:
    """
    单因子有效性分析主函数
    
    Args:
        factor_data: 因子值，可以是:
                     - 单只股票的Series
                     - 多只股票的DataFrame(索引为日期，列为股票代码)
                     - 多只股票的字典，key为股票代码，value为该股票的因子值Series
        price_data: 价格数据，可以是:
                    - 单只股票的DataFrame，包含close列
                    - 多只股票的字典，key为股票代码，value为该股票的价格DataFrame
        factor_name: 因子名称，用于显示
        date_col: 日期列名称，当索引不是日期时使用此列作为日期索引
        need_plot: 是否需要画图
        
    Returns:
        因子分析器对象
    """
    # 判断是否为多股票数据
    is_panel_data = isinstance(factor_data, dict) or (
            isinstance(factor_data, pd.DataFrame) and factor_data.shape[1] > 1)

    if is_panel_data:
        # 处理多只股票数据的情况

        # 处理因子数据已经是面板格式的情况
        if isinstance(factor_data, pd.DataFrame) and factor_data.shape[1] > 1:
            factor_panel = factor_data

            # 如果需要，转换price_data为面板格式
            if isinstance(price_data, dict) and all(isinstance(df, pd.DataFrame) for df in price_data.values()):
                # 确保所有DataFrame都使用日期索引
                all_dates = set()
                for code, df in price_data.items():
                    if date_col in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                        df.set_index(pd.to_datetime(df[date_col]), inplace=True)
                    all_dates.update(df.index)

                date_index = pd.DatetimeIndex(sorted(all_dates))
                price_panel = {'close': pd.DataFrame(index=date_index)}

                for code, df in price_data.items():
                    price_panel['close'][code] = df['close']
            else:
                # 如果price_data已经是面板格式
                price_panel = price_data

        # 处理因子数据是字典格式的情况
        else:
            # 第一步：获取所有日期
            all_dates = set()
            for code, df in price_data.items():
                if date_col in df.columns:
                    all_dates.update(pd.to_datetime(df[date_col]))
                else:
                    all_dates.update(df.index)

            date_index = pd.DatetimeIndex(sorted(all_dates))

            # 第二步：合并因子数据到面板
            factor_data_dict = {}  # 收集所有股票的因子数据

            for code, series in factor_data.items():
                if date_col in price_data[code].columns:
                    # 获取日期映射
                    date_map = dict(zip(
                        price_data[code].index,
                        pd.to_datetime(price_data[code][date_col])
                    ))
                    # 创建临时Series并重新索引
                    temp_series = pd.Series(
                        series.values,
                        index=[date_map.get(idx) for idx in series.index],
                        name=code
                    ).dropna()
                else:
                    # 如果价格数据已经使用日期索引
                    temp_series = series.copy()
                    temp_series.index = pd.to_datetime(temp_series.index)

                # 添加到数据字典而不是直接添加到面板
                factor_data_dict[code] = temp_series

            # 一次性构建DataFrame，避免内存碎片化
            factor_panel = pd.DataFrame(factor_data_dict, index=date_index)

            # 第三步：合并价格数据到面板
            price_panel = {}

            # 收集所有股票的收盘价数据
            close_data = {}
            for code, df in price_data.items():
                # 设置日期索引
                temp_df = df.copy()
                if date_col in temp_df.columns and not isinstance(temp_df.index, pd.DatetimeIndex):
                    temp_df.set_index(pd.to_datetime(temp_df[date_col]), inplace=True)
                # 添加到临时字典
                close_data[code] = temp_df['close']

            # 一次性构建DataFrame，避免内存碎片化
            price_panel['close'] = pd.DataFrame(close_data, index=date_index)

        # 创建分析器对象
        analyzer = FactorAnalyzer(factor_panel, price_panel, is_panel_data=True)

    else:
        # 处理单只股票数据的情况（原有逻辑）
        price_data_copy = price_data.copy()

        # 保存原始索引和trade_date的对应关系
        if date_col in price_data_copy.columns and not isinstance(price_data_copy.index, pd.DatetimeIndex):
            index_date_map = dict(zip(price_data_copy.index, pd.to_datetime(price_data_copy[date_col])))

            # 设置trade_date为索引
            price_data_copy.set_index(date_col, inplace=True)
            price_data_copy.index = pd.to_datetime(price_data_copy.index)

            # 重建factor_data，使其索引为对应的trade_date
            new_factor_data = pd.Series(
                factor_data.values,
                index=[index_date_map.get(idx) for idx in factor_data.index],
                name=factor_data.name
            )
        else:
            # 如果已经是日期索引或没有trade_date列，则不做处理
            new_factor_data = factor_data

        # 创建分析器对象
        analyzer = FactorAnalyzer(new_factor_data, price_data_copy, is_panel_data=False)

    # 计算未来收益率
    analyzer.calculate_forward_returns()

    # 打印报告
    logging.info(f"\n{factor_name}因子分析报告" if factor_name else "\n因子分析报告")
    analyzer.print_summary(need_plot=need_plot)

    return analyzer


if __name__ == "__main__":
    # 使用示例代码 - 函数式编程方法计算因子
    from backend.quant.core.factor_engine.factor_generator import get_registered_factors
    from backend.data.stock_data_fetcher import StockDataFetcher
    import pandas as pd

    # 创建股票数据获取器
    fetcher = StockDataFetcher()

    # 获取所有注册的因子
    try:
        factors = get_registered_factors()
        logging.info(f"获取到以下因子: {list(factors.keys())}")
    except Exception as e:
        logging.exception(f"获取因子时出错: {e}")
        # 使用备用方案
        factors = {}

    # 准备股票池和日期范围 - 增加股票数量以便更好地进行分组
    stock_codes = ["sh.605300", "sz.300490", "sh.603336", "sh.600519", "sz.000858",
                   "sh.601398", "sz.000651", "sh.601318", "sz.000333", "sh.600036"]
    start_date = "2022-01-01"
    end_date = "2023-03-01"

    logging.info(f"分析周期: {start_date} 至 {end_date}")
    logging.info(f"股票样本: {stock_codes}")

    # 获取股票数据并计算因子
    price_data = {}
    factor_values = {}
    valid_stocks = 0

    for code in stock_codes:
        try:
            # 获取股票数据
            df = fetcher.fetch_stock_data(code=code, start_date=start_date, end_date=end_date)

            if df is None or df.empty:
                logging.info(f"警告: 无法获取股票 {code} 的数据")
                continue

            valid_stocks += 1
            price_data[code] = df

            # 计算因子
            factor_values[code] = factors['macd'](close=df.close)
        except Exception as e:
            logging.exception(f"处理股票 {code} 时出错: {e}")

    logging.info(f"成功处理 {valid_stocks} 只股票数据")

    # 对因子进行有效性分析
    try:
        logging.info("\n开始因子有效性分析...")
        analyzer = analyze_single_factor(
            factor_data=factor_values,
            price_data=price_data,
            factor_name='macd',
            date_col='trade_date'
        )

    except Exception as e:
        logging.exception(f"因子分析过程中出错: {e}")
        import traceback

        traceback.print_exc()
