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
                # 对每只股票计算未来收益率
                pct_change = close_panel.pct_change(period).shift(-period)
                returns[f'return_{period}d'] = pct_change
        else:
            # 单只股票数据的处理
            returns = pd.DataFrame(index=self.price_data.index)

            for period in periods:
                pct_change = self.price_data['close'].pct_change(period).shift(-period)
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
            print(f"因子数据形状: {self.factor_data.shape}")
            print(f"收益率数据周期: {list(self.returns_data.keys())}")

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
                        print(f"计算{date}的IC值时出错: {e}")

                # 输出统计信息
                print(f"周期 {period} 统计：")
                print(f"  - 有效交易日: {valid_dates}")
                print(f"  - 因子值为常数的交易日: {constant_dates}")
                print(f"  - 总交易日: {len(self.factor_data.index)}")

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
                    print(f"警告: {period} 周期无有效IC值")
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

        # 添加多空组合收益
        if num_quantiles > 1 and not self.group_returns.empty:
            long_short = self.group_returns.loc[num_quantiles] - self.group_returns.loc[1]
            self.group_returns.loc['long_short'] = long_short

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
                print(f"返回周期 {period} 不存在")
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
            report = {
                'ic_data': self.ic_series,
                'group_returns': self.group_returns,
                'win_rates': win_rates,
                'summary': {
                    'ic_mean': self.ic_series.loc[
                        'return_20d', 'ic'] if 'return_20d' in self.ic_series.index else np.nan,
                    'ic_pos_rate': self.ic_series.loc[
                        'return_20d', 'pos_rate'] if 'return_20d' in self.ic_series.index else np.nan,
                    'top_group_return': self.group_returns.loc[
                        5, 'return_20d'] if 5 in self.group_returns.index and 'return_20d' in self.group_returns.columns else np.nan,
                    'bottom_group_return': self.group_returns.loc[
                        1, 'return_20d'] if 1 in self.group_returns.index and 'return_20d' in self.group_returns.columns else np.nan,
                    'long_short_return': self.group_returns.loc[
                        'long_short', 'return_20d'] if 'long_short' in self.group_returns.index and 'return_20d' in self.group_returns.columns else np.nan,
                    'top_group_win_rate': win_rates.loc[
                        5, 'return_20d'] if 5 in win_rates.index and 'return_20d' in win_rates.columns else np.nan
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

    def print_summary(self) -> None:
        """打印因子分析摘要"""
        report = self.generate_report()
        summary = report['summary']

        print("\n" + "=" * 50)
        print("因子有效性分析摘要")
        print("=" * 50)

        try:
            # ====== IC分析结果解读 ======
            print("\n【IC分析结果】(Information Coefficient，因子预测能力)")
            print("-" * 50)

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
                separator = "─"*max_period_len + "┼" + \
                            "─"*col_widths['ic'] + "┼" + \
                            "─"*col_widths['ic_std'] + "┼" + \
                            "─"*col_widths['ic_ir'] + "┼" + \
                            "─"*col_widths['pos_rate'] + "┼" + \
                            "─"*col_widths['abs_ic']

                print("\nIC统计表：")
                print(header)
                print(separator)

                # 输出每行数据
                for period, row in self.ic_series.iterrows():
                    parts = [
                        f"{period:<{max_period_len}}",
                        f"{row['ic']:^{col_widths['ic']}.4f}",
                        f"{row['ic_std']:^{col_widths['ic_std']}.4f}",
                        f"{row['ic_ir'] if not pd.isna(row['ic_ir']) else 'N/A':^{col_widths['ic_ir']}.2f}",
                        f"{row['pos_rate']:^{col_widths['pos_rate']}.1%}",
                        f"{row['abs_ic']:^{col_widths['abs_ic']}.4f}"
                    ]
                    print("│".join(parts))

                # 添加单位说明
                print("\n* 单位说明:")
                print("  - IC均值和标准差保留4位小数")
                print("  - IR(信息比率) = IC均值 / IC标准差")
                print("  - 正比例显示为百分比格式")
            else:
                print("未能计算有效IC值，请检查数据")

            # ====== 分组收益结果解读 ======
            print("\n\n【分组收益分析】(按因子值分组后各组平均收益)")
            print("-" * 50)

            if self.group_returns is not None and not self.group_returns.empty:
                # 打印格式化的分组收益表
                pd.set_option('display.float_format', '{:.2%}'.format)
                print(self.group_returns)
                pd.reset_option('display.float_format')

                # 提取关键信息并解读
                if 'long_short' in self.group_returns.index:
                    print("\n多空组合收益:", end=' ')
                    for col in self.group_returns.columns:
                        print(f"{col}: {self.group_returns.loc['long_short', col]:.2%}", end='  ')

                print("\n\n* 分组解读: 若高分位组收益显著高于低分位组，表明因子有区分能力")
                print("* 多空组合: 顶层组减底层组的收益，越大表示区分能力越强")
            else:
                print("未能计算有效分组收益，请检查数据")

            # ====== 胜率分析结果解读 ======
            win_rates = self.calculate_win_rate()
            print("\n\n【胜率分析】(收益为正的比例)")
            print("-" * 50)

            if win_rates is not None and not win_rates.empty:
                pd.set_option('display.float_format', '{:.2%}'.format)
                print(win_rates)
                pd.reset_option('display.float_format')

                print("\n* 胜率解读: 顶层组胜率>55%通常认为因子有效")
            else:
                print("未能计算有效胜率，请检查数据")

        except Exception as e:
            print(f"打印摘要时出错: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 50)

        # 绘图，添加异常处理
        try:
            self.plot_ic_series()
        except Exception as e:
            print(f"绘制IC序列时出错: {e}")

        try:
            self.plot_quantile_returns()
        except Exception as e:
            print(f"绘制分位数收益时出错: {e}")

        try:
            self.plot_cumulative_returns()
        except Exception as e:
            print(f"绘制累积收益时出错: {e}")


def analyze_single_factor(factor_data: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]],
                          price_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                          factor_name: str = '',
                          date_col: str = 'trade_date') -> FactorAnalyzer:
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
            factor_panel = pd.DataFrame(index=date_index)
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

                # 添加到面板
                factor_panel[code] = temp_series

            # 第三步：合并价格数据到面板
            price_panel = {}
            price_panel['close'] = pd.DataFrame(index=date_index)

            for code, df in price_data.items():
                # 设置日期索引
                temp_df = df.copy()
                if date_col in temp_df.columns and not isinstance(temp_df.index, pd.DatetimeIndex):
                    temp_df.set_index(pd.to_datetime(temp_df[date_col]), inplace=True)
                # 添加到面板
                price_panel['close'][code] = temp_df['close']

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
    print(f"\n{factor_name}因子分析报告" if factor_name else "\n因子分析报告")
    analyzer.print_summary()

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
        print(f"获取到以下因子: {list(factors.keys())}")
    except Exception as e:
        print(f"获取因子时出错: {e}")
        # 使用备用方案
        factors = {}

    # 准备股票池和日期范围 - 增加股票数量以便更好地进行分组
    stock_codes = ["sh.605300", "sz.300490", "sh.603336", "sh.600519", "sz.000858",
                   "sh.601398", "sz.000651", "sh.601318", "sz.000333", "sh.600036"]
    start_date = "2022-01-01"
    end_date = "2023-03-01"

    print(f"分析周期: {start_date} 至 {end_date}")
    print(f"股票样本: {stock_codes}")

    # 获取股票数据并计算因子
    price_data = {}
    factor_values = {}
    valid_stocks = 0

    for code in stock_codes:
        try:
            # 获取股票数据
            df = fetcher.fetch_stock_data(code=code, start_date=start_date, end_date=end_date)

            if df is None or df.empty:
                print(f"警告: 无法获取股票 {code} 的数据")
                continue

            valid_stocks += 1
            price_data[code] = df

            # 计算因子
            factor_values[code] = factors['macd'](close=df.close)
        except Exception as e:
            print(f"处理股票 {code} 时出错: {e}")

    print(f"成功处理 {valid_stocks} 只股票数据")

    # 对因子进行有效性分析
    try:
        print("\n开始因子有效性分析...")
        analyzer = analyze_single_factor(
            factor_data=factor_values,
            price_data=price_data,
            factor_name='macd',
            date_col='trade_date'
        )

    except Exception as e:
        print(f"因子分析过程中出错: {e}")
        import traceback

        traceback.print_exc()
