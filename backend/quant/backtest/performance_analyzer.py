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
        1. 计算因子IC值（因子值与未来收益的秩相关性）
        2. 按因子分位数分组统计各组平均收益
        3. 生成月度胜率报告

    输出示例：
        PE因子近5年IC均值 = 0.12
        前10%低PE组5日平均收益 = 1.2%
        胜率 = 58%
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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

    def __init__(self, factor_data: pd.Series, price_data: pd.DataFrame):
        """
        初始化因子分析器
        
        Args:
            factor_data: 因子值序列，包含日期索引
            price_data: 价格数据，至少包含close列和日期索引
        """
        self.factor_data = factor_data
        self.price_data = price_data
        self.returns_data = None
        self.ic_series = None
        self.group_returns = None

    def calculate_forward_returns(self, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        计算未来收益率
        
        Args:
            periods: 未来收益率的周期列表，单位为交易日
            
        Returns:
            未来收益率数据框
        """
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
            method: 相关系数计算方法，'pearson' 或 'spearman'(秩相关)
            
        Returns:
            IC值数据框，包含每个周期的IC值
        """
        if self.returns_data is None:
            self.calculate_forward_returns()

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

        return pd.DataFrame({
            'ic': self.ic_series,
            'abs_ic': self.ic_series.abs(),
            'pos_rate': (monthly_ic > 0).mean(),
            'mean_monthly_ic': monthly_ic.mean()
        })

    def _calculate_monthly_ic(self, method: str = 'spearman') -> pd.Series:
        """
        计算月度IC值
        
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
            lambda x: x['factor'].corr(x['return'], method=method),
            include_groups=False  # 添加此参数以避免警告
        )

        return monthly_ic

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

        # 计算多空组合收益
        long_short = {}
        for col in self.returns_data.columns:
            # 最高分位组减去最低分位组
            long_short[col] = self.group_returns.loc[num_quantiles, col] - self.group_returns.loc[1, col]

        # 添加多空组合收益
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
        if self.ic_series is None:
            self.calculate_ic()

        plt.figure(figsize=(12, 6))
        self.ic_series.plot(kind='bar')
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
        ic_data = self.calculate_ic()
        group_returns = self.calculate_quantile_returns()
        win_rates = self.calculate_win_rate()

        # 汇总报告
        report = {
            'ic_data': ic_data,
            'group_returns': group_returns,
            'win_rates': win_rates,
            'summary': {
                'ic_mean': self.ic_series.mean(),
                'ic_std': self.ic_series.std(),
                'ic_ir': self.ic_series.mean() / self.ic_series.std() if self.ic_series.std() != 0 else np.nan,
                'top_group_return': group_returns.loc[5, 'return_20d'],
                'bottom_group_return': group_returns.loc[1, 'return_20d'],
                'long_short_return': group_returns.loc['long_short', 'return_20d'],
                'top_group_win_rate': win_rates.loc[5, 'monthly_return_20d']
            }
        }

        return report

    def print_summary(self) -> None:
        """打印因子分析摘要"""
        report = self.generate_report()
        summary = report['summary']

        print("=" * 50)
        print("因子有效性分析摘要")
        print("=" * 50)
        print(f"因子IC均值: {summary['ic_mean']:.4f}")
        print(f"因子IC标准差: {summary['ic_std']:.4f}")
        print(f"因子信息比率(IR): {summary['ic_ir']:.4f}")
        print(f"顶层分位组20日收益: {summary['top_group_return']:.2%}")
        print(f"底层分位组20日收益: {summary['bottom_group_return']:.2%}")
        print(f"多空组合收益: {summary['long_short_return']:.2%}")
        print(f"顶层分位组月度胜率: {summary['top_group_win_rate']:.2%}")
        print("=" * 50)

        # 绘图
        self.plot_ic_series()
        self.plot_quantile_returns()
        self.plot_cumulative_returns()


def analyze_single_factor(factor_data: pd.Series, price_data: pd.DataFrame, factor_name: str = '', date_col: str = 'trade_date') -> FactorAnalyzer:
    """
    单因子有效性分析主函数
    
    Args:
        factor_data: 因子值序列，包含索引（对应price_data的索引）
        price_data: 价格数据，至少包含close列和日期列
        factor_name: 因子名称，用于显示
        date_col: 日期列名称，当索引不是日期时使用此列作为日期索引
    """
    # 处理数据索引，使trade_date成为索引
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
    
    # 创建分析器并执行分析
    analyzer = FactorAnalyzer(new_factor_data, price_data_copy)
    analyzer.calculate_forward_returns()

    # 打印报告
    print(f"\n{factor_name}因子分析报告" if factor_name else "\n因子分析报告")
    analyzer.print_summary()

    return analyzer


if __name__ == "__main__":
    # 使用示例代码
    from backend.quant.core.factor_engine.factor_generator import get_registered_factors

    # 假设已经从数据库加载了股票数据
    from backend.data.stock_data_fetcher import StockDataFetcher

    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(code="sh.605300", start_date="2022-01-01", end_date="2025-03-01")

    # 获取所有注册的因子
    factors = get_registered_factors()

    # 计算RSI因子
    rsi_factor = factors['rsi'](df['close'])

    # 对RSI因子进行有效性分析
    analyzer = analyze_single_factor(rsi_factor, df, factor_name='RSI')

    # 查看详细结果
    ic_data = analyzer.ic_series
    group_returns = analyzer.group_returns
    win_rates = analyzer.calculate_win_rate()

    # 自定义分析参数
    analyzer.calculate_quantile_returns(num_quantiles=10)  # 十分位分组
    analyzer.plot_cumulative_returns(period='return_5d')  # 绘制5日累积收益曲线
