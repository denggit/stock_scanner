#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:56 PM
@File       : report.py
@Description: 
"""
# backend/business/backtest/analysis/report.py

import pandas as pd
import quantstats as qs
from typing import List, Dict


def generate_report(daily_net_values: List[Dict], output_filename: str = "backtest_report.html"):
    """
    使用 quantstats 生成一份全面的回测分析报告。

    Args:
        daily_net_values (List[Dict]): 每日净值列表，
                                      每个元素是 {'date': pd.Timestamp, 'net_value': float}。
        output_filename (str, optional): 输出的HTML报告文件名。
                                         默认为 "backtest_report.html"。
    """
    if not daily_net_values:
        print("无法生成报告：每日净值列表为空。")
        return

    print(f"\n--- 正在生成 QuantStats 分析报告 ---")

    # 1. 将我们的回测结果转换成 quantstats 期望的格式：一个Pandas Series，包含每日收益率
    results_df = pd.DataFrame(daily_net_values).set_index('date')

    # 确保索引是 DatetimeIndex 类型
    results_df.index = pd.to_datetime(results_df.index)

    # 从每日净值计算每日收益率
    returns_series = results_df['net_value'].pct_change().fillna(0)

    # 2. 生成HTML报告
    try:
        # benchmark 参数可以添加一个基准进行比较，例如沪深300
        qs.reports.html(returns_series, output=output_filename, title='我的策略回测报告')
        print(f"报告生成成功: {output_filename}")
        print("您现在可以在浏览器中打开此文件查看完整的分析结果。")
    except Exception as e:
        print(f"生成 quantstats 报告时出错: {e}")