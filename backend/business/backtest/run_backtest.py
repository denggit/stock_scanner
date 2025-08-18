#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:41 PM
@File       : run_backtest.py
@Description: 
"""
# backend/business/backtest/run_backtest.py

import os
import sys

# 这是一个技巧，用于解决在项目内部运行时Python的模块导入路径问题
# 它将项目的根目录添加到了Python的搜索路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# backend/business/backtest/run_backtest.py

# ... (保留 sys, os, pandas 的 import)

from backend.business.backtest.core.engine import BacktestEngine
# 修改这里：导入我们新的策略
from backend.business.backtest.strategies.implementations.rising_channel_strategy import RisingChannelStrategy
from backend.business.backtest.analysis.report import generate_report
# 如果需要获取全市场股票，可以从这里获取
from backend.business.data.data_fetcher import StockDataFetcher


def get_all_stock_codes():
    """获取所有A股代码列表的辅助函数"""
    fetcher = StockDataFetcher()
    stock_list_df = fetcher.get_stock_list(pool_name="no_st")
    fetcher.close()
    return stock_list_df['code'].tolist()


def main():
    """回测主函数。"""
    print("--- Setting up backtest parameters for Rising Channel Strategy ---")

    # --- 1. 设置回测参数 ---
    params = {
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",  # 先用半年数据测试，速度更快
        "initial_cash": 1000000,
        # 从数据库动态获取全市场股票，或使用一个固定的较小列表进行测试
        # "stock_codes": get_all_stock_codes(),
        "stock_codes": ["sh.600036", "sh.600519", "sz.000001", "sz.300750", "sh.601318", "sh.600000"],  # 使用一个小的测试池
        "adjust": "1",  # 使用后复权数据
        # --- 策略专属参数 ---
        "strategy_params": {
            'lookback_period': 60,  # 计算通道的回溯期
            'min_channel_width': 0.08,  # 最小通道宽度
            'target_positions': 5,  # 目标持仓股票数量
        },
    }

    print(f"Parameters: {params}")

    # --- 2. 实例化并运行回测引擎 ---
    engine = BacktestEngine(
        start_date=params["start_date"],
        end_date=params["end_date"],
        initial_cash=params["initial_cash"]
    )

    results = engine.run(
        # 修改这里：传入新的策略类
        strategy_class=RisingChannelStrategy,
        stock_codes=params["stock_codes"],
        strategy_params=params["strategy_params"],
        adjust=params["adjust"]
    )

    # --- 3. 生成报告 ---
    if results:
        generate_report(results, "rising_channel_report.html")


if __name__ == '__main__':
    main()
