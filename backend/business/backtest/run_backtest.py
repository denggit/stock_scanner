#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/18/25 9:41 PM
@File       : run_backtest.py
@Description: 
"""
# backend/business/backtest/run_backtest.py

import sys
import os
import pandas as pd

# 这是一个技巧，用于解决在项目内部运行时Python的模块导入路径问题
# 它将项目的根目录添加到了Python的搜索路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from backend.business.backtest.core.engine import BacktestEngine
from backend.business.backtest.strategies.implementations.buy_and_hold import BuyAndHoldStrategy


def main():
    """
    回测主函数。
    """
    print("--- Setting up backtest parameters ---")

    # --- 1. 设置回测参数 ---
    params = {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_cash": 1000000,
        # 选择几只常见的股票作为测试股票池
        "stock_codes": ["sh.600036", "sh.600519", "sz.000001", "sz.300750"],
        "strategy_params": {},  # 本策略无特殊参数
        "adjust": "1"  # 使用后复权数据
    }

    print(f"Parameters: {params}")

    # --- 2. 实例化并运行回测引擎 ---
    engine = BacktestEngine(
        start_date=params["start_date"],
        end_date=params["end_date"],
        initial_cash=params["initial_cash"]
    )

    results = engine.run(
        strategy_class=BuyAndHoldStrategy,
        stock_codes=params["stock_codes"],
        strategy_params=params["strategy_params"],
        adjust=params["adjust"]
    )

    # --- 3. 打印结果 ---
    if results:
        print("\n--- Backtest Results (Daily Net Values) ---")
        results_df = pd.DataFrame(results).set_index('date')
        print(results_df.tail(10))  # 打印最后10天的净值

        # 简单计算最终收益率
        initial_value = results_df['net_value'].iloc[0]
        final_value = results_df['net_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        print(f"\nInitial Value: {initial_value:,.2f}")
        print(f"Final Value:   {final_value:,.2f}")
        print(f"Total Return:  {total_return:.2f}%")


if __name__ == '__main__':
    main()