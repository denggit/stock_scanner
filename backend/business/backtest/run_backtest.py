#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用回测运行脚本
用于执行各种策略的回测

使用方法:
    python run_backtest.py
"""

import os
import sys
from typing import List, Dict, Any, Optional, Type

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.business.backtest.core.engine import BacktestEngine
from backend.business.backtest.strategies.base import BaseStrategy
from backend.business.backtest.analysis.report import generate_report
from backend.business.data.data_fetcher import StockDataFetcher


def get_stock_codes(pool_name: str = "no_st", max_stocks: Optional[int] = None) -> List[str]:
    """
    获取股票代码列表
    
    Args:
        pool_name: 股票池名称
        max_stocks: 最大股票数量，None表示不限制
        
    Returns:
        股票代码列表
    """
    try:
        fetcher = StockDataFetcher()
        stock_list_df = fetcher.get_stock_list(pool_name=pool_name)
        fetcher.close()

        stock_codes = stock_list_df['code'].tolist()

        if max_stocks and len(stock_codes) > max_stocks:
            # 随机选择指定数量的股票
            import random
            random.shuffle(stock_codes)
            stock_codes = stock_codes[:max_stocks]

        return stock_codes
    except Exception as e:
        print(f"ERROR: Failed to get stock codes: {e}")
        # 返回默认测试股票
        return ["sh.600036", "sh.600519", "sz.000001", "sz.300750", "sh.601318", "sh.600000"]


def run_backtest(
    strategy_class: Type[BaseStrategy],
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    initial_cash: float = 1000000.0,
    stock_codes: Optional[List[str]] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    adjust: str = "1",
    generate_report_flag: bool = True,
    report_name: str = "backtest_report"
) -> Optional[List[Dict]]:
    """
    运行回测
    
    Args:
        strategy_class: 策略类
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
        stock_codes: 股票代码列表，None表示自动获取
        strategy_params: 策略参数，None表示使用默认参数
        adjust: 复权类型
        generate_report_flag: 是否生成报告
        report_name: 报告名称
        
    Returns:
        回测结果列表，失败时返回None
    """
    try:
        strategy_name = strategy_class.__name__
        print(f"=== {strategy_name} 回测 ===")
        print(f"回测期间: {start_date} 到 {end_date}")
        print(f"初始资金: {initial_cash:,.2f}")

        # 获取股票代码
        if stock_codes is None:
            print("正在获取股票列表...")
            stock_codes = get_stock_codes(max_stocks=500)  # 限制股票数量以提高速度

        print(f"股票数量: {len(stock_codes)}")

        # 获取策略参数
        if strategy_params is None:
            strategy_params = {}

        print(f"策略参数: {strategy_params}")

        # 创建回测引擎
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash
        )

        # 运行回测
        print("\n开始运行回测...")
        results = engine.run(
            strategy_class=strategy_class,
            stock_codes=stock_codes,
            strategy_params=strategy_params,
            adjust=adjust
        )

        if not results:
            print("ERROR: 回测未返回结果")
            return None

        # 计算基本统计信息
        if len(results) > 1:
            initial_value = results[0]['net_value']
            final_value = results[-1]['net_value']
            total_return = ((final_value - initial_value) / initial_value) * 100

            print(f"\n=== 回测结果 ===")
            print(f"初始净值: {initial_value:,.2f}")
            print(f"最终净值: {final_value:,.2f}")
            print(f"总收益率: {total_return:.2f}%")
            print(f"回测天数: {len(results)}")

        # 生成报告
        if generate_report_flag and results:
            print("\n正在生成报告...")
            generate_report(results, report_name)

        return results

    except Exception as e:
        print(f"ERROR: 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数 - 示例用法"""
    print("请导入具体的策略类并调用 run_backtest 函数")
    print("示例:")
    print("from backend.business.backtest.strategies.implementations.rising_channel_strategy import RisingChannelStrategy")
    print("results = run_backtest(RisingChannelStrategy, ...)")


if __name__ == '__main__':
    main()
