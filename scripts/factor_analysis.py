#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 3/11/25 12:15 AM
@File       : factor_analysis.py
@Description: 单因子有效性分析脚本

功能：对quantitative因子库中的因子进行单因子有效性分析
使用方法：
    python factor_analysis.py --factor_name macd --stocks sh.605300,sz.300490 --start_date 2022-01-01 --end_date 2023-03-01
    python factor_analysis.py --factor_type TechnicalFactors --stocks sh.605300,sz.300490 --start_date 2022-01-01 --end_date 2023-03-01
    python factor_analysis.py --factor_name all --stocks sh.605300,sz.300490 --start_date 2022-01-01 --end_date 2023-03-01

参数说明：
    factor_name: 具体因子名称或"all"代表所有因子
    factor_type: 因子类型(如MomentumFactors, TechnicalFactors等)
    stocks: 股票代码列表，以逗号分隔
    start_date: 起始日期
    end_date: 结束日期
"""
import datetime
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
os.chdir(root_dir)

import argparse
from typing import List, Optional
import importlib

from backend.data.stock_data_fetcher import StockDataFetcher
from backend.quant.backtest.performance_analyzer import analyze_single_factor
from backend.quant.core.factor_engine.factor_generator import (
    get_registered_factors, FACTOR_REGISTRY
)


def run_factor_analysis(
        factor_name: str,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        factor_type: Optional[str] = None
) -> None:
    """
    执行单因子有效性分析

    Args:
        factor_name: 因子名称或"all"代表所有因子
        stock_codes: 股票代码列表
        start_date: 起始日期
        end_date: 结束日期
        factor_type: 因子类别名称(可选)
    """
    # 创建股票数据获取器
    fetcher = StockDataFetcher()

    # 记录所有因子分析结果，用于最后的汇总
    factor_results = []

    # 确定要分析的因子
    if factor_type is not None:
        # 如果指定了因子类型
        try:
            # 动态导入指定的因子类
            module = importlib.import_module('backend.quant.core.factor_engine.factor_generator')
            factor_class = getattr(module, factor_type)
            
            # 获取该类的所有注册因子，而不是通过方法名匹配
            factors_to_analyze = {
                name: func for name, func in FACTOR_REGISTRY.items()
                if any(name == getattr(factor_class, attr).__name__ or 
                       (hasattr(func, '__wrapped__') and func.__wrapped__.__qualname__.startswith(f"{factor_type}."))
                       for attr in dir(factor_class) if not attr.startswith('_') and attr != 'register_factor')
            }
            print(f"将分析{factor_type}类的所有因子: {list(factors_to_analyze.keys())}")
        except (KeyError, AttributeError, ImportError) as e:
            print(f"错误: 未找到因子类型 '{factor_type}': {e}")
            return
    elif factor_name.lower() == 'all':
        # 分析所有注册的因子
        factors_to_analyze = get_registered_factors()
        print(f"将分析所有注册的因子: {list(factors_to_analyze.keys())}")
    else:
        # 分析特定的因子
        factors = get_registered_factors()
        if factor_name in factors:
            factors_to_analyze = {factor_name: factors[factor_name]}
            print(f"将分析因子: {factor_name}")
        else:
            print(f"错误: 未找到因子 '{factor_name}'")
            return

    # 获取股票数据
    print(f"分析周期: {start_date} 至 {end_date}")
    print(f"股票样本: {stock_codes}")

    price_data = {}
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
        except Exception as e:
            print(f"处理股票 {code} 时出错: {e}")

    print(f"成功处理 {valid_stocks} 只股票数据")

    # 依次分析每个因子
    for factor_name, factor_func in factors_to_analyze.items():
        print(f"\n{'=' * 80}")
        print(f"开始分析因子: {factor_name}")
        print(f"{'=' * 80}")

        # 计算因子值
        factor_values = {}

        try:
            for code, df in price_data.items():
                # 基于因子函数的参数名确定要传递的数据列
                import inspect
                params = inspect.signature(factor_func).parameters
                param_names = list(params.keys())

                # 准备参数
                args = {}
                for param in param_names:
                    if param in df.columns:
                        args[param] = df[param]
                    elif param == "open_price":
                        # 因为open是一个built_in参数，所以尽量不直接使用它
                        args["open_price"] = df["open"]

                # 计算因子
                if args:
                    factor_values[code] = factor_func(**args)

            # 判断是否成功计算了因子
            if not factor_values:
                print(f"错误: 无法为因子 {factor_name} 计算有效值，请检查所需数据列是否存在")
                continue

            # 对因子进行有效性分析
            print(f"\n开始 {factor_name} 因子有效性分析...")
            analyzer = analyze_single_factor(
                factor_data=factor_values,
                price_data=price_data,
                factor_name=factor_name,
                date_col='trade_date',
                need_plot=False
            )

            # 获取分析结果并记录
            report_data = analyzer.generate_report()

            # 记录因子分析结果
            factor_result = {
                'factor_name': factor_name,
                'ic_mean': report_data['summary'].get('ic_mean', float('nan')),
                'ic_ir': report_data['summary'].get('ic_ir', float('nan')),
                'ic_pos_rate': report_data['summary'].get('ic_pos_rate', float('nan')),
                'long_short_return': report_data['summary'].get('long_short_return', float('nan')),
                'top_group_win_rate': report_data['summary'].get('top_group_win_rate', float('nan'))
            }
            factor_results.append(factor_result)

            # 保存分析结果
            report_name = os.path.join("results", datetime.date.today().strftime("%Y%m%d"),
                                       f"factor_analysis_{factor_name}.html")
            try:
                analyzer.save_report(report_name)
                print(f"分析报告已保存为: {report_name}")
            except Exception as e:
                print(f"保存报告时出错: {e}")

        except Exception as e:
            print(f"分析因子 {factor_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 输出优秀因子汇总
    if factor_results:
        print_excellent_factors(factor_results)


def print_excellent_factors(factor_results: List[dict]) -> None:
    """
    输出优秀因子汇总

    Args:
        factor_results: 因子分析结果列表
    """
    import pandas as pd

    # 转换为DataFrame方便处理
    results_df = pd.DataFrame(factor_results)

    # 定义优秀因子标准
    # 1. 强有效: IC均值 > 0.05
    # 2. 优秀因子: IR > 1.0
    # 3. 方向一致: IC正比例 > 0.55
    # 4. 区分能力强: 多空组合收益 > 0.5%
    # 5. 有效: 顶层组胜率 > 0.55

    # 标记每个指标是否达到优秀标准
    results_df['强有效'] = results_df['ic_mean'] > 0.05
    results_df['高IR值'] = results_df['ic_ir'] > 1.0
    results_df['方向一致'] = results_df['ic_pos_rate'] > 0.55
    results_df['区分能力'] = results_df['long_short_return'] > 0.005
    results_df['高胜率'] = results_df['top_group_win_rate'] > 0.55

    # 计算综合得分 (满足的标准数量)
    results_df['优秀度'] = results_df[['强有效', '高IR值', '方向一致', '区分能力', '高胜率']].sum(axis=1)

    # 按优秀度排序
    results_df = results_df.sort_values('优秀度', ascending=False)

    # 格式化输出数据
    formatted_df = results_df[['factor_name', 'ic_mean', 'ic_ir', 'ic_pos_rate',
                               'long_short_return', 'top_group_win_rate', '优秀度']].copy()

    # 格式化显示
    formatted_df['ic_mean'] = formatted_df['ic_mean'].map(lambda x: f"{x:.4f}")
    formatted_df['ic_ir'] = formatted_df['ic_ir'].map(lambda x: f"{x:.4f}")
    formatted_df['ic_pos_rate'] = formatted_df['ic_pos_rate'].map(lambda x: f"{x:.2%}")
    formatted_df['long_short_return'] = formatted_df['long_short_return'].map(lambda x: f"{x:.2%}")
    formatted_df['top_group_win_rate'] = formatted_df['top_group_win_rate'].map(lambda x: f"{x:.2%}")

    # 重命名列
    formatted_df.columns = ['因子名称', 'IC均值', 'IR值', 'IC正比例', '多空收益', '顶层胜率', '优秀度']

    print("\n\n" + "=" * 100)
    print("因子有效性排名")
    print("=" * 100)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    print(formatted_df)

    # 输出优秀因子 (优秀度 >= 3)
    excellent_factors = formatted_df[formatted_df['优秀度'] >= 3]
    if not excellent_factors.empty:
        print("\n\n" + "=" * 100)
        print("优秀因子汇总 (满足至少3项标准)")
        print("=" * 100)
        print(excellent_factors)

    # 输出评价标准说明
    print("\n评价标准:")
    print("- 强有效: IC均值 > 0.05")
    print("- 高IR值: IR > 1.0")
    print("- 方向一致: IC正比例 > 55%")
    print("- 区分能力: 多空组合收益 > 0.5%")
    print("- 高胜率: 顶层组胜率 > 55%")
    print("- 优秀度: 满足上述标准的数量")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="单因子有效性分析工具")

    # 添加因子名称和类型参数(互斥)
    factor_group = parser.add_mutually_exclusive_group(required=True)
    factor_group.add_argument("--factor_name", type=str, help="因子名称或'all'代表所有因子")
    factor_group.add_argument("--factor_type", type=str, help="因子类型名称(如TechnicalFactors)")

    # 添加其他必需参数
    parser.add_argument("--stocks", type=str, required=True, help="股票代码列表，以逗号分隔")
    parser.add_argument("--start_date", type=str, required=True, help="起始日期(YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期(YYYY-MM-DD)")

    args = parser.parse_args()

    # 处理股票代码列表
    stock_codes = args.stocks.split(',')

    # 执行因子分析
    if args.factor_name:
        run_factor_analysis(
            factor_name=args.factor_name,
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        run_factor_analysis(
            factor_name="",  # 当使用factor_type时，这个参数不使用
            stock_codes=stock_codes,
            start_date=args.start_date,
            end_date=args.end_date,
            factor_type=args.factor_type
        )


if __name__ == "__main__":
    # main()

    # stock_codes = ["sh.605300", "sz.300490", "sh.603336", "sh.600519", "sz.000858",
    #                "sh.601398", "sz.000651", "sh.601318", "sz.000333", "sh.600036"]
    start_date = "2024-01-01"
    end_date = "2025-03-01"
    fetcher = StockDataFetcher()
    all_stock_codes = fetcher.get_stock_list()
    stock_codes = all_stock_codes[
        all_stock_codes['ipo_date'] < datetime.datetime.strptime(start_date, "%Y-%m-%d").date()].code.to_list()

    # from backend.quant.core.factor_engine.factor_generator import ShortTermFactors
    run_factor_analysis(
        factor_name="",
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        factor_type="ShortTermFactors"
    )
