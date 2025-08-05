#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新架构回测框架演示脚本
展示如何使用重新设计的回测框架
"""

from backend.backtest import (
    run_backtest,
    run_multi_strategy_backtest,
    optimize_parameters,
    MAStrategy,
    RSIStrategy,
    MACDStrategy,
    DualThrustStrategy,
    DataUtils,
    ReportUtils
)


def demo_simple_backtest():
    """演示简单回测"""
    print("=== 简单回测演示 ===")
    
    # 创建示例数据
    data = DataUtils.create_sample_data(days=100)
    print(f"数据范围: {data.index[0].date()} 到 {data.index[-1].date()}")
    print(f"数据条数: {len(data)}")
    
    # 验证数据
    validation_result = DataUtils.validate_data(data)
    print(f"数据验证结果: {'通过' if validation_result['is_valid'] else '失败'}")
    if validation_result['warnings']:
        print(f"警告: {validation_result['warnings']}")
    
    # 运行移动平均策略回测
    print("\n--- 移动平均策略回测 ---")
    results = run_backtest(
        data=data,
        strategy_class=MAStrategy,
        initial_cash=100000.0,
        commission=0.0003,
        strategy_params={'short_period': 10, 'long_period': 30},
        strategy_name="移动平均策略"
    )
    
    # 打印报告
    print(results['report'])
    
    return results


def demo_multi_strategy_backtest():
    """演示多策略回测"""
    print("\n=== 多策略回测演示 ===")
    
    # 创建示例数据
    data = DataUtils.create_sample_data(days=100)
    
    # 定义策略列表
    strategies = [
        {
            'name': '移动平均策略',
            'class': MAStrategy,
            'params': {'short_period': 10, 'long_period': 30}
        },
        {
            'name': 'RSI策略',
            'class': RSIStrategy,
            'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
        },
        {
            'name': 'MACD策略',
            'class': MACDStrategy,
            'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        }
    ]
    
    # 运行多策略回测
    results = run_multi_strategy_backtest(
        data=data,
        strategies=strategies,
        initial_cash=100000.0,
        commission=0.0003
    )
    
    # 打印比较报告
    print(results['comparison_report'])
    
    return results


def demo_parameter_optimization():
    """演示参数优化"""
    print("\n=== 参数优化演示 ===")
    
    # 创建示例数据
    data = DataUtils.create_sample_data(days=100)
    
    # 定义参数范围
    parameter_ranges = {
        'short_period': [5, 10, 15],
        'long_period': [20, 30, 40]
    }
    
    # 运行参数优化
    results = optimize_parameters(
        data=data,
        strategy_class=MAStrategy,
        parameter_ranges=parameter_ranges,
        initial_cash=100000.0,
        commission=0.0003,
        optimization_target="总收益率"
    )
    
    # 打印优化结果
    print(f"总参数组合数: {results['total_combinations']}")
    print(f"最优参数: {results['best_params']}")
    print(f"最优收益率: {results['best_value']:.2f}%")
    
    # 打印前5个最优结果
    print("\n前5个最优结果:")
    sorted_results = sorted(results['optimization_results'], 
                           key=lambda x: x['target_value'], reverse=True)
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. 参数: {result['parameters']}, 收益率: {result['target_value']:.2f}%")
    
    return results


def demo_data_utils():
    """演示数据工具"""
    print("\n=== 数据工具演示 ===")
    
    # 创建不同参数的数据
    data1 = DataUtils.create_sample_data(days=50, volatility=0.01, trend=0.0005)
    data2 = DataUtils.create_sample_data(days=50, volatility=0.03, trend=0.002)
    
    # 计算收益率
    data1_with_returns = DataUtils.calculate_returns(data1)
    data2_with_returns = DataUtils.calculate_returns(data2)
    
    print(f"数据1平均日收益率: {data1_with_returns['daily_return'].mean():.4f}")
    print(f"数据2平均日收益率: {data2_with_returns['daily_return'].mean():.4f}")
    
    # 重采样数据
    weekly_data = DataUtils.resample_data(data1, freq='W')
    print(f"周度数据条数: {len(weekly_data)}")


def demo_report_utils():
    """演示报告工具"""
    print("\n=== 报告工具演示 ===")
    
    # 运行一个回测获取结果
    data = DataUtils.create_sample_data(days=100)
    results = run_backtest(
        data=data,
        strategy_class=MAStrategy,
        strategy_params={'short_period': 10, 'long_period': 30},
        strategy_name="演示策略"
    )
    
    # 创建性能汇总表
    performance_summary = ReportUtils.create_performance_summary(results)
    print("\n性能汇总表:")
    print(performance_summary.to_string(index=False))
    
    # 创建交易汇总表
    if results['trades']:
        trade_summary = ReportUtils.create_trade_summary(results['trades'])
        print("\n交易汇总表:")
        print(trade_summary.to_string(index=False))
        
        # 创建月度收益表
        monthly_returns = ReportUtils.create_monthly_returns(results['trades'])
        if not monthly_returns.empty:
            print("\n月度收益表:")
            print(monthly_returns.to_string(index=False))


def main():
    """主函数"""
    print("新架构回测框架演示")
    print("=" * 50)
    
    try:
        # 演示各个功能
        demo_simple_backtest()
        demo_multi_strategy_backtest()
        demo_parameter_optimization()
        demo_data_utils()
        demo_report_utils()
        
        print("\n所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 