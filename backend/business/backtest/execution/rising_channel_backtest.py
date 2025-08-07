#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道策略回测示例
展示如何使用回测框架和数据获取工具进行策略回测
"""

import logging
import os
import sys
from datetime import datetime
import pandas as pd
import backtrader as bt
from typing import Dict, List

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from backend.business.backtest import (
    run_backtest,
    run_multi_strategy_backtest,
    optimize_parameters,
    DataUtils,
    ReportUtils
)
from backend.business.backtest.strategies.rising_channel import RisingChannelBacktestStrategy
from backend.business.backtest.core.backtest_engine import BacktestEngine
from backend.utils.logger import setup_logger


class MultiStockDataFeed(bt.feeds.PandasData):
    """
    多股票数据源
    用于在backtrader中处理多股票数据
    """
    
    def __init__(self, stock_code: str, **kwargs):
        """
        初始化数据源
        
        Args:
            stock_code: 股票代码
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self._name = stock_code


class RisingChannelBacktestRunner:
    """
    上升通道策略回测运行器
    提供完整的回测流程管理
    """

    def __init__(self, log_level=logging.INFO):
        """
        初始化回测运行器
        
        Args:
            log_level: 日志级别
        """
        # 设置日志记录器
        self.logger = setup_logger("backtest", log_level=log_level)

        # 回测配置
        self.config = {
            'initial_cash': 1000000.0,  # 初始资金100万
            'commission': 0.0003,  # 手续费率
            'stock_pool': 'no_st',  # 股票池：非ST股票
            'start_date': '2024-01-01',  # 开始日期
            'end_date': datetime.today().strftime("%Y-%m-%d"),  # 结束日期
            'max_stocks': 100,  # 最大股票数量
            'min_data_days': 120  # 最小数据天数
        }

        # 策略参数
        self.strategy_params = {
            'max_positions': 50,  # 最大持仓数量（50只股票）
            'min_channel_score': 60.0,  # 最小通道评分
            'k': 2.0,  # 通道斜率参数
            'L_max': 120,  # 最大通道长度
            'gain_trigger': 0.30,  # 收益触发阈值
            'beta_delta': 0.15,  # Beta变化阈值
            'R2_min': 0.20,  # 最小R²值
            'width_pct_min': 0.04,  # 最小通道宽度
            'width_pct_max': 0.15  # 最大通道宽度
        }

    def run_basic_backtest(self):
        """
        运行基础回测 - 真正的多股票策略回测
        """
        self.logger.info("开始运行上升通道策略多股票回测...")

        try:
            # 1. 获取股票数据
            self.logger.info("正在获取股票数据...")
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                max_stocks=self.config['max_stocks'],
                min_data_days=self.config['min_data_days']
            )

            self.logger.info(f"成功获取 {len(stock_data_dict)} 只股票的数据")

            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            # 2. 创建多股票回测引擎
            results = self._run_multi_stock_backtest(stock_data_dict)

            # 3. 输出结果
            self._print_results(results)

            # 4. 保存报告
            self._save_report(results, "multi_stock_backtest")

            return results

        except Exception as e:
            self.logger.exception(f"多股票回测失败: {e}")
            return None

    def _run_multi_stock_backtest(self, stock_data_dict: Dict[str, pd.DataFrame], strategy_params: Dict = None):
        """
        运行多股票回测
        
        Args:
            stock_data_dict: 股票数据字典
            strategy_params: 策略参数字典，如果为None则使用默认参数
            
        Returns:
            回测结果
        """
        self.logger.info("开始创建多股票回测引擎...")
        
        # 使用传入的策略参数或默认参数
        if strategy_params is None:
            strategy_params = self.strategy_params
        
        # 创建回测引擎
        engine = BacktestEngine(
            initial_cash=self.config['initial_cash'],
            commission=self.config['commission']
        )
        
        # 选择一只股票作为主数据源（用于时间轴）
        # 选择数据最完整的股票
        main_stock_code = self._select_main_stock(stock_data_dict)
        main_stock_data = stock_data_dict[main_stock_code]
        
        self.logger.info(f"选择主数据源: {main_stock_code}")
        
        # 添加主数据源
        engine.add_data(main_stock_data, name=main_stock_code)
        
        # 添加策略到引擎，同时传入股票数据
        engine.add_strategy(RisingChannelBacktestStrategy, stock_data_dict=stock_data_dict, **strategy_params)
        
        # 运行回测
        self.logger.info("开始运行回测...")
        results = engine.run("上升通道多股票策略")
        
        # 获取策略实例（用于获取详细信息）
        strategy_instance = self._get_strategy_instance(engine)
        if strategy_instance:
            # 添加策略详细信息到结果中
            results['strategy_info'] = strategy_instance.get_strategy_info()
            results['performance'] = strategy_instance.get_performance_summary()
        
        return results

    def _select_main_stock(self, stock_data_dict: Dict[str, pd.DataFrame]) -> str:
        """
        选择主数据源股票（数据最完整的）
        
        Args:
            stock_data_dict: 股票数据字典
            
        Returns:
            主股票代码
        """
        best_stock = None
        max_length = 0
        
        for stock_code, data in stock_data_dict.items():
            if len(data) > max_length:
                max_length = len(data)
                best_stock = stock_code
        
        return best_stock

    def _get_strategy_instance(self, engine: BacktestEngine):
        """
        获取策略实例
        
        Args:
            engine: 回测引擎
            
        Returns:
            策略实例
        """
        try:
            # 从引擎中获取策略实例
            if hasattr(engine, 'cerebro') and engine.cerebro:
                strategies = engine.cerebro.strats
                if strategies and len(strategies) > 0:
                    return strategies[0][0]  # 返回第一个策略实例
        except Exception as e:
            self.logger.warning(f"获取策略实例失败: {e}")
        
        return None

    def run_parameter_optimization(self):
        """
        运行参数优化
        """
        self.logger.info("开始运行参数优化...")

        try:
            # 获取股票数据
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                max_stocks=20,  # 优化时使用较少股票以提高速度
                min_data_days=self.config['min_data_days']
            )

            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            # 选择主数据源
            main_stock_code = self._select_main_stock(stock_data_dict)
            main_stock_data = stock_data_dict[main_stock_code]

            # 定义参数范围
            parameter_ranges = {
                'max_positions': [30, 50, 70],  # 持仓数量范围
                'min_channel_score': [50.0, 60.0, 70.0],  # 通道评分范围
                'k': [1.5, 2.0, 2.5],  # 通道斜率范围
                'gain_trigger': [0.25, 0.30, 0.35],  # 收益触发阈值范围
                'R2_min': [0.15, 0.20, 0.25],  # 最小R²值范围
                'width_pct_min': [0.03, 0.04, 0.05]  # 最小通道宽度范围
            }

            # 运行参数优化
            optimization_results = self._run_parameter_optimization(
                stock_data_dict, parameter_ranges
            )

            # 输出优化结果
            self._print_optimization_results(optimization_results)

            # 保存优化报告
            self._save_optimization_report(optimization_results)

            return optimization_results

        except Exception as e:
            self.logger.exception(f"参数优化失败: {e}")
            return None

    def _run_parameter_optimization(self, stock_data_dict: Dict[str, pd.DataFrame], parameter_ranges: Dict):
        """
        运行参数优化
        
        Args:
            stock_data_dict: 股票数据字典
            parameter_ranges: 参数范围
            
        Returns:
            优化结果
        """
        self.logger.info("开始参数优化...")
        
        # 生成参数组合
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"总共 {len(param_combinations)} 个参数组合")
        
        optimization_results = []
        best_result = None
        best_value = float('-inf')
        
        for i, param_combo in enumerate(param_combinations):
            try:
                # 构建参数字典
                params = dict(zip(param_names, param_combo))
                
                self.logger.info(f"测试参数组合 {i+1}/{len(param_combinations)}: {params}")
                
                # 运行回测
                result = self._run_multi_stock_backtest(stock_data_dict, params)
                
                if result and 'summary' in result:
                    # 获取目标指标
                    target_value = result['summary'].get('total_return', 0)
                    
                    optimization_results.append({
                        'parameters': params,
                        'result': result,
                        'target_value': target_value
                    })
                    
                    # 更新最佳结果
                    if target_value > best_value:
                        best_value = target_value
                        best_result = {
                            'parameters': params,
                            'result': result,
                            'target_value': target_value
                        }
                
            except Exception as e:
                self.logger.warning(f"参数组合 {params} 测试失败: {e}")
                continue
        
        return {
            'optimization_results': optimization_results,
            'best_params': best_result['parameters'] if best_result else {},
            'best_value': best_result['target_value'] if best_result else 0,
            'total_combinations': len(param_combinations)
        }

    def run_comparison_backtest(self):
        """
        运行对比回测
        """
        self.logger.info("开始运行对比回测...")

        try:
            # 获取股票数据
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                max_stocks=50,
                min_data_days=self.config['min_data_days']
            )

            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            # 定义不同参数配置的策略
            strategies = [
                {
                    'name': '保守策略',
                    'params': {
                        'max_positions': 30,  # 较少持仓
                        'min_channel_score': 70.0,  # 更高评分要求
                        'k': 1.5,  # 较低斜率
                        'gain_trigger': 0.25,  # 较低收益触发
                        'R2_min': 0.25,  # 更高R²要求
                        'width_pct_min': 0.05  # 更宽通道要求
                    }
                },
                {
                    'name': '标准策略',
                    'params': self.strategy_params
                },
                {
                    'name': '激进策略',
                    'params': {
                        'max_positions': 70,  # 更多持仓
                        'min_channel_score': 50.0,  # 较低评分要求
                        'k': 2.5,  # 较高斜率
                        'gain_trigger': 0.35,  # 较高收益触发
                        'R2_min': 0.15,  # 较低R²要求
                        'width_pct_min': 0.03  # 较窄通道要求
                    }
                }
            ]

            # 运行对比回测
            comparison_results = self._run_comparison_backtest(stock_data_dict, strategies)

            # 输出对比结果
            self._print_comparison_results(comparison_results)

            # 保存对比报告
            self._save_comparison_report(comparison_results)

            return comparison_results

        except Exception as e:
            self.logger.exception(f"对比回测失败: {e}")
            return None

    def _run_comparison_backtest(self, stock_data_dict: Dict[str, pd.DataFrame], strategies: List[Dict]):
        """
        运行对比回测
        
        Args:
            stock_data_dict: 股票数据字典
            strategies: 策略配置列表
            
        Returns:
            对比结果
        """
        self.logger.info("开始对比回测...")
        
        strategy_results = {}
        
        for strategy_config in strategies:
            strategy_name = strategy_config['name']
            strategy_params = strategy_config['params']
            
            self.logger.info(f"运行策略: {strategy_name}")
            
            try:
                # 运行回测
                result = self._run_multi_stock_backtest(stock_data_dict, strategy_params)
                strategy_results[strategy_name] = result
                
            except Exception as e:
                self.logger.error(f"策略 {strategy_name} 运行失败: {e}")
                continue
        
        # 生成比较报告
        comparison_report = self._generate_comparison_report(strategy_results)
        
        return {
            'strategy_results': strategy_results,
            'comparison_report': comparison_report
        }

    def _generate_comparison_report(self, strategy_results: Dict) -> str:
        """
        生成对比报告
        
        Args:
            strategy_results: 策略结果字典
            
        Returns:
            对比报告
        """
        report_lines = []
        report_lines.append("=== 策略对比报告 ===")
        report_lines.append("")
        
        # 创建对比表格
        report_lines.append("策略名称\t总收益率\t夏普比率\t最大回撤\t交易次数\t胜率")
        report_lines.append("-" * 80)
        
        for strategy_name, result in strategy_results.items():
            if result and 'summary' in result:
                summary = result['summary']
                report_lines.append(
                    f"{strategy_name}\t"
                    f"{summary.get('total_return', 0):.2f}%\t"
                    f"{summary.get('sharpe_ratio', 0):.4f}\t"
                    f"{summary.get('max_drawdown', 0):.2f}%\t"
                    f"{summary.get('total_trades', 0)}\t"
                    f"{summary.get('win_rate', 0):.2f}%"
                )
        
        return "\n".join(report_lines)

    def _print_results(self, results):
        """
        打印回测结果
        
        Args:
            results: 回测结果
        """
        if not results:
            return

        print("\n" + "=" * 60)
        print("上升通道多股票策略回测结果")
        print("=" * 60)

        # 打印报告
        if 'report' in results:
            print(results['report'])

        # 打印关键指标
        metrics = results.get('metrics', {})
        if metrics:
            # 安全获取指标值，处理None值
            def safe_get(key, default=0):
                value = metrics.get(key, default)
                return value if value is not None else default
            
            print("\n关键指标:")
            print(f"  总收益率: {safe_get('总收益率', 0):.2f}%")
            print(f"  夏普比率: {safe_get('夏普比率', 0):.4f}")
            print(f"  最大回撤: {safe_get('最大回撤', 0):.2f}%")
            print(f"  交易次数: {safe_get('交易次数', 0)}")
            print(f"  胜率: {safe_get('胜率', 0):.2f}%")

        # 打印策略信息
        if 'strategy_info' in results:
            strategy_info = results['strategy_info']
            print(f"\n策略信息:")
            print(f"  当前持仓数量: {strategy_info.get('current_status', {}).get('position_count', 0)}")
            print(f"  策略参数: {strategy_info.get('parameters', {})}")

    def _print_optimization_results(self, results):
        """
        打印参数优化结果
        
        Args:
            results: 优化结果
        """
        if not results:
            return

        print("\n" + "=" * 60)
        print("参数优化结果")
        print("=" * 60)

        print(f"总参数组合数: {results['total_combinations']}")
        print(f"最优参数: {results['best_params']}")
        print(f"最优收益率: {results['best_value']:.2f}%")

        # 打印前5个最优结果
        print("\n前5个最优结果:")
        sorted_results = sorted(results['optimization_results'],
                                key=lambda x: x['target_value'], reverse=True)

        for i, result in enumerate(sorted_results[:5]):
            print(f"{i + 1}. 参数: {result['parameters']}")
            print(f"   收益率: {result['target_value']:.2f}%")
            print()

    def _print_comparison_results(self, results):
        """
        打印对比回测结果
        
        Args:
            results: 对比结果
        """
        if not results:
            return

        print("\n" + "=" * 60)
        print("策略对比结果")
        print("=" * 60)

        # 打印比较报告
        if 'comparison_report' in results:
            print(results['comparison_report'])

        # 打印各策略结果
        strategy_results = results.get('strategy_results', {})
        for name, result in strategy_results.items():
            metrics = result.get('metrics', {})
            
            # 安全获取指标值，处理None值
            def safe_get(key, default=0):
                value = metrics.get(key, default)
                return value if value is not None else default
            
            print(f"\n{name}:")
            print(f"  总收益率: {safe_get('总收益率', 0):.2f}%")
            print(f"  夏普比率: {safe_get('夏普比率', 0):.4f}")
            print(f"  最大回撤: {safe_get('最大回撤', 0):.2f}%")
            print(f"  交易次数: {safe_get('交易次数', 0)}")
            print(f"  胜率: {safe_get('胜率', 0):.2f}%")

    def _save_report(self, results, report_type):
        """
        保存回测报告
        
        Args:
            results: 回测结果
            report_type: 报告类型
        """
        if not results:
            return

        try:
            # 创建报告目录
            report_dir = "backtest_reports"
            os.makedirs(report_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_dir}/rising_channel_{report_type}_{timestamp}.xlsx"

            # 保存报告
            ReportUtils.save_report_to_excel(results, filename)

            self.logger.info(f"报告已保存: {filename}")

        except Exception as e:
            self.logger.exception(f"保存报告失败: {e}")

    def _save_optimization_report(self, results):
        """
        保存优化报告
        
        Args:
            results: 优化结果
        """
        if not results:
            return

        try:
            # 创建报告目录
            report_dir = "backtest_reports"
            os.makedirs(report_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_dir}/rising_channel_optimization_{timestamp}.xlsx"

            # 创建优化结果DataFrame
            optimization_data = []
            for result in results['optimization_results']:
                # 安全获取指标值，处理None值
                def safe_get_metrics(key, default=0):
                    metrics = result['result'].get('metrics', {})
                    value = metrics.get(key, default)
                    return value if value is not None else default
                
                row = {
                    '参数组合': str(result['parameters']),
                    '收益率': result['target_value'],
                    '夏普比率': safe_get_metrics('夏普比率', 0),
                    '最大回撤': safe_get_metrics('最大回撤', 0),
                    '交易次数': safe_get_metrics('交易次数', 0)
                }
                optimization_data.append(row)

            df = pd.DataFrame(optimization_data)
            df = df.sort_values('收益率', ascending=False)

            # 保存到Excel
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='优化结果', index=False)

                # 添加最优结果
                best_result = pd.DataFrame([{
                    '最优参数': str(results['best_params']),
                    '最优收益率': results['best_value'],
                    '总组合数': results['total_combinations']
                }])
                best_result.to_excel(writer, sheet_name='最优结果', index=False)

            self.logger.info(f"优化报告已保存: {filename}")

        except Exception as e:
            self.logger.exception(f"保存优化报告失败: {e}")

    def _save_comparison_report(self, results):
        """
        保存对比报告
        
        Args:
            results: 对比结果
        """
        if not results:
            return

        try:
            # 创建报告目录
            report_dir = "backtest_reports"
            os.makedirs(report_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_dir}/rising_channel_comparison_{timestamp}.xlsx"

            # 保存报告
            ReportUtils.save_report_to_excel(results, filename)

            self.logger.info(f"对比报告已保存: {filename}")

        except Exception as e:
            self.logger.exception(f"保存对比报告失败: {e}")


def main():
    """
    主函数
    """
    print("上升通道多股票策略回测示例")
    print("=" * 50)

    # 创建回测运行器
    runner = RisingChannelBacktestRunner(log_level=logging.INFO)

    try:
        # 1. 运行基础回测
        print("\n1. 运行多股票回测...")
        basic_results = runner.run_basic_backtest()

        # 2. 运行参数优化
        print("\n2. 运行参数优化...")
        optimization_results = runner.run_parameter_optimization()

        # 3. 运行对比回测
        print("\n3. 运行对比回测...")
        comparison_results = runner.run_comparison_backtest()

        print("\n" + "=" * 50)
        print("所有回测完成！")

        # 总结
        if basic_results:
            print("✓ 多股票回测成功")
        if optimization_results:
            print("✓ 参数优化成功")
        if comparison_results:
            print("✓ 对比回测成功")

    except Exception as e:
        print(f"\n❌ 回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
