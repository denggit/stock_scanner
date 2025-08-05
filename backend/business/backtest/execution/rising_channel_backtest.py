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
from backend.utils.logger import setup_logger


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
            'end_date': '2024-12-31',  # 结束日期
            'max_stocks': 50,  # 最大股票数量（用于测试）
            'min_data_days': 120  # 最小数据天数
        }

        # 策略参数
        self.strategy_params = {
            'max_positions': 10,  # 最大持仓数量
            'min_channel_score': 60.0,  # 最小通道评分
            'k': 2.0,  # 通道斜率参数
            'L_max': 120,  # 最大通道长度
            'gain_trigger': 0.30,  # 收益触发阈值
            'beta_delta': 0.15  # Beta变化阈值
        }

    def run_basic_backtest(self):
        """
        运行基础回测
        """
        self.logger.info("开始运行上升通道策略基础回测...")

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

            # 2. 选择一只股票进行单股票回测（演示用）
            if stock_data_dict:
                test_stock_code = list(stock_data_dict.keys())[0]
                test_stock_data = stock_data_dict[test_stock_code]

                self.logger.info(f"使用股票 {test_stock_code} 进行回测演示")

                # 3. 运行回测
                results = run_backtest(
                    data=test_stock_data,
                    strategy_class=RisingChannelBacktestStrategy,
                    initial_cash=self.config['initial_cash'],
                    commission=self.config['commission'],
                    strategy_params=self.strategy_params,
                    strategy_name="上升通道策略",
                    plot=False  # 暂时关闭绘图
                )

                # 4. 输出结果
                self._print_results(results)

                # 5. 保存报告
                self._save_report(results, "basic_backtest")

                return results
            else:
                self.logger.error("没有获取到有效的股票数据")
                return None

        except Exception as e:
            self.logger.error(f"基础回测失败: {e}")
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
                max_stocks=5,  # 优化时使用较少股票以提高速度
                min_data_days=self.config['min_data_days']
            )

            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            # 选择一只股票进行优化
            test_stock_code = list(stock_data_dict.keys())[0]
            test_stock_data = stock_data_dict[test_stock_code]

            # 定义参数范围
            parameter_ranges = {
                'max_positions': [5, 10, 15],
                'min_channel_score': [50.0, 60.0, 70.0],
                'k': [1.5, 2.0, 2.5],
                'gain_trigger': [0.25, 0.30, 0.35]
            }

            # 运行参数优化
            optimization_results = optimize_parameters(
                data=test_stock_data,
                strategy_class=RisingChannelBacktestStrategy,
                parameter_ranges=parameter_ranges,
                initial_cash=self.config['initial_cash'],
                commission=self.config['commission'],
                optimization_target="总收益率"
            )

            # 输出优化结果
            self._print_optimization_results(optimization_results)

            # 保存优化报告
            self._save_optimization_report(optimization_results)

            return optimization_results

        except Exception as e:
            self.logger.error(f"参数优化失败: {e}")
            return None

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
                max_stocks=5,
                min_data_days=self.config['min_data_days']
            )

            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            # 选择一只股票进行对比
            test_stock_code = list(stock_data_dict.keys())[0]
            test_stock_data = stock_data_dict[test_stock_code]

            # 定义不同参数配置的策略
            strategies = [
                {
                    'name': '保守策略',
                    'class': RisingChannelBacktestStrategy,
                    'params': {
                        'max_positions': 5,
                        'min_channel_score': 70.0,
                        'k': 1.5,
                        'gain_trigger': 0.25
                    }
                },
                {
                    'name': '标准策略',
                    'class': RisingChannelBacktestStrategy,
                    'params': self.strategy_params
                },
                {
                    'name': '激进策略',
                    'class': RisingChannelBacktestStrategy,
                    'params': {
                        'max_positions': 15,
                        'min_channel_score': 50.0,
                        'k': 2.5,
                        'gain_trigger': 0.35
                    }
                }
            ]

            # 运行多策略回测
            comparison_results = run_multi_strategy_backtest(
                data=test_stock_data,
                strategies=strategies,
                initial_cash=self.config['initial_cash'],
                commission=self.config['commission']
            )

            # 输出对比结果
            self._print_comparison_results(comparison_results)

            # 保存对比报告
            self._save_comparison_report(comparison_results)

            return comparison_results

        except Exception as e:
            self.logger.error(f"对比回测失败: {e}")
            return None

    def _print_results(self, results):
        """
        打印回测结果
        
        Args:
            results: 回测结果
        """
        if not results:
            return

        print("\n" + "=" * 60)
        print("上升通道策略回测结果")
        print("=" * 60)

        # 打印报告
        if 'report' in results:
            print(results['report'])

        # 打印关键指标
        metrics = results.get('metrics', {})
        if metrics:
            print("\n关键指标:")
            print(f"  总收益率: {metrics.get('总收益率', 0):.2f}%")
            print(f"  夏普比率: {metrics.get('夏普比率', 0):.4f}")
            print(f"  最大回撤: {metrics.get('最大回撤', 0):.2f}%")
            print(f"  交易次数: {metrics.get('交易次数', 0)}")
            print(f"  胜率: {metrics.get('胜率', 0):.2f}%")

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
            print(f"\n{name}:")
            print(f"  总收益率: {metrics.get('总收益率', 0):.2f}%")
            print(f"  夏普比率: {metrics.get('夏普比率', 0):.4f}")
            print(f"  最大回撤: {metrics.get('最大回撤', 0):.2f}%")

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
            self.logger.error(f"保存报告失败: {e}")

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
                row = {
                    '参数组合': str(result['parameters']),
                    '收益率': result['target_value'],
                    '夏普比率': result['result']['metrics'].get('夏普比率', 0),
                    '最大回撤': result['result']['metrics'].get('最大回撤', 0),
                    '交易次数': result['result']['metrics'].get('交易次数', 0)
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
            self.logger.error(f"保存优化报告失败: {e}")

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
            self.logger.error(f"保存对比报告失败: {e}")


def main():
    """
    主函数
    """
    print("上升通道策略回测示例")
    print("=" * 50)

    # 创建回测运行器
    runner = RisingChannelBacktestRunner(log_level=logging.INFO)

    try:
        # 1. 运行基础回测
        print("\n1. 运行基础回测...")
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
            print("✓ 基础回测成功")
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
