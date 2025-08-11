#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道策略回测运行器

这是一个专用的回测运行器，用于执行上升通道交易策略。
继承自BaseBacktestRunner，提供了完整的回测功能包括：
- 基础回测执行
- 参数优化
- 策略对比
- 报告生成

使用示例：
    # 开发环境快速测试
    runner = RisingChannelBacktestRunner(environment='development')
    results = runner.run_basic_backtest()
    
    # 参数优化
    runner = RisingChannelBacktestRunner(environment='optimization')
    optimization_results = runner.run_parameter_optimization()
    
    # 策略对比
    runner = RisingChannelBacktestRunner(environment='production')
    comparison_results = runner.run_comparison_backtest()
"""

import logging
import os
import sys
from typing import Optional, Dict, Any

# 添加项目根目录到路径（保留以兼容独立运行）
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from backend.business.backtest.strategies.implementations.channel.rising_channel import \
    RisingChannelStrategy as RisingChannelBacktestStrategy
from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.backtest.execution.base.base_runner import BaseBacktestRunner


class RisingChannelBacktestRunner(BaseBacktestRunner):
    """
    上升通道策略回测运行器
    
    专门用于上升通道策略的回测执行器，通过依赖注入的方式
    将策略类和配置类传递给基础运行器，实现了策略与运行逻辑的解耦。
    
    特点：
    - 继承了BaseBacktestRunner的所有功能
    - 专门配置了上升通道策略和相关配置
    - 支持多种运行环境（development/optimization/production/full_backtest）
    - 自动化的报告生成和结果保存
    
    Attributes:
        strategy_class: 上升通道策略类
        config_cls: 上升通道配置类
        logger: 日志记录器
        environment: 当前运行环境
        config: 环境配置字典
        strategy_params: 策略参数字典
    """

    def __init__(self, log_level: int = logging.INFO, environment: Optional[str] = None):
        """
        初始化上升通道回测运行器
        
        Args:
            log_level: 日志级别，默认为INFO级别
            environment: 运行环境，可选值：
                - 'development': 开发环境，快速验证策略逻辑
                - 'optimization': 优化环境，用于参数调优
                - 'production': 生产环境，完整回测
                - 'full_backtest': 完整回测，包含所有功能
                - None: 使用环境变量BACKTEST_ENV或默认为development
        """
        super().__init__(
            strategy_class=RisingChannelBacktestStrategy,
            config_cls=RisingChannelConfig,
            log_level=log_level,
            environment=environment,
        )

        # 记录运行器特定信息
        self.logger.info(f"上升通道回测运行器初始化完成")
        self.logger.info(f"策略类: {RisingChannelBacktestStrategy.__name__}")
        self.logger.info(f"配置类: {RisingChannelConfig.__name__}")

        # 记录详细的策略参数
        self.logger.info("=" * 60)
        self.logger.info("📊 策略参数配置:")
        self.logger.info(f"  最大持仓数量: {self.strategy_params.get('max_positions')}")
        self.logger.info(f"  最小通道评分: {self.strategy_params.get('min_channel_score')}")
        self.logger.info(f"  最小数据点数: {self.strategy_params.get('min_data_points')}")
        self.logger.info(f"  通道斜率参数(k): {self.strategy_params.get('k')}")
        self.logger.info(f"  最大回看天数(L_max): {self.strategy_params.get('L_max')}")
        self.logger.info(f"  切割参数(delta_cut): {self.strategy_params.get('delta_cut')}")
        self.logger.info(f"  枢轴参数(pivot_m): {self.strategy_params.get('pivot_m')}")
        self.logger.info(f"  收益触发阈值(gain_trigger): {self.strategy_params.get('gain_trigger')}")
        self.logger.info(f"  Beta变化阈值(beta_delta): {self.strategy_params.get('beta_delta')}")
        self.logger.info(f"  突破天数(break_days): {self.strategy_params.get('break_days')}")
        self.logger.info(f"  重锚定失败最大次数: {self.strategy_params.get('reanchor_fail_max')}")
        self.logger.info(f"  最小R²值: {self.strategy_params.get('R2_min')}")
        self.logger.info(f"  最大R²值: {self.strategy_params.get('R2_max')}")
        self.logger.info(f"  最小通道宽度: {self.strategy_params.get('width_pct_min')}")
        self.logger.info(f"  最大通道宽度: {self.strategy_params.get('width_pct_max')}")
        self.logger.info(f"  距离下沿最大百分比: {self.strategy_params.get('max_distance_from_lower')}%")
        self.logger.info(f"  启用日志: {self.strategy_params.get('enable_logging')}")
        self.logger.info("=" * 60)

    def get_runner_info(self) -> Dict[str, Any]:
        """
        获取运行器信息
        
        Returns:
            运行器信息字典
        """
        return {
            'runner_type': 'RisingChannelBacktestRunner',
            'strategy_class': RisingChannelBacktestStrategy.__name__,
            'config_class': RisingChannelConfig.__name__,
            'environment': self.environment,
            'description': self.config.get('description', ''),
            'max_stocks': self.config.get('max_stocks'),
            'max_positions': self.config.get('max_positions'),
            'initial_cash': self.config.get('initial_cash'),
            'commission': self.config.get('commission'),
            'date_range': {
                'start': self.config.get('start_date'),
                'end': self.config.get('end_date')
            }
        }


def _print_header():
    """打印程序头部信息"""
    print("🚀 上升通道策略回测系统")
    print("=" * 60)
    print("功能: 基于上升通道技术分析的量化交易策略回测")
    print("作者: Ubiquant Team")
    print("=" * 60)


def _print_runner_info(runner: RisingChannelBacktestRunner):
    """打印运行器信息"""
    info = runner.get_runner_info()
    print(f"\n📊 运行环境信息:")
    print(f"  环境: {info['environment']}")
    print(f"  描述: {info['description']}")
    print(f"  最大股票数: {info['max_stocks'] or '无限制'}")
    print(f"  最大持仓数: {info['max_positions']}")
    print(f"  初始资金: {info['initial_cash']:,.0f} 元")
    print(f"  手续费率: {info['commission']:.4f}")
    print(f"  回测时间: {info['date_range']['start']} 至 {info['date_range']['end']}")


def _execute_backtest_by_environment(runner: RisingChannelBacktestRunner, environment: str):
    """根据环境执行相应的回测"""
    execution_map = {
        "development": _run_development_backtest,
        "optimization": _run_optimization_backtest,
        "production": _run_production_backtest,
        "full_backtest": _run_full_backtest
    }

    executor = execution_map.get(environment, _run_default_backtest)
    executor(runner)


def _run_development_backtest(runner: RisingChannelBacktestRunner):
    """开发环境回测"""
    print("\n🔍 开发环境 - 快速验证策略逻辑...")
    runner.run_basic_backtest()


def _run_optimization_backtest(runner: RisingChannelBacktestRunner):
    """优化环境回测"""
    print("\n⚙️ 优化环境 - 参数优化...")
    runner.run_parameter_optimization()


def _run_production_backtest(runner: RisingChannelBacktestRunner):
    """生产环境回测"""
    print("\n🏭 生产环境 - 完整回测...")
    runner.run_basic_backtest()
    print("\n📈 生产环境 - 策略对比...")
    runner.run_comparison_backtest()


def _run_full_backtest(runner: RisingChannelBacktestRunner):
    """完整回测"""
    print("\n🎯 完整回测 - 多股票回测...")
    runner.run_basic_backtest()
    print("\n⚙️ 完整回测 - 参数优化...")
    runner.run_parameter_optimization()
    print("\n📊 完整回测 - 策略对比...")
    runner.run_comparison_backtest()


def _run_default_backtest(runner: RisingChannelBacktestRunner):
    """默认回测"""
    print("\n📝 默认环境 - 基础回测...")
    runner.run_basic_backtest()


def _print_completion_message():
    """打印完成信息"""
    print("\n" + "=" * 60)
    print("✅ 回测完成！请查看生成的报告文件。")
    print("=" * 60)


def _print_error_message(error: Exception):
    """打印错误信息"""
    print(f"\n❌ 回测过程中出现错误: {error}")
    print("\n🔍 详细错误信息:")
    import traceback
    traceback.print_exc()
    print("\n💡 建议检查:")
    print("  1. 数据库连接是否正常")
    print("  2. 股票数据是否完整")
    print("  3. 策略参数是否合理")
    print("  4. 系统资源是否充足")


def main():
    """
    主函数 - 展示不同环境的使用方式

    提供了四种运行模式的示例：
    1. development: 开发环境，快速验证
    2. optimization: 优化环境，参数调优
    3. production: 生产环境，完整回测和对比
    4. full_backtest: 全功能回测
    """
    _print_header()

    # 可以修改这里来测试不同环境
    environment = 'development'

    try:
        runner = RisingChannelBacktestRunner(log_level=logging.INFO, environment=environment)
        _print_runner_info(runner)
        _execute_backtest_by_environment(runner, environment)
        _print_completion_message()

    except Exception as e:
        _print_error_message(e)


if __name__ == "__main__":
    main()
