#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道策略回测示例
展示如何使用回测框架和数据获取工具进行策略回测
"""

import logging
import os
import sys

# 添加项目根目录到路径（保留以兼容独立运行）
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from backend.business.backtest.strategies.rising_channel_strategy import RisingChannelBacktestStrategy
from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.backtest.execution.base.base_runner import BaseBacktestRunner


class RisingChannelBacktestRunner(BaseBacktestRunner):
    """
    上升通道策略回测运行器
    继承通用`BaseBacktestRunner`，仅绑定策略类与配置类。
    """

    def __init__(self, log_level=logging.INFO, environment: str | None = None):
        """
        初始化回测运行器
        
        Args:
            log_level: 日志级别
            environment: 运行环境 (development/production/optimization/full_backtest)
        """
        super().__init__(
            strategy_class=RisingChannelBacktestStrategy,
            config_cls=RisingChannelConfig,
            log_level=log_level,
            environment=environment,
        )


def main():
    """
    主函数 - 展示不同环境的使用方式
    """
    print("上升通道多股票策略回测示例")
    print("=" * 50)

    environment = 'optimization'
    print(f"当前运行环境: {environment}")

    runner = RisingChannelBacktestRunner(log_level=logging.INFO, environment=environment)

    print(f"环境配置: {runner.config['description']}")
    print(f"股票数量限制: {runner.config['max_stocks']}")

    try:
        if environment == "development":
            print("\n1. 开发环境 - 快速验证策略逻辑...")
            _ = runner.run_basic_backtest()
        elif environment == "optimization":
            print("\n1. 优化环境 - 参数优化...")
            _ = runner.run_parameter_optimization()
        elif environment == "production":
            print("\n1. 生产环境 - 完整回测...")
            _ = runner.run_basic_backtest()
            print("\n2. 生产环境 - 对比回测...")
            _ = runner.run_comparison_backtest()
        elif environment == "full_backtest":
            print("\n1. 完整回测 - 多股票回测...")
            _ = runner.run_basic_backtest()
            print("\n2. 完整回测 - 参数优化...")
            _ = runner.run_parameter_optimization()
            print("\n3. 完整回测 - 对比回测...")
            _ = runner.run_comparison_backtest()
        else:
            print("\n1. 默认环境 - 基础回测...")
            _ = runner.run_basic_backtest()

        print("\n" + "=" * 50)
        print("回测完成！")
    except Exception as e:
        print(f"\n❌ 回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
