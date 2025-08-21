#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:00 PM
@File       : __init__.py
@Description: 因子研究框架 - 整合因子开发到回测全流程

重构后的架构:

├── core/                          # 核心模块
│   ├── factor/                    # 因子定义和计算
│   │   ├── base_factor.py         # 基础因子类
│   │   ├── factor_registry.py     # 因子注册管理
│   │   └── factor_engine.py       # 因子计算引擎
│   ├── data/                      # 数据管理
│   │   ├── data_manager.py        # 数据管理器
│   │   ├── data_validator.py      # 数据验证
│   │   └── data_cleaner.py        # 数据清洗
│   ├── backtest/                  # 回测系统 (整合自backtest_factor)
│   │   ├── backtest_engine.py     # 回测引擎
│   │   ├── portfolio_manager.py   # 组合管理
│   │   └── risk_manager.py        # 风险管理
│   ├── analysis/                  # 因子分析
│   │   ├── factor_analyzer.py     # 因子有效性分析
│   │   ├── ic_analyzer.py         # IC分析
│   │   └── performance_analyzer.py # 绩效分析
│   └── reporting/                 # 报告生成
│       ├── report_generator.py    # 报告生成器
│       └── visualization.py       # 可视化
├── library/                       # 因子库 (用户主要修改的地方)
│   ├── technical_factors.py       # 技术因子
│   ├── fundamental_factors.py     # 基本面因子
│   ├── worldquant_factors.py      # WorldQuant Alpha因子
│   └── channel_factors.py         # 通道分析因子
│   ├── data/                      # 数据管理
│   │   ├── data_manager.py        # 数据管理器
│   │   ├── data_validator.py      # 数据验证
│   │   └── data_cleaner.py        # 数据清洗
│   ├── backtest/                  # 回测系统 (整合自backtest_factor)
│   │   ├── backtest_engine.py     # 回测引擎
│   │   ├── portfolio_manager.py   # 组合管理
│   │   └── risk_manager.py        # 风险管理
│   ├── analysis/                  # 因子分析
│   │   ├── factor_analyzer.py     # 因子有效性分析
│   │   ├── ic_analyzer.py         # IC分析
│   │   └── performance_analyzer.py # 绩效分析
│   └── reporting/                 # 报告生成
│       ├── report_generator.py    # 报告生成器
│       └── visualization.py       # 可视化
├── configs/                       # 配置文件
│   ├── factor_config.yaml         # 因子配置
│   ├── backtest_config.yaml       # 回测配置
│   └── strategy_templates/        # 策略模板
├── storage/                       # 数据存储
│   ├── factor_data/               # 因子数据
│   ├── backtest_results/          # 回测结果
│   └── reports/                   # 报告文件
├── utils/                         # 工具函数
│   ├── logger.py                  # 日志工具
│   └── helpers.py                 # 辅助函数
└── main.py                        # 主入口文件
"""

# 导入因子库，自动注册所有因子
from . import library
from .core.analysis.factor_analyzer import FactorAnalyzer
from .core.backtest.backtest_engine import FactorBacktestEngine
from .core.data.data_manager import FactorDataManager
from .core.factor.base_factor import BaseFactor
from .core.factor.factor_engine import FactorEngine
from .core.factor.factor_registry import FactorRegistry
from .core.reporting.report_generator import FactorReportGenerator
from .main import FactorResearchFramework, create_factor_research_framework, run_quick_factor_analysis

__all__ = [
    'BaseFactor',
    'FactorRegistry',
    'FactorEngine',
    'FactorDataManager',
    'FactorBacktestEngine',
    'FactorAnalyzer',
    'FactorReportGenerator',
    'FactorResearchFramework',
    'create_factor_research_framework',
    'run_quick_factor_analysis'
]

__version__ = "2.0.0"
