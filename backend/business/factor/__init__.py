#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:00 PM
@File       : __init__.py.py
@Description: 架构

├── configs/                          # 新增策略配置目录
│   ├── strategy_templates/           # 策略模板（YAML/JSON）
│   └── risk_constraints/             # 风控规则配置
├── core/
│   ├── data_preprocessing/           # 更名为core
│   │   ├── data_validator.py         # 原data_splitter功能升级
│   │   ├── data_cleaner.py           # 清理异常数据
│   │   └── feature_engineer.py       # 原data_normalizer功能扩展
│   ├── engine/
│   │   ├── generator.py       # 单因子计算
│   │   ├── factor_analyzer.py        # 新增：因子有效性分析
│   │   └── factor_optimizer.py       # 因子组合优化
│   ├── scoring_model/
│   │   ├── model_factory.py          # 模型动态加载
│   │   ├── static_models.py          # 等权/IC加权等传统模型
│   │   └── ml_models/                # 机器学习模型子包
│   ├── execution/                    # 新增执行层
│   │   ├── order_generator.py        # 信号转订单
│   │   └── cost_model.py             # 交易成本估算
├── storage/
│   ├── metastore/                    # 元数据存储
│   ├── timeseries/                   # 时序数据存储
│   └── results/                      # 回测结果存储
├── visualization/                    # 独立可视化层
│   ├── factor_visualizer.py
│   └── portfolio_analyzer.py
├── monitoring/
│   ├── anomaly_detector.py           # 异常检测升级
│   └── performance_monitor.py
└── backtest_event/
    ├── event_driven_engine.py        # 事件驱动回测
    └── risk_analyzer.py
"""
