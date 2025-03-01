#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 3/1/25 10:58 PM
@File       : validate_factor.py.py
@Description: 
"""
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
os.chdir(root_dir)

import dotenv

dotenv.load_dotenv()

from backend.quant.backtest.performance_analyzer import analyze_single_factor
from backend.quant.backtest.event_driven_engine import run_backtest


def main():
    # 统计分析
    ic_report = analyze_single_factor(factor="PE")

    # 事件回测（可选）
    backtest_report = run_backtest(strategy_config="pe_validation.yaml")

    # 生成可视化报告
    generate_html_report(ic_report, backtest_report)