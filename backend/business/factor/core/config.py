#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : config.py
@Description: 因子研究框架配置
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import warnings

import pandas as pd


def setup_environment():
    """
    设置运行环境，减少警告信息
    """
    # 设置pandas选项，减少FutureWarning
    pd.set_option('future.no_silent_downcasting', True)

    # 忽略特定的警告
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

    # 设置日志级别
    import logging
    logging.getLogger('vectorbt').setLevel(logging.WARNING)
    logging.getLogger('quantstats').setLevel(logging.WARNING)


# 自动设置环境
setup_environment()
