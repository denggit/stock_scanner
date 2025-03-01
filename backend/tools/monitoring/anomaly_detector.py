#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:34 PM
@File       : anomaly_detector.py
@Description: 异常检测升级
"""


def detect_factor_decay(current_ic, threshold=0.1):
    if current_ic < threshold:
        alert(f"因子IC值衰减至{current_ic}，建议重新评估!")


def check_ma_strategy_health(strategy_output):
    # 检查选股数量突增（可能因子失效）
    if len(strategy_output) > 30 and historical_avg < 15:
        send_alert("均线策略选股数量异常!")
    # 检查评分分布偏移
    if strategy_output['total_score'].std() < 0.5:
        send_alert("策略评分区分度不足!")
