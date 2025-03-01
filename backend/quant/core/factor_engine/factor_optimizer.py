#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:00 PM
@File       : factor_optimizer.py
@Description: 使用CVaR方法优化因子权重，控制极端风险，只负责权重分配，不涉及具体交易规则

优化因子权重（IC加权、均值方差优化等），输出最优因子组合
- 输入：单因子历史收益数据
- 输出：优化后的因子权重（如：{"PE": 0.4, "MOM": 0.6}）
"""
