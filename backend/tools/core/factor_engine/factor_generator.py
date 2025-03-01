#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:00 PM
@File       : factor_generator.py
@Description: 实现因子注册机制，支持动态加载因子类（如FactorMomentum）


设计要点：
1. 性能优化 - 因子并行计算：使用Dask或Ray实现跨CPU核的因子并行计算

2. 灵活拓展 - 插件式因子：通过装饰器自动注册新因子
@register_factor(name='momentum_12m')
class Momentum12M:
    def calculate(self, close_prices):
        return close_prices.pct_change(252)
"""
