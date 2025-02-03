#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:32 PM
@File       : breakout.py
@Description: 
"""
import pandas as pd

from backend.strategies.base import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="突破策略", description="突破高点打板")
        self._params = {

        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        pass
