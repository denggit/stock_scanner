  #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:33 PM
@File       : long_term_uptrend.py
@Description: 
"""



import pandas as pd

from backend.strategies.base import BaseStrategy


class LongTermUpTrendStrategy(BaseStrategy):
    """
    长期上涨策略
    """
    def __init__(self):
        super().__init__(name="长期上涨策略", description="多头排列")
        self._params = {}

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        pass

