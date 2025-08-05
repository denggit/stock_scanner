#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/16/25 3:52 PM
@File       : continuous_rise.py
@Description: 
"""
import pandas as pd

from backend.business.data.source.akshare_src import AKShareSource
from backend.business.strategies.base import BaseStrategy


class ContinuousRiseStrategy(BaseStrategy):
    """
    放量上涨策略
    """

    def __init__(self):
        super().__init__(name="放量上涨策略", description="持续放量上涨")
        self._params = {
            "continuous_days": 3,  # 持续量价齐升天数
        }

    def generate_signal(self, data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
        """生成交易信号"""
        ak = AKShareSource()
        df = ak.get_technical_indicators(indicator_type="volume_price_rise").drop(columns="序号")
        df = df[df["量价齐升天数"] >= self._params["continuous_days"]]

        return df
