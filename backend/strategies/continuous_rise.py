#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/16/25 3:52 PM
@File       : continuous_rise.py
@Description: 
"""
import pandas as pd

from backend.data_source.akshare_source import AKShareSource
from backend.strategies.base import BaseStrategy


class ContinuousRiseStrategy(BaseStrategy):
    """
    放量上涨策略
    """

    def __init__(self):
        super().__init__(name="放量上涨策略", description="持续放量上涨")
        self._params = {
            "volume_up_days": 3,  # 持续放量天数
            "price_up_days": 3,  # 持续上涨天数
        }

    def generate_signal(self, data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
        """生成交易信号"""
        ak = AKShareSource()
        volume_up = ak.get_technical_indicators(indicator_type="volume_up").drop(columns="序号")
        price_up = ak.get_technical_indicators(indicator_type="continuous_rise").drop(columns="序号")

        # 合并两个df
        df = pd.merge(volume_up, price_up, on="股票代码")
        df = df.drop(columns=['股票简称_y', '所属行业_y'])
        df = df.rename(columns={"股票简称_x": "股票简称", "所属行业_x": "所属行业"})

        return df
