#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 1:45 AM
@File       : base.py
@Description: 
"""
from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_stock_data(self, code: str, start_date: str, end_date: str, period: str = 'daily') -> pd.DataFrame:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass
