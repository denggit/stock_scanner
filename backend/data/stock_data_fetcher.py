#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 6:33 PM
@File       : stock_data_fetcher.py
@Description: 
"""
import datetime
from typing import Optional

import pandas as pd

from backend.data.database import DatabaseManager
from backend.data_source.baostock_source import BaostockSource


class StockDataFetcher:

    def __init__(self):
        self.db = DatabaseManager()

    def fetch_stock_data(
            self,
            code: str,
            period: str = 'daily',
            start_date: Optional[str] = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            end_date: Optional[str] = datetime.date.today().strftime("%Y-%m-%d"),
            adjust: str = '3'
    ) -> pd.DataFrame:
        """
        获取股票数据
        :param code: 股票代码
        :param period: 数据周期，可选值：daily, weekly, monthly
        :param start_date: 开始日期，格式为YYYY-MM-DD
        :param end_date: YYYY-MM-DD
        :param adjust: 复权类型，1:后复权，2:前复权，3:不复权
        """
        if period.lower().startswith('d'):
            df = self.db.get_stock_daily(code=code, start_date=start_date, end_date=end_date, adjust=adjust)
        # elif period.lower().startswith('w'):
        #     df = self.db.get_stock_weekly(code=code, start_date=start_date, end_date=end_date)
        # elif period.lower().startswith('m'):
        #     df = self.db.get_stock_monthly(code=code, start_date=start_date, end_date=end_date)
        else:
            raise ValueError(f"Invalid period: {period}")

        # 把数字型数据改为float
        numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'pe_ttm',
                           'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm']
        for column in numeric_columns:
            df[column] = df[column].astype(float)

        return df

    def get_stock_list(self, pool_name: str = "full") -> pd.DataFrame:
        """
        获取股票列表
        :return: 股票列表
        """
        if pool_name in ("full", "全量股票"):
            return self.db.get_stock_list()
        elif pool_name in ("no_st", "非ST股票"):
            stock_list = self.db.get_stock_list()
            return stock_list[~stock_list['name'].str.contains("ST")]
        elif pool_name == "st":
            stock_list = self.db.get_stock_list()
            return stock_list[stock_list['name'].str.contains("ST")]
        elif pool_name in ("sz50", "上证50"):
            bs = BaostockSource()
            rs = bs.get_sz50().rename(columns={"code_name": "name"})
            return rs
        elif pool_name in ("hs300", "沪深300"):
            bs = BaostockSource()
            rs = bs.get_hs300().rename(columns={"code_name": "name"})
            return rs
        elif pool_name in ("zz500", "中证500"):
            bs = BaostockSource()
            rs = bs.get_zz500().rename(columns={"code_name": "name"})
            return rs

        return self.db.get_stock_list()
