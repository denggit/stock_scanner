#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 6:33 PM
@File       : stock_data_fetcher.py
@Description: 
"""
import datetime
import logging
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
            tradestatus: int = 1,
            adjust: str = '3'
    ) -> pd.DataFrame:
        """
        获取股票数据
        :param code: 股票代码
        :param period: 数据周期，可选值：daily, weekly, monthly
        :param start_date: 开始日期，格式为YYYY-MM-DD
        :param end_date: YYYY-MM-DD
        :param adjust: 复权类型，1:后复权，2:前复权，3:不复权
        :param tradestatus: 1: 正常交易 0: 停牌

        """
        if period.lower().startswith('d'):
            df = self.db.get_stock_daily(code=code, start_date=start_date, end_date=end_date, adjust=adjust)
            # 把数字型数据改为float
            numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg',
                               'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm', 'vwap']
        # elif period.lower().startswith('w'):
        #     df = self.db.get_stock_weekly(code=code, start_date=start_date, end_date=end_date)
        # elif period.lower().startswith('m'):
        #     df = self.db.get_stock_monthly(code=code, start_date=start_date, end_date=end_date)
        elif period.lower() in ('5min', '5'):
            df = self.db.get_stock_5min(code=code, start_date=start_date, end_date=end_date, adjust=adjust)
            # 把数字型数据改为float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap']
        else:
            raise ValueError(f"Invalid period: {period}")

        if len(df) == 0:
            logging.warning(f"No data found for code: {code}. Please Update Database")
            return df
        for column in numeric_columns:
            df[column] = df[column].astype(float)
        # 过滤交易状态
        df = df[df.tradestatus == tradestatus]

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

    def get_stock_list_with_cond(self, pool_name: str = "full", ipo_date: Optional[str] = None,
                                 min_amount: Optional[float] = None,
                                 end_date: Optional[datetime] = datetime.date.today()) -> pd.DataFrame:
        """
        获取股票列表，带条件

        Parameters:
        pool_name: 股票池
        ipo_date: 必须要在这个日期前上市的股票 (YYYY-MM-DD)
        daily_volume: 5日日均成交额不低于该值

        :return: 股票列表
        """
        stock_list = self.get_stock_list(pool_name=pool_name)
        if ipo_date is not None:
            if "ipo_date" not in stock_list.columns:
                temp_stock_list = self.get_stock_list()
                stock_list = temp_stock_list[temp_stock_list.code.isin(stock_list.code)]
            stock_list = stock_list[stock_list['ipo_date'] <= datetime.datetime.strptime(ipo_date, "%Y-%m-%d").date()]

        if min_amount is not None:
            start_date = (end_date - datetime.timedelta(days=20)).strftime("%Y-%m-%d")
            match_codes = []
            for code in stock_list.code.to_list():
                stock_data = self.fetch_stock_data(code, start_date=start_date).tail(5)
                if stock_data.empty:
                    continue
                avg_amount = stock_data['amount'].mean()
                if avg_amount > min_amount:
                    match_codes.append(code)
            stock_list = stock_list[stock_list['code'].isin(match_codes)]
        return stock_list
