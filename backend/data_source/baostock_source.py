#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 1:45 AM
@File       : baostock_source.py
@Description: 
"""
import logging
import time

import baostock as bs
import pandas as pd

from .base import DataSource


class BaostockSource(DataSource):
    # 定义错误码
    NETWORK_ERROR_CODES = ['10002007', '10002009', '10002010']
    LOGIN_ERROR_CODES = ['10001001', '10001002', '10001003']
    SUCCESS_CODE = '0'

    def __init__(self):
        self._connected = False

    def connect(self):
        try:
            if self._connected:
                bs.logout()
                time.sleep(1)

            login_result = bs.login()
            self._connected = login_result.error_code == self.SUCCESS_CODE
            if self._connected:
                logging.info("Baostock login successful")
            else:
                logging.error(f"Baostock login failed: {login_result.error_msg}")
        except Exception as e:
            logging.error(f"Baostock login failed: {e}")
            self._connected = False
        return self._connected

    def disconnect(self):
        try:
            bs.logout()
            self._connected = False
            logging.info("Baostock logout successful")
        except Exception as e:
            logging.warning(f"Baostock logout failed: {e}")

    def is_connected(self) -> bool:
        return self._connected

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表（仅上市的A股）"""
        if not self._connected:
            self.connect()

        rs = bs.query_stock_basic()
        if rs.error_code != self.SUCCESS_CODE:
            raise Exception(f"Failed to get stock list: {rs.error_msg}")

        df = rs.get_data()

        # 重命名列
        df = df.rename(columns={
            'code_name': 'name',
            'ipoDate': 'ipo_date',
            'outDate': 'out_date',
        })

        # 过滤
        # type='1' 表示A股
        # status='1' 表示上市
        df = df[(df['type'] == '1') & (df['status'] == '1')].copy()

        # 转换列类型
        df['ipo_date'] = pd.to_datetime(df['ipo_date'])
        df['out_date'] = pd.to_datetime(df['out_date'])

        return df

    def get_stock_data(self, code: str, start_date: str, end_date: str, period: str = 'daily',
                       adjust: str = '3') -> pd.DataFrame:
        """获取股票数据
        
        :param adjust: 复权类型，1:后复权，2:前复权，3:不复权
        """
        if not self._connected:
            self.connect()

        fields = 'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST'

        rs = bs.query_history_k_data_plus(code, fields=fields, start_date=start_date, end_date=end_date,
                                          frequency=period[0], adjustflag=adjust)
        if rs.error_code != self.SUCCESS_CODE:
            if rs.error_code in self.NETWORK_ERROR_CODES + self.LOGIN_ERROR_CODES:
                self.connect()
                return self.get_stock_data(code, start_date, end_date, period, adjust)
            else:
                raise Exception(f"Failed to get stock data: {rs.error_msg}")

        data_list = []
        while (rs.error_code == self.SUCCESS_CODE) & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        # 创建DataFrame并重命名列
        columns = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn',
                   'tradestatus', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'isST']
        df = pd.DataFrame(data_list, columns=columns)

        # 重命名列以匹配数据库字段
        column_mapping = {
            'date': 'trade_date',
            'pctChg': 'pct_chg',
            'peTTM': 'pe_ttm',
            'pbMRQ': 'pb_mrq',
            'psTTM': 'ps_ttm',
            'pcfNcfTTM': 'pcf_ncf_ttm',
            'isST': 'is_st',
        }
        df = df.rename(columns=column_mapping)

        # 转换数据类型
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['tradestatus'] = df['tradestatus'].astype(int)
        df['is_st'] = df['is_st'].astype(int)

        numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg', 'pe_ttm',
                           'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 确保所有必须的列都存在
        required_columns = ['code', 'trade_date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                            'turn', 'tradestatus', 'pct_chg', 'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm', 'is_st']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据源返回的数据缺少必须的列：{missing_columns}")

        return df[required_columns]  # 只返回必须的列，并按照固定顺去配置
