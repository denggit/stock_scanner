#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 8:23 PM
@File       : data_manager.py
@Description: 
"""
import datetime as dt
import logging
import time
from typing import Optional

import pandas as pd

from backend.data.database import DatabaseManager
from backend.data_source.baostock_source import BaostockSource
from backend.data_source.base import DataSource


class DataUpdateManager:
    def __init__(self, data_source: Optional[DataSource] = None):
        self.db = DatabaseManager()
        self.data_source = data_source or BaostockSource()
        self._init_connection()

    def _init_connection(self):
        if not self.data_source.connect():
            raise Exception("Failed to connect to data source")

    def _retry_operation(self, operation, max_retries: int = 3, retry_delay: int = 5, **kwargs):
        for i in range(max_retries):
            try:
                return operation(**kwargs)
            except Exception as e:
                logging.warning(
                    f"Operation {operation.__name__} with params {kwargs} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                if not self.data_source.is_connected():
                    self._init_connection()
        raise Exception("Operation failed after multiple retries")

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self._retry_operation(self.data_source.get_stock_list)

    def update_stock_list(self, stock_list: pd.DataFrame):
        """更新股票列表"""
        self._retry_operation(self.db.save_stock_basic, stock_basic=stock_list)

    def update_all_stocks(self, force_full_update: bool = False, progress_callback=None):
        """更新所有股票数据
        
        :param force_full_update: 是否强制全量更新
        :param progress_callback: 进度回调函数
        """
        stock_list = self.get_stock_list()
        total_stocks = len(stock_list)
        updated_count = 0
        failed_count = 0
        failed_codes = []

        logging.info(f"开始更新 {total_stocks} 只股票的数据...")

        # 一次性获取所有股票的最新日期
        lastest_dates = {} if force_full_update else self._retry_operation(self.db.get_all_update_time)

        for code in stock_list['code']:
            try:
                self.update_stock_data(code, force_full_update, lastest_dates.get(code))
                updated_count += 1
                if progress_callback:
                    progress_callback()
            except Exception as e:
                failed_count += 1
                failed_codes.append(code)
                logging.exception(f"更新股票 {code} 数据失败: {e}")
                if not self.data_source.is_connected():
                    self._init_connection()
                if progress_callback:
                    progress_callback()
                continue

        logging.info(f"更新完成，成功更新 {updated_count} 只股票，失败 {failed_count} 只股票")

        return {
            "total": total_stocks,
            "updated": updated_count,
            "failed": failed_count,
            "failed_codes": failed_codes
        }

    def update_stock_data(self, code: str, force_full_update: bool = False, latest_date: Optional[str] = None):
        """更新单个股票数据"""
        today_6pm = dt.datetime.combine(dt.datetime.today(), dt.time(18, 0))
        if force_full_update or latest_date is None:
            # 如果强制全量更新或者是新股票（没有历史数据），则从5年前开始更新
            start_date = (dt.datetime.now() - dt.timedelta(days=5 * 365)).strftime('%Y-%m-%d')
        elif dt.datetime.strptime(latest_date, '%Y-%m-%d %H:%M:%S') > today_6pm:
            # 如果是今天下午六点后更新的，无需更新
            logging.warning(f"股票 {code} 今天已经更新，无需重复更新")
            return
        else:
            # 如果最新更新日期为交易日，则选择当天为start_date，否则向前选择一个交易日
            pre_15_days = (dt.datetime.strptime(latest_date, '%Y-%m-%d %H:%M:%S') - dt.timedelta(days=15)).strftime(
                '%Y-%m-%d')
            trading_calendar = self.data_source.get_trading_calendar(start_date=pre_15_days, end_date=latest_date)
            start_date = "2025-01-01"
            for i in range(1, len(trading_calendar) + 1):
                if trading_calendar.is_trading_day.iloc[-i] == "1":
                    start_date = trading_calendar.calendar_date.iloc[-i]
                    break

        end_date = dt.datetime.now().strftime('%Y-%m-%d')

        # 如果开始日期晚于结束日期，则不更新
        if start_date > end_date:
            logging.warning(f"股票 {code} 的开始日期晚于结束日期，不更新")
            return

        df = self._retry_operation(self.data_source.get_stock_data, code=code, start_date=start_date, end_date=end_date)

        if df.empty:
            logging.warning(f"股票 {code} 没有更新数据")
            return

        self.db.update_stock_daily(code, df)

    def update_financial_data(self, code: str, year: int, quarter: int = None):
        """更新单个股票的财务数据
        
        Args:
            code: 股票代码
            year: 年份
            quarter: 季度（可选）
        """
        # 获取各类财务数据
        profit_data = self._retry_operation(
            self.data_source.get_profit_data,
            code=code,
            year=year,
            quarter=quarter
        )
        
        balance_data = self._retry_operation(
            self.data_source.get_balance_data,
            code=code,
            year=year,
            quarter=quarter
        )
        
        cashflow_data = self._retry_operation(
            self.data_source.get_cashflow_data,
            code=code,
            year=year,
            quarter=quarter
        )
        
        growth_data = self._retry_operation(
            self.data_source.get_growth_data,
            code=code,
            year=year,
            quarter=quarter
        )
        
        operation_data = self._retry_operation(
            self.data_source.get_operation_data,
            code=code,
            year=year,
            quarter=quarter
        )
        
        dupont_data = self._retry_operation(
            self.data_source.get_dupont_data,
            code=code,
            year=year,
            quarter=quarter
        )

        dividend_data = self._retry_operation(
            self.data_source.get_dividend_data,
            code=code,
            year=year
        )
        
        # 保存到数据库
        if not profit_data.empty:
            self.db.save_profit_data(profit_data)
        if not balance_data.empty:
            self.db.save_balance_data(balance_data)
        if not cashflow_data.empty:
            self.db.save_cashflow_data(cashflow_data)
        if not growth_data.empty:
            self.db.save_growth_data(growth_data)
        if not operation_data.empty:
            self.db.save_operation_data(operation_data)
        if not dupont_data.empty:
            self.db.save_dupont_data(dupont_data)
        if not dividend_data.empty:
            self.db.save_dividend_data(dividend_data)

    def update_all_financial_data(self, start_year: int, end_year: int = None, progress_callback=None):
        """更新所有股票的财务数据
        
        Args:
            start_year: 开始年份
            end_year: 结束年份（默认为当前年份）
            progress_callback: 进度回调函数
        """
        if end_year is None:
            end_year = dt.datetime.now().year
        
        stock_list = self.get_stock_list()
        total_stocks = len(stock_list)
        updated_count = 0
        failed_count = 0
        failed_codes = []
        
        for code in stock_list['code']:
            try:
                for year in range(start_year, end_year + 1):
                    self.update_financial_data(code, year)
                updated_count += 1
                if progress_callback:
                    progress_callback()
            except Exception as e:
                failed_count += 1
                failed_codes.append(code)
                logging.exception(f"更新股票 {code} 财务数据失败: {e}")
                if progress_callback:
                    progress_callback()
                continue
        
        return {
            "total": total_stocks,
            "updated": updated_count,
            "failed": failed_count,
            "failed_codes": failed_codes
        }
