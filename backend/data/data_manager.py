#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 8:23 PM
@File       : data_manager.py
@Description: 
"""
import logging
import time
from datetime import datetime, timedelta
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

    def update_stock_data(self, code: str, force_full_update: bool = False, lastest_date: Optional[str] = None):
        """更新单个股票数据"""
        if force_full_update or lastest_date is None:
            # 如果强制全量更新或者是新股票（没有历史数据），则从5年前开始更新
            start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
        else:
            # 从最新数据的前三天开始更新
            start_date = (datetime.strptime(lastest_date, '%Y-%m-%d') - timedelta(days=3)).strftime('%Y-%m-%d')

        end_date = datetime.now().strftime('%Y-%m-%d')

        # 如果开始日期晚于结束日期，则不更新
        if start_date > end_date:
            logging.warning(f"股票 {code} 的开始日期晚于结束日期，不更新")
            return

        df = self._retry_operation(self.data_source.get_stock_data, code=code, start_date=start_date, end_date=end_date)

        if df.empty:
            logging.warning(f"股票 {code} 没有更新数据")
            return

        self.db.update_stock_daily(code, df)
