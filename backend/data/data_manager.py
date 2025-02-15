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
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.data.database import DatabaseManager
from backend.data_source.baostock_source import BaostockSource
from backend.data_source.base import DataSource


class DataUpdateManager:
    def __init__(self, data_source: Optional[DataSource] = None):
        self.db = DatabaseManager()
        self.data_source = data_source or BaostockSource()
        self._init_connection()
        self.trading_calendar = self.data_source.get_trading_calendar(start_date="2025-01-01")
        self.data_queue = queue.Queue(maxsize=100)  # 限制队列大小以控制内存使用
        self.producer_done = threading.Event()
        self.stats_lock = threading.Lock()  # 添加线程锁用于保护统计信息

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

    def update_all_stocks(self, force_full_update: bool = False, adjust: str = "3", progress_callback=None):
        """更新所有股票数据

        Args:
            force_full_update (bool): 是否强制全量更新
            adjust（str）: 是否复权 1->后复权，2->前复权，3->不复权，None->不复权和后复权一起
            progress_callback (callable): 进度回调函数

        Returns:
            Dict: 更新统计信息
        """
        stock_list = self.get_stock_list()
        total_stocks = len(stock_list)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 获取最新日期
        if not adjust:
            latest_dates = {} if force_full_update else self._retry_operation(self.db.get_all_update_time, adjust='3')
            latest_dates_back = {} if force_full_update else self._retry_operation(self.db.get_all_update_time,
                                                                                   adjust='1')
        else:
            latest_dates = {} if force_full_update else self._retry_operation(self.db.get_all_update_time,
                                                                              adjust=adjust)
            latest_dates_back = {}

        # 启动消费者线程
        consumer = threading.Thread(
            target=self._consume_stock_data,
            args=(update_stats, progress_callback)
        )
        consumer.start()

        # 生产者：获取股票数据
        try:
            for code in stock_list['code']:
                try:
                    if not adjust:
                        # 获取不复权数据
                        df = self.__fetch_stock_data(code, force_full_update, latest_dates.get(code), adjust='3')
                        if not df.empty:
                            self.data_queue.put((code, df, '3'))
                        elif progress_callback:
                            progress_callback()

                        # 获取后复权数据
                        df_back = self.__fetch_stock_data(code, force_full_update, latest_dates_back.get(code),
                                                          adjust='1')
                        if not df_back.empty:
                            self.data_queue.put((code, df_back, '1'))
                        elif progress_callback:
                            progress_callback()
                    else:
                        # 获取数据
                        df = self.__fetch_stock_data(code, force_full_update, latest_dates.get(code), adjust=adjust)
                        if not df.empty:
                            self.data_queue.put((code, df, adjust))
                        elif progress_callback:
                            progress_callback()
                except Exception as e:
                    with self.stats_lock:  # 使用线程锁保护统计信息的更新
                        update_stats["failed"] += 1
                        update_stats["failed_codes"].append(code)
                    # 即使获取数据失败也需要回复进度
                    if progress_callback:
                        progress_callback()
                    logging.exception(f"获取股票 {code} 数据失败: {e}")
                    if not self.data_source.is_connected():
                        self._init_connection()
        finally:
            # 标记生产者完成
            self.producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        update_stats["total"] = total_stocks
        return update_stats

    def _consume_stock_data(self, update_stats: Dict, progress_callback=None):
        """消费者：处理股票数据并保存到数据库

        Args:
            update_stats (Dict): 更新统计信息
            progress_callback (callable): 进度回调函数
        """
        while not (self.producer_done.is_set() and self.data_queue.empty()):
            try:
                # 等待5秒获取数据，如果超时则检查生产者是否完成
                code, df, adjust = self.data_queue.get(timeout=5)
                try:
                    self.db.update_stock_daily(code, df, adjust)
                    with self.stats_lock:  # 使用线程锁保护统计信息的更新
                        update_stats["updated"] += 1
                except Exception as e:
                    with self.stats_lock:  # 使用线程锁保护统计信息的更新
                        update_stats["failed"] += 1
                        update_stats["failed_codes"].append(code)
                    logging.exception(f"保存股票 {code} 数据失败: {e}")
                finally:
                    if progress_callback:
                        progress_callback()
                    self.data_queue.task_done()
            except queue.Empty:
                logging.info("队列为空，等待数据...")
                continue

    def __fetch_stock_data(self, code: str, force_full_update: bool = False, latest_date: Optional[str] = None,
                           adjust: str = '3') -> pd.DataFrame:
        """获取单只股票的数据"""
        if isinstance(latest_date, str):
            latest_date = dt.datetime.strptime(latest_date, '%Y-%m-%d %H:%M:%S')
        today_6pm = dt.datetime.combine(dt.datetime.today(), dt.time(18, 0))
        if force_full_update or latest_date is None:
            # 如果强制全量更新或者是新股票（没有历史数据），则从5年前开始更新
            start_date = (dt.datetime.now() - dt.timedelta(days=5 * 365)).strftime('%Y-%m-%d')
        elif latest_date > today_6pm:
            # 如果是今天下午六点后更新的，无需更新
            logging.warning(f"股票 {code}_{adjust} 今天已经更新，无需重复更新")
            return pd.DataFrame()
        else:
            # 如果最新更新日期为交易日，则选择start_date该日期，否则选择前一个交易日为start_date，避免出现跨0点运行代码导致的数据缺失
            days = (dt.datetime.today() - latest_date).days + 1
            start_date = "2025-01-01"
            for i in range(days, len(self.trading_calendar) + 1):
                if self.trading_calendar.is_trading_day.iloc[-i] == "1":
                    start_date = self.trading_calendar.calendar_date.iloc[-i]
                    break

        end_date = dt.datetime.now().strftime('%Y-%m-%d')

        # 如果开始日期晚于结束日期，则不更新
        if start_date > end_date:
            logging.warning(f"股票 {code}_{adjust} 的开始日期晚于结束日期，不更新")
            return pd.DataFrame()

        df = self._retry_operation(self.data_source.get_stock_data, code=code, start_date=start_date, end_date=end_date,
                                   adjust=adjust)

        if df.empty:
            logging.warning(f"股票 {code}_{adjust} 没有更新数据")
            return pd.DataFrame()

        return df

    def update_financial_data(self, code: str, year: int, quarter: int = None):
        """更新单个股票的财务数据
        
        Args:
            code: 股票代码
            year: 年份
            quarter: 季度（可选）

        """
        # 获取各类财务数据
        profit_data = self.__fetch_profit_data(code, year, quarter)
        balance_data = self.__fetch_balance_data(code, year, quarter)
        cashflow_data = self.__fetch_cashflow_data(code, year, quarter)
        growth_data = self.__fetch_growth_data(code, year, quarter)
        operation_data = self.__fetch_operation_data(code, year, quarter)
        dupont_data = self.__fetch_dupont_data(code, year, quarter)
        dividend_data = self.__fetch_dividend_data(code, year)

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

    @staticmethod
    def __fetch_all_financial_data(stock_list, fetcher, start_year, end_year, data_queue, update_stats, progress_callback):
        """财务数据生产者：通过fetcher来获取每一只股票的财务数据"""
        for code in stock_list['code']:
            try:
                for year in range(start_year, end_year + 1):
                    profit_data = fetcher(code, year)
                    if not profit_data.empty:
                        data_queue.put((code, profit_data))
                        update_stats["updated"] += 1
                        if progress_callback:
                            progress_callback()
            except Exception as e:
                logging.exception(f"获取股票 {code} 利润表数据失败: {e}")
                update_stats["failed"] += 1
                update_stats["failed_codes"].append(code)
                if progress_callback:
                    progress_callback()

    @staticmethod
    def __consume_financial_data(producer_done, data_queue, data_saver, update_stats, progress_callback):
        """财务数据消费者：消费对应data_queue里的财务数据"""
        while not (producer_done.is_set() and data_queue.empty()):
            try:
                code, data = data_queue.get(timeout=5)
                try:
                    data_saver(data)
                    update_stats["updated"] += 1
                except Exception as e:
                    logging.exception(f"保存利润数据失败：{e}")
                    update_stats["failed"] += 1
                    update_stats["failed_codes"].append(code)
                finally:
                    data_queue.task_done()
                    if progress_callback:
                        progress_callback()
            except queue.Empty:
                logging.info(f"利润数据队列为空，等待数据...")
                continue

    def _update_profit_data(self, stock_list, start_year, end_year, progress_callback):
        """更新单个股票的利润数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 启动消费者线程
        consumer = threading.Thread(
            target=self.__consume_financial_data,
            args=(producer_done, data_queue, self.db.save_profit_data, update_stats, progress_callback)
        )
        consumer.start()

        try:
            self.__fetch_all_financial_data(
                stock_list=stock_list,
                fetcher=self.__fetch_profit_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                progress_callback=progress_callback
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    def _update_balance_data(self, stock_list, start_year, end_year, progress_callback):
        """更新单个股票的资产负债表数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 启动消费者线程
        consumer = threading.Thread(
            target=self.__consume_financial_data,
            args=(producer_done, data_queue, self.db.save_balance_data, update_stats, progress_callback)
        )
        consumer.start()

        try:
            self.__fetch_all_financial_data(
                stock_list=stock_list,
                fetcher=self.__fetch_balance_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                progress_callback=progress_callback
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    def _update_cashflow_data(self, stock_list, start_year, end_year, progress_callback):
        """更新单个股票的现金流量表数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 启动消费者线程
        consumer = threading.Thread(
            target=self.__consume_financial_data,
            args=(producer_done, data_queue, self.db.save_cashflow_data, update_stats, progress_callback)
        )
        consumer.start()

        try:
            self.__fetch_all_financial_data(
                stock_list=stock_list,
                fetcher=self.__fetch_cashflow_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                progress_callback=progress_callback
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    def _update_growth_data(self, stock_list, start_year, end_year, progress_callback):
        """更新单个股票的成长数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 启动消费者线程
        consumer = threading.Thread(
            target=self.__consume_financial_data,
            args=(producer_done, data_queue, self.db.save_growth_data, update_stats, progress_callback)
        )
        consumer.start()

        try:
            self.__fetch_all_financial_data(
                stock_list=stock_list,
                fetcher=self.__fetch_growth_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                progress_callback=progress_callback
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    def _update_operation_data(self, stock_list, start_year, end_year, progress_callback):
        """更新单个股票的经营数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 启动消费者线程
        consumer = threading.Thread(
            target=self.__consume_financial_data,
            args=(producer_done, data_queue, self.db.save_operation_data, update_stats, progress_callback)
        )
        consumer.start()

        try:
            self.__fetch_all_financial_data(
                stock_list=stock_list,
                fetcher=self.__fetch_operation_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                progress_callback=progress_callback
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    def _update_dupont_data(self, stock_list, start_year, end_year, progress_callback):
        """更新单个股票的杜邦分析数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 启动消费者线程
        consumer = threading.Thread(
            target=self.__consume_financial_data,
            args=(producer_done, data_queue, self.db.save_dupont_data, update_stats, progress_callback)
        )
        consumer.start()

        try:
            self.__fetch_all_financial_data(
                stock_list=stock_list,
                fetcher=self.__fetch_dupont_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                progress_callback=progress_callback
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    def _update_dividend_data(self, stock_list, start_year, end_year, progress_callback):
        """更新单个股票的分红数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }

        # 启动消费者线程
        consumer = threading.Thread(
            target=self.__consume_financial_data,
            args=(producer_done, data_queue, self.db.save_dividend_data, update_stats, progress_callback)
        )
        consumer.start()

        try:
            self.__fetch_all_financial_data(
                stock_list=stock_list,
                fetcher=self.__fetch_dividend_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                progress_callback=progress_callback
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    def update_all_financial_data(self, start_year: int, end_year: int = None):
        """更新所有股票的财务数据

        Args:
            start_year: 开始年份
            end_year: 结束年份（默认为当前年份）
        """
        if end_year is None:
            end_year = dt.datetime.now().year

        stock_list = self.get_stock_list()
        total_stocks = len(stock_list)
        years_count = end_year - start_year + 1
        # 每一年一份数据，7个财务表
        total_updates = total_stocks * years_count * 7

        # 添加进度条
        pbar = tqdm(total=total_updates, desc="更新财务数据")

        results = {}
        # 创建进程池来并行更新不同类型的财务数据
        with ProcessPoolExecutor() as executor:
            # 定义要更新的所有财务数据类型及其对应的更新方法
            update_tasks = [
                (self._update_profit_data, "利润表"),
                (self._update_balance_data, "资产负债表"),
                (self._update_cashflow_data, "现金流量表"),
                (self._update_growth_data, "成长能力"),
                (self._update_operation_data, "营运能力"),
                (self._update_dupont_data, "杜邦分析"),
                (self._update_dividend_data, "分红数据")
            ]

            # 提交所有任务到进程池
            futures = []
            for update_func, data_type in update_tasks:
                future = executor.submit(
                    update_func,
                    stock_list=stock_list,
                    start_year=start_year,
                    end_year=end_year,
                    progress_callback=lambda: pbar.update(1)
                )
                futures.append((future, data_type))

            # 等待所有任务完成
            for future, data_type in futures:
                try:
                    results[data_type] = future.result()  # 等待任务完成
                    logging.info(f"{data_type}数据更新完成")
                except Exception as e:
                    logging.exception(f"更新{data_type}数据时发生错误: {e}")

        pbar.close()  # 关闭进度条
        for data_type, result in results.items():
            logging.info(f"{data_type} 数据更新结果：")
            for key, value in result.items():
                logging.info(f"{key}: {value}")

        return results

    @staticmethod
    def __format_financial_df(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
        """格式化财务数据DataFrame"""
        if df.empty:
            return df

        # 创建一个新的 DataFrame 来存储转换后的数据
        result_df = df.copy()

        # 转换日期列
        date_columns = [col for col in result_df.columns if 'Date' in col]
        for col in date_columns:
            # 将 NaT 转换为 None
            result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
            result_df[col] = result_df[col].apply(lambda x: x.date() if pd.notna(x) else None)

        # 只处理指定的数值列
        for col in numeric_columns:
            if col in result_df.columns:
                # 将列转换为数值类型，无效值变为 NaN
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

                # 将无效的数值转换为 None
                result_df[col] = result_df[col].apply(
                    lambda x: float(x) if pd.notna(x) and not np.isinf(x) else None
                )

        # 最后的安全检查：确保数值列没有任何 NaN 或 inf 值
        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].replace([np.inf, -np.inf, np.nan], None)

        return result_df

    def __fetch_profit_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的利润数据"""
        profit_data = self._retry_operation(
            self.data_source.get_profit_data,
            code=code,
            year=year,
            quarter=quarter
        )
        profit_data = self.__format_financial_df(profit_data, [
            'roeAvg', 'npMargin', 'gpMargin', 'netProfit',
            'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare'
        ])
        return profit_data

    def __fetch_balance_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的资产负债表数据"""
        balance_data = self._retry_operation(
            self.data_source.get_balance_data,
            code=code,
            year=year,
            quarter=quarter
        )
        balance_data = self.__format_financial_df(balance_data, [
            'currentRatio', 'quickRatio', 'cashRatio',
            'YOYLiability', 'liabilityToAsset', 'assetToEquity'
        ])
        return balance_data

    def __fetch_cashflow_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的现金流量表数据"""
        cashflow_data = self._retry_operation(
            self.data_source.get_cashflow_data,
            code=code,
            year=year,
            quarter=quarter
        )
        cashflow_data = self.__format_financial_df(cashflow_data, [
            'CAToAsset', 'NCAToAsset', 'tangibleAssetToAsset',
            'ebitToInterest', 'CFOToOR', 'CFOToNP', 'CFOToGr'
        ])
        return cashflow_data

    def __fetch_growth_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的成长数据"""
        growth_data = self._retry_operation(
            self.data_source.get_growth_data,
            code=code,
            year=year,
            quarter=quarter
        )
        growth_data = self.__format_financial_df(growth_data, [
            'YOYEquity', 'YOYAsset', 'YOYNI',
            'YOYEPSBasic', 'YOYPNI'
        ])
        return growth_data

    def __fetch_operation_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的经营数据"""
        operation_data = self._retry_operation(
            self.data_source.get_operation_data,
            code=code,
            year=year,
            quarter=quarter
        )
        operation_data = self.__format_financial_df(operation_data, [
            'NRTurnRatio', 'NRTurnDays', 'INVTurnRatio',
            'INVTurnDays', 'CATurnRatio', 'AssetTurnRatio'
        ])
        return operation_data

    def __fetch_dupont_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的杜邦分析数据"""
        dupont_data = self._retry_operation(
            self.data_source.get_dupont_data,
            code=code,
            year=year,
            quarter=quarter
        )
        dupont_data = self.__format_financial_df(dupont_data, [
            'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn',
            'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden',
            'dupontIntburden', 'dupontEbittogr'
        ])
        return dupont_data

    def __fetch_dividend_data(self, code: str, year: int, quarter=None) -> pd.DataFrame:
        """获取单个股票的分红数据"""
        dividend_data = self._retry_operation(
            self.data_source.get_dividend_data,
            code=code,
            year=year,
        )
        dividend_data = self.__format_financial_df(dividend_data, [
            'dividCashPsBeforeTax', 'dividCashPsAfterTax',
            'dividStocksPs', 'dividCashStock', 'dividReserveToStockPs'
        ])
        return dividend_data
