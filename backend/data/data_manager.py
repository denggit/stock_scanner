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
import multiprocessing
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.data.database import DatabaseManager
from backend.data_source.baostock_source import BaostockSource
from backend.data_source.base import DataSource


class DataUpdateManager:
    """数据更新管理器，负责管理和协调所有数据的更新操作
    
    该类提供了以下主要功能：
    1. 股票日线数据的更新
    2. 财务数据的更新（包括利润表、资产负债表等）
    3. 数据更新进度追踪
    4. 错误重试和异常处理
    
    Attributes:
        db (DatabaseManager): 数据库管理器实例
        data_source (DataSource): 数据源实例
        data_queue (queue.Queue): 数据更新队列
        producer_done (threading.Event): 生产者完成标志
        stats_lock (threading.Lock): 统计信息线程锁
    """
    DEFAULT_CALENDAR_START: str = "2015-01-01"
    MAX_QUEUE_SIZE: int = 100
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    QUEUE_TIMEOUT: int = 5

    def __init__(self, data_source: Optional[DataSource] = None):
        """初始化数据更新管理器
        
        Args:
            data_source (Optional[DataSource]): 数据源，默认使用 BaostockSource
        """
        self.db = DatabaseManager()
        self.data_source = data_source or BaostockSource()
        self._init_connection()

        # 下面这些是给更新股票行情数据用的，多线程
        self.data_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)  # 限制队列大小以控制内存使用
        self.producer_done = threading.Event()
        self.stats_lock = threading.Lock()  # 添加线程锁用于保护统计信息

    def _init_connection(self):
        if not self.data_source.connect():
            raise Exception("Failed to connect to data source")

    @staticmethod
    def _retry_operation(operation, max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY, **kwargs):
        for i in range(max_retries):
            try:
                return operation(**kwargs)
            except Exception as e:
                logging.warning(
                    f"Operation {operation.__name__} with params {kwargs} failed: {e}. "
                    f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        raise Exception("Operation failed after multiple retries")

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self._retry_operation(self.data_source.get_stock_list)

    def update_stock_list(self, stock_list: pd.DataFrame):
        """更新股票列表"""
        self._retry_operation(self.db.save_stock_basic, stock_basic=stock_list)

    def update_all_stocks(self, force_full_update: bool = False, progress_callback=None):
        """更新所有股票数据

        Args:
            force_full_update (bool): 是否强制全量更新
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
        latest_dates = {} if force_full_update else self._retry_operation(self.db.get_all_update_time, adjust='3')
        latest_dates_back = {} if force_full_update else self._retry_operation(self.db.get_all_update_time, adjust='1')

        # 启动消费者线程
        consumer = threading.Thread(
            target=self._consume_stock_data,
            args=(update_stats, progress_callback)
        )
        consumer.start()

        trading_calendar = self.data_source.get_trading_calendar(start_date=self.DEFAULT_CALENDAR_START)
        # 生产者：获取股票数据
        try:
            for code in stock_list['code']:
                try:
                    # 获取不复权数据
                    df = self.__fetch_stock_data(code, trading_calendar, force_full_update, latest_dates.get(code),
                                                 adjust='3')
                    if not df.empty:
                        self.data_queue.put((code, df, '3'))
                    elif progress_callback:
                        progress_callback()
                    # 获取后复权数据
                    df_back = self.__fetch_stock_data(code, trading_calendar, force_full_update,
                                                      latest_dates_back.get(code),
                                                      adjust='1')
                    if not df_back.empty:
                        self.data_queue.put((code, df_back, '1'))
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
                code, df, adjust = self.data_queue.get(timeout=self.QUEUE_TIMEOUT)
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

    def __fetch_stock_data(self, code: str, trading_calendar: pd.DataFrame, force_full_update: bool = False,
                           latest_date: Optional[str] = None, adjust: str = '3') -> pd.DataFrame:
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
            start_date = '2025-01-01'
            for i in range(days, len(trading_calendar) + 1):
                if trading_calendar.is_trading_day.iloc[-i] == "1":
                    start_date = trading_calendar.calendar_date.iloc[-i]
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

    def update_all_financial_data(self, start_year: int, end_year: int = None) -> Dict[str, Dict[str, Any]]:
        """更新所有股票的财务数据

        Args:
            start_year: 开始年份
            end_year: 结束年份（默认为当前年份）
        """
        if end_year is None or end_year > dt.date.today().year:
            end_year = dt.date.today().year

        stock_list = self.db.get_stock_list()
        total_stocks = len(stock_list)
        years_count = end_year - start_year + 1
        # 每一年一份数据，7个财务表
        total_updates = total_stocks * years_count * 7

        # 创建共享队列用于进度更新
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        update_queue = manager.Queue()
        progress_done = threading.Event()

        # 启动进度监听线程
        progress_thread = threading.Thread(
            target=self._handle_progress_updates,
            args=(progress_queue, total_updates, progress_done)
        )
        progress_thread.start()

        results = {}

        # 启动更新财务数据时间的进程
        update_time_process = multiprocessing.Process(
            target=self._update_financial_update_time,
            args=(update_queue,)
        )
        update_time_process.start()

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
                    progress_queue=progress_queue,  # 传递共享队列
                    update_queue=update_queue  # 传递共享队列
                )
                futures.append((future, data_type))

            # 等待所有任务完成
            for future, data_type in futures:
                try:
                    results[data_type] = future.result()  # 等待任务完成
                    logging.info(f"{data_type}数据更新完成")
                except Exception as e:
                    logging.exception(f"更新{data_type}数据时发生错误: {e}")

        # 任务已完成
        # 终止更新时间的进程
        update_time_process.terminate()

        # 等待进度监听线程完成
        progress_done.set()
        progress_thread.join()

        for data_type, result in results.items():
            logging.info(f"{data_type} 数据更新结果：")
            for key, value in result.items():
                logging.info(f"{key}: {value}")

        return results

    @staticmethod
    def _handle_progress_updates(progress_queue: multiprocessing.Queue, total: int, progress_done: threading.Event):
        """处理进度更新的线程函数"""
        with tqdm(total=total, desc="更新财务数据") as pbar:
            while not (progress_done.is_set() and progress_queue.empty()):
                try:
                    # 从队列获取更新信号
                    msg = progress_queue.get(timeout=5)
                    if msg == "COMPLETE":
                        # 但这里目前是不会运行到的，因为有7个不同的进程同时发送queue过来，等progress_done.is_set()再退出
                        break
                    pbar.update(1)
                except queue.Empty:
                    continue
            pbar.close()

    @staticmethod
    def __fetch_all_financial_data(data_source: DataSource, stock_list: pd.DataFrame, data_type: str, fetcher,
                                   start_year: int, end_year: int,
                                   data_queue: queue.Queue, update_stats: dict,
                                   update_queue: multiprocessing.Queue = None,
                                   progress_queue: multiprocessing.Queue = None):
        """财务数据生产者：通过fetcher来获取每一只股票的财务数据"""
        current_year = dt.date.today().year
        for _, row in stock_list.iterrows():
            code, updated_year = row.code, row.get(f"update_time_{data_type}")
            try:
                for year in range(start_year, end_year + 1):
                    if updated_year is not None and year < updated_year:
                        # 如果已经这一年已经更新过也可以覆盖更新，避免遗漏季度，但过去的年份无需重新更新
                        logging.info(f"该数据过去已更新，无需重复更新：{code}_{year}_{data_type}")
                        if progress_queue:
                            progress_queue.put(1)
                        continue
                    profit_data = fetcher(data_source, code, year)
                    if not profit_data.empty:
                        data_queue.put((code, profit_data, year))
                        update_stats["updated"] += 1
                    else:
                        logging.warning(f"该数据为空: {code}_{year}_{data_type}")
                        # 尽管数据为空也已经更新过了
                        if year < current_year:
                            update_queue.put((code, year, data_type))
                        else:
                            # 避免前一年的季度数据还未更新
                            update_queue.put((code, year - 1, data_type))
                    if progress_queue:
                        progress_queue.put(1)
            except Exception as e:
                logging.exception(f"获取股票 {code} 利润表数据失败: {e}")
                update_stats["failed"] += 1
                update_stats["failed_codes"].append(code)
                if progress_queue:
                    progress_queue.put(1)

    @staticmethod
    def __consume_financial_data(producer_done: threading.Event, data_queue: queue.Queue,
                                 data_saver: Callable, data_type: str,
                                 update_stats: dict, progress_queue: multiprocessing.Queue = None,
                                 update_queue: multiprocessing.Queue = None):
        """财务数据消费者：消费对应data_queue里的财务数据

        Args:
            producer_done (threading.Event): 生产者完成标志
            data_queue (queue.Queue): 财务数据队列
            data_saver (callable): 数据保存方法
            data_type (str): 数据类型（profit/balance/cashflow等）
            update_stats (dict): 更新统计信息
            progress_queue (Queue): 共享队列，用于更新财务数据的时间
            update_queue (Queue): 共享队列，用于更新财务数据的时间
        """
        while not (producer_done.is_set() and data_queue.empty()):
            try:
                code, data, year = data_queue.get(timeout=5)
                try:
                    data_saver(data)
                    update_stats["updated"] += 1
                    logging.info(f"成功保存 {code} {year}年 {data_type} 数据")
                    # 将更新信息放入共享队列
                    update_queue.put((code, year, data_type))
                except Exception as e:
                    logging.exception(f"保存数据失败：{e}")
                    update_stats["failed"] += 1
                    update_stats["failed_codes"].append(code)
                finally:
                    data_queue.task_done()
                    if progress_queue:
                        progress_queue.put(1)
            except queue.Empty:
                logging.info(f"{data_type}数据队列为空，等待数据...")
                continue

    @staticmethod
    def _update_financial_update_time(update_queue: multiprocessing.Queue):
        """进程：更新财务数据的更新时间

        Args:
            update_queue (Queue): 共享队列，包含需要更新的数据
        """
        # 在子进程中创建新的数据库连接
        db = DatabaseManager()

        try:
            while True:
                try:
                    # 从队列中获取数据
                    code, year, data_type = update_queue.get(timeout=5)
                    # 调用数据库方法更新数据
                    db.update_financial_update_time(
                        pd.DataFrame([{"code": code, "year": year}]),
                        data_type=data_type
                    )
                except queue.Empty:
                    if not multiprocessing.parent_process().is_alive():
                        logging.info("父进程已终止，更新时间进程退出")
                        break
                    logging.debug("财务更新时间队列为空，等待数据...")
                    continue
                except (ValueError, KeyError) as e:
                    logging.error(f"数据格式错误: {e}")
                except pd.errors.EmptyDataError:
                    logging.error("更新数据为空")
        except Exception as e:
            logging.exception(f"更新财务数据时间时发生错误: {e}")
        finally:
            # 确保关闭数据库连接
            db.close()

    @staticmethod
    def _update_profit_data(stock_list: pd.DataFrame, start_year: int, end_year: int,
                            update_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue = None):
        """更新单个股票的利润数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }
        db = DatabaseManager()

        # 启动消费者线程
        consumer = threading.Thread(
            target=DataUpdateManager.__consume_financial_data,
            kwargs={
                'producer_done': producer_done,
                'data_queue': data_queue,
                'data_saver': db.save_profit_data,
                'data_type': "profit",
                'update_stats': update_stats,
                'progress_queue': progress_queue,
                'update_queue': update_queue
            }
        )
        consumer.start()

        try:
            data_source = BaostockSource()
            DataUpdateManager.__fetch_all_financial_data(
                data_source=data_source,
                stock_list=stock_list,
                data_type="profit",
                fetcher=DataUpdateManager.__fetch_profit_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                update_queue=update_queue,
                progress_queue=progress_queue
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    @staticmethod
    def _update_balance_data(stock_list: pd.DataFrame, start_year: int, end_year: int,
                             update_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue = None):
        """更新单个股票的资产负债表数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }
        db = DatabaseManager()

        # 启动消费者线程
        consumer = threading.Thread(
            target=DataUpdateManager.__consume_financial_data,
            kwargs={
                'producer_done': producer_done,
                'data_queue': data_queue,
                'data_saver': db.save_balance_data,
                'data_type': "balance",
                'update_stats': update_stats,
                'progress_queue': progress_queue,
                'update_queue': update_queue
            }
        )
        consumer.start()

        try:
            data_source = BaostockSource()
            DataUpdateManager.__fetch_all_financial_data(
                data_source=data_source,
                stock_list=stock_list,
                data_type="balance",
                fetcher=DataUpdateManager.__fetch_balance_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                update_queue=update_queue,
                progress_queue=progress_queue
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    @staticmethod
    def _update_cashflow_data(stock_list: pd.DataFrame, start_year: int, end_year: int,
                              update_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue = None):
        """更新单个股票的现金流量表数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }
        db = DatabaseManager()

        # 启动消费者线程
        consumer = threading.Thread(
            target=DataUpdateManager.__consume_financial_data,
            kwargs={
                'producer_done': producer_done,
                'data_queue': data_queue,
                'data_saver': db.save_cashflow_data,
                'data_type': "cashflow",
                'update_stats': update_stats,
                'progress_queue': progress_queue,
                'update_queue': update_queue
            }
        )
        consumer.start()

        try:
            data_source = BaostockSource()
            DataUpdateManager.__fetch_all_financial_data(
                data_source=data_source,
                stock_list=stock_list,
                data_type="cashflow",
                fetcher=DataUpdateManager.__fetch_cashflow_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                update_queue=update_queue,
                progress_queue=progress_queue
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    @staticmethod
    def _update_growth_data(stock_list: pd.DataFrame, start_year: int, end_year: int,
                            update_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue = None):
        """更新单个股票的成长数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }
        db = DatabaseManager()

        # 启动消费者线程
        consumer = threading.Thread(
            target=DataUpdateManager.__consume_financial_data,
            kwargs={
                'producer_done': producer_done,
                'data_queue': data_queue,
                'data_saver': db.save_growth_data,
                'data_type': 'growth',
                'update_stats': update_stats,
                'progress_queue': progress_queue,
                'update_queue': update_queue
            }
        )
        consumer.start()

        try:
            data_source = BaostockSource()
            DataUpdateManager.__fetch_all_financial_data(
                data_source=data_source,
                stock_list=stock_list,
                data_type="growth",
                fetcher=DataUpdateManager.__fetch_growth_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                update_queue=update_queue,
                progress_queue=progress_queue
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    @staticmethod
    def _update_operation_data(stock_list: pd.DataFrame, start_year: int, end_year: int,
                               update_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue = None):
        """更新单个股票的经营数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }
        db = DatabaseManager()

        # 启动消费者线程
        consumer = threading.Thread(
            target=DataUpdateManager.__consume_financial_data,
            kwargs={
                'producer_done': producer_done,
                'data_queue': data_queue,
                'data_saver': db.save_operation_data,
                'data_type': 'operation',
                'update_stats': update_stats,
                'progress_queue': progress_queue,
                'update_queue': update_queue
            }
        )
        consumer.start()

        try:
            data_source = BaostockSource()
            DataUpdateManager.__fetch_all_financial_data(
                data_source=data_source,
                stock_list=stock_list,
                data_type="operation",
                fetcher=DataUpdateManager.__fetch_operation_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                update_queue=update_queue,
                progress_queue=progress_queue
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    @staticmethod
    def _update_dupont_data(stock_list: pd.DataFrame, start_year: int, end_year: int,
                            update_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue = None):
        """更新单个股票的杜邦分析数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }
        db = DatabaseManager()

        # 启动消费者线程
        consumer = threading.Thread(
            target=DataUpdateManager.__consume_financial_data,
            kwargs={
                'producer_done': producer_done,
                'data_queue': data_queue,
                'data_saver': db.save_dupont_data,
                'data_type': 'dupont',
                'update_stats': update_stats,
                'progress_queue': progress_queue,
                'update_queue': update_queue
            }
        )
        consumer.start()

        try:
            data_source = BaostockSource()
            DataUpdateManager.__fetch_all_financial_data(
                data_source=data_source,
                stock_list=stock_list,
                data_type="dupont",
                fetcher=DataUpdateManager.__fetch_dupont_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                update_queue=update_queue,
                progress_queue=progress_queue
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

    @staticmethod
    def _update_dividend_data(stock_list: pd.DataFrame, start_year: int, end_year: int,
                              update_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue = None):
        """更新单个股票的分红数据"""
        producer_done = threading.Event()
        data_queue = queue.Queue(maxsize=100)
        update_stats = {
            "updated": 0,
            "failed": 0,
            "failed_codes": []
        }
        db = DatabaseManager()

        # 启动消费者线程
        consumer = threading.Thread(
            target=DataUpdateManager.__consume_financial_data,
            kwargs={
                'producer_done': producer_done,
                'data_queue': data_queue,
                'data_saver': db.save_dividend_data,
                'data_type': 'dividend',
                'update_stats': update_stats,
                'progress_queue': progress_queue,
                'update_queue': update_queue
            }
        )
        consumer.start()

        try:
            data_source = BaostockSource()
            DataUpdateManager.__fetch_all_financial_data(
                data_source=data_source,
                stock_list=stock_list,
                data_type="dividend",
                fetcher=DataUpdateManager.__fetch_dividend_data,
                start_year=start_year,
                end_year=end_year,
                data_queue=data_queue,
                update_stats=update_stats,
                update_queue=update_queue,
                progress_queue=progress_queue
            )
        finally:
            # 标记生产者完成
            producer_done.set()

        # 等待消费者处理完所有数据
        consumer.join()

        return update_stats

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

    @staticmethod
    def __fetch_profit_data(data_source, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的利润数据"""
        profit_data = DataUpdateManager._retry_operation(
            data_source.get_profit_data,
            code=code,
            year=year,
            quarter=quarter
        )
        profit_data = DataUpdateManager.__format_financial_df(profit_data, [
            'roeAvg', 'npMargin', 'gpMargin', 'netProfit',
            'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare'
        ])
        return profit_data

    @staticmethod
    def __fetch_balance_data(data_source, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的资产负债表数据"""
        balance_data = DataUpdateManager._retry_operation(
            data_source.get_balance_data,
            code=code,
            year=year,
            quarter=quarter
        )
        balance_data = DataUpdateManager.__format_financial_df(balance_data, [
            'currentRatio', 'quickRatio', 'cashRatio',
            'YOYLiability', 'liabilityToAsset', 'assetToEquity'
        ])
        return balance_data

    @staticmethod
    def __fetch_cashflow_data(data_source, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的现金流量表数据"""
        cashflow_data = DataUpdateManager._retry_operation(
            data_source.get_cashflow_data,
            code=code,
            year=year,
            quarter=quarter
        )
        cashflow_data = DataUpdateManager.__format_financial_df(cashflow_data, [
            'CAToAsset', 'NCAToAsset', 'tangibleAssetToAsset',
            'ebitToInterest', 'CFOToOR', 'CFOToNP', 'CFOToGr'
        ])
        return cashflow_data

    @staticmethod
    def __fetch_growth_data(data_source, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的成长数据"""
        growth_data = DataUpdateManager._retry_operation(
            data_source.get_growth_data,
            code=code,
            year=year,
            quarter=quarter
        )
        growth_data = DataUpdateManager.__format_financial_df(growth_data, [
            'YOYEquity', 'YOYAsset', 'YOYNI',
            'YOYEPSBasic', 'YOYPNI'
        ])
        return growth_data

    @staticmethod
    def __fetch_operation_data(data_source, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的经营数据"""
        operation_data = DataUpdateManager._retry_operation(
            data_source.get_operation_data,
            code=code,
            year=year,
            quarter=quarter
        )
        operation_data = DataUpdateManager.__format_financial_df(operation_data, [
            'NRTurnRatio', 'NRTurnDays', 'INVTurnRatio',
            'INVTurnDays', 'CATurnRatio', 'AssetTurnRatio'
        ])
        return operation_data

    @staticmethod
    def __fetch_dupont_data(data_source, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取单个股票的杜邦分析数据"""
        dupont_data = DataUpdateManager._retry_operation(
            data_source.get_dupont_data,
            code=code,
            year=year,
            quarter=quarter
        )
        dupont_data = DataUpdateManager.__format_financial_df(dupont_data, [
            'dupontROE', 'dupontAssetStoEquity', 'dupontAssetTurn',
            'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden',
            'dupontIntburden', 'dupontEbittogr'
        ])
        return dupont_data

    @staticmethod
    def __fetch_dividend_data(data_source, code: str, year: int, quarter=None) -> pd.DataFrame:
        """获取单个股票的分红数据"""
        dividend_data = DataUpdateManager._retry_operation(
            data_source.get_dividend_data,
            code=code,
            year=year,
        )
        dividend_data = DataUpdateManager.__format_financial_df(dividend_data, [
            'dividCashPsBeforeTax', 'dividCashPsAfterTax',
            'dividStocksPs', 'dividCashStock', 'dividReserveToStockPs'
        ])
        return dividend_data
