#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 6:33 PM
@File       : data_fetcher.py
@Description: 
"""
import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from backend.business.data.data_manager import DatabaseManager
from backend.business.data.source.baostock_src import BaostockSource


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

        # 数据清理：处理无穷大和NaN值
        for column in numeric_columns:
            if column in df.columns:
                df[column] = df[column].astype(float)
                # 替换无穷大值为NaN
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)

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
                                 end_date: Optional[datetime.date] = datetime.date.today()) -> pd.DataFrame:
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

    def fetch_financial_data(
            self,
            code: str,
            data_type: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取财务数据
        
        Args:
            code: 股票代码
            data_type: 财务数据类型，可选值：
                - 'profit': 利润表数据 (roeAvg, npMargin, gpMargin, epsTTM等)
                - 'balance': 资产负债表数据 (currentRatio, liabilityToAsset等)
                - 'cashflow': 现金流量表数据 (CFOToOR, CFOToNP等)
                - 'growth': 成长能力数据 (YOYAsset, YOYNI等)
                - 'operation': 营运能力数据 (AssetTurnRatio, INVTurnRatio等)
                - 'dupont': 杜邦分析数据
                - 'dividend': 分红数据
            start_date: 开始日期，格式为YYYY-MM-DD，可选
            end_date: 结束日期，格式为YYYY-MM-DD，可选
            
        Returns:
            包含财务数据的DataFrame
        """
        # 首先尝试从数据库获取
        df = self.db.get_financial_data(code, data_type, start_date, end_date)
        
        if df.empty:
            logging.warning(f"No {data_type} data found for code: {code} in database, trying alternative sources...")
            
            # 如果数据库中没有数据，尝试从其他数据源获取
            try:
                # 尝试从数据源直接获取（绕过数据库）
                from backend.business.data.source.baostock_src import BaostockSource
                
                # 创建BaostockSource实例
                baostock_source = BaostockSource()
                
                # 根据数据类型调用相应的数据源方法
                if data_type == 'profit':
                    # 尝试获取最近几年的利润表数据
                    current_year = datetime.date.today().year
                    for year in range(current_year, current_year - 3, -1):
                        try:
                            df = baostock_source.get_profit_data(code, year)
                            if not df.empty:
                                logging.info(f"Successfully retrieved {data_type} data for {code} from Baostock for year {year}")
                                break
                        except Exception as e:
                            logging.debug(f"Failed to get {data_type} data for {code} from Baostock for year {year}: {e}")
                            continue
                            
                elif data_type == 'balance':
                    current_year = datetime.date.today().year
                    for year in range(current_year, current_year - 3, -1):
                        try:
                            df = baostock_source.get_balance_data(code, year)
                            if not df.empty:
                                logging.info(f"Successfully retrieved {data_type} data for {code} from Baostock for year {year}")
                                break
                        except Exception as e:
                            logging.debug(f"Failed to get {data_type} data for {code} from Baostock for year {year}: {e}")
                            continue
                            
                elif data_type == 'cashflow':
                    current_year = datetime.date.today().year
                    for year in range(current_year, current_year - 3, -1):
                        try:
                            df = baostock_source.get_cashflow_data(code, year)
                            if not df.empty:
                                logging.info(f"Successfully retrieved {data_type} data for {code} from Baostock for year {year}")
                                break
                        except Exception as e:
                            logging.debug(f"Failed to get {data_type} data for {code} from Baostock for year {year}: {e}")
                            continue
                            
                elif data_type == 'growth':
                    current_year = datetime.date.today().year
                    for year in range(current_year, current_year - 3, -1):
                        try:
                            df = baostock_source.get_growth_data(code, year)
                            if not df.empty:
                                logging.info(f"Successfully retrieved {data_type} data for {code} from Baostock for year {year}")
                                break
                        except Exception as e:
                            logging.debug(f"Failed to get {data_type} data for {code} from Baostock for year {year}: {e}")
                            continue
                            
                elif data_type == 'operation':
                    current_year = datetime.date.today().year
                    for year in range(current_year, current_year - 3, -1):
                        try:
                            df = baostock_source.get_operation_data(code, year)
                            if not df.empty:
                                logging.info(f"Successfully retrieved {data_type} data for {code} from Baostock for year {year}")
                                break
                        except Exception as e:
                            logging.debug(f"Failed to get {data_type} data for {code} from Baostock for year {year}: {e}")
                            continue
                            
                elif data_type == 'dupont':
                    current_year = datetime.date.today().year
                    for year in range(current_year, current_year - 3, -1):
                        try:
                            df = baostock_source.get_dupont_data(code, year)
                            if not df.empty:
                                logging.info(f"Successfully retrieved {data_type} data for {code} from Baostock for year {year}")
                                break
                        except Exception as e:
                            logging.debug(f"Failed to get {data_type} data for {code} from Baostock for year {year}: {e}")
                            continue
                            
                elif data_type == 'dividend':
                    current_year = datetime.date.today().year
                    for year in range(current_year, current_year - 3, -1):
                        try:
                            df = baostock_source.get_dividend_data(code, year)
                            if not df.empty:
                                # 分红数据使用dividOperateDate而不是statDate
                                if 'dividOperateDate' in df.columns:
                                    # 重命名列以保持一致性
                                    df = df.rename(columns={'dividOperateDate': 'statDate'})
                                logging.info(f"Successfully retrieved {data_type} data for {code} from Baostock for year {year}")
                                break
                        except Exception as e:
                            logging.debug(f"Failed to get {data_type} data for {code} from Baostock for year {year}: {e}")
                            continue
                
                if df.empty:
                    logging.warning(f"No {data_type} data found for code: {code} from any source")
                    
            except Exception as e:
                logging.error(f"Failed to get {data_type} data for {code} from alternative source: {e}")
        
        # 数据清理：处理无穷大和NaN值
        if not df.empty:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                df[column] = df[column].astype(float)
                # 替换无穷大值为NaN
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        
        return df

    def fetch_profit_data(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取利润表数据
        
        Returns:
            包含以下字段的DataFrame:
            - roeAvg: 净资产收益率
            - npMargin: 净利润率
            - gpMargin: 毛利率
            - epsTTM: 每股收益(TTM)
            - netProfit: 净利润
            - MBRevenue: 主营收入
        """
        return self.fetch_financial_data(code, 'profit', start_date, end_date)

    def fetch_balance_data(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取资产负债表数据
        
        Returns:
            包含以下字段的DataFrame:
            - currentRatio: 流动比率
            - liabilityToAsset: 资产负债率
            - assetToEquity: 权益乘数
            - quickRatio: 速动比率
            - cashRatio: 现金比率
        """
        return self.fetch_financial_data(code, 'balance', start_date, end_date)

    def fetch_cashflow_data(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取现金流量表数据
        
        Returns:
            包含以下字段的DataFrame:
            - CFOToOR: 经营现金流/营业收入
            - CFOToNP: 经营现金流/净利润
            - CFOToGr: 经营现金流/毛利润
        """
        return self.fetch_financial_data(code, 'cashflow', start_date, end_date)

    def fetch_growth_data(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取成长能力数据
        
        Returns:
            包含以下字段的DataFrame:
            - YOYAsset: 资产同比增长率
            - YOYNI: 净利润同比增长率
            - YOYEquity: 净资产同比增长率
            - YOYEPSBasic: 每股收益同比增长率
        """
        return self.fetch_financial_data(code, 'growth', start_date, end_date)

    def fetch_operation_data(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取营运能力数据
        
        Returns:
            包含以下字段的DataFrame:
            - AssetTurnRatio: 总资产周转率
            - INVTurnRatio: 存货周转率
            - NRTurnRatio: 应收账款周转率
            - CATurnRatio: 流动资产周转率
        """
        return self.fetch_financial_data(code, 'operation', start_date, end_date)

    def fetch_dupont_data(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取杜邦分析数据
        
        Returns:
            包含以下字段的DataFrame:
            - dupontROE: 杜邦ROE
            - dupontAssetStoEquity: 权益乘数
            - dupontAssetTurn: 总资产周转率
            - dupontPnitoni: 净利润/营业利润
        """
        return self.fetch_financial_data(code, 'dupont', start_date, end_date)

    def fetch_dividend_data(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取分红数据
        
        Returns:
            包含以下字段的DataFrame:
            - dividCashPsBeforeTax: 税前每股现金分红
            - dividCashPsAfterTax: 税后每股现金分红
            - dividStocksPs: 每股股票分红
        """
        return self.fetch_financial_data(code, 'dividend', start_date, end_date)

    def fetch_all_financial_data(
            self,
            code: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> dict:
        """
        获取所有财务数据
        
        Args:
            code: 股票代码
            start_date: 开始日期，格式为YYYY-MM-DD，可选
            end_date: 结束日期，格式为YYYY-MM-DD，可选
            
        Returns:
            包含所有财务数据的字典，键为数据类型，值为对应的DataFrame
        """
        data_types = ['profit', 'balance', 'cashflow', 'growth', 'operation', 'dupont', 'dividend']
        result = {}
        
        for data_type in data_types:
            try:
                df = self.fetch_financial_data(code, data_type, start_date, end_date)
                result[data_type] = df
            except Exception as e:
                logging.warning(f"Failed to fetch {data_type} data for {code}: {e}")
                result[data_type] = pd.DataFrame()
        
        return result
