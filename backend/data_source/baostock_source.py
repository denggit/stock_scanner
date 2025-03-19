#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 1:45 AM
@File       : baostock_source.py
@Description: 
"""
import logging
import re
import time

import baostock as bs
import pandas as pd

from .base import DataSource


def ensure_connection(func):
    """确保在执行方法前已经建立连接的装饰器
    
    Args:
        func: 需要确保连接的方法
        
    Returns:
        wrapper: 包装后的方法
    """

    def wrapper(self, *args, **kwargs):
        if not self._connected:
            self.connect()
        return func(self, *args, **kwargs)

    return wrapper


class BaostockSource(DataSource):
    # 定义错误码
    NETWORK_ERROR_CODES = ['10002007', '10002009', '10002010']
    LOGIN_ERROR_CODES = ['10001001', '10001002', '10001003']
    SUCCESS_CODE = '0'

    def __init__(self):
        self._connected = False

    def connect(self):
        for i in range(3):
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
                return self._connected
            except Exception as e:
                logging.error(f"Baostock login failed: {e}")
                self._connected = False
                if i == 2:
                    raise Exception(f"Baostock login failed: {e}, 尝试{i + 1}次失败，退出")

    def disconnect(self):
        try:
            bs.logout()
            self._connected = False
            logging.info("Baostock logout successful")
        except Exception as e:
            logging.warning(f"Baostock logout failed: {e}")

    def is_connected(self) -> bool:
        return self._connected

    @ensure_connection
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表（仅上市的A股）"""
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

    @ensure_connection
    def get_stock_data(self, code: str, start_date: str, end_date: str, frequency: str = 'daily',
                       adjust: str = '3') -> pd.DataFrame:
        """获取股票数据
        
        :param adjust: 复权类型，1:后复权，2:前复权，3:不复权

        RETURN:
            date	交易所行情日期
            code	证券代码
            open	开盘价
            high	最高价
            low	最低价
            close	收盘价
            preclose	前收盘价	见表格下方详细说明
            volume	成交量（累计 单位：股）
            amount	成交额（单位：人民币元）
            adjustflag	复权状态(1：后复权， 2：前复权，3：不复权）
            turn	换手率	[指定交易日的成交量(股)/指定交易日的股票的流通股总股数(股)]*100%
            tradestatus	交易状态(1：正常交易 0：停牌）
            pctChg	涨跌幅（百分比）	日涨跌幅=[(指定交易日的收盘价-指定交易日前收盘价)/指定交易日前收盘价]*100%
            peTTM	滚动市盈率	(指定交易日的股票收盘价/指定交易日的每股盈余TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/归属母公司股东净利润TTM
            pbMRQ	市净率	(指定交易日的股票收盘价/指定交易日的每股净资产)=总市值/(最近披露的归属母公司股东的权益-其他权益工具)
            psTTM	滚动市销率	(指定交易日的股票收盘价/指定交易日的每股销售额)=(指定交易日的股票收盘价*截至当日公司总股本)/营业总收入TTM
            pcfNcfTTM	滚动市现率	(指定交易日的股票收盘价/指定交易日的每股现金流TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/现金以及现金等价物净增加额TTM
            isST	是否ST股，1是，0否
        """
        if frequency in ['daily', 'weekly', 'monthly', 'd', 'w', 'm']:
            fields = 'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST'
            rs = bs.query_history_k_data_plus(code, fields=fields, start_date=start_date, end_date=end_date,
                                              frequency=frequency[0], adjustflag=adjust)
            columns = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn',
                       'tradestatus', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'isST']
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
            # 确保所有必须的列都存在
            required_columns = ['code', 'trade_date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                                'turn', 'tradestatus', 'pct_chg', 'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm', 'is_st']
            numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pct_chg',
                               'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm']
        else:
            # 分钟线
            match = re.match(r'^\d+', frequency)
            if match:
                frequency = match.group()
            else:
                raise ValueError(f"无效的频率参数: {frequency}")
            fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
            rs = bs.query_history_k_data_plus(code, fields=fields, start_date=start_date, end_date=end_date,
                                              frequency=frequency, adjustflag=adjust)
            columns = ['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adjustflag']
            # 重命名列以匹配数据库字段
            column_mapping = {
                'date': 'trade_date',
            }
            # 确保所有必须的列都存在
            required_columns = ['code', 'trade_date', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']

        if rs.error_code != self.SUCCESS_CODE:
            if rs.error_code in self.NETWORK_ERROR_CODES + self.LOGIN_ERROR_CODES:
                self.connect()
                return self.get_stock_data(code, start_date, end_date, frequency, adjust)
            else:
                raise Exception(f"Failed to get stock data: {rs.error_msg}")

        data_list = []
        while (rs.error_code == self.SUCCESS_CODE) & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        # 创建DataFrame
        df = pd.DataFrame(data_list, columns=columns)

        # 重命名列
        df = df.rename(columns=column_mapping)

        # 转换数据类型
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        if frequency in ['daily', 'weekly', 'monthly', 'd', 'w', 'm']:
            df['tradestatus'] = df['tradestatus'].astype(int)
            df['is_st'] = df['is_st'].astype(int)
        else:
            # 分钟级别数据
            df['time'] = df['time'].str[8:14].apply(lambda x: f"{x[0:2]}:{x[2:4]}:{x[4:6]}")

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据源返回的数据缺少必须的列：{missing_columns}")

        return df[required_columns]  # 只返回必须的列，并按照固定顺去配置

    @ensure_connection
    def get_stock_industry(self, code: str):
        """"""
        df = bs.query_stock_industry(code=code).get_data()


    @ensure_connection
    def get_sz50(self):
        """获取上证50股票"""
        return bs.query_sz50_stocks().get_data()

    @ensure_connection
    def get_hs300(self):
        """获取沪深300股票"""
        return bs.query_hs300_stocks().get_data()

    @ensure_connection
    def get_zz500(self):
        """获取中证500股票"""
        return bs.query_zz500_stocks().get_data()

    @ensure_connection
    def get_profit_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取上市公司利润表数据
        
        返回值：
            code	证券代码	
            pubDate	公司发布财报的日期	
            statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30	
            roeAvg	净资产收益率(平均)(%)	归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
            npMargin	销售净利率(%)	净利润/营业收入*100%
            gpMargin	销售毛利率(%)	毛利/营业收入*100%=(营业收入-营业成本)/营业收入*100%
            netProfit	净利润(元)	
            epsTTM	每股收益	归属母公司股东的净利润TTM/最新总股本
            MBRevenue	主营营业收入(元)	
            totalShare	总股本	
            liqaShare	流通股本

        """
        if quarter:
            return bs.query_profit_data(code=code, year=year, quarter=quarter).get_data()
        else:
            quarter_1 = bs.query_profit_data(code=code, year=year, quarter=1).get_data()
            quarter_2 = bs.query_profit_data(code=code, year=year, quarter=2).get_data()
            quarter_3 = bs.query_profit_data(code=code, year=year, quarter=3).get_data()
            quarter_4 = bs.query_profit_data(code=code, year=year, quarter=4).get_data()
            return pd.concat([quarter_1, quarter_2, quarter_3, quarter_4])

    @ensure_connection
    def get_balance_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取上市公司资产负债表数据
        
        RETURN：
            code	证券代码	
            pubDate	公司发布财报的日期	
            statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30	
            currentRatio	流动比率	流动资产/流动负债
            quickRatio	速动比率	(流动资产-存货净额)/流动负债
            cashRatio	现金比率	(货币资金+交易性金融资产)/流动负债
            YOYLiability	总负债同比增长率	(本期总负债-上年同期总负债)/上年同期中负债的绝对值*100%
            liabilityToAsset	资产负债率	负债总额/资产总额
            assetToEquity	权益乘数	资产总额/股东权益总额=1/(1-资产负债率)
        
        """
        if quarter:
            return bs.query_balance_data(code=code, year=year, quarter=quarter).get_data()
        else:
            quarter_1 = bs.query_balance_data(code=code, year=year, quarter=1).get_data()
            quarter_2 = bs.query_balance_data(code=code, year=year, quarter=2).get_data()
            quarter_3 = bs.query_balance_data(code=code, year=year, quarter=3).get_data()
            quarter_4 = bs.query_balance_data(code=code, year=year, quarter=4).get_data()
            return pd.concat([quarter_1, quarter_2, quarter_3, quarter_4])

    @ensure_connection
    def get_cashflow_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取上市公司现金流量表数据
        
        RETURN：
            code	证券代码	
            pubDate	公司发布财报的日期	
            statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30	
            CAToAsset	流动资产除以总资产	
            NCAToAsset	非流动资产除以总资产	
            tangibleAssetToAsset	有形资产除以总资产	
            ebitToInterest	已获利息倍数	息税前利润/利息费用
            CFOToOR	经营活动产生的现金流量净额除以营业收入	
            CFOToNP	经营性现金净流量除以净利润	
            CFOToGr	经营性现金净流量除以营业总收入	
        
        """
        if quarter:
            return bs.query_cash_flow_data(code=code, year=year, quarter=quarter).get_data()
        else:
            quarter_1 = bs.query_cash_flow_data(code=code, year=year, quarter=1).get_data()
            quarter_2 = bs.query_cash_flow_data(code=code, year=year, quarter=2).get_data()
            quarter_3 = bs.query_cash_flow_data(code=code, year=year, quarter=3).get_data()
            quarter_4 = bs.query_cash_flow_data(code=code, year=year, quarter=4).get_data()
            return pd.concat([quarter_1, quarter_2, quarter_3, quarter_4])

    @ensure_connection
    def get_dupont_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取上市公司杜邦分析数据
        
        RETURN：
            code	证券代码	
            pubDate	公司发布财报的日期	
            statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30	
            dupontROE	净资产收益率	归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
            dupontAssetStoEquity	权益乘数，反映企业财务杠杆效应强弱和财务风险	平均总资产/平均归属于母公司的股东权益
            dupontAssetTurn	总资产周转率，反映企业资产管理效率的指标	营业总收入/[(期初资产总额+期末资产总额)/2]
            dupontPnitoni	归属母公司股东的净利润/净利润，反映母公司控股子公司百分比。如果企业追加投资，扩大持股比例，则本指标会增加。	
            dupontNitogr	净利润/营业总收入，反映企业销售获利率	
            dupontTaxBurden	净利润/利润总额，反映企业税负水平，该比值高则税负较低。净利润/利润总额=1-所得税/利润总额	
            dupontIntburden	利润总额/息税前利润，反映企业利息负担，该比值高则税负较低。利润总额/息税前利润=1-利息费用/息税前利润
            dupontEbittogr	息税前利润/营业总收入，反映企业经营利润率，是企业经营获得的可供全体投资人（股东和债权人）分配的盈利占企业全部营收收入的百分比	
            
        """
        if quarter:
            return bs.query_dupont_data(code=code, year=year, quarter=quarter).get_data()
        else:
            quarter_1 = bs.query_dupont_data(code=code, year=year, quarter=1).get_data()
            quarter_2 = bs.query_dupont_data(code=code, year=year, quarter=2).get_data()
            quarter_3 = bs.query_dupont_data(code=code, year=year, quarter=3).get_data()
            quarter_4 = bs.query_dupont_data(code=code, year=year, quarter=4).get_data()
            return pd.concat([quarter_1, quarter_2, quarter_3, quarter_4])

    @ensure_connection
    def get_growth_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取上市公司成长能力数据
        
        RETURN：
            code	证券代码	
            pubDate	公司发布财报的日期	
            statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30	
            YOYEquity	净资产同比增长率	(本期净资产-上年同期净资产)/上年同期净资产的绝对值*100%
            YOYAsset	总资产同比增长率	(本期总资产-上年同期总资产)/上年同期总资产的绝对值*100%
            YOYNI	净利润同比增长率	(本期净利润-上年同期净利润)/上年同期净利润的绝对值*100%
            YOYEPSBasic	基本每股收益同比增长率	(本期基本每股收益-上年同期基本每股收益)/上年同期基本每股收益的绝对值*100%
            YOYPNI	归属母公司股东净利润同比增长率	(本期归属母公司股东净利润-上年同期归属母公司股东净利润)/上年同期归属母公司股东净利润的绝对值*100%
        
        """
        if quarter:
            return bs.query_growth_data(code=code, year=year, quarter=quarter).get_data()
        else:
            quarter_1 = bs.query_growth_data(code=code, year=year, quarter=1).get_data()
            quarter_2 = bs.query_growth_data(code=code, year=year, quarter=2).get_data()
            quarter_3 = bs.query_growth_data(code=code, year=year, quarter=3).get_data()
            quarter_4 = bs.query_growth_data(code=code, year=year, quarter=4).get_data()
            return pd.concat([quarter_1, quarter_2, quarter_3, quarter_4])

    @ensure_connection
    def get_operation_data(self, code: str, year: int, quarter: int = None) -> pd.DataFrame:
        """获取上市公司营运能力数据
        
        RETURN：
            code	证券代码	
            pubDate	公司发布财报的日期	
            statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30	
            NRTurnRatio	应收账款周转率(次)	营业收入/[(期初应收票据及应收账款净额+期末应收票据及应收账款净额)/2]
            NRTurnDays	应收账款周转天数(天)	季报天数/应收账款周转率(一季报：90天，中报：180天，三季报：270天，年报：360天)
            INVTurnRatio	存货周转率(次)	营业成本/[(期初存货净额+期末存货净额)/2]
            INVTurnDays	存货周转天数(天)	季报天数/存货周转率(一季报：90天，中报：180天，三季报：270天，年报：360天)
            CATurnRatio	流动资产周转率(次)	营业总收入/[(期初流动资产+期末流动资产)/2]
            AssetTurnRatio	总资产周转率	营业总收入/[(期初资产总额+期末资产总额)/2]
        
        """
        if quarter:
            return bs.query_operation_data(code=code, year=year, quarter=quarter).get_data()
        else:
            quarter_1 = bs.query_operation_data(code=code, year=year, quarter=1).get_data()
            quarter_2 = bs.query_operation_data(code=code, year=year, quarter=2).get_data()
            quarter_3 = bs.query_operation_data(code=code, year=year, quarter=3).get_data()
            quarter_4 = bs.query_operation_data(code=code, year=year, quarter=4).get_data()
            return pd.concat([quarter_1, quarter_2, quarter_3, quarter_4])

    @ensure_connection
    def get_dividend_data(self, code: str, year: int, year_type: str = "report") -> pd.DataFrame:
        """获取上市公司股息分红数据

        @param code: 证券代码，不可为空
        @param year: 年份，为空时默认当前年份
        @param year_type: 年份类别，默认为"report":预案公告年份，可选项"operate":除权除息年份


        RETURN：
            code	证券代码
            dividPreNoticeDate	预批露公告日
            dividAgmPumDate	股东大会公告日期
            dividPlanAnnounceDate	预案公告日
            dividPlanDate	分红实施公告日
            dividRegistDate	股权登记告日
            dividOperateDate	除权除息日期
            dividPayDate	派息日
            dividStockMarketDate	红股上市交易日
            dividCashPsBeforeTax	每股股利税前	派息比例分子(税前)/派息比例分母
            dividCashPsAfterTax	每股股利税后	派息比例分子(税后)/派息比例分母
            dividStocksPs	每股红股
            dividCashStock	分红送转	每股派息数(税前)+每股送股数+每股转增股本数
            dividReserveToStockPs	每股转增资本

        """
        return bs.query_dividend_data(code=code, year=year, yearType=year_type).get_data()

    @ensure_connection
    def get_trading_calendar(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取交易日历数据"""
        return bs.query_trade_dates(start_date=start_date, end_date=end_date).get_data()
