#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 1:45 AM
@File       : akshare_src.py
@Description: AKShare数据源封装，提供股票市场相关数据的统一访问接口
"""
import logging

import akshare as ak
import pandas as pd

from backend.utils import format_info
from .base import DataSource


class AKShareSource(DataSource):
    """AKShare数据源封装类
    
    提供对AKShare接口的统一封装，包括：
    1. 基础数据（股票列表、交易日历等）
    2. 行情数据（日线、周线、月线）
    3. 财务数据（财务报表、分红数据等）
    4. 市场数据（机构参与、综合评分等）
    5. 技术指标（创新高、突破等）
    """

    def __init__(self):
        self._connected = True  # AKShare 不需要登录，直接设置为 True

    def connect(self) -> bool:
        """AKShare 不需要连接操作"""
        return True

    def disconnect(self):
        """AKShare 不需要断开连接操作"""
        pass

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected

    @staticmethod
    def get_stock_list() -> pd.DataFrame:
        """获取A股上市公司基本信息"""
        try:
            # 获取A股上市公司基本信息
            df = ak.stock_info_a_code_name()

            # 重命名列
            df = df.rename(columns={
                'code': 'code',
                'name': 'name'
            })

            # 添加额外信息
            df['type'] = '1'  # 1表示A股
            df['status'] = '1'  # 1表示上市状态

            return df
        except Exception as e:
            logging.error(f"Failed to get stock list: {e}")
            raise

    @staticmethod
    def get_financial_data(code: str, report_type: str) -> pd.DataFrame:
        """获取财务报表数据
        
        :param code: 股票代码
        :param report_type: 报表类型：'income'利润表，'balance'资产负债表，'cash'现金流量表
        """
        try:
            if report_type == 'income':
                df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            elif report_type == 'balance':
                df = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
            elif report_type == 'cash':
                df = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")
            else:
                raise ValueError(f"Unsupported report type: {report_type}")

            return df
        except Exception as e:
            logging.error(f"Failed to get financial data for {code}: {e}")
            raise

    @staticmethod
    def get_main_business_composition(code: str) -> pd.DataFrame:
        """获取单只股票主营构成所有历史数据

        输入参数:
            code (str): 股票代码，例如"SH688041"

        输出参数:
            DataFrame:
                股票代码: object - 股票代码
                报告日期: object - 报告期
                分类类型: object - 业务分类
                主营构成: int64 - 构成名称
                主营收入: float64 - 收入金额（元）
                收入比例: float64 - 收入占比
                主营成本: float64 - 成本金额（元）
                成本比例: float64 - 成本占比
                主营利润: float64 - 利润金额（元）
                利润比例: float64 - 利润占比
                毛利率: float64 - 毛利率
        """
        code = format_info.stock_code(code)
        try:
            df = ak.stock_zygc_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get main business composition for {code}: {e}")
            raise

    @staticmethod
    def get_stock_account_statistics() -> pd.DataFrame:
        """获取股票账户统计月度数据

        输入参数: 无

        输出参数:
            DataFrame:
                数据日期: object - 统计日期
                新增投资者-数量: float64 - 新增账户数
                新增投资者-环比: float64 - 环比变化率
                新增投资者-同比: float64 - 同比变化率
                期末投资者-总量: float64 - 期末总账户数
                期末投资者-A股账户: float64 - A股账户数
                期末投资者-B股账户: float64 - B股账户数
                期末投资者-基金账户: float64 - 基金账户数
        """
        try:
            df = ak.stock_account_statistics_em()
            return df
        except Exception as e:
            logging.error(f"Failed to get stock account statistics: {e}")
            raise

    @staticmethod
    def get_institution_participation(code: str) -> pd.DataFrame:
        """获取机构参与度数据

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                日期: object - 交易日期
                机构参与度: float64 - 机构参与交易程度
                机构买入占比: float64 - 机构买入金额占比
                机构卖出占比: float64 - 机构卖出金额占比
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get institution participation for {code}: {e}")
            raise

    @staticmethod
    def get_stock_comprehensive_score(code: str) -> pd.DataFrame:
        """获取东方财富综合评分

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                日期: object - 评分日期
                综合评分: float64 - 股票综合评分
                评分级别: object - 评分等级
                技术评分: float64 - 技术面评分
                市场评分: float64 - 市场表现评分
                资金评分: float64 - 资金面评分
                基本面评分: float64 - 基本面评分
                消息面评分: float64 - 消息面评分
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_comment_detail_zhpj_lspf_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get comprehensive score for {code}: {e}")
            raise

    @staticmethod
    def get_market_attention(code: str) -> pd.DataFrame:
        """获取用户关注度数据

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                日期: object - 统计日期
                关注度: float64 - 用户关注程度
                关注度变化: float64 - 关注度变化情况
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_comment_detail_scrd_focus_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get market attention for {code}: {e}")
            raise

    @staticmethod
    def get_market_participation_willingness(code: str) -> pd.DataFrame:
        """获取市场参与意愿数据

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                日期: object - 统计日期
                市场参与意愿: float64 - 市场参与意愿指数
                市场参与意愿变化: float64 - 意愿指数变化
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_comment_detail_scrd_desire_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get market participation willingness for {code}: {e}")
            raise

    @staticmethod
    def get_stock_news(code: str) -> pd.DataFrame:
        """获取个股新闻数据

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                新闻标题: object - 新闻标题
                发布时间: object - 新闻发布时间
                新闻链接: object - 新闻详情链接
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_news_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get stock news for {code}: {e}")
            raise

    @staticmethod
    def get_shareholder_changes(symbol: str) -> pd.DataFrame:
        """获取股东增减持数据

        输入参数:
            symbol (str): choice of {"全部", "股东增持", "股东减持"}

        输出参数:
            DataFrame:
                股东名称: object - 股东名称
                变动截止日期: object - 变动截止日期
                变动价格: float64 - 变动价格（元）
                变动数量: float64 - 变动数量（股）
                变动后持股数: float64 - 变动后持股数（股）
                变动后持股比例: float64 - 变动后持股比例（%）
                变动原因: object - 变动原因
        """
        try:
            df = ak.stock_ggcg_em(symbol=symbol)
            return df
        except Exception as e:
            logging.error(f"Failed to get shareholder changes for {symbol}: {e}")
            raise

    @staticmethod
    def get_dividend_detail(code: str) -> pd.DataFrame:
        """获取分红详细数据-同花顺

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                实施年度: object - 分红年度
                送股: float64 - 每10股送股数
                转增: float64 - 每10股转增数
                派息: float64 - 每10股派息金额（税前）
                进度: object - 方案进度
                除权除息日: object - 除权除息日期
                股权登记日: object - 股权登记日期
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_fhps_detail_ths(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get dividend details for {code}: {e}")
            raise

    @staticmethod
    def get_chip_distribution(code: str, adjust: str = "") -> pd.DataFrame:
        """获取筹码分布数据

        输入参数:
            code (str): 股票代码，例如"688041"
            adjust: choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}

        输出参数:
            DataFrame:
                日期: object - 交易日期
                价格: float64 - 股票价格
                占比: float64 - 筹码占比（%）
                成本: float64 - 成本价格
                集中度: float64 - 筹码集中度
                获利比例: float64 - 获利盘比例（%）
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_cyq_em(symbol=code, adjust=adjust)
            return df
        except Exception as e:
            logging.error(f"Failed to get chip distribution for {code}: {e}")
            raise

    @staticmethod
    def get_research_reports(code: str) -> pd.DataFrame:
        """获取个股研报数据

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                发布日期: object - 研报发布日期
                标题: object - 研报标题
                研究机构: object - 发布机构
                分析师: object - 分析师姓名
                评级: object - 投资评级
                目标价: float64 - 目标价格
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_research_report_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get research reports for {code}: {e}")
            raise

    @staticmethod
    def get_key_indicators(code: str, indicator: str = "按报告期") -> pd.DataFrame:
        """获取关键指标数据-同花顺

        输入参数:
            code (str): 股票代码，例如"688041"
            indicator: 指标; choice of {"按报告期", "按年度", "按单季度"}

        输出参数:
            DataFrame:
                日期: object - 统计日期
                基本每股收益: float64 - 每股收益（元）
                净资产收益率: float64 - ROE（%）
                营业总收入: float64 - 营收（元）
                营业总收入同比增长: float64 - 营收增长率（%）
                净利润: float64 - 净利润（元）
                净利润同比增长: float64 - 净利润增长率（%）
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator=indicator)
            return df
        except Exception as e:
            logging.error(f"Failed to get key indicators for {code}: {e}")
            raise

    @staticmethod
    def get_stock_indicators(code: str) -> pd.DataFrame:
        """获取A股个股指标数据

        输入参数:
            code (str): 股票代码，例如"688041"

        输出参数:
            DataFrame:
                日期: object - 交易日期
                市盈率: float64 - PE
                市净率: float64 - PB
                股息率: float64 - 股息率（%）
                总市值: float64 - 总市值（元）
                流通市值: float64 - 流通市值（元）
        """
        code = format_info.stock_code_plain(code)
        try:
            df = ak.stock_a_indicator_lg(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get stock indicators for {code}: {e}")
            raise

    @staticmethod
    def get_high_low_statistics(symbol: str = "all") -> pd.DataFrame:
        """获取创新高和新低的股票数量统计

        输入参数:
            symbol: choice of {"all", "sz50", "hs300", "zz500"}

        输出参数:
            DataFrame:
                日期: object - 统计日期
                创新高股票数: int64 - 创新高股票数量
                创新低股票数: int64 - 创新低股票数量
        """
        try:
            df = ak.stock_a_high_low_statistics(symbol=symbol)
            return df
        except Exception as e:
            logging.error(f"Failed to get high low statistics: {e}")
            raise

    @staticmethod
    def get_profit_forecast(code: str = None, indicator: str = "预测年报每股收益") -> pd.DataFrame:
        """获取盈利预测数据

        输入参数:
            code (str): 股票代码，例如"688041"。None 则获取所有
            indicator: choice of {"预测年报每股收益", "预测年报净利润", "业绩预测详表-机构", "业绩预测详表-详细指标预测"}

        输出参数:
            DataFrame:
                股票代码: object - 股票代码
                股票简称: object - 股票名称
                预测年度: object - 预测年份
                预测每股收益: float64 - 预测EPS（元）
                预测净利润: float64 - 预测净利润（元）
                预测营业收入: float64 - 预测营业收入（元）
                机构数量: int64 - 预测机构数量
        """
        code = format_info.stock_code_plain(code)
        try:
            if code:
                df = ak.stock_profit_forecast_ths(symbol=code, indicator=indicator)
            else:
                df = ak.stock_profit_forecast_em()
            return df
        except Exception as e:
            logging.error(f"Failed to get profit forecast: {e}")
            raise

    @staticmethod
    def get_stock_hot_rank(code: str) -> pd.DataFrame:
        """获取股票热度排行数据

        输入参数:
            code (str): 股票代码，例如"SH688041"

        输出参数:
            DataFrame:
                日期: object - 统计日期
                热度值: float64 - 热度指数
                排名: int64 - 热度排名
                关注人数: int64 - 关注人数
                粉丝特征: object - 粉丝画像
        """
        code = format_info.stock_code(code)
        try:
            df = ak.stock_hot_rank_detail_em(symbol=code)
            return df
        except Exception as e:
            logging.error(f"Failed to get stock hot rank for {code}: {e}")
            raise

    @staticmethod
    def get_technical_indicators(indicator_type: str, symbol: str = None) -> pd.DataFrame:
        """获取技术指标数据

        Args:
            indicator_type: 指标类型，可选值：
                'new_high': 创新高
                'new_low': 创新低
                'continuous_rise': 连续上涨
                'continuous_fall': 连续下跌
                'volume_up': 连续放量
                'volume_down': 连续缩量
                'break_up': 向上突破
                'break_down': 向下突破
                'volume_price_rise': 量价齐升
                'volume_price_fall': 量价齐跌
            symbol:
                'new_high': choice of {"创月新高", "半年新高", "一年新高", "历史新高"}
                'new_low': choice of {"创月新低", "半年新低", "一年新低", "历史新低"}
                'break_up': choice of {"5日均线", "10日均线", "20日均线", "30日均线", "60日均线", "90日均线", "250日均线", "500日均线"}
                'break_down': choice of {"5日均线", "10日均线", "20日均线", "30日均线", "60日均线", "90日均线", "250日均线", "500日均线"}

        Returns:
            DataFrame包含技术指标数据
        """
        try:
            indicator_mapping = {
                'new_high': ak.stock_rank_cxg_ths,
                'new_low': ak.stock_rank_cxd_ths,
                'continuous_rise': ak.stock_rank_lxsz_ths,
                'continuous_fall': ak.stock_rank_lxxd_ths,
                'volume_up': ak.stock_rank_cxfl_ths,
                'volume_down': ak.stock_rank_cxsl_ths,
                'break_up': ak.stock_rank_xstp_ths,
                'break_down': ak.stock_rank_xxtp_ths,
                'volume_price_rise': ak.stock_rank_ljqs_ths,
                'volume_price_fall': ak.stock_rank_ljqd_ths
            }
            type_with_symbol = ('new_high', 'new_low', 'break_up', 'break_down')

            if indicator_type not in indicator_mapping:
                raise ValueError(f"Unsupported indicator type: {indicator_type}")

            if indicator_type in type_with_symbol:
                df = indicator_mapping[indicator_type](symbol=symbol)
            else:
                df = indicator_mapping[indicator_type]()
            return df
        except Exception as e:
            logging.error(f"Failed to get technical indicators for {indicator_type}: {e}")
            raise
