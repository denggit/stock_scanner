#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : data_manager.py
@Description: 因子数据管理器，负责数据准备和预处理
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
from backend.business.data.data_fetcher import StockDataFetcher
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FactorDataManager:
    """
    因子数据管理器，负责数据准备和预处理
    
    功能：
    1. 获取全市场行情数据
    2. 获取财务数据
    3. 数据清洗和预处理
    4. 数据对齐和合并
    """
    
    def __init__(self, data_fetcher: Optional[StockDataFetcher] = None):
        """
        初始化数据管理器
        
        Args:
            data_fetcher: 数据获取器实例
        """
        self.data_fetcher = data_fetcher or StockDataFetcher()
        self._market_data = None
        self._financial_data = None
        self._processed_data = None
        
    def get_market_data(self, 
                       start_date: str,
                       end_date: str,
                       stock_codes: Optional[List[str]] = None,
                       stock_pool: str = "hs300",
                       fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取市场行情数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            stock_codes: 股票代码列表，优先使用此参数
            stock_pool: 股票池名称，当stock_codes为None时使用
            fields: 需要的字段列表
            
        Returns:
            市场行情数据DataFrame
        """
        logger.info(f"开始获取市场行情数据: {start_date} 到 {end_date}")
        
        if fields is None:
            # 使用实际数据库中的字段
            fields = ['code', 'trade_date', 'open', 'high', 'low', 'close', 'preclose',
                     'volume', 'amount', 'turn', 'tradestatus', 'pct_chg', 'pe_ttm', 
                     'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm', 'is_st', 'vwap']
        
        # 获取股票列表
        if stock_codes is None:
            # 使用data_fetcher的股票池选择功能
            stock_list_df = self.data_fetcher.get_stock_list(pool_name=stock_pool)
            if stock_list_df.empty:
                logger.warning(f"股票池 {stock_pool} 为空，使用默认股票")
                stock_codes = ['sz.000001', 'sz.000002', 'sz.000858', 'sz.002415', 'sz.300059']
            else:
                stock_codes = stock_list_df['code'].tolist()
                logger.info(f"从股票池 {stock_pool} 获取到 {len(stock_codes)} 只股票")
        
        market_data_list = []
        failed_count = 0
        
        for code in stock_codes:
            try:
                df = self.data_fetcher.fetch_stock_data(
                    code, 
                    start_date=start_date, 
                    end_date=end_date,
                    period='daily'
                )
                if df is not None and not df.empty:
                    # 确保trade_date是datetime类型
                    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                    
                    # 过滤日期范围
                    mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
                    df = df[mask]
                    
                    if not df.empty:
                        # 只选择存在的字段
                        available_fields = [field for field in fields if field in df.columns]
                        if available_fields:
                            market_data_list.append(df[available_fields])
                        
            except Exception as e:
                failed_count += 1
                logger.warning(f"获取股票 {code} 数据失败: {e}")
                continue
        
        if not market_data_list:
            raise ValueError("未能获取到任何市场数据")
        
        # 合并所有股票数据
        market_data = pd.concat(market_data_list, ignore_index=True)
        
        # 数据清洗
        market_data = self._clean_market_data(market_data)
        
        self._market_data = market_data
        logger.info(f"成功获取 {len(market_data)} 条市场数据，失败 {failed_count} 只股票")
        
        return market_data
    
    def get_financial_data(self,
                          start_date: str,
                          end_date: str,
                          stock_codes: Optional[List[str]] = None,
                          fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取财务数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
            fields: 需要的字段列表
            
        Returns:
            财务数据DataFrame
        """
        logger.info(f"开始获取财务数据: {start_date} 到 {end_date}")
        
        if fields is None:
            # 使用实际数据库中的财务字段
            fields = ['code', 'trade_date', 'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm', 'is_st']
        
        # 财务数据目前包含在市场数据中的财务指标
        if self._market_data is not None:
            # 检查哪些字段在数据中存在
            available_fields = [field for field in fields if field in self._market_data.columns]
            if available_fields:
                financial_data = self._market_data[available_fields].copy()
                self._financial_data = financial_data
                logger.info(f"从市场数据中提取 {len(available_fields)} 个财务指标字段")
                return financial_data
            else:
                logger.warning("市场数据中没有财务指标字段")
                return pd.DataFrame()
        else:
            logger.warning("请先获取市场数据，财务指标包含在市场数据中")
            return pd.DataFrame()
    
    def prepare_factor_data(self,
                           start_date: str,
                           end_date: str,
                           stock_codes: Optional[List[str]] = None,
                           stock_pool: str = "hs300",
                           **kwargs) -> pd.DataFrame:
        """
        准备因子计算所需的数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表，优先使用此参数
            stock_pool: 股票池名称，当stock_codes为None时使用
            **kwargs: 其他参数
            
        Returns:
            处理后的数据DataFrame
        """
        logger.info(f"开始准备因子计算数据: {start_date} 到 {end_date}")
        
        # 获取市场数据
        market_data = self.get_market_data(
            start_date=start_date, 
            end_date=end_date, 
            stock_codes=stock_codes,
            stock_pool=stock_pool
        )
        
        # 获取财务数据（包含在市场数据中）
        financial_data = self.get_financial_data(start_date, end_date, stock_codes)
        
        # 由于财务数据已经包含在市场数据中，直接使用市场数据
        processed_data = market_data.copy()
        
        # 数据预处理
        processed_data = self._preprocess_data(processed_data)
        
        self._processed_data = processed_data
        logger.info(f"数据准备完成，共 {len(processed_data)} 条记录，包含 {processed_data['code'].nunique()} 只股票")
        
        return processed_data
    
    def get_stock_pool(self, pool_name: str = "hs300", **kwargs) -> pd.DataFrame:
        """
        获取股票池列表
        
        Args:
            pool_name: 股票池名称
                - "full" 或 "全量股票": 全部股票
                - "no_st" 或 "非ST股票": 非ST股票
                - "st": ST股票
                - "sz50" 或 "上证50": 上证50
                - "hs300" 或 "沪深300": 沪深300
                - "zz500" 或 "中证500": 中证500
            **kwargs: 其他参数，如ipo_date、min_amount等
            
        Returns:
            股票列表DataFrame
        """
        logger.info(f"获取股票池: {pool_name}")
        
        if kwargs:
            # 如果有额外条件，使用get_stock_list_with_cond
            return self.data_fetcher.get_stock_list_with_cond(
                pool_name=pool_name, 
                **kwargs
            )
        else:
            # 直接获取股票池
            return self.data_fetcher.get_stock_list(pool_name=pool_name)
    
    def _clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗市场数据
        
        Args:
            data: 原始市场数据
            
        Returns:
            清洗后的数据
        """
        # 移除停牌数据
        if 'tradestatus' in data.columns:
            data = data[data['tradestatus'] == 1]
        
        # 根据板块和股票类型设置涨跌停规则
        if 'pct_chg' in data.columns and 'code' in data.columns:
            data = self._filter_price_limit_data(data)
        
        # 移除异常价格数据
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # 移除异常成交量数据
        if 'volume' in data.columns:
            data = data[data['volume'] > 0]
        
        # 按日期和代码排序
        data = data.sort_values(['trade_date', 'code']).reset_index(drop=True)
        
        return data
    
    def _filter_price_limit_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根据板块和股票类型过滤涨跌停数据
        
        Args:
            data: 原始数据
            
        Returns:
            过滤后的数据
        """
        # 定义涨跌停规则
        price_limit_rules = {
            "sh_main": {"pattern": r"^60\d{4}$", "limit": 0.10},  # 上海主板: ±10%
            "sz_main": {"pattern": r"^000\d{3}$|^001\d{3}$", "limit": 0.10},  # 深圳主板: ±10%
            "gem": {"pattern": r"^300\d{3}$", "limit": 0.20},  # 创业板: ±20%
            "star": {"pattern": r"^688\d{3}$", "limit": 0.20},  # 科创板: ±20%
            "beijing": {"pattern": r"^83\d{4}$|^87\d{4}$|^43\d{4}$", "limit": 0.30},  # 北交所: ±30%
        }
        
        # 为每个股票代码确定涨跌停限制
        def get_price_limit(code: str, is_st: bool = False) -> float:
            """获取股票的涨跌停限制"""
            limit = 0.10  # 默认10%
            
            # 根据股票代码确定板块
            for rule_name, rule in price_limit_rules.items():
                if re.match(rule["pattern"], code):
                    limit = rule["limit"]
                    break
            
            # ST股票涨跌停限制减半
            if is_st:
                limit = limit / 2
            
            return limit * 100  # 转换为百分比
        
        # 创建股票代码到涨跌停限制的映射
        code_limit_map = {}
        for code in data['code'].unique():
            stock_data = data[data['code'] == code].iloc[0]
            is_st = stock_data.get('is_st', 0) == 1 if 'is_st' in stock_data else False
            code_limit_map[code] = get_price_limit(code, is_st)
        
        # 应用过滤条件
        def filter_by_limit(row):
            code = row['code']
            pct_chg = row['pct_chg']
            limit = code_limit_map.get(code, 10.0)  # 默认10%
            return abs(pct_chg) < limit
        
        # 过滤数据
        mask = data.apply(filter_by_limit, axis=1)
        filtered_data = data[mask]
        
        logger.info(f"涨跌停过滤: 原始数据 {len(data)} 条，过滤后 {len(filtered_data)} 条")
        
        return filtered_data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        # 计算收益率
        if 'close' in data.columns:
            data['returns'] = data.groupby('code')['close'].pct_change()
        
        # 计算对数收益率
        if 'close' in data.columns:
            log_returns_list = []
            for code in data['code'].unique():
                stock_data = data[data['code'] == code].copy()
                stock_data = stock_data.sort_values('trade_date')
                log_returns = np.log(stock_data['close'] / stock_data['close'].shift(1))
                stock_data['log_returns'] = log_returns
                log_returns_list.append(stock_data)
            
            data = pd.concat(log_returns_list, ignore_index=True)
        
        # 处理财务指标的异常值
        financial_cols = ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm']
        for col in financial_cols:
            if col in data.columns:
                # 移除极端值
                q1 = data[col].quantile(0.01)
                q99 = data[col].quantile(0.99)
                data[col] = data[col].clip(q1, q99)
        
        return data
    
    def get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取单个股票的数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据DataFrame
        """
        if self._processed_data is not None:
            mask = (self._processed_data['code'] == stock_code) & \
                   (self._processed_data['trade_date'] >= start_date) & \
                   (self._processed_data['trade_date'] <= end_date)
            return self._processed_data[mask].copy()
        else:
            return self.data_fetcher.fetch_stock_data(stock_code, start_date=start_date)
    
    def get_cross_section_data(self, date: str) -> pd.DataFrame:
        """
        获取指定日期的横截面数据
        
        Args:
            date: 日期字符串
            
        Returns:
            横截面数据DataFrame
        """
        if self._processed_data is not None:
            mask = self._processed_data['trade_date'] == date
            return self._processed_data[mask].copy()
        else:
            raise ValueError("请先调用 prepare_factor_data 准备数据")
    
    def get_time_series_data(self, stock_code: str) -> pd.DataFrame:
        """
        获取单个股票的时间序列数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            时间序列数据DataFrame
        """
        if self._processed_data is not None:
            mask = self._processed_data['code'] == stock_code
            return self._processed_data[mask].copy()
        else:
            raise ValueError("请先调用 prepare_factor_data 准备数据")
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据信息
        
        Returns:
            数据信息字典
        """
        info = {
            'market_data_shape': self._market_data.shape if self._market_data is not None else None,
            'financial_data_shape': self._financial_data.shape if self._financial_data is not None else None,
            'processed_data_shape': self._processed_data.shape if self._processed_data is not None else None,
        }
        
        if self._processed_data is not None:
            info.update({
                'date_range': [
                    self._processed_data['trade_date'].min(),
                    self._processed_data['trade_date'].max()
                ],
                'stock_count': self._processed_data['code'].nunique(),
                'columns': list(self._processed_data.columns)
            })
        
        return info
