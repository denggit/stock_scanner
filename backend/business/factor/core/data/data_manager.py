#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : data_manager.py
@Description: 因子数据管理器，负责数据准备和预处理
"""

import re
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

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
                        fields: Optional[List[str]] = None,
                        optimize_data_fetch: bool = True) -> pd.DataFrame:
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
            if optimize_data_fetch:
                # 使用数据需求分析器优化字段选择
                from .data_requirement_analyzer import data_requirement_analyzer
                logger.info("使用智能数据获取优化，只获取必要的字段")
                # 默认获取基础字段，具体优化在prepare_factor_data中实现
                fields = ['code', 'trade_date', 'close']
            else:
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
        获取财务数据（季度数据）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
            fields: 需要的字段列表
            
        Returns:
            财务数据DataFrame，包含季度财务指标
        """
        logger.info(f"开始获取财务数据: {start_date} 到 {end_date}")

        # 获取股票列表
        if stock_codes is None:
            if self._market_data is not None:
                stock_codes = self._market_data['code'].unique().tolist()
            else:
                logger.warning("请先获取市场数据或提供股票代码列表")
                return pd.DataFrame()

        # 扩展时间范围以获取足够的财务数据（财务数据按季度发布）
        # 通常需要提前1-2年获取财务数据以确保有足够的历史数据
        extended_start_date = self._extend_date_for_financial_data(start_date)

        financial_data_list = []
        failed_count = 0

        for code in stock_codes:
            try:
                # 获取各种财务数据
                all_financial_data = self.data_fetcher.fetch_all_financial_data(
                    code,
                    start_date=extended_start_date,
                    end_date=end_date
                )

                # 合并所有财务数据
                code_financial_data = self._merge_financial_data_for_stock(code, all_financial_data)

                if not code_financial_data.empty:
                    financial_data_list.append(code_financial_data)
                else:
                    failed_count += 1

            except Exception as e:
                logger.warning(f"获取股票 {code} 财务数据失败: {e}")
                failed_count += 1

        if financial_data_list:
            # 合并所有股票的财务数据
            financial_data = pd.concat(financial_data_list, ignore_index=True)

            # 数据清洗和预处理
            financial_data = self._clean_financial_data(financial_data)

            self._financial_data = financial_data
            logger.info(f"成功获取 {len(financial_data)} 条财务数据，失败 {failed_count} 只股票")

            return financial_data
        else:
            logger.warning("没有获取到任何财务数据")
            return pd.DataFrame()

    def _extend_date_for_financial_data(self, start_date: str) -> str:
        """
        为财务数据扩展开始日期（财务数据按季度发布）
        
        Args:
            start_date: 原始开始日期
            
        Returns:
            扩展后的开始日期
        """
        start_dt = pd.to_datetime(start_date)
        # 提前2年获取财务数据，确保有足够的历史数据用于因子计算
        extended_dt = start_dt - pd.DateOffset(years=2)
        return extended_dt.strftime('%Y-%m-%d')

    def _merge_financial_data_for_stock(self, code: str, all_financial_data: dict) -> pd.DataFrame:
        """
        合并单只股票的所有财务数据
        
        Args:
            code: 股票代码
            all_financial_data: 所有财务数据字典
            
        Returns:
            合并后的财务数据DataFrame
        """
        merged_data = []

        # 获取所有财务数据的日期
        all_dates = set()
        for data_type, df in all_financial_data.items():
            if not df.empty:
                if 'statDate' in df.columns:
                    all_dates.update(df['statDate'].tolist())

        # 按日期合并数据
        for date in sorted(all_dates, reverse=True):  # 按日期倒序排列
            row_data = {'code': code, 'statDate': date}

            # 合并各种财务数据
            for data_type, df in all_financial_data.items():
                if not df.empty and 'statDate' in df.columns:
                    date_data = df[df['statDate'] == date]
                    if not date_data.empty:
                        # 添加财务指标（排除code和statDate字段）
                        for col in date_data.columns:
                            if col not in ['code', 'statDate', 'pubDate']:
                                row_data[col] = date_data[col].iloc[0]

            merged_data.append(row_data)

        return pd.DataFrame(merged_data)

    def _clean_financial_data(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗财务数据
        
        Args:
            financial_data: 原始财务数据
            
        Returns:
            清洗后的财务数据
        """
        if financial_data.empty:
            return financial_data

        # 确保日期格式正确
        if 'statDate' in financial_data.columns:
            financial_data['statDate'] = pd.to_datetime(financial_data['statDate'])

        # 处理decimal.Decimal类型的数据，转换为float
        for col in financial_data.columns:
            if col not in ['code', 'statDate']:
                if financial_data[col].dtype == 'object':
                    # 尝试转换为float，处理decimal.Decimal
                    try:
                        financial_data[col] = pd.to_numeric(financial_data[col], errors='coerce')
                    except:
                        pass

        # 处理数值字段的异常值
        numeric_columns = financial_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['code']:
                # 替换无穷大值
                financial_data[col] = financial_data[col].replace([np.inf, -np.inf], np.nan)

                # 处理异常大的值（可能是数据错误）
                q99 = financial_data[col].quantile(0.99)
                q01 = financial_data[col].quantile(0.01)
                if pd.notna(q99) and pd.notna(q01):
                    financial_data.loc[financial_data[col] > q99 * 10, col] = np.nan
                    financial_data.loc[financial_data[col] < q01 / 10, col] = np.nan

        # 按股票代码和日期排序
        financial_data = financial_data.sort_values(['code', 'statDate'], ascending=[True, False])

        return financial_data

    def prepare_factor_data(self,
                            start_date: str,
                            end_date: str,
                            stock_codes: Optional[List[str]] = None,
                            stock_pool: str = "hs300",
                            factor_names: Optional[List[str]] = None,
                            optimize_data_fetch: bool = True,
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
            处理后的数据DataFrame，包含市场数据和财务数据
        """
        logger.info(f"开始准备因子计算数据: {start_date} 到 {end_date}")

        # 智能数据获取优化
        if optimize_data_fetch and factor_names:
            from .data_requirement_analyzer import data_requirement_analyzer
            from ..factor.factor_registry import factor_registry

            logger.info("使用智能数据获取优化...")
            fetch_plan = data_requirement_analyzer.generate_data_fetch_plan(factor_names, factor_registry)

            logger.info(f"数据获取计划: {fetch_plan}")
            logger.info(f"预计内存节省: {fetch_plan['memory_saving_ratio']:.2%}")
            logger.info(f"预计性能提升: {fetch_plan['estimated_performance_improvement']:.2%}")

            # 根据优化配置获取数据
            optimized_config = fetch_plan['optimized_config']
            required_fields = ['code', 'trade_date'] + optimized_config['market_fields'] + optimized_config[
                'valuation_fields']

            logger.info(f"优化后的字段列表: {required_fields}")
        else:
            required_fields = None
            fetch_plan = None

        # 获取市场数据
        market_data = self.get_market_data(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            stock_pool=stock_pool,
            fields=required_fields,
            optimize_data_fetch=optimize_data_fetch
        )

        if market_data.empty:
            logger.error("无法获取市场数据")
            return pd.DataFrame()

        # 获取财务数据（季度数据）
        if fetch_plan and fetch_plan['fetch_strategy']['financial_data']:
            logger.info(f"需要财务数据，类型: {fetch_plan['fetch_strategy']['financial_types']}")
            financial_data = self.get_financial_data(start_date, end_date, stock_codes)
        else:
            logger.info("不需要财务数据，跳过财务数据获取")
            financial_data = pd.DataFrame()

        # 合并市场数据和财务数据
        processed_data = self._merge_market_and_financial_data(market_data, financial_data)

        # 数据预处理
        processed_data = self._preprocess_data(processed_data)

        self._processed_data = processed_data
        logger.info(f"数据准备完成，共 {len(processed_data)} 条记录，包含 {processed_data['code'].nunique()} 只股票")

        return processed_data

    def _merge_market_and_financial_data(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        合并市场数据和财务数据
        
        使用 pandas.merge_asof 进行高效的时间序列数据合并，替代原来的嵌套循环方式。
        
        Args:
            market_data: 市场数据
            financial_data: 财务数据
            
        Returns:
            合并后的数据DataFrame
        """
        if financial_data.empty:
            logger.info("没有财务数据，仅使用市场数据")
            return market_data.copy()

        logger.info(f"开始合并市场数据({len(market_data)}条)和财务数据({len(financial_data)}条)")

        # 1. 确保日期格式一致并转换为datetime类型
        market_data = market_data.copy()
        financial_data = financial_data.copy()
        
        market_data['trade_date'] = pd.to_datetime(market_data['trade_date'])
        financial_data['statDate'] = pd.to_datetime(financial_data['statDate'])

        # 2. 确保数据已按日期排序（merge_asof的要求）
        market_data = market_data.sort_values(['code', 'trade_date']).reset_index(drop=True)
        financial_data = financial_data.sort_values(['code', 'statDate']).reset_index(drop=True)

        # 3. 使用 merge_asof 进行高效合并
        try:
            # 重命名财务数据的日期列以便合并
            financial_data_renamed = financial_data.rename(columns={'statDate': 'trade_date'})
            
            # 确保数据按合并键排序（merge_asof的要求）
            # merge_asof要求整个数据框按trade_date排序，而不仅仅是按code和trade_date排序
            market_data_sorted = market_data.sort_values('trade_date').reset_index(drop=True)
            financial_data_sorted = financial_data_renamed.sort_values('trade_date').reset_index(drop=True)
            
            # 验证排序是否正确
            if not market_data_sorted['trade_date'].is_monotonic_increasing:
                logger.warning("市场数据日期排序不正确")
                raise ValueError("市场数据日期排序失败")
            
            if not financial_data_sorted['trade_date'].is_monotonic_increasing:
                logger.warning("财务数据日期排序不正确")
                raise ValueError("财务数据日期排序失败")
            
            # 使用 merge_asof 进行合并
            # left: market_data (交易日数据)
            # right: financial_data_renamed (财务报告日数据)
            # on: 'trade_date' (日期列)
            # by: 'code' (按股票代码分组)
            # direction: 'backward' (匹配最近的、不晚于交易日的财务报告)
            merged_data = pd.merge_asof(
                left=market_data_sorted,
                right=financial_data_sorted,
                on='trade_date',
                by='code',
                direction='backward',
                suffixes=('', '_financial')
            )
            
            # 4. 清理合并后的数据
            # 移除财务数据特有的列（如pubDate等）
            columns_to_drop = [col for col in merged_data.columns 
                             if col.endswith('_financial') or col in ['pubDate']]
            merged_data = merged_data.drop(columns=columns_to_drop, errors='ignore')
            
            # 5. 确保结果按原始顺序排列
            merged_data = merged_data.sort_values(['code', 'trade_date']).reset_index(drop=True)
            
            logger.info(f"数据合并完成，共 {len(merged_data)} 条记录")
            logger.info(f"合并效率提升: 使用 merge_asof 替代嵌套循环")
            
            return merged_data
            
        except Exception as e:
            logger.error(f"使用 merge_asof 合并失败: {e}")
            logger.info("回退到原始合并方法")
            
            # 回退到原始方法（如果 merge_asof 失败）
            return self._merge_market_and_financial_data_fallback(market_data, financial_data)
    
    def _merge_market_and_financial_data_fallback(self, market_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        回退的合并方法（原始嵌套循环方式）
        
        Args:
            market_data: 市场数据
            financial_data: 财务数据
            
        Returns:
            合并后的数据DataFrame
        """
        logger.info("使用回退方法合并数据")
        
        # 为每个交易日找到最近的财务数据
        merged_data_list = []

        for code in market_data['code'].unique():
            code_market_data = market_data[market_data['code'] == code].copy()
            code_financial_data = financial_data[financial_data['code'] == code].copy()

            if code_financial_data.empty:
                # 如果没有财务数据，使用市场数据
                merged_data_list.append(code_market_data)
                continue

            # 为每个交易日找到最近的财务数据
            code_merged_data = []
            for _, market_row in code_market_data.iterrows():
                trade_date = market_row['trade_date']

                # 找到最近的财务数据（财务数据日期 <= 交易日期）
                available_financial_data = code_financial_data[code_financial_data['statDate'] <= trade_date]

                if not available_financial_data.empty:
                    # 选择最近的财务数据
                    latest_financial_data = available_financial_data.iloc[0]  # 已经按日期倒序排列

                    # 合并数据
                    merged_row = market_row.copy()
                    for col in latest_financial_data.index:
                        if col not in ['code', 'statDate', 'pubDate']:
                            merged_row[col] = latest_financial_data[col]

                    code_merged_data.append(merged_row)
                else:
                    # 如果没有可用的财务数据，使用市场数据
                    code_merged_data.append(market_row)

            # 合并该股票的所有数据
            if code_merged_data:
                code_result = pd.DataFrame(code_merged_data)
                merged_data_list.append(code_result)

        if merged_data_list:
            result = pd.concat(merged_data_list, ignore_index=True)
            logger.info(f"回退方法数据合并完成，共 {len(result)} 条记录")
            return result
        else:
            logger.warning("回退方法数据合并失败，返回市场数据")
            return market_data.copy()

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
