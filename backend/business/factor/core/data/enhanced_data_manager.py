#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : enhanced_data_manager.py
@Description: 增强的数据管理器 - 集成AKShare数据源
@Author     : Zijun Deng
@Date       : 2025-08-21
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from backend.business.data.source.akshare_src import AKShareSource
from backend.utils.logger import setup_logger
from .data_manager import FactorDataManager

logger = setup_logger("backtest_factor")


class EnhancedFactorDataManager(FactorDataManager):
    """
    增强的因子数据管理器
    
    继承自FactorDataManager，并集成AKShare数据源，
    提供更丰富的数据支持因子回测
    """

    def __init__(self):
        super().__init__()
        self.akshare_source = AKShareSource()
        self._enhanced_data_cache = {}

    def get_enhanced_stock_data(self,
                                codes: List[str],
                                start_date: str,
                                end_date: str,
                                include_akshare_data: bool = True) -> pd.DataFrame:
        """
        获取增强的股票数据，包括基础行情数据和AKShare补充数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            include_akshare_data: 是否包含AKShare数据
            
        Returns:
            增强的股票数据DataFrame
        """
        # 获取基础行情数据
        base_data = self.get_stock_data(codes, start_date, end_date)

        if not include_akshare_data:
            return base_data

        # 获取AKShare补充数据
        enhanced_data = self._get_akshare_enhanced_data(base_data, codes, start_date, end_date)

        # 合并数据
        merged_data = self._merge_enhanced_data(base_data, enhanced_data)

        return merged_data

    def _get_akshare_enhanced_data(self,
                                   base_data: pd.DataFrame,
                                   codes: List[str],
                                   start_date: str,
                                   end_date: str) -> Dict[str, pd.DataFrame]:
        """
        获取AKShare增强数据
        
        Args:
            base_data: 基础行情数据
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            AKShare增强数据字典
        """
        enhanced_data = {}

        for code in codes:
            try:
                logger.info(f"获取股票 {code} 的AKShare增强数据...")

                # 获取个股指标数据（PE、PB、市值等）
                try:
                    indicators_data = self.akshare_source.get_stock_indicators(code)
                    enhanced_data[f'{code}_indicators'] = indicators_data
                except Exception as e:
                    logger.warning(f"获取股票 {code} 指标数据失败: {e}")

                # 获取机构参与度数据
                try:
                    institution_data = self.akshare_source.get_institution_participation(code)
                    enhanced_data[f'{code}_institution'] = institution_data
                except Exception as e:
                    logger.warning(f"获取股票 {code} 机构参与度数据失败: {e}")

                # 获取综合评分数据
                try:
                    score_data = self.akshare_source.get_stock_comprehensive_score(code)
                    enhanced_data[f'{code}_score'] = score_data
                except Exception as e:
                    logger.warning(f"获取股票 {code} 综合评分数据失败: {e}")

                # 获取市场关注度数据
                try:
                    attention_data = self.akshare_source.get_market_attention(code)
                    enhanced_data[f'{code}_attention'] = attention_data
                except Exception as e:
                    logger.warning(f"获取股票 {code} 市场关注度数据失败: {e}")

                # 获取筹码分布数据
                try:
                    chip_data = self.akshare_source.get_chip_distribution(code)
                    enhanced_data[f'{code}_chip'] = chip_data
                except Exception as e:
                    logger.warning(f"获取股票 {code} 筹码分布数据失败: {e}")

            except Exception as e:
                logger.error(f"获取股票 {code} AKShare数据时发生错误: {e}")
                continue

        return enhanced_data

    def _merge_enhanced_data(self,
                             base_data: pd.DataFrame,
                             enhanced_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并基础数据和增强数据
        
        Args:
            base_data: 基础行情数据
            enhanced_data: AKShare增强数据
            
        Returns:
            合并后的增强数据
        """
        merged_data = base_data.copy()

        # 按股票代码分组处理
        for code in merged_data['code'].unique():
            code_data = merged_data[merged_data['code'] == code].copy()

            # 合并个股指标数据
            if f'{code}_indicators' in enhanced_data:
                indicators = enhanced_data[f'{code}_indicators']
                code_data = self._merge_indicators_data(code_data, indicators)

            # 合并机构参与度数据
            if f'{code}_institution' in enhanced_data:
                institution = enhanced_data[f'{code}_institution']
                code_data = self._merge_institution_data(code_data, institution)

            # 合并综合评分数据
            if f'{code}_score' in enhanced_data:
                score = enhanced_data[f'{code}_score']
                code_data = self._merge_score_data(code_data, score)

            # 合并市场关注度数据
            if f'{code}_attention' in enhanced_data:
                attention = enhanced_data[f'{code}_attention']
                code_data = self._merge_attention_data(code_data, attention)

            # 合并筹码分布数据
            if f'{code}_chip' in enhanced_data:
                chip = enhanced_data[f'{code}_chip']
                code_data = self._merge_chip_data(code_data, chip)

            # 更新合并后的数据
            merged_data.loc[merged_data['code'] == code] = code_data

        return merged_data

    def _merge_indicators_data(self,
                               base_data: pd.DataFrame,
                               indicators_data: pd.DataFrame) -> pd.DataFrame:
        """合并个股指标数据"""
        if indicators_data.empty:
            return base_data

        # 重命名列以避免冲突
        indicators_data = indicators_data.rename(columns={
            '日期': 'trade_date',
            '市盈率': 'pe_akshare',
            '市净率': 'pb_akshare',
            '股息率': 'dividend_yield_akshare',
            '总市值': 'total_market_cap_akshare',
            '流通市值': 'circulating_market_cap_akshare'
        })

        # 确保日期格式一致
        indicators_data['trade_date'] = pd.to_datetime(indicators_data['trade_date'])
        base_data['trade_date'] = pd.to_datetime(base_data['trade_date'])

        # 合并数据
        merged = pd.merge(base_data, indicators_data, on='trade_date', how='left')

        return merged

    def _merge_institution_data(self,
                                base_data: pd.DataFrame,
                                institution_data: pd.DataFrame) -> pd.DataFrame:
        """合并机构参与度数据"""
        if institution_data.empty:
            return base_data

        # 重命名列
        institution_data = institution_data.rename(columns={
            '日期': 'trade_date',
            '机构参与度': 'institution_participation',
            '机构买入占比': 'institution_buy_ratio',
            '机构卖出占比': 'institution_sell_ratio'
        })

        # 确保日期格式一致
        institution_data['trade_date'] = pd.to_datetime(institution_data['trade_date'])
        base_data['trade_date'] = pd.to_datetime(base_data['trade_date'])

        # 合并数据
        merged = pd.merge(base_data, institution_data, on='trade_date', how='left')

        return merged

    def _merge_score_data(self,
                          base_data: pd.DataFrame,
                          score_data: pd.DataFrame) -> pd.DataFrame:
        """合并综合评分数据"""
        if score_data.empty:
            return base_data

        # 重命名列
        score_data = score_data.rename(columns={
            '日期': 'trade_date',
            '综合评分': 'comprehensive_score',
            '评分级别': 'score_level',
            '技术评分': 'technical_score',
            '市场评分': 'market_score',
            '资金评分': 'fund_score',
            '基本面评分': 'fundamental_score',
            '消息面评分': 'news_score'
        })

        # 确保日期格式一致
        score_data['trade_date'] = pd.to_datetime(score_data['trade_date'])
        base_data['trade_date'] = pd.to_datetime(base_data['trade_date'])

        # 合并数据
        merged = pd.merge(base_data, score_data, on='trade_date', how='left')

        return merged

    def _merge_attention_data(self,
                              base_data: pd.DataFrame,
                              attention_data: pd.DataFrame) -> pd.DataFrame:
        """合并市场关注度数据"""
        if attention_data.empty:
            return base_data

        # 重命名列
        attention_data = attention_data.rename(columns={
            '日期': 'trade_date',
            '关注度': 'market_attention',
            '关注度变化': 'attention_change'
        })

        # 确保日期格式一致
        attention_data['trade_date'] = pd.to_datetime(attention_data['trade_date'])
        base_data['trade_date'] = pd.to_datetime(base_data['trade_date'])

        # 合并数据
        merged = pd.merge(base_data, attention_data, on='trade_date', how='left')

        return merged

    def _merge_chip_data(self,
                         base_data: pd.DataFrame,
                         chip_data: pd.DataFrame) -> pd.DataFrame:
        """合并筹码分布数据"""
        if chip_data.empty:
            return base_data

        # 重命名列
        chip_data = chip_data.rename(columns={
            '日期': 'trade_date',
            '价格': 'chip_price',
            '占比': 'chip_ratio',
            '成本': 'chip_cost',
            '集中度': 'chip_concentration',
            '获利比例': 'profit_ratio'
        })

        # 确保日期格式一致
        chip_data['trade_date'] = pd.to_datetime(chip_data['trade_date'])
        base_data['trade_date'] = pd.to_datetime(base_data['trade_date'])

        # 合并数据
        merged = pd.merge(base_data, chip_data, on='trade_date', how='left')

        return merged

    def get_market_sentiment_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取市场情绪数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            市场情绪数据
        """
        try:
            # 获取创新高/新低统计
            high_low_stats = self.akshare_source.get_high_low_statistics()

            # 获取账户统计
            account_stats = self.akshare_source.get_stock_account_statistics()

            # 合并市场情绪数据
            sentiment_data = pd.merge(high_low_stats, account_stats,
                                      left_on='日期', right_on='数据日期', how='outer')

            # 过滤日期范围
            sentiment_data['日期'] = pd.to_datetime(sentiment_data['日期'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            sentiment_data = sentiment_data[
                (sentiment_data['日期'] >= start_dt) &
                (sentiment_data['日期'] <= end_dt)
                ]

            return sentiment_data

        except Exception as e:
            logger.error(f"获取市场情绪数据失败: {e}")
            return pd.DataFrame()

    def get_technical_breakout_data(self,
                                    indicator_type: str = 'break_up',
                                    symbol: str = '20日均线') -> pd.DataFrame:
        """
        获取技术突破数据
        
        Args:
            indicator_type: 指标类型
            symbol: 突破类型
            
        Returns:
            技术突破数据
        """
        try:
            breakout_data = self.akshare_source.get_technical_indicators(
                indicator_type=indicator_type,
                symbol=symbol
            )
            return breakout_data
        except Exception as e:
            logger.error(f"获取技术突破数据失败: {e}")
            return pd.DataFrame()

    def get_enhanced_factor_data(self,
                                 codes: List[str],
                                 start_date: str,
                                 end_date: str,
                                 factor_names: List[str]) -> pd.DataFrame:
        """
        获取增强的因子数据，包括基础因子和基于AKShare数据的新因子
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            factor_names: 因子名称列表
            
        Returns:
            增强的因子数据
        """
        # 获取基础因子数据
        base_factor_data = self.calculate_factors(codes, start_date, end_date, factor_names)

        # 获取增强数据
        enhanced_data = self.get_enhanced_stock_data(codes, start_date, end_date, True)

        # 计算基于AKShare数据的新因子
        new_factors = self._calculate_akshare_based_factors(enhanced_data)

        # 合并因子数据
        if not new_factors.empty:
            factor_data = pd.merge(base_factor_data, new_factors,
                                   on=['code', 'trade_date'], how='left')
        else:
            factor_data = base_factor_data

        return factor_data

    def _calculate_akshare_based_factors(self, enhanced_data: pd.DataFrame) -> pd.DataFrame:
        """
        基于AKShare数据计算新因子
        
        Args:
            enhanced_data: 增强的股票数据
            
        Returns:
            新因子数据
        """
        factor_data = enhanced_data[['code', 'trade_date']].copy()

        # 机构参与度因子
        if 'institution_participation' in enhanced_data.columns:
            factor_data['institution_participation_factor'] = enhanced_data['institution_participation']
            factor_data['institution_buy_sell_ratio'] = (
                    enhanced_data['institution_buy_ratio'] /
                    (enhanced_data['institution_sell_ratio'] + 1e-8)
            )

        # 综合评分因子
        if 'comprehensive_score' in enhanced_data.columns:
            factor_data['comprehensive_score_factor'] = enhanced_data['comprehensive_score']
            factor_data['technical_score_factor'] = enhanced_data['technical_score']
            factor_data['fundamental_score_factor'] = enhanced_data['fundamental_score']

        # 市场关注度因子
        if 'market_attention' in enhanced_data.columns:
            factor_data['market_attention_factor'] = enhanced_data['market_attention']
            factor_data['attention_change_factor'] = enhanced_data['attention_change']

        # 筹码分布因子
        if 'chip_concentration' in enhanced_data.columns:
            factor_data['chip_concentration_factor'] = enhanced_data['chip_concentration']
            factor_data['profit_ratio_factor'] = enhanced_data['profit_ratio']

        # 估值因子（基于AKShare数据）
        if 'pe_akshare' in enhanced_data.columns:
            factor_data['pe_akshare_factor'] = 1 / (enhanced_data['pe_akshare'] + 1e-8)
            factor_data['pb_akshare_factor'] = 1 / (enhanced_data['pb_akshare'] + 1e-8)
            factor_data['dividend_yield_factor'] = enhanced_data['dividend_yield_akshare']

        # 市值因子
        if 'total_market_cap_akshare' in enhanced_data.columns:
            factor_data['market_cap_factor'] = np.log(enhanced_data['total_market_cap_akshare'])
            factor_data['circulating_market_cap_factor'] = np.log(enhanced_data['circulating_market_cap_akshare'])

        return factor_data
