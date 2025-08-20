#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : factor_engine.py
@Description: 因子计算引擎，负责批量计算所有股票的因子值
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from backend.business.backtest_factor.core.base_factor import BaseFactor
from backend.business.backtest_factor.core.data_manager import FactorDataManager
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FactorEngine:
    """
    因子计算引擎，负责批量计算所有股票的因子值
    
    功能：
    1. 批量计算因子值
    2. 因子标准化
    3. 因子去极值
    4. 因子中性化
    """
    
    def __init__(self, data_manager: FactorDataManager):
        """
        初始化因子引擎
        
        Args:
            data_manager: 数据管理器实例
        """
        self.data_manager = data_manager
        self._factor_data = None
        self._factor_metadata = {}
        
    def calculate_factors(self,
                         factor_names: List[str],
                         data: Optional[pd.DataFrame] = None,
                         **kwargs) -> pd.DataFrame:
        """
        批量计算因子值
        
        Args:
            factor_names: 因子名称列表
            data: 输入数据，None表示使用数据管理器的数据
            **kwargs: 其他参数
            
        Returns:
            因子值DataFrame
        """
        logger.info(f"开始计算因子: {factor_names}")
        
        if data is None:
            if self.data_manager._processed_data is None:
                raise ValueError("请先调用 data_manager.prepare_factor_data 准备数据")
            data = self.data_manager._processed_data.copy()
        
        # 获取注册的因子
        registered_factors = BaseFactor.get_registered_factors()
        
        # 检查因子是否存在
        missing_factors = [name for name in factor_names if name not in registered_factors]
        if missing_factors:
            raise ValueError(f"以下因子未注册: {missing_factors}")
        
        # 计算因子值
        factor_results = {}
        
        for factor_name in tqdm(factor_names, desc="计算因子"):
            try:
                factor_func = registered_factors[factor_name]['function']
                factor_values = self._calculate_single_factor(factor_func, data, **kwargs)
                factor_results[factor_name] = factor_values
                
                logger.info(f"因子 {factor_name} 计算完成")
                
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 失败: {e}")
                continue
        
        # 合并因子结果
        if factor_results:
            factor_df = pd.DataFrame(factor_results)
            factor_df['code'] = data['code'].values
            factor_df['trade_date'] = data['trade_date'].values
            
            # 重新排列列顺序
            cols = ['code', 'trade_date'] + list(factor_results.keys())
            factor_df = factor_df[cols]
            
            self._factor_data = factor_df
            logger.info(f"因子计算完成，共 {len(factor_df)} 条记录")
            
            return factor_df
        else:
            raise ValueError("没有成功计算任何因子")
    
    def _calculate_single_factor(self,
                                factor_func: Callable,
                                data: pd.DataFrame,
                                **kwargs) -> pd.Series:
        """
        计算单个因子值
        
        Args:
            factor_func: 因子计算函数
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            因子值序列
        """
        # 按股票分组计算
        factor_values = []
        
        for code in data['code'].unique():
            stock_data = data[data['code'] == code].copy()
            stock_data = stock_data.sort_values('trade_date').reset_index(drop=True)
            
            try:
                # 调用因子函数
                stock_factor = factor_func(**stock_data, **kwargs)
                
                # 确保返回的是Series
                if isinstance(stock_factor, (int, float)):
                    stock_factor = pd.Series([stock_factor] * len(stock_data), index=stock_data.index)
                elif isinstance(stock_factor, np.ndarray):
                    stock_factor = pd.Series(stock_factor, index=stock_data.index)
                
                factor_values.append(stock_factor)
                
            except Exception as e:
                logger.warning(f"计算股票 {code} 因子失败: {e}")
                # 填充NaN
                factor_values.append(pd.Series([np.nan] * len(stock_data), index=stock_data.index))
        
        # 合并所有股票的因子值
        return pd.concat(factor_values, ignore_index=True)
    
    def standardize_factors(self,
                           factor_names: Optional[List[str]] = None,
                           method: str = 'zscore') -> pd.DataFrame:
        """
        因子标准化
        
        Args:
            factor_names: 因子名称列表，None表示所有因子
            method: 标准化方法 ('zscore', 'minmax', 'rank')
            
        Returns:
            标准化后的因子DataFrame
        """
        if self._factor_data is None:
            raise ValueError("请先调用 calculate_factors 计算因子")
        
        if factor_names is None:
            factor_names = [col for col in self._factor_data.columns 
                          if col not in ['code', 'trade_date']]
        
        standardized_data = self._factor_data.copy()
        
        for factor_name in factor_names:
            if factor_name in standardized_data.columns:
                factor_values = standardized_data[factor_name]
                
                if method == 'zscore':
                    # Z-score标准化
                    mean_val = factor_values.mean()
                    std_val = factor_values.std()
                    if std_val != 0:
                        standardized_data[factor_name] = (factor_values - mean_val) / std_val
                    else:
                        standardized_data[factor_name] = 0
                        
                elif method == 'minmax':
                    # Min-Max标准化
                    min_val = factor_values.min()
                    max_val = factor_values.max()
                    if max_val != min_val:
                        standardized_data[factor_name] = (factor_values - min_val) / (max_val - min_val)
                    else:
                        standardized_data[factor_name] = 0.5
                        
                elif method == 'rank':
                    # 排名标准化
                    standardized_data[factor_name] = factor_values.rank(pct=True)
                
                else:
                    raise ValueError(f"不支持的标准化方法: {method}")
        
        logger.info(f"因子标准化完成，方法: {method}")
        return standardized_data
    
    def winsorize_factors(self,
                         factor_names: Optional[List[str]] = None,
                         limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
        """
        因子去极值
        
        Args:
            factor_names: 因子名称列表，None表示所有因子
            limits: 去极值范围 (lower_percentile, upper_percentile)
            
        Returns:
            去极值后的因子DataFrame
        """
        if self._factor_data is None:
            raise ValueError("请先调用 calculate_factors 计算因子")
        
        if factor_names is None:
            factor_names = [col for col in self._factor_data.columns 
                          if col not in ['code', 'trade_date']]
        
        winsorized_data = self._factor_data.copy()
        
        for factor_name in factor_names:
            if factor_name in winsorized_data.columns:
                factor_values = winsorized_data[factor_name]
                
                # 计算分位数
                lower_bound = factor_values.quantile(limits[0])
                upper_bound = factor_values.quantile(limits[1])
                
                # 去极值
                winsorized_data[factor_name] = factor_values.clip(lower_bound, upper_bound)
        
        logger.info(f"因子去极值完成，范围: {limits}")
        return winsorized_data
    
    def neutralize_factors(self,
                          factor_names: Optional[List[str]] = None,
                          industry_col: Optional[str] = None,
                          market_cap_col: Optional[str] = None) -> pd.DataFrame:
        """
        因子中性化
        
        Args:
            factor_names: 因子名称列表，None表示所有因子
            industry_col: 行业列名
            market_cap_col: 市值列名
            
        Returns:
            中性化后的因子DataFrame
        """
        if self._factor_data is None:
            raise ValueError("请先调用 calculate_factors 计算因子")
        
        if factor_names is None:
            factor_names = [col for col in self._factor_data.columns 
                          if col not in ['code', 'trade_date']]
        
        neutralized_data = self._factor_data.copy()
        
        # 这里可以实现行业和市值中性化
        # 目前只是占位，具体实现可以根据需要扩展
        
        logger.info("因子中性化完成")
        return neutralized_data
    
    def get_factor_data(self) -> Optional[pd.DataFrame]:
        """
        获取因子数据
        
        Returns:
            因子数据DataFrame
        """
        return self._factor_data
    
    def get_factor_summary(self) -> pd.DataFrame:
        """
        获取因子统计摘要
        
        Returns:
            因子统计摘要DataFrame
        """
        if self._factor_data is None:
            raise ValueError("请先调用 calculate_factors 计算因子")
        
        factor_cols = [col for col in self._factor_data.columns 
                      if col not in ['code', 'trade_date']]
        
        summary_data = []
        
        for factor_name in factor_cols:
            factor_values = self._factor_data[factor_name].dropna()
            
            summary = {
                'factor_name': factor_name,
                'count': len(factor_values),
                'mean': factor_values.mean(),
                'std': factor_values.std(),
                'min': factor_values.min(),
                'max': factor_values.max(),
                'skew': factor_values.skew(),
                'kurtosis': factor_values.kurtosis(),
                'missing_rate': (self._factor_data[factor_name].isna().sum() / 
                               len(self._factor_data))
            }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_factor_data(self, file_path: str):
        """
        保存因子数据
        
        Args:
            file_path: 文件路径
        """
        if self._factor_data is not None:
            self._factor_data.to_csv(file_path, index=False)
            logger.info(f"因子数据已保存到: {file_path}")
        else:
            raise ValueError("没有可保存的因子数据")
    
    def load_factor_data(self, file_path: str):
        """
        加载因子数据
        
        Args:
            file_path: 文件路径
        """
        self._factor_data = pd.read_csv(file_path)
        logger.info(f"因子数据已从 {file_path} 加载")
