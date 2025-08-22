#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : factor_engine.py
@Description: 因子计算引擎
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import traceback
import warnings
from datetime import datetime
from typing import List, Optional, Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.business.factor.core.data.data_manager import FactorDataManager
from backend.business.factor.core.config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_STOCK_POOL,
    DEFAULT_TOP_N, DEFAULT_N_GROUPS, DEFAULT_USE_PARALLEL, DEFAULT_MAX_WORKERS,
    DEFAULT_OPTIMIZE_DATA_FETCH
)
from backend.utils.logger import setup_logger
from .factor_registry import factor_registry

# 全局警告处理配置
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = setup_logger("backtest_factor")


def _calculate_single_factor_parallel(factor_name: str, 
                                     data: pd.DataFrame,
                                     start_date: str,
                                     end_date: str,
                                     stock_pool: str,
                                     top_n: int,
                                     n_groups: int,
                                     **kwargs) -> tuple:
    """
    并行计算单个因子的辅助函数
    
    Args:
        factor_name: 因子名称（将在此函数内部获取实际函数）
        data: 输入数据
        start_date: 开始日期
        end_date: 结束日期
        stock_pool: 股票池
        top_n: 选股数量
        n_groups: 分组数量
        **kwargs: 其他参数
        
    Returns:
        (factor_name, factor_values) 元组
    """
    try:
        # 在子进程中重新获取因子函数，避免pickle问题
        from .factor_registry import factor_registry
        factor_func = factor_registry.get_factor(factor_name)
        
        if factor_func is None:
            raise ValueError(f"因子 {factor_name} 未注册")
        
        # 确保数据按股票代码和交易日期排序
        data = data.sort_values(['code', 'trade_date']).reset_index(drop=True)
        
        def calculate_factor_for_group(group_data):
            """为单个股票组计算因子值"""
            try:
                # 获取股票代码用于错误日志
                code = group_data['code'].iloc[0] if 'code' in group_data.columns else 'unknown'
                
                # 准备因子函数的参数
                factor_kwargs = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'stock_pool': stock_pool,
                    'top_n': top_n,
                    'n_groups': n_groups,
                    **kwargs
                }
                
                # 调用因子函数，传入整个DataFrame
                stock_factor = factor_func(group_data, **factor_kwargs)
                
                # 确保返回的是Series，并且索引与输入数据对齐
                if isinstance(stock_factor, (int, float)):
                    stock_factor = pd.Series([stock_factor] * len(group_data), 
                                           index=group_data.index, 
                                           name=factor_name)
                elif isinstance(stock_factor, np.ndarray):
                    stock_factor = pd.Series(stock_factor, 
                                           index=group_data.index, 
                                           name=factor_name)
                elif isinstance(stock_factor, pd.Series):
                    # 确保索引对齐
                    if len(stock_factor) == len(group_data):
                        stock_factor.index = group_data.index
                        stock_factor.name = factor_name
                    else:
                        # 长度不匹配，用NaN填充
                        stock_factor = pd.Series([np.nan] * len(group_data), 
                                               index=group_data.index, 
                                               name=factor_name)
                else:
                    # 其他类型，尝试转换
                    try:
                        stock_factor = pd.Series(stock_factor, 
                                               index=group_data.index, 
                                               name=factor_name)
                    except:
                        stock_factor = pd.Series([np.nan] * len(group_data), 
                                               index=group_data.index, 
                                               name=factor_name)
                
                return stock_factor
                
            except Exception as e:
                # 捕获所有异常，返回NaN序列
                code = group_data['code'].iloc[0] if 'code' in group_data.columns and len(group_data) > 0 else 'unknown'
                return pd.Series([np.nan] * len(group_data), 
                               index=group_data.index, 
                               name=factor_name)
        
        # 使用向量化的groupby().apply()计算所有股票的因子值
        factor_result = data.groupby('code', group_keys=False).apply(calculate_factor_for_group)
        
        # 处理可能的多级索引
        if isinstance(factor_result.index, pd.MultiIndex):
            factor_result = factor_result.droplevel(0)
        
        # 确保结果与原始数据长度一致
        if len(factor_result) != len(data):
            # 重建结果，确保长度一致
            result = pd.Series([np.nan] * len(data), index=data.index, name=factor_name)
            if len(factor_result) > 0:
                # 尽可能对齐数据
                common_indices = result.index.intersection(factor_result.index)
                if len(common_indices) > 0:
                    result.loc[common_indices] = factor_result.loc[common_indices]
            factor_result = result
        
        return factor_name, factor_result
        
    except Exception as e:
        # 返回错误信息
        return factor_name, pd.Series([np.nan] * len(data), index=data.index, name=factor_name)


def ensure_pandas_series(data: Any, index: Optional[pd.Index] = None) -> pd.Series:
    """
    确保数据是pandas Series类型
    
    Args:
        data: 输入数据
        index: 索引
        
    Returns:
        pandas Series
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, np.ndarray):
        return pd.Series(data, index=index)
    elif isinstance(data, (list, tuple)):
        return pd.Series(data, index=index)
    elif isinstance(data, (int, float)):
        if index is not None:
            return pd.Series([data] * len(index), index=index)
        else:
            return pd.Series([data])
    else:
        # 其他类型，尝试转换为Series
        try:
            return pd.Series(data, index=index)
        except:
            # 如果转换失败，返回空Series
            if index is not None:
                return pd.Series([np.nan] * len(index), index=index)
            else:
                return pd.Series([])


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
                         start_date: str = DEFAULT_START_DATE,
                         end_date: str = None,
                         stock_pool: str = DEFAULT_STOCK_POOL,
                         top_n: int = DEFAULT_TOP_N,
                         n_groups: int = DEFAULT_N_GROUPS,
                         optimize_data_fetch: bool = DEFAULT_OPTIMIZE_DATA_FETCH,
                         use_parallel: bool = DEFAULT_USE_PARALLEL,
                         max_workers: Optional[int] = DEFAULT_MAX_WORKERS,
                         **kwargs) -> pd.DataFrame:
        """
        批量计算因子值
        
        Args:
            factor_names: 因子名称列表
            data: 输入数据，None表示使用数据管理器的数据
            start_date: 开始日期，格式为YYYY-MM-DD，默认为2025-01-01
            end_date: 结束日期，格式为YYYY-MM-DD，默认为当天
            stock_pool: 股票池，默认为'no_st'
            top_n: 选股数量，默认为10
            n_groups: 分组数量，默认为5
            optimize_data_fetch: 是否优化数据获取，默认为True
            use_parallel: 是否使用并行计算，默认为True
            max_workers: 最大工作进程数，None表示使用CPU核心数
            **kwargs: 其他参数
            
        Returns:
            因子值DataFrame
        """
        logger.info(f"开始计算因子: {factor_names}")

        # 处理日期参数
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 验证日期格式
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"日期格式错误: {e}")

        # 验证其他参数
        if top_n <= 0:
            raise ValueError("top_n必须大于0")
        if n_groups <= 0:
            raise ValueError("n_groups必须大于0")



        logger.info(
            f"参数设置: start_date={start_date}, end_date={end_date}, stock_pool={stock_pool}, top_n={top_n}, n_groups={n_groups}")

        if data is None:
            if self.data_manager._processed_data is None:
                logger.info("数据未准备，开始智能数据获取...")
                # 使用智能数据获取准备数据
                self.data_manager.prepare_factor_data(
                    start_date=start_date,
                    end_date=end_date,
                    stock_pool=stock_pool,
                    factor_names=factor_names,
                    optimize_data_fetch=optimize_data_fetch
                )
                data = self.data_manager._processed_data.copy()
            else:
                data = self.data_manager._processed_data.copy()

        # 获取注册的因子并检查是否存在
        missing_factors = []
        factor_funcs = {}
        
        for factor_name in factor_names:
            factor_func = factor_registry.get_factor(factor_name)
            if factor_func is None:
                missing_factors.append(factor_name)
            else:
                factor_funcs[factor_name] = factor_func
        
        if missing_factors:
            raise ValueError(f"以下因子未注册: {missing_factors}")
        
        if not factor_funcs:
            raise ValueError("没有可计算的因子")
        
        factor_results = {}
        
        # 确定工作进程数
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(factor_funcs))
        else:
            max_workers = min(max_workers, len(factor_funcs))
        
        if use_parallel and max_workers > 1:
            # 并行计算模式
            logger.info(f"使用并行计算模式，工作进程数: {max_workers}")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_factor = {}
                for factor_name in factor_funcs.keys():
                    future = executor.submit(
                        _calculate_single_factor_parallel,
                        factor_name,  # 只传递因子名称，避免pickle问题
                        data,
                        start_date,
                        end_date,
                        stock_pool,
                        top_n,
                        n_groups,
                        **kwargs
                    )
                    future_to_factor[future] = factor_name
                
                # 收集结果
                with tqdm(total=len(factor_funcs), desc="并行计算因子") as pbar:
                    for future in as_completed(future_to_factor):
                        factor_name = future_to_factor[future]
                        try:
                            result_factor_name, factor_values = future.result()
                            factor_results[result_factor_name] = factor_values
                            logger.debug(f"因子 {result_factor_name} 计算完成")
                        except Exception as e:
                            logger.error(f"计算因子 {factor_name} 失败: {e}")
                            # 创建空的因子结果
                            factor_results[factor_name] = pd.Series([np.nan] * len(data), 
                                                                   index=data.index, 
                                                                   name=factor_name)
                        pbar.update(1)
        else:
            # 串行计算模式
            logger.info("使用串行计算模式")
            
            for factor_name, factor_func in tqdm(factor_funcs.items(), desc="计算因子"):
                try:
                    factor_values = self._calculate_single_factor(factor_func, data, start_date, end_date, stock_pool,
                                                                  top_n, n_groups, **kwargs)
                    factor_results[factor_name] = factor_values
                    logger.debug(f"因子 {factor_name} 计算完成")
                except Exception as e:
                    logger.error(f"计算因子 {factor_name} 失败: {e}")
                    # 创建空的因子结果
                    factor_results[factor_name] = pd.Series([np.nan] * len(data), 
                                                           index=data.index, 
                                                           name=factor_name)

        # 合并因子结果
        if factor_results:
            factor_df = pd.DataFrame(factor_results)
            factor_df['code'] = data['code'].values
            factor_df['trade_date'] = data['trade_date'].values

            # 确保因子列按照原始factor_names的顺序排列
            factor_cols = [name for name in factor_names if name in factor_results.keys()]
            cols = ['code', 'trade_date'] + factor_cols
            factor_df = factor_df[cols]

            self._factor_data = factor_df
            logger.info(f"因子计算完成，共 {len(factor_df)} 条记录")

            return factor_df
        else:
            raise ValueError("没有成功计算任何因子")

    def _calculate_single_factor(self,
                                 factor_func: Callable,
                                 data: pd.DataFrame,
                                 start_date: str,
                                 end_date: str,
                                 stock_pool: str,
                                 top_n: int,
                                 n_groups: int,
                                 **kwargs) -> pd.Series:
        """
        计算单个因子值（向量化版本）
        
        Args:
            factor_func: 因子计算函数
            data: 输入数据
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            top_n: 选股数量
            n_groups: 分组数量
            **kwargs: 其他参数
            
        Returns:
            因子值序列
        """
        # 获取因子名称用于错误日志
        factor_name = factor_func.__name__ if hasattr(factor_func, '__name__') else 'unknown_factor'
        
        # 确保数据按股票代码和交易日期排序
        data = data.sort_values(['code', 'trade_date']).reset_index(drop=True)
        
        def calculate_factor_for_group(group_data):
            """
            为单个股票组计算因子值
            
            Args:
                group_data: 单个股票的历史数据 DataFrame
                
            Returns:
                因子值序列
            """
            try:
                # 获取股票代码用于错误日志
                code = group_data['code'].iloc[0] if 'code' in group_data.columns else 'unknown'
                
                # 准备因子函数的参数
                factor_kwargs = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'stock_pool': stock_pool,
                    'top_n': top_n,
                    'n_groups': n_groups,
                    **kwargs
                }
                
                # 调用因子函数，传入整个DataFrame
                stock_factor = factor_func(group_data, **factor_kwargs)
                
                # 确保返回的是Series，并且索引与输入数据对齐
                if isinstance(stock_factor, (int, float)):
                    stock_factor = pd.Series([stock_factor] * len(group_data), 
                                           index=group_data.index, 
                                           name=factor_name)
                elif isinstance(stock_factor, np.ndarray):
                    stock_factor = pd.Series(stock_factor, 
                                           index=group_data.index, 
                                           name=factor_name)
                elif isinstance(stock_factor, pd.Series):
                    # 确保索引对齐
                    if len(stock_factor) == len(group_data):
                        stock_factor.index = group_data.index
                        stock_factor.name = factor_name
                    else:
                        # 长度不匹配，用NaN填充
                        logger.warning(f"股票 {code} 的因子 {factor_name} 计算结果长度不匹配，使用NaN填充")
                        stock_factor = pd.Series([np.nan] * len(group_data), 
                                               index=group_data.index, 
                                               name=factor_name)
                else:
                    # 其他类型，尝试转换
                    try:
                        stock_factor = pd.Series(stock_factor, 
                                               index=group_data.index, 
                                               name=factor_name)
                    except:
                        stock_factor = pd.Series([np.nan] * len(group_data), 
                                               index=group_data.index, 
                                               name=factor_name)
                
                return stock_factor
                
            except (ValueError, KeyError, ZeroDivisionError, TypeError, IndexError) as e:
                # 捕获具体的常见异常
                code = group_data['code'].iloc[0] if 'code' in group_data.columns and len(group_data) > 0 else 'unknown'
                error_msg = f"计算股票 {code} 的因子 {factor_name} 时发生 {type(e).__name__} 错误: {str(e)}"
                logger.error(f"{error_msg}\n完整错误堆栈:\n{traceback.format_exc()}")
                # 填充NaN并继续执行
                return pd.Series([np.nan] * len(group_data), 
                               index=group_data.index, 
                               name=factor_name)
                
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                # 捕获pandas相关异常
                code = group_data['code'].iloc[0] if 'code' in group_data.columns and len(group_data) > 0 else 'unknown'
                error_msg = f"计算股票 {code} 的因子 {factor_name} 时发生pandas数据错误: {str(e)}"
                logger.error(f"{error_msg}\n完整错误堆栈:\n{traceback.format_exc()}")
                # 填充NaN并继续执行
                return pd.Series([np.nan] * len(group_data), 
                               index=group_data.index, 
                               name=factor_name)
                
            except (np.linalg.LinAlgError, np.core._exceptions._UFuncNoLoopError) as e:
                # 捕获numpy相关异常
                code = group_data['code'].iloc[0] if 'code' in group_data.columns and len(group_data) > 0 else 'unknown'
                error_msg = f"计算股票 {code} 的因子 {factor_name} 时发生numpy计算错误: {str(e)}"
                logger.error(f"{error_msg}\n完整错误堆栈:\n{traceback.format_exc()}")
                # 填充NaN并继续执行
                return pd.Series([np.nan] * len(group_data), 
                               index=group_data.index, 
                               name=factor_name)
                
            except Exception as e:
                # 捕获其他未预期的异常
                code = group_data['code'].iloc[0] if 'code' in group_data.columns and len(group_data) > 0 else 'unknown'
                error_msg = f"计算股票 {code} 的因子 {factor_name} 时发生未预期的错误 {type(e).__name__}: {str(e)}"
                logger.error(f"{error_msg}\n完整错误堆栈:\n{traceback.format_exc()}")
                # 填充NaN并继续执行
                return pd.Series([np.nan] * len(group_data), 
                               index=group_data.index, 
                               name=factor_name)
        
        try:
            # 使用向量化的groupby().apply()计算所有股票的因子值
            logger.debug(f"开始向量化计算因子 {factor_name}，共 {data['code'].nunique()} 只股票")
            factor_result = data.groupby('code', group_keys=False).apply(calculate_factor_for_group)
            
            # 处理可能的多级索引
            if isinstance(factor_result.index, pd.MultiIndex):
                # 如果是多级索引，取第二级（原始行索引）
                factor_result = factor_result.droplevel(0)
            
            # 确保结果与原始数据长度一致
            if len(factor_result) != len(data):
                logger.warning(f"因子 {factor_name} 计算结果长度 {len(factor_result)} 与原始数据长度 {len(data)} 不一致")
                # 重建结果，确保长度一致
                result = pd.Series([np.nan] * len(data), index=data.index, name=factor_name)
                if len(factor_result) > 0:
                    # 尽可能对齐数据
                    common_indices = result.index.intersection(factor_result.index)
                    if len(common_indices) > 0:
                        result.loc[common_indices] = factor_result.loc[common_indices]
                factor_result = result
            
            logger.debug(f"因子 {factor_name} 向量化计算完成，结果长度: {len(factor_result)}")
            return factor_result
            
        except Exception as e:
            logger.error(f"向量化计算因子 {factor_name} 失败: {e}\n完整错误堆栈:\n{traceback.format_exc()}")
            # 返回全NaN的Series作为fallback
            return pd.Series([np.nan] * len(data), index=data.index, name=factor_name)

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
