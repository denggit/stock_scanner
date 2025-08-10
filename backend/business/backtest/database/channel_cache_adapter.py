#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道缓存适配器

该模块提供上升通道分析器的缓存包装器，实现：
- 透明的缓存机制
- 自动缓存未命中时的数据计算
- 批量数据预加载
- 与现有分析器的无缝集成

作者: AI Assistant
日期: 2024-12-20
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd

from backend.business.factor.core.engine.library.channel_analysis.rising_channel import AscendingChannelRegression
from backend.utils.logger import setup_logger
from .cache_manager import ChannelDataCache, CacheConfig


class ChannelCacheAdapter:
    """
    上升通道缓存适配器
    
    为上升通道分析器提供透明的缓存层，自动处理缓存命中/未命中的情况。
    """
    
    def __init__(self, 
                 cache_config: CacheConfig = None,
                 enable_batch_processing: bool = True,
                 enable_auto_cache: bool = True,
                 cache_batch_size: int = 500):
        """
        初始化缓存适配器
        
        Args:
            cache_config: 缓存配置
            enable_batch_processing: 是否启用批量处理
            enable_auto_cache: 是否启用自动缓存
            cache_batch_size: 批量更新缓存的大小，默认500只股票更新一次
        """
        self.cache = ChannelDataCache(cache_config)
        self.logger = setup_logger(__name__)
        self.enable_batch_processing = enable_batch_processing
        self.enable_auto_cache = enable_auto_cache
        self.cache_batch_size = cache_batch_size
        
        # 分析器实例缓存
        self._analyzer_cache: Dict[str, AscendingChannelRegression] = {}
        
        self.logger.info("上升通道缓存适配器初始化完成")
    
    def get_channel_history_data(self,
                                stock_data_dict: Dict[str, pd.DataFrame],
                                params: Dict[str, Any],
                                start_date: str = None,
                                end_date: str = None,
                                min_window_size: int = 60) -> Dict[str, pd.DataFrame]:
        """
        获取上升通道历史数据（批量处理，支持缓存）
        
        Args:
            stock_data_dict: 股票数据字典 {股票代码: DataFrame}
            params: 通道参数
            start_date: 开始日期
            end_date: 结束日期
            min_window_size: 最小窗口大小
            
        Returns:
            Dict[str, pd.DataFrame]: 通道历史数据 {股票代码: 通道DataFrame}
        """
        stock_codes = list(stock_data_dict.keys())
        
        # 智能分析缓存命中情况，考虑时间范围
        cache_analysis = self._analyze_cache_coverage(params, stock_codes, start_date, end_date)
        
        result_data = {}
        
        # 处理完全缓存命中的股票
        if cache_analysis['full_hits']:
            self.logger.info(f"完全缓存命中: {len(cache_analysis['full_hits'])} 只股票")
            for stock_code, stock_data in cache_analysis['full_hits'].items():
                df = pd.DataFrame(stock_data)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                result_data[stock_code] = df
        
        # 处理部分缓存命中的股票
        partial_hits_stocks = []
        if cache_analysis['partial_hits']:
            self.logger.info(f"部分缓存命中: {len(cache_analysis['partial_hits'])} 只股票")
            for stock_code, hit_info in cache_analysis['partial_hits'].items():
                partial_hits_stocks.append(stock_code)
                # 使用缓存的部分数据
                if hit_info['cached_data']:
                    cached_df = pd.DataFrame(hit_info['cached_data'])
                    cached_df['trade_date'] = pd.to_datetime(cached_df['trade_date'])
                    result_data[stock_code] = cached_df
        
        # 处理完全未命中的股票
        full_miss_stocks = list(cache_analysis['full_misses'])
        
        # 合并需要计算的股票（完全未命中 + 部分命中）
        stocks_to_compute = full_miss_stocks + partial_hits_stocks
        
        if stocks_to_compute:
            self.logger.info(f"需要计算通道数据的股票: {len(stocks_to_compute)} 只")
            
            # 计算新的通道数据
            new_channel_data = self._compute_channel_data_for_stocks(
                stocks_to_compute, stock_data_dict, params, 
                start_date, end_date, min_window_size, cache_analysis,
                self.cache_batch_size
            )
            
            # 合并计算结果到最终结果
            for stock_code, computed_df in new_channel_data.items():
                if stock_code in result_data:
                    # 部分命中的情况，需要合并数据
                    result_data[stock_code] = self._merge_channel_dataframes(
                        result_data[stock_code], computed_df
                    )
                else:
                    # 完全未命中的情况，直接使用计算结果
                    result_data[stock_code] = computed_df
        
        self.logger.info(f"最终获取到 {len(result_data)} 只股票的通道数据")
        return result_data
    
    def _analyze_cache_coverage(self, params: Dict[str, Any], stock_codes: List[str], 
                               start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        智能分析缓存覆盖情况
        
        Returns:
            Dict: 包含full_hits, partial_hits, full_misses的分析结果
        """
        # 获取所有缓存数据（不进行时间过滤）
        cached_data = self.cache.get_channel_data(params, stock_codes)
        
        analysis = {
            'full_hits': {},      # 完全命中（时间范围完全覆盖）
            'partial_hits': {},   # 部分命中（时间范围部分覆盖）
            'full_misses': set(stock_codes)  # 完全未命中
        }
        
        if cached_data is None or not cached_data['value']:
            return analysis
        
        # 将字符串日期转换为 pandas Timestamp 用于比较
        req_start = pd.to_datetime(start_date) if start_date else None
        req_end = pd.to_datetime(end_date) if end_date else None
        
        for stock_code in stock_codes:
            if stock_code not in cached_data['value']:
                continue
                
            stock_cache_data = cached_data['value'][stock_code]
            if not stock_cache_data:
                continue
            
            # 获取缓存数据的时间范围
            cache_dates = [pd.to_datetime(item['trade_date']) for item in stock_cache_data]
            cache_start = min(cache_dates)
            cache_end = max(cache_dates)
            
            # 分析覆盖情况
            coverage_type, filtered_data, missing_ranges = self._analyze_time_coverage(
                stock_cache_data, cache_start, cache_end, req_start, req_end
            )
            
            if coverage_type == 'full':
                analysis['full_hits'][stock_code] = filtered_data
                analysis['full_misses'].discard(stock_code)
            elif coverage_type == 'partial':
                analysis['partial_hits'][stock_code] = {
                    'cached_data': filtered_data,
                    'missing_ranges': missing_ranges,
                    'cache_start': cache_start,
                    'cache_end': cache_end
                }
                analysis['full_misses'].discard(stock_code)
        
        return analysis
    
    def _analyze_time_coverage(self, stock_cache_data: List[Dict], 
                              cache_start: pd.Timestamp, cache_end: pd.Timestamp,
                              req_start: pd.Timestamp = None, req_end: pd.Timestamp = None):
        """分析时间覆盖情况"""
        # 如果没有指定请求时间范围，返回所有缓存数据
        if req_start is None and req_end is None:
            return 'full', stock_cache_data, []
        
        # 设置默认值
        if req_start is None:
            req_start = cache_start
        if req_end is None:
            req_end = cache_end
        
        # 检查覆盖情况
        full_coverage = cache_start <= req_start and cache_end >= req_end
        no_coverage = cache_end < req_start or cache_start > req_end
        
        if no_coverage:
            return 'none', [], [(req_start, req_end)]
        
        if full_coverage:
            # 完全覆盖，过滤数据
            filtered_data = [
                item for item in stock_cache_data
                if req_start <= pd.to_datetime(item['trade_date']) <= req_end
            ]
            return 'full', filtered_data, []
        
        # 部分覆盖
        overlap_start = max(cache_start, req_start)
        overlap_end = min(cache_end, req_end)
        
        # 过滤重叠部分的数据
        filtered_data = [
            item for item in stock_cache_data
            if overlap_start <= pd.to_datetime(item['trade_date']) <= overlap_end
        ]
        
        # 计算缺失的时间范围
        missing_ranges = []
        if req_start < cache_start:
            # 确保时间类型一致性 - 都转换为pd.Timestamp
            missing_start = pd.to_datetime(req_start)
            missing_end = pd.to_datetime(cache_start - pd.Timedelta(days=1))
            missing_ranges.append((missing_start, missing_end))
        if req_end > cache_end:
            # 确保时间类型一致性 - 都转换为pd.Timestamp  
            missing_start = pd.to_datetime(cache_end + pd.Timedelta(days=1))
            missing_end = pd.to_datetime(req_end)
            missing_ranges.append((missing_start, missing_end))
        
        return 'partial', filtered_data, missing_ranges
    
    def _compute_channel_data_for_stocks(self, stocks_to_compute: List[str],
                                        stock_data_dict: Dict[str, pd.DataFrame],
                                        params: Dict[str, Any],
                                        start_date: str, end_date: str,
                                        min_window_size: int,
                                        cache_analysis: Dict[str, Any],
                                        batch_size: int = 500) -> Dict[str, pd.DataFrame]:
        """
        为指定股票计算通道数据
        
        Args:
            stocks_to_compute: 需要计算的股票代码列表
            stock_data_dict: 股票数据字典
            params: 通道参数
            start_date: 开始日期
            end_date: 结束日期
            min_window_size: 最小窗口大小
            cache_analysis: 缓存分析结果
            batch_size: 批量更新缓存的大小，默认500只股票更新一次
        """
        computed_data = {}
        new_cache_data = {}
        analyzer = self._get_analyzer(params)
        
        for i, stock_code in enumerate(stocks_to_compute, 1):
            try:
                if stock_code not in stock_data_dict:
                    self.logger.warning(f"股票 {stock_code} 的原始数据不存在")
                    continue
                
                stock_df = stock_data_dict[stock_code]
                if len(stock_df) < min_window_size:
                    self.logger.warning(f"股票 {stock_code} 数据不足，跳过 (需要 {min_window_size}，实际 {len(stock_df)})")
                    continue
                
                # 确保stock_df中的trade_date列是pd.Timestamp类型
                if 'trade_date' in stock_df.columns:
                    stock_df = stock_df.copy()
                    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
                
                # 对于部分命中的股票，只计算缺失的时间范围
                if stock_code in cache_analysis['partial_hits']:
                    computed_df = self._compute_missing_time_ranges(
                        stock_code, stock_df, analyzer, min_window_size,
                        cache_analysis['partial_hits'][stock_code]['missing_ranges']
                    )
                else:
                    # 完全未命中，计算整个时间范围
                    computed_df = analyzer.fit_channel_history_optimized(stock_df, min_window_size)
                
                if not computed_df.empty:
                    # 过滤掉无效数据（beta为空的记录）
                    valid_df = computed_df.dropna(subset=['beta'])
                    
                    if not valid_df.empty:
                        computed_data[stock_code] = valid_df
                        
                        # 准备缓存数据（只存储有效数据）
                        cache_records = valid_df.to_dict('records')
                        for record in cache_records:
                            if 'trade_date' in record and isinstance(record['trade_date'], pd.Timestamp):
                                record['trade_date'] = record['trade_date'].strftime('%Y-%m-%d')
                        
                        new_cache_data[stock_code] = cache_records
                    else:
                        self.logger.info(f"股票 {stock_code} 计算的通道数据全部无效，跳过缓存")
                
                # 每计算batch_size只股票就更新一次缓存，避免进程中断导致数据丢失
                if i % batch_size == 0 and new_cache_data and self.enable_auto_cache:
                    self.logger.info(f"批量更新缓存: 第 {i} 只股票，更新 {len(new_cache_data)} 只股票的计算数据")
                    self.cache.update_channel_data(params, new_cache_data)
                    # 清空已更新的缓存数据，避免重复更新
                    new_cache_data = {}
                
                if i % 50 == 0:
                    self.logger.info(f"已计算 {i}/{len(stocks_to_compute)} 只股票")
                    
            except Exception as e:
                self.logger.error(f"计算股票 {stock_code} 的通道数据失败: {e}")
                continue
        
        # 更新剩余的缓存数据
        if new_cache_data and self.enable_auto_cache:
            self.logger.info(f"最终更新缓存: {len(new_cache_data)} 只股票的新计算数据")
            self.cache.update_channel_data(params, new_cache_data)
        
        return computed_data
    
    def _compute_missing_time_ranges(self, stock_code: str, stock_df: pd.DataFrame,
                                   analyzer, min_window_size: int,
                                   missing_ranges: List[Tuple]) -> pd.DataFrame:
        """计算缺失时间范围的通道数据"""
        if not missing_ranges:
            return pd.DataFrame()
        
        all_computed = []
        
        for start_date, end_date in missing_ranges:
            # 确保时间类型一致性 - 转换为pd.Timestamp
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # 为了计算通道，需要包含足够的历史数据
            extended_start = start_date - pd.Timedelta(days=min_window_size + 30)
            
            # 确保stock_df中的trade_date也是pd.Timestamp类型
            if 'trade_date' in stock_df.columns:
                stock_df = stock_df.copy()
                stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
            
            # 过滤数据到指定时间范围
            mask = (stock_df['trade_date'] >= extended_start) & (stock_df['trade_date'] <= end_date)
            range_df = stock_df[mask].copy()
            
            if len(range_df) >= min_window_size:
                computed_df = analyzer.fit_channel_history_optimized(range_df, min_window_size)
                
                if not computed_df.empty:
                    # 确保computed_df中的trade_date也是pd.Timestamp类型
                    computed_df = computed_df.copy()
                    computed_df['trade_date'] = pd.to_datetime(computed_df['trade_date'])
                    
                    # 只保留请求时间范围内的数据
                    result_mask = (computed_df['trade_date'] >= start_date) & (computed_df['trade_date'] <= end_date)
                    result_df = computed_df[result_mask]
                    
                    if not result_df.empty:
                        all_computed.append(result_df)
        
        if all_computed:
            return pd.concat(all_computed, ignore_index=True).sort_values('trade_date')
        
        return pd.DataFrame()
    
    def _merge_channel_dataframes(self, cached_df: pd.DataFrame, computed_df: pd.DataFrame) -> pd.DataFrame:
        """合并缓存数据和计算数据"""
        if cached_df.empty:
            return computed_df
        if computed_df.empty:
            return cached_df
        
        # 合并数据并按日期排序
        merged_df = pd.concat([cached_df, computed_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['trade_date'], keep='last')
        merged_df = merged_df.sort_values('trade_date').reset_index(drop=True)
        
        return merged_df
    
    def get_single_channel_data(self,
                               stock_code: str,
                               stock_df: pd.DataFrame,
                               params: Dict[str, Any],
                               target_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取单只股票的通道数据
        
        Args:
            stock_code: 股票代码
            stock_df: 股票数据
            params: 通道参数
            target_date: 目标日期，None表示获取所有
            
        Returns:
            Optional[pd.DataFrame]: 通道数据
        """
        # 先尝试从缓存获取
        cached_data = self.cache.get_channel_data(params, [stock_code], target_date, target_date)
        
        if cached_data and stock_code in cached_data['value']:
            channel_records = cached_data['value'][stock_code]
            if channel_records:
                df = pd.DataFrame(channel_records)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df
        
        # 缓存未命中，计算数据
        self.logger.debug(f"为股票 {stock_code} 计算通道数据")
        
        try:
            analyzer = self._get_analyzer(params)
            channel_df = analyzer.fit_channel_history_optimized(stock_df)
            
            # 自动缓存
            if not channel_df.empty and self.enable_auto_cache:
                channel_records = channel_df.to_dict('records')
                for record in channel_records:
                    if 'trade_date' in record and isinstance(record['trade_date'], pd.Timestamp):
                        record['trade_date'] = record['trade_date'].strftime('%Y-%m-%d')
                
                self.cache.update_channel_data(params, {stock_code: channel_records})
            
            return channel_df
            
        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 的通道数据失败: {e}")
            return None
    
    def preload_cache_for_backtest(self,
                                  stock_data_dict: Dict[str, pd.DataFrame],
                                  params_list: List[Dict[str, Any]],
                                  start_date: str,
                                  end_date: str) -> bool:
        """
        为回测预加载缓存数据
        
        Args:
            stock_data_dict: 股票数据字典
            params_list: 参数组合列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        Returns:
            bool: 预加载是否成功
        """
        self.logger.info(f"开始为回测预加载缓存数据")
        self.logger.info(f"参数组合数: {len(params_list)}, 股票数: {len(stock_data_dict)}")
        
        try:
            for i, params in enumerate(params_list, 1):
                self.logger.info(f"预加载参数组合 {i}/{len(params_list)}")
                
                # 获取通道数据（会自动计算缺失的并缓存）
                channel_data = self.get_channel_history_data(
                    stock_data_dict, params, start_date, end_date
                )
                
                self.logger.info(f"参数组合 {i} 预加载完成，获取到 {len(channel_data)} 只股票数据")
            
            self.logger.info("所有参数组合预加载完成")
            return True
            
        except Exception as e:
            self.logger.error(f"预加载缓存数据失败: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.cache.get_cache_statistics()
    
    def clear_cache(self, params: Dict[str, Any] = None) -> bool:
        """清理缓存"""
        return self.cache.clear_cache(params)
    
    def _get_analyzer(self, params: Dict[str, Any]) -> AscendingChannelRegression:
        """获取分析器实例（带缓存）"""
        # 生成参数键
        param_key = str(sorted(params.items()))
        
        if param_key not in self._analyzer_cache:
            self._analyzer_cache[param_key] = AscendingChannelRegression(**params)
        
        return self._analyzer_cache[param_key]
    
    def validate_cache_consistency(self, 
                                  stock_code: str,
                                  stock_df: pd.DataFrame,
                                  params: Dict[str, Any],
                                  sample_dates: List[str] = None) -> Dict[str, bool]:
        """
        验证缓存数据一致性
        
        Args:
            stock_code: 股票代码
            stock_df: 股票数据
            params: 通道参数
            sample_dates: 抽样验证的日期列表
            
        Returns:
            Dict[str, bool]: 验证结果
        """
        results = {
            'cache_exists': False,
            'data_consistent': False,
            'sample_dates_match': False
        }
        
        try:
            # 检查缓存是否存在
            cached_data = self.cache.get_channel_data(params, [stock_code])
            if not cached_data or stock_code not in cached_data['value']:
                return results
            
            results['cache_exists'] = True
            
            # 重新计算数据进行对比
            analyzer = self._get_analyzer(params)
            fresh_df = analyzer.fit_channel_history_optimized(stock_df)
            
            if fresh_df.empty:
                return results
            
            # 比较数据一致性
            cached_df = pd.DataFrame(cached_data['value'][stock_code])
            cached_df['trade_date'] = pd.to_datetime(cached_df['trade_date'])
            
            # 检查数据量是否一致
            if len(cached_df) == len(fresh_df):
                results['data_consistent'] = True
            
            # 抽样验证特定日期
            if sample_dates and results['data_consistent']:
                matches = 0
                for date in sample_dates:
                    cached_row = cached_df[cached_df['trade_date'] == date]
                    fresh_row = fresh_df[fresh_df['trade_date'] == date]
                    
                    if not cached_row.empty and not fresh_row.empty:
                        # 比较关键字段
                        key_fields = ['beta', 'sigma', 'mid_today', 'upper_today', 'lower_today']
                        field_matches = 0
                        for field in key_fields:
                            if field in cached_row.columns and field in fresh_row.columns:
                                cached_val = cached_row.iloc[0][field]
                                fresh_val = fresh_row.iloc[0][field]
                                if abs(cached_val - fresh_val) < 1e-6:  # 浮点数比较
                                    field_matches += 1
                        
                        if field_matches == len(key_fields):
                            matches += 1
                
                results['sample_dates_match'] = matches == len(sample_dates)
            
        except Exception as e:
            self.logger.error(f"验证缓存一致性失败: {e}")
        
        return results
