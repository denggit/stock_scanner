#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据管理器
负责多股票数据的管理、缓存和获取
使用观察者模式通知数据更新
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

import pandas as pd


class DataObserver(ABC):
    """
    数据观察者接口
    用于观察者模式，当数据更新时通知观察者
    """
    
    @abstractmethod
    def on_data_updated(self, stock_code: str, data: pd.DataFrame):
        """
        数据更新通知
        
        Args:
            stock_code: 股票代码
            data: 更新的数据
        """
        pass


class DataCache:
    """
    数据缓存类
    管理历史数据的缓存和清理
    """
    
    def __init__(self, max_cache_size: int = 1000):
        """
        初始化数据缓存
        
        Args:
            max_cache_size: 最大缓存大小（天数）
        """
        self.max_cache_size = max_cache_size
        self._cache = {}  # {股票代码: 数据列表}
        self.logger = logging.getLogger("backtest")
    
    def add_data(self, stock_code: str, data_record: Dict[str, Any]):
        """
        添加数据记录
        
        Args:
            stock_code: 股票代码
            data_record: 数据记录
        """
        if stock_code not in self._cache:
            self._cache[stock_code] = []
        
        self._cache[stock_code].append(data_record)
        
        # 清理过期数据
        self._cleanup_cache(stock_code)
    
    def get_data(self, stock_code: str, days: int = None) -> List[Dict[str, Any]]:
        """
        获取缓存数据
        
        Args:
            stock_code: 股票代码
            days: 获取天数（None表示全部）
            
        Returns:
            数据记录列表
        """
        if stock_code not in self._cache:
            return []
        
        data = self._cache[stock_code]
        
        if days is None:
            return data.copy()
        else:
            return data[-days:] if len(data) >= days else data.copy()
    
    def _cleanup_cache(self, stock_code: str):
        """清理缓存，保持在最大大小限制内"""
        if stock_code in self._cache:
            cache_data = self._cache[stock_code]
            if len(cache_data) > self.max_cache_size:
                # 保留最新的数据
                self._cache[stock_code] = cache_data[-self.max_cache_size:]
                self.logger.debug(f"清理 {stock_code} 的缓存数据")
    
    def clear_cache(self, stock_code: str = None):
        """
        清理缓存
        
        Args:
            stock_code: 股票代码，None表示清理所有
        """
        if stock_code:
            if stock_code in self._cache:
                del self._cache[stock_code]
        else:
            self._cache.clear()


class DataManager:
    """
    数据管理器
    
    功能：
    - 管理多股票数据
    - 提供数据查询接口
    - 缓存历史数据
    - 数据预处理和验证
    """

    def __init__(self, cache_size: int = 1000):
        """
        初始化数据管理器
        
        Args:
            cache_size: 缓存大小
        """
        # 原始股票数据 {股票代码: DataFrame}
        self.stock_data = {}
        
        # 股票代码列表
        self.stock_codes = []
        
        # 数据缓存
        self.cache = DataCache(cache_size)
        
        # 观察者列表
        self.observers = []
        
        # 日志记录器
        self.logger = logging.getLogger("backtest")
        
        # 数据统计
        self.data_stats = {
            'total_stocks': 0,
            'data_date_range': {},
            'missing_data_count': 0,
            'invalid_data_count': 0
        }

    def set_stock_data(self, stock_data_dict: Dict[str, pd.DataFrame]):
        """
        设置股票数据
        
        Args:
            stock_data_dict: 股票数据字典 {股票代码: DataFrame}
        """
        self.stock_data = stock_data_dict.copy()
        self.stock_codes = list(stock_data_dict.keys())
        
        # 验证和预处理数据
        self._validate_and_preprocess_data()
        
        # 更新统计信息
        self._update_data_statistics()
        
        self.logger.info(f"设置股票数据: {len(self.stock_codes)} 只股票")

    def add_observer(self, observer: DataObserver):
        """
        添加数据观察者
        
        Args:
            observer: 数据观察者
        """
        if observer not in self.observers:
            self.observers.append(observer)

    def remove_observer(self, observer: DataObserver):
        """
        移除数据观察者
        
        Args:
            observer: 数据观察者
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def _notify_observers(self, stock_code: str, data: pd.DataFrame):
        """
        通知所有观察者数据更新
        
        Args:
            stock_code: 股票代码
            data: 更新的数据
        """
        for observer in self.observers:
            try:
                observer.on_data_updated(stock_code, data)
            except Exception as e:
                self.logger.error(f"通知观察者失败: {e}")

    def get_stock_price(self, stock_code: str, date: datetime) -> float:
        """
        获取指定股票在指定日期的价格
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            股票价格，获取失败返回0.0
        """
        if stock_code not in self.stock_data:
            return 0.0

        stock_df = self.stock_data[stock_code]

        # 确保date是datetime类型
        if isinstance(date, datetime):
            target_date = date
        elif isinstance(date, pd.Timestamp):
            target_date = date.to_pydatetime()
        else:
            target_date = pd.to_datetime(date).to_pydatetime()

        # 精确匹配
        current_data = stock_df[stock_df['trade_date'] == target_date.date()]
        if not current_data.empty:
            return float(current_data.iloc[0]['close'])

        # 如果精确匹配失败，获取最近的日期
        if len(stock_df) > 0:
            return float(stock_df.iloc[-1]['close'])

        return 0.0

    def get_stock_data_until(self, stock_code: str, date: datetime, 
                           min_data_points: int = 60) -> Optional[pd.DataFrame]:
        """
        获取指定股票截至指定日期的历史数据
        
        Args:
            stock_code: 股票代码
            date: 截止日期
            min_data_points: 最小数据点数
            
        Returns:
            历史数据DataFrame，数据不足返回None
        """
        if stock_code not in self.stock_data:
            return None

        stock_df = self.stock_data[stock_code].copy()

        # 确保date是datetime类型
        if isinstance(date, datetime):
            target_datetime = date
        elif isinstance(date, pd.Timestamp):
            target_datetime = date.to_pydatetime()
        else:
            target_datetime = pd.to_datetime(date).to_pydatetime()

        # 过滤到指定日期之前的数据
        filtered_df = stock_df[stock_df['trade_date'] <= target_datetime.date()].copy()

        if len(filtered_df) < min_data_points:
            self.logger.debug(f"股票 {stock_code} 数据不足: {len(filtered_df)} < {min_data_points}")
            return None

        return filtered_df

    def get_all_stock_prices(self, date: datetime) -> Dict[str, float]:
        """
        获取所有股票在指定日期的价格
        
        Args:
            date: 日期
            
        Returns:
            股票价格字典 {股票代码: 价格}
        """
        prices = {}
        for stock_code in self.stock_codes:
            price = self.get_stock_price(stock_code, date)
            if price > 0:
                prices[stock_code] = price
        
        return prices

    def record_current_data(self, stock_code: str, date: datetime, 
                          ohlcv_data: Dict[str, float]):
        """
        记录当前交易日数据
        
        Args:
            stock_code: 股票代码
            date: 交易日期
            ohlcv_data: OHLCV数据字典
        """
        # 确保时间格式一致
        if isinstance(date, datetime):
            trade_date = date
        else:
            trade_date = datetime.combine(date, datetime.min.time())

        data_record = {
            'trade_date': trade_date,
            'open': ohlcv_data.get('open', 0.0),
            'high': ohlcv_data.get('high', 0.0),
            'low': ohlcv_data.get('low', 0.0),
            'close': ohlcv_data.get('close', 0.0),
            'volume': ohlcv_data.get('volume', 0)
        }

        # 添加到缓存
        self.cache.add_data(stock_code, data_record)

    def get_cached_data(self, stock_code: str, days: int = None) -> List[Dict[str, Any]]:
        """
        获取缓存的历史数据
        
        Args:
            stock_code: 股票代码
            days: 获取天数
            
        Returns:
            历史数据记录列表
        """
        return self.cache.get_data(stock_code, days)

    def _validate_and_preprocess_data(self):
        """验证和预处理数据"""
        invalid_stocks = []
        
        for stock_code, data in self.stock_data.items():
            try:
                # 检查必需的列
                required_columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = []
                
                for col in required_columns:
                    if col not in data.columns:
                        missing_columns.append(col)
                
                if missing_columns:
                    self.logger.warning(f"股票 {stock_code} 缺少列: {missing_columns}")
                    # 尝试填充缺失列
                    for col in missing_columns:
                        if col == 'volume':
                            data[col] = 1000000  # 默认成交量
                        else:
                            data[col] = data.get('close', 0)  # 使用收盘价填充
                
                # 检查数据质量
                null_counts = data[required_columns].isnull().sum()
                if null_counts.any():
                    self.logger.warning(f"股票 {stock_code} 存在空值: {null_counts.to_dict()}")
                    # 前向填充空值（使用更兼容的方法）
                    data[required_columns] = data[required_columns].fillna(method='ffill').fillna(method='bfill')
                
                # 检查价格数据合理性
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if (data[col] <= 0).any():
                        self.logger.warning(f"股票 {stock_code} 存在非正价格数据")
                        # 移除非正价格的行
                        data = data[data[col] > 0]
                
                # 更新处理后的数据
                self.stock_data[stock_code] = data
                
            except Exception as e:
                self.logger.error(f"处理股票 {stock_code} 数据失败: {e}")
                invalid_stocks.append(stock_code)
        
        # 移除无效股票
        for stock_code in invalid_stocks:
            del self.stock_data[stock_code]
            if stock_code in self.stock_codes:
                self.stock_codes.remove(stock_code)
            self.data_stats['invalid_data_count'] += 1

    def _update_data_statistics(self):
        """更新数据统计信息"""
        self.data_stats['total_stocks'] = len(self.stock_codes)
        
        # 计算日期范围
        date_ranges = {}
        for stock_code, data in self.stock_data.items():
            if not data.empty:
                min_date = data['trade_date'].min()
                max_date = data['trade_date'].max()
                date_ranges[stock_code] = {
                    'start_date': min_date,
                    'end_date': max_date,
                    'data_points': len(data)
                }
        
        self.data_stats['data_date_range'] = date_ranges

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            数据统计字典
        """
        stats = self.data_stats.copy()
        
        # 添加当前状态
        stats['cache_status'] = {
            'cached_stocks': len(self.cache._cache),
            'total_cache_size': sum(len(data) for data in self.cache._cache.values())
        }
        
        return stats

    def validate_data_availability(self, date: datetime, 
                                 required_stocks: List[str] = None) -> Dict[str, bool]:
        """
        验证指定日期的数据可用性
        
        Args:
            date: 目标日期
            required_stocks: 需要检查的股票列表，None表示检查所有股票
            
        Returns:
            数据可用性字典 {股票代码: 是否可用}
        """
        if required_stocks is None:
            required_stocks = self.stock_codes
        
        availability = {}
        
        for stock_code in required_stocks:
            price = self.get_stock_price(stock_code, date)
            availability[stock_code] = price > 0
        
        return availability

    def get_stock_codes_list(self) -> List[str]:
        """获取所有股票代码列表"""
        return self.stock_codes.copy()

    def has_stock_data(self, stock_code: str) -> bool:
        """
        检查是否有指定股票的数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            是否有数据
        """
        return stock_code in self.stock_data and not self.stock_data[stock_code].empty 