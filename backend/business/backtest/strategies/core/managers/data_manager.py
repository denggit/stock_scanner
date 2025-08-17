#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据管理器
负责多股票数据的管理、缓存和获取
使用观察者模式通知数据更新
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from backend.utils.logger import setup_logger


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
        self.logger = setup_logger("backtest")

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
        self.logger = setup_logger("backtest")

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
        
        修改说明：
        - 支持NaN数据：当股票在指定日期未上市时，返回0.0
        - 兼容前向填充：正确处理前向填充产生的NaN数据
        - 保持健壮性：确保方法不会因为NaN数据而失败
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            股票价格，获取失败或股票未上市返回0.0
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

        # 统一为日期比较对象
        try:
            from pandas.api import types as ptypes
            # 若列为datetime64，转为日期再比较；否则直接比较date
            if ptypes.is_datetime64_any_dtype(stock_df['trade_date']):
                trade_dates = pd.to_datetime(stock_df['trade_date']).dt.date
            else:
                trade_dates = stock_df['trade_date']
        except Exception:
            trade_dates = pd.to_datetime(stock_df['trade_date'], errors='coerce').dt.date

        # 仅精确匹配"当日"价格；若当日无该股票记录（如停牌/未上市），则不回退到未来数据，返回0.0
        mask_exact = trade_dates == target_date.date()
        if mask_exact.any():
            # 取当日第一条（正常应唯一）
            row_idx = stock_df.index[mask_exact][0]
            try:
                close_price = stock_df.loc[row_idx, 'close']
                # 检查是否为NaN（表示股票未上市）
                if pd.isna(close_price):
                    return 0.0
                return float(close_price)
            except Exception:
                pass

        # 当日无数据：严格返回0，避免使用未来价格导致穿越
        return 0.0

    def get_stock_open_price(self, stock_code: str, date: datetime) -> float:
        """
        获取指定股票在指定日期的开盘价
        
        修改说明：
        - 支持NaN数据：当股票在指定日期未上市时，返回0.0
        - 兼容前向填充：正确处理前向填充产生的NaN数据
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            股票开盘价，获取失败或股票未上市返回0.0
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

        # 统一为日期比较对象
        try:
            from pandas.api import types as ptypes
            # 若列为datetime64，转为日期再比较；否则直接比较date
            if ptypes.is_datetime64_any_dtype(stock_df['trade_date']):
                trade_dates = pd.to_datetime(stock_df['trade_date']).dt.date
            else:
                trade_dates = stock_df['trade_date']
        except Exception:
            trade_dates = pd.to_datetime(stock_df['trade_date'], errors='coerce').dt.date

        # 仅精确匹配"当日"开盘价；若当日无该股票记录（如停牌/未上市），则不回退到未来数据，返回0.0
        mask_exact = trade_dates == target_date.date()
        if mask_exact.any():
            # 取当日第一条（正常应唯一）
            row_idx = stock_df.index[mask_exact][0]
            try:
                open_price = stock_df.loc[row_idx, 'open']
                # 检查是否为NaN（表示股票未上市）
                if pd.isna(open_price):
                    return 0.0
                return float(open_price)
            except Exception:
                pass

        # 当日无数据：严格返回0，避免使用未来价格导致穿越
        return 0.0

    def get_stock_volume(self, stock_code: str, date: datetime) -> float:
        """
        获取指定股票在指定日期的成交量
        
        修改说明：
        - 支持NaN数据：当股票在指定日期未上市时，返回0.0
        - 兼容前向填充：正确处理前向填充产生的NaN数据
        
        Args:
            stock_code: 股票代码
            date: 日期
            
        Returns:
            股票成交量，获取失败或股票未上市返回0.0
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

        # 统一为日期比较对象
        try:
            from pandas.api import types as ptypes
            # 若列为datetime64，转为日期再比较；否则直接比较date
            if ptypes.is_datetime64_any_dtype(stock_df['trade_date']):
                trade_dates = pd.to_datetime(stock_df['trade_date']).dt.date
            else:
                trade_dates = stock_df['trade_date']
        except Exception:
            trade_dates = pd.to_datetime(stock_df['trade_date'], errors='coerce').dt.date

        # 仅精确匹配"当日"成交量；若当日无该股票记录（如停牌/未上市），则不回退到未来数据，返回0.0
        mask_exact = trade_dates == target_date.date()
        if mask_exact.any():
            # 取当日第一条（正常应唯一）
            row_idx = stock_df.index[mask_exact][0]
            try:
                volume = stock_df.loc[row_idx, 'volume']
                # 检查是否为NaN（表示股票未上市）
                if pd.isna(volume):
                    return 0.0
                return float(volume)
            except Exception:
                pass

        # 当日无数据：严格返回0，避免使用未来数据导致穿越
        return 0.0

    def get_stock_avg_volume(self, stock_code: str, date: datetime, days: int = 5) -> float:
        """
        获取指定股票在指定日期前N天的平均成交量
        
        修改说明：
        - 支持NaN数据：正确处理前向填充产生的NaN数据
        - 过滤NaN值：在计算平均值时排除NaN值
        
        Args:
            stock_code: 股票代码
            date: 日期
            days: 计算平均的天数，默认5天
            
        Returns:
            平均成交量，获取失败或数据不足返回0.0
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

        # 获取指定日期前的数据
        try:
            from pandas.api import types as ptypes
            if ptypes.is_datetime64_any_dtype(stock_df['trade_date']):
                cutoff = pd.to_datetime(target_date)
                filtered_df = stock_df[stock_df['trade_date'] < cutoff]
            else:
                filtered_df = stock_df[stock_df['trade_date'] < target_date.date()]
        except Exception:
            cutoff = pd.to_datetime(target_date)
            try:
                filtered_df = stock_df[stock_df['trade_date'] < cutoff]
            except Exception:
                filtered_df = stock_df[stock_df['trade_date'] < cutoff.date()]

        # 取最近N天的数据，并过滤掉NaN值
        if len(filtered_df) >= days:
            recent_df = filtered_df.tail(days)
            try:
                # 过滤掉NaN值，只计算有效数据的平均值
                volumes = recent_df['volume'].dropna().astype(float)
                if len(volumes) > 0:
                    return float(volumes.mean())
            except Exception:
                pass

        return 0.0

    def get_stock_data_until(self, stock_code: str, date: datetime,
                             min_data_points: int = 60) -> Optional[pd.DataFrame]:
        """
        获取指定股票截至指定日期的历史数据
        
        修改说明：
        - 支持NaN数据：正确处理前向填充产生的NaN数据
        - 有效数据计数：在计算数据点数时排除NaN值
        - 保持健壮性：确保方法不会因为NaN数据而失败
        
        Args:
            stock_code: 股票代码
            date: 截止日期
            min_data_points: 最小数据点数（有效数据点）
            
        Returns:
            历史数据DataFrame，数据不足返回None
        """
        if stock_code not in self.stock_data:
            return None

        stock_df = self.stock_data[stock_code]

        # 确保date是datetime类型
        if isinstance(date, datetime):
            target_datetime = date
        elif isinstance(date, pd.Timestamp):
            target_datetime = date.to_pydatetime()
        else:
            target_datetime = pd.to_datetime(date).to_pydatetime()

        # 过滤到指定日期之前的数据（支持 datetime64 与 date 比较）
        try:
            from pandas.api import types as ptypes
            if ptypes.is_datetime64_any_dtype(stock_df['trade_date']):
                cutoff = pd.to_datetime(target_datetime)
                filtered_df = stock_df[stock_df['trade_date'] <= cutoff]
            else:
                filtered_df = stock_df[stock_df['trade_date'] <= target_datetime.date()]
        except Exception:
            cutoff = pd.to_datetime(target_datetime)
            try:
                filtered_df = stock_df[stock_df['trade_date'] <= cutoff]
            except Exception:
                filtered_df = stock_df[stock_df['trade_date'] <= cutoff.date()]

        # 计算有效数据点数（排除NaN值）
        valid_data_count = filtered_df[['open', 'high', 'low', 'close', 'volume']].dropna().shape[0]
        
        if valid_data_count < min_data_points:
            self.logger.debug(f"股票 {stock_code} 有效数据不足: {valid_data_count} < {min_data_points}")
            return None

        return filtered_df

    def get_all_stock_prices(self, date: datetime) -> Dict[str, float]:
        """
        获取所有股票在指定日期的价格
        
        修改说明：
        - 支持NaN数据：正确处理前向填充产生的NaN数据
        - 过滤无效价格：只返回有效的股票价格（>0）
        
        Args:
            date: 日期
            
        Returns:
            股票价格字典 {股票代码: 价格}，只包含有效价格的股票
        """
        prices = {}
        for stock_code in self.stock_codes:
            price = self.get_stock_price(stock_code, date)
            # 只包含有效价格的股票（价格>0表示股票已上市且有交易）
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
        """
        验证和预处理数据
        
        修改说明：
        - 保留NaN数据：不自动填充NaN值，保留NaN值用于表示股票未上市期间
        - 支持前向填充：正确处理前向填充产生的NaN数据
        - 保持健壮性：确保验证逻辑不会因为NaN数据而失败
        """
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

                # 检查数据质量 - 修改为支持NaN数据
                null_counts = data[required_columns].isnull().sum()
                if null_counts.any():
                    # 记录NaN数据统计，但不自动填充
                    self.logger.debug(f"股票 {stock_code} NaN数据统计: {null_counts.to_dict()}")
                    # 注意：不再自动填充NaN值，保留NaN值用于表示股票未上市期间

                # 统一 trade_date 类型，避免后续重复转换开销
                try:
                    from pandas.api import types as ptypes
                    if not ptypes.is_datetime64_any_dtype(data['trade_date']):
                        data['trade_date'] = pd.to_datetime(data['trade_date'], errors='coerce')
                except Exception:
                    # 保底转换
                    data['trade_date'] = pd.to_datetime(data['trade_date'], errors='coerce')

                # 排序，确保按日期升序，便于切片
                try:
                    data = data.sort_values('trade_date').reset_index(drop=True)
                except Exception:
                    pass

                # 裁剪不必要的列，降低内存占用（仅保留策略与通道分析所需最小集合）
                try:
                    keep_cols = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
                    extra_cols = [c for c in data.columns if c not in keep_cols]
                    if extra_cols:
                        data = data[keep_cols]
                except Exception:
                    # 若裁剪失败，不中断流程
                    pass

                # 检查价格数据合理性 - 只检查非NaN数据
                non_null_data = data[['open', 'high', 'low', 'close']].dropna()
                if len(non_null_data) > 0:
                    invalid_prices = (
                            (non_null_data['high'] < non_null_data['low']) |
                            (non_null_data['open'] > non_null_data['high']) |
                            (non_null_data['close'] > non_null_data['high']) |
                            (non_null_data['open'] < non_null_data['low']) |
                            (non_null_data['close'] < non_null_data['low'])
                    )
                    if invalid_prices.any():
                        invalid_count = invalid_prices.sum()
                        self.logger.warning(f"股票 {stock_code} 发现 {invalid_count} 条价格逻辑错误的数据")

            except Exception as e:
                self.logger.error(f"预处理股票 {stock_code} 数据时发生错误: {e}")
                invalid_stocks.append(stock_code)

        # 移除无效股票
        for stock_code in invalid_stocks:
            if stock_code in self.stock_data:
                del self.stock_data[stock_code]
                self.logger.warning(f"移除无效股票: {stock_code}")

        # 更新股票代码列表
        self.stock_codes = list(self.stock_data.keys())

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

    def get_all_stock_data(self) -> Dict[str, pd.DataFrame]:
        """获取所有股票数据字典的拷贝"""
        return {k: v.copy() for k, v in self.stock_data.items()}
