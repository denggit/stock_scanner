#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据管理器
使用单例模式管理回测数据
"""

from typing import Dict, Any, Optional, List

import backtrader as bt
import pandas as pd

from backend.utils.logger import setup_logger


class DataManager:
    """
    数据管理器
    负责数据的加载、验证和转换
    """

    _instance = None

    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化数据管理器"""
        if self._initialized:
            return

        # 统一使用backtest主日志记录器，便于全局日志管理和追踪
        self.logger = setup_logger("backtest")
        self.data_cache = {}
        self._initialized = True

    def load_data(self, data: pd.DataFrame, name: str = "data") -> bt.feeds.PandasData:
        """
        加载数据并转换为backtrader格式
        
        Args:
            data: pandas DataFrame格式的数据
            name: 数据名称
            
        Returns:
            backtrader数据源
        """
        # 验证数据格式
        self._validate_data(data)

        # 处理时间戳格式
        processed_data = self._process_timestamp(data)

        # 创建数据源
        data_feed = bt.feeds.PandasData(
            dataname=processed_data,
            name=name,
            datetime=None,  # 使用索引作为日期
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # 不使用持仓量
        )

        # 缓存数据
        self.data_cache[name] = {
            'data': processed_data,
            'feed': data_feed,
            'info': self._get_data_info(processed_data)
        }

        self.logger.info(f"数据加载成功: {name}, 数据条数: {len(processed_data)}")
        return data_feed

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        验证数据格式
        
        Args:
            data: 待验证的数据
            
        Raises:
            ValueError: 数据格式不正确
        """
        if data.empty:
            raise ValueError("数据不能为空")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"数据缺少必需列: {missing_columns}")

        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"列 {col} 必须是数值类型")

        # 检查数据完整性
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            raise ValueError(f"数据包含空值: {null_counts[null_counts > 0].to_dict()}")

        # 检查价格逻辑
        invalid_prices = (
                (data['high'] < data['low']) |
                (data['open'] > data['high']) |
                (data['close'] > data['high']) |
                (data['open'] < data['low']) |
                (data['close'] < data['low'])
        )

        if invalid_prices.any():
            invalid_count = invalid_prices.sum()
            raise ValueError(f"发现 {invalid_count} 条价格逻辑错误的数据")

    def _get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据信息
        
        Args:
            data: 数据
            
        Returns:
            数据信息字典
        """
        return {
            "start_date": data.index[0] if len(data) > 0 else None,
            "end_date": data.index[-1] if len(data) > 0 else None,
            "total_days": len(data),
            "price_range": {
                "min": data['close'].min(),
                "max": data['close'].max(),
                "mean": data['close'].mean()
            },
            "volume_info": {
                "total": data['volume'].sum(),
                "mean": data['volume'].mean(),
                "max": data['volume'].max()
            }
        }

    def get_data_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定数据的信息
        
        Args:
            name: 数据名称
            
        Returns:
            数据信息字典
        """
        if name in self.data_cache:
            return self.data_cache[name]['info']
        return None

    def get_cached_data(self, name: str) -> Optional[bt.feeds.PandasData]:
        """
        获取缓存的数据
        
        Args:
            name: 数据名称
            
        Returns:
            数据源
        """
        if name in self.data_cache:
            return self.data_cache[name]['feed']
        return None

    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            name: 数据名称，None表示清除所有缓存
        """
        if name is None:
            self.data_cache.clear()
            self.logger.info("清除所有数据缓存")
        elif name in self.data_cache:
            del self.data_cache[name]
            self.logger.info(f"清除数据缓存: {name}")

    def list_cached_data(self) -> List[str]:
        """
        列出所有缓存的数据名称
        
        Returns:
            数据名称列表
        """
        return list(self.data_cache.keys())

    def create_sample_data(self, days: int = 252, start_price: float = 100.0) -> pd.DataFrame:
        """
        创建示例数据
        
        Args:
            days: 数据天数
            start_price: 起始价格
            
        Returns:
            示例数据
        """
        import numpy as np
        from datetime import datetime, timedelta

        # 生成日期序列
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(days)]

        # 生成价格数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, days)
        prices = [start_price]

        for i in range(1, days):
            price = prices[-1] * (1 + returns[i])
            prices.append(price)

        # 生成OHLCV数据
        data = []
        for date, close in zip(dates, prices):
            open_price = close * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 10000000)

            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data, index=dates)

    def _process_timestamp(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理时间戳格式，确保符合backtrader要求
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        processed_data = data.copy()

        # 检查是否有trade_date列
        if 'trade_date' in processed_data.columns:
            # 将trade_date转换为datetime格式
            processed_data['trade_date'] = pd.to_datetime(processed_data['trade_date'])
            # 设置为索引
            processed_data.set_index('trade_date', inplace=True)
        elif not isinstance(processed_data.index, pd.DatetimeIndex):
            # 如果索引不是DatetimeIndex，尝试转换
            try:
                processed_data.index = pd.to_datetime(processed_data.index)
            except Exception as e:
                self.logger.warning(f"无法转换索引为datetime格式: {e}")
                # 创建默认的日期索引
                processed_data.index = pd.date_range(
                    start='2024-01-01',
                    periods=len(processed_data),
                    freq='D'
                )

        # 确保索引是DatetimeIndex
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是datetime格式")

        # 按日期排序
        processed_data.sort_index(inplace=True)

        return processed_data
