#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测数据缓存系统

该模块提供高效的上升通道数据缓存机制，避免重复计算，提升回测性能。

主要功能：
- 基于参数组合的数据缓存
- 增量数据更新
- 批量数据加载
- 自动缓存管理

使用示例：
    from backend.business.backtest.database import ChannelDataCache
    
    cache = ChannelDataCache()
    data = cache.get_channel_data(params, stock_codes, date_range)
"""

from .cache_manager import ChannelDataCache, CacheConfig
from .storage_engine import JsonStorageEngine, PickleStorageEngine

__all__ = [
    'ChannelDataCache',
    'CacheConfig', 
    'JsonStorageEngine',
    'PickleStorageEngine'
]
