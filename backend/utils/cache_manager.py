#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/4/2025 3:30 PM
@File       : cache_manager.py
@Description: 缓存管理器 - 提供内存和Redis缓存功能
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import pandas as pd
import redis
from functools import wraps


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, redis_password: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            redis_host: Redis服务器地址
            redis_port: Redis端口
            redis_db: Redis数据库编号
            redis_password: Redis密码
        """
        self.memory_cache = {}  # 内存缓存
        self.redis_client = None
        self.redis_available = False
        
        # 初始化Redis连接
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 测试连接
            self.redis_client.ping()
            self.redis_available = True
            logging.info("Redis缓存连接成功")
        except Exception as e:
            logging.warning(f"Redis连接失败，将使用内存缓存: {e}")
            self.redis_available = False

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将参数转换为字符串
        key_parts = [prefix]
        
        # 添加位置参数
        for arg in args:
            key_parts.append(str(arg))
        
        # 添加关键字参数（排序以确保一致性）
        for key in sorted(kwargs.keys()):
            key_parts.append(f"{key}:{kwargs[key]}")
        
        # 生成MD5哈希
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存的值或默认值
        """
        try:
            # 先尝试Redis
            if self.redis_available:
                value = self.redis_client.get(key)
                if value is not None:
                    return json.loads(value)
            
            # 再尝试内存缓存
            if key in self.memory_cache:
                cache_item = self.memory_cache[key]
                if cache_item['expire_time'] > time.time():
                    return cache_item['value']
                else:
                    # 清理过期缓存
                    del self.memory_cache[key]
            
            return default
            
        except Exception as e:
            logging.error(f"获取缓存失败: {e}")
            return default

    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
            
        Returns:
            是否设置成功
        """
        try:
            # 处理pandas DataFrame
            if isinstance(value, pd.DataFrame):
                value = value.to_dict('records')
            
            # 尝试Redis
            if self.redis_available:
                self.redis_client.setex(key, expire, json.dumps(value))
            
            # 同时设置内存缓存
            self.memory_cache[key] = {
                'value': value,
                'expire_time': time.time() + expire
            }
            
            return True
            
        except Exception as e:
            logging.error(f"设置缓存失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        try:
            # 删除Redis缓存
            if self.redis_available:
                self.redis_client.delete(key)
            
            # 删除内存缓存
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            return True
            
        except Exception as e:
            logging.error(f"删除缓存失败: {e}")
            return False

    def clear(self, pattern: str = None) -> bool:
        """
        清理缓存
        
        Args:
            pattern: 匹配模式（仅Redis支持）
            
        Returns:
            是否清理成功
        """
        try:
            # 清理Redis缓存
            if self.redis_available and pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            
            # 清理内存缓存
            if pattern:
                # 简单的模式匹配
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
            else:
                self.memory_cache.clear()
            
            return True
            
        except Exception as e:
            logging.error(f"清理缓存失败: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        try:
            # 检查Redis
            if self.redis_available and self.redis_client.exists(key):
                return True
            
            # 检查内存缓存
            if key in self.memory_cache:
                cache_item = self.memory_cache[key]
                if cache_item['expire_time'] > time.time():
                    return True
                else:
                    del self.memory_cache[key]
            
            return False
            
        except Exception as e:
            logging.error(f"检查缓存存在性失败: {e}")
            return False

    def get_memory_usage(self) -> Dict[str, int]:
        """
        获取内存使用情况
        
        Returns:
            内存使用统计
        """
        try:
            stats = {
                'memory_cache_size': len(self.memory_cache),
                'redis_keys': 0
            }
            
            if self.redis_available:
                stats['redis_keys'] = self.redis_client.dbsize()
            
            return stats
            
        except Exception as e:
            logging.error(f"获取内存使用情况失败: {e}")
            return {'memory_cache_size': 0, 'redis_keys': 0}

    def cleanup_expired(self) -> int:
        """
        清理过期缓存
        
        Returns:
            清理的缓存数量
        """
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, cache_item in self.memory_cache.items():
                if cache_item['expire_time'] <= current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            return len(expired_keys)
            
        except Exception as e:
            logging.error(f"清理过期缓存失败: {e}")
            return 0


# 全局缓存管理器实例
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(prefix: str, expire: int = 3600):
    """
    缓存装饰器
    
    Args:
        prefix: 缓存键前缀
        expire: 过期时间（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            cache_key = cache_manager._generate_key(prefix, *args, **kwargs)
            
            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logging.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache_manager.set(cache_key, result, expire)
            logging.debug(f"缓存设置: {cache_key}")
            
            return result
        return wrapper
    return decorator


# 股票数据缓存装饰器
def stock_data_cache(expire: int = 1800):
    """股票数据缓存装饰器（30分钟过期）"""
    return cached("stock_data", expire)


# 策略结果缓存装饰器
def strategy_result_cache(expire: int = 3600):
    """策略结果缓存装饰器（1小时过期）"""
    return cached("strategy_result", expire)


# 技术指标缓存装饰器
def indicator_cache(expire: int = 7200):
    """技术指标缓存装饰器（2小时过期）"""
    return cached("indicator", expire) 