#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道数据缓存管理器

该模块实现了基于参数组合的上升通道数据缓存系统，提供：
- 智能参数哈希生成
- 批量数据缓存和读取
- 增量数据更新
- 自动过期管理

作者: AI Assistant
日期: 2024-12-20
"""

import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd

from backend.utils.logger import setup_logger


class CacheConfig:
    """缓存配置类"""
    
    def __init__(self, 
                 cache_dir: str = None,
                 storage_format: str = 'pickle',  # 'pickle' or 'json'
                 max_cache_age_days: int = 30,
                 max_cache_size_mb: int = 1000,
                 enable_compression: bool = True):
        """
        初始化缓存配置
        
        Args:
            cache_dir: 缓存目录路径
            storage_format: 存储格式 ('pickle' 或 'json')
            max_cache_age_days: 缓存最大保存天数
            max_cache_size_mb: 缓存最大大小（MB）
            enable_compression: 是否启用压缩
        """
        if cache_dir is None:
            # 默认在backtest目录下创建database文件夹
            current_dir = Path(__file__).parent.parent
            cache_dir = current_dir / "database" / "channel_cache"
        
        self.cache_dir = Path(cache_dir)
        self.storage_format = storage_format
        self.max_cache_age_days = max_cache_age_days
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_compression = enable_compression
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_file_path(self, params_hash: str) -> Path:
        """获取缓存文件路径"""
        ext = 'pkl' if self.storage_format == 'pickle' else 'json'
        return self.cache_dir / f"channel_data_{params_hash}.{ext}"
    
    def get_metadata_file_path(self, params_hash: str) -> Path:
        """获取元数据文件路径"""
        return self.cache_dir / f"metadata_{params_hash}.json"


class ParameterHasher:
    """参数哈希生成器"""
    
    @staticmethod
    def generate_hash(params: Dict[str, Any]) -> str:
        """
        生成参数组合的唯一哈希值
        
        Args:
            params: 参数字典
            
        Returns:
            str: 参数哈希值
        """
        # 确保参数键值对排序，保证一致性
        sorted_params = dict(sorted(params.items()))
        
        # 将参数转换为JSON字符串（确保可序列化）
        param_str = json.dumps(sorted_params, sort_keys=True, ensure_ascii=False)
        
        # 生成MD5哈希
        hash_obj = hashlib.md5(param_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # 取前16位，足够避免冲突
    
    @staticmethod
    def get_readable_params_key(params: Dict[str, Any]) -> str:
        """
        生成可读的参数键
        
        Args:
            params: 参数字典
            
        Returns:
            str: 可读的参数键
        """
        key_parts = []
        for key, value in sorted(params.items()):
            if isinstance(value, float):
                key_parts.append(f"{key}_{value:.3f}")
            else:
                key_parts.append(f"{key}_{value}")
        
        return "_".join(key_parts)


class ChannelDataCache:
    """
    上升通道数据缓存管理器
    
    提供高效的上升通道历史数据缓存和查询功能，
    支持基于参数组合的智能缓存机制。
    """
    
    def __init__(self, config: CacheConfig = None):
        """
        初始化缓存管理器
        
        Args:
            config: 缓存配置，为None时使用默认配置
        """
        self.config = config or CacheConfig()
        self.logger = setup_logger(__name__)
        self.hasher = ParameterHasher()
        
        # 内存缓存，用于加速同一回测会话中的重复访问
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"上升通道数据缓存系统初始化完成")
        self.logger.info(f"缓存目录: {self.config.cache_dir}")
        self.logger.info(f"存储格式: {self.config.storage_format}")
    
    def get_channel_data(self, 
                        params: Dict[str, Any],
                        stock_codes: List[str] = None,
                        start_date: str = None,
                        end_date: str = None) -> Optional[Dict[str, Any]]:
        """
        获取上升通道数据
        
        Args:
            params: 通道参数
            stock_codes: 股票代码列表，None表示获取所有
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            
        Returns:
            Dict: 包含params和value的数据字典，未找到时返回None
        """
        params_hash = self.hasher.generate_hash(params)
        
        # 先检查内存缓存
        if params_hash in self._memory_cache:
            self.logger.debug(f"从内存缓存获取数据: {params_hash}")
            cached_data = self._memory_cache[params_hash]
            return self._filter_data(cached_data, stock_codes, start_date, end_date)
        
        # 检查磁盘缓存
        cache_file = self.config.get_cache_file_path(params_hash)
        if not cache_file.exists():
            self.logger.debug(f"缓存文件不存在: {cache_file}")
            return None
        
        try:
            # 检查缓存是否过期
            if self._is_cache_expired(params_hash):
                self.logger.info(f"缓存已过期，删除: {params_hash}")
                self._delete_cache(params_hash)
                return None
            
            # 加载缓存数据
            cached_data = self._load_cache_file(cache_file)
            if cached_data is None:
                return None
            
            # 加载到内存缓存
            self._memory_cache[params_hash] = cached_data
            
            self.logger.info(f"从磁盘缓存获取数据: {params_hash}")
            return self._filter_data(cached_data, stock_codes, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"读取缓存文件失败: {e}")
            return None
    
    def save_channel_data(self, 
                         params: Dict[str, Any], 
                         data: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        保存上升通道数据到缓存
        
        Args:
            params: 通道参数
            data: 通道数据 {股票代码: [历史数据]}
            
        Returns:
            bool: 保存是否成功
        """
        params_hash = self.hasher.generate_hash(params)
        
        try:
            # 构建缓存数据结构
            cache_data = {
                'params': params,
                'value': data,
                'created_at': datetime.now().isoformat(),
                'stock_count': len(data),
                'data_points': sum(len(stock_data) for stock_data in data.values())
            }
            
            # 保存到磁盘
            cache_file = self.config.get_cache_file_path(params_hash)
            if self._save_cache_file(cache_file, cache_data):
                # 保存元数据
                self._save_metadata(params_hash, params, cache_data)
                
                # 更新内存缓存
                self._memory_cache[params_hash] = cache_data
                
                self.logger.info(f"成功保存缓存数据: {params_hash}")
                self.logger.info(f"包含 {len(data)} 只股票，{cache_data['data_points']} 个数据点")
                return True
            
        except Exception as e:
            self.logger.error(f"保存缓存数据失败: {e}")
            
        return False
    
    def update_channel_data(self, 
                           params: Dict[str, Any],
                           new_data: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        增量更新缓存数据
        
        Args:
            params: 通道参数
            new_data: 新的通道数据
            
        Returns:
            bool: 更新是否成功
        """
        params_hash = self.hasher.generate_hash(params)
        
        try:
            # 获取现有数据
            existing_data = self.get_channel_data(params)
            if existing_data is None:
                # 没有现有数据，直接保存
                return self.save_channel_data(params, new_data)
            
            # 合并数据
            merged_data = existing_data['value'].copy()
            for stock_code, stock_data in new_data.items():
                if stock_code in merged_data:
                    # 合并同一股票的数据，按日期去重
                    existing_dates = {item['trade_date'] for item in merged_data[stock_code]}
                    new_items = [item for item in stock_data 
                               if item['trade_date'] not in existing_dates]
                    merged_data[stock_code].extend(new_items)
                    # 按日期排序
                    merged_data[stock_code].sort(key=lambda x: x['trade_date'])
                else:
                    merged_data[stock_code] = stock_data
            
            # 保存合并后的数据
            return self.save_channel_data(params, merged_data)
            
        except Exception as e:
            self.logger.error(f"增量更新缓存数据失败: {e}")
            return False
    
    def get_cached_params_list(self) -> List[Dict[str, Any]]:
        """
        获取所有已缓存的参数组合
        
        Returns:
            List: 参数组合列表
        """
        cached_params = []
        
        try:
            for metadata_file in self.config.cache_dir.glob("metadata_*.json"):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    cached_params.append({
                        'params': metadata['params'],
                        'hash': metadata['hash'],
                        'created_at': metadata['created_at'],
                        'stock_count': metadata.get('stock_count', 0),
                        'data_points': metadata.get('data_points', 0)
                    })
        except Exception as e:
            self.logger.error(f"获取缓存参数列表失败: {e}")
        
        return cached_params
    
    def clear_cache(self, params: Dict[str, Any] = None) -> bool:
        """
        清理缓存
        
        Args:
            params: 指定参数组合，None表示清理所有缓存
            
        Returns:
            bool: 清理是否成功
        """
        try:
            if params is None:
                # 清理所有缓存
                for cache_file in self.config.cache_dir.glob("*"):
                    cache_file.unlink()
                self._memory_cache.clear()
                self.logger.info("已清理所有缓存")
            else:
                # 清理指定参数的缓存
                params_hash = self.hasher.generate_hash(params)
                self._delete_cache(params_hash)
                self.logger.info(f"已清理缓存: {params_hash}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        stats = {
            'total_cache_files': 0,
            'total_size_mb': 0.0,
            'memory_cache_size': len(self._memory_cache),
            'oldest_cache': None,
            'newest_cache': None
        }
        
        try:
            cache_files = list(self.config.cache_dir.glob("channel_data_*.pkl")) + \
                         list(self.config.cache_dir.glob("channel_data_*.json"))
            
            stats['total_cache_files'] = len(cache_files)
            
            if cache_files:
                total_size = sum(f.stat().st_size for f in cache_files)
                stats['total_size_mb'] = total_size / (1024 * 1024)
                
                # 获取最老和最新的缓存
                cache_times = [(f.stat().st_mtime, f.name) for f in cache_files]
                cache_times.sort()
                
                stats['oldest_cache'] = datetime.fromtimestamp(cache_times[0][0]).isoformat()
                stats['newest_cache'] = datetime.fromtimestamp(cache_times[-1][0]).isoformat()
        
        except Exception as e:
            self.logger.error(f"获取缓存统计信息失败: {e}")
        
        return stats
    
    def _filter_data(self, 
                    cached_data: Dict[str, Any],
                    stock_codes: List[str] = None,
                    start_date: str = None,
                    end_date: str = None) -> Dict[str, Any]:
        """筛选缓存数据"""
        filtered_data = cached_data.copy()
        value = cached_data['value']
        
        # 筛选股票
        if stock_codes is not None:
            value = {code: data for code, data in value.items() if code in stock_codes}
        
        # 筛选日期范围
        if start_date is not None or end_date is not None:
            filtered_value = {}
            for stock_code, stock_data in value.items():
                filtered_stock_data = []
                for item in stock_data:
                    item_date = item['trade_date']
                    # 处理不同的日期格式
                    if isinstance(item_date, str):
                        if len(item_date) == 8:  # YYYYMMDD
                            item_date = f"{item_date[:4]}-{item_date[4:6]}-{item_date[6:8]}"
                    
                    # 检查日期范围
                    if start_date and item_date < start_date:
                        continue
                    if end_date and item_date > end_date:
                        continue
                    
                    filtered_stock_data.append(item)
                
                if filtered_stock_data:
                    filtered_value[stock_code] = filtered_stock_data
            
            value = filtered_value
        
        filtered_data['value'] = value
        return filtered_data
    
    def _is_cache_expired(self, params_hash: str) -> bool:
        """检查缓存是否过期"""
        metadata_file = self.config.get_metadata_file_path(params_hash)
        if not metadata_file.exists():
            return True
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            created_at = datetime.fromisoformat(metadata['created_at'])
            age = datetime.now() - created_at
            
            return age.days > self.config.max_cache_age_days
            
        except Exception:
            return True
    
    def _delete_cache(self, params_hash: str):
        """删除指定的缓存"""
        cache_file = self.config.get_cache_file_path(params_hash)
        metadata_file = self.config.get_metadata_file_path(params_hash)
        
        for file_path in [cache_file, metadata_file]:
            if file_path.exists():
                file_path.unlink()
        
        # 从内存缓存中删除
        self._memory_cache.pop(params_hash, None)
    
    def _load_cache_file(self, cache_file: Path) -> Optional[Dict[str, Any]]:
        """加载缓存文件"""
        try:
            if self.config.storage_format == 'pickle':
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:  # json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"加载缓存文件失败: {e}")
            return None
    
    def _save_cache_file(self, cache_file: Path, data: Dict[str, Any]) -> bool:
        """保存缓存文件"""
        try:
            if self.config.storage_format == 'pickle':
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            else:  # json
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"保存缓存文件失败: {e}")
            return False
    
    def _save_metadata(self, params_hash: str, params: Dict[str, Any], cache_data: Dict[str, Any]):
        """保存元数据"""
        metadata = {
            'hash': params_hash,
            'params': params,
            'readable_key': self.hasher.get_readable_params_key(params),
            'created_at': cache_data['created_at'],
            'stock_count': cache_data['stock_count'],
            'data_points': cache_data['data_points']
        }
        
        metadata_file = self.config.get_metadata_file_path(params_hash)
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")
