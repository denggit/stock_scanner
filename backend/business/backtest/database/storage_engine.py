#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缓存存储引擎

提供不同格式的数据存储引擎，支持JSON和Pickle格式。

作者: AI Assistant
日期: 2024-12-20
"""

import json
import pickle
import gzip
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from backend.utils.logger import setup_logger


class StorageEngine(ABC):
    """存储引擎抽象基类"""
    
    @abstractmethod
    def save(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """保存数据"""
        pass
    
    @abstractmethod
    def load(self, file_path: Path) -> Dict[str, Any]:
        """加载数据"""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """获取文件扩展名"""
        pass


class JsonStorageEngine(StorageEngine):
    """JSON存储引擎"""
    
    def __init__(self, compressed: bool = False):
        """
        初始化JSON存储引擎
        
        Args:
            compressed: 是否启用压缩
        """
        self.compressed = compressed
        self.logger = setup_logger(__name__)
    
    def save(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """保存JSON数据"""
        try:
            json_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            
            if self.compressed:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(json_str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存JSON文件失败: {e}")
            return False
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """加载JSON数据"""
        try:
            if self.compressed:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            self.logger.error(f"加载JSON文件失败: {e}")
            raise
    
    def get_file_extension(self) -> str:
        """获取文件扩展名"""
        return 'json.gz' if self.compressed else 'json'


class PickleStorageEngine(StorageEngine):
    """Pickle存储引擎"""
    
    def __init__(self, compressed: bool = False, protocol: int = pickle.HIGHEST_PROTOCOL):
        """
        初始化Pickle存储引擎
        
        Args:
            compressed: 是否启用压缩
            protocol: Pickle协议版本
        """
        self.compressed = compressed
        self.protocol = protocol
        self.logger = setup_logger(__name__)
    
    def save(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """保存Pickle数据"""
        try:
            if self.compressed:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=self.protocol)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=self.protocol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存Pickle文件失败: {e}")
            return False
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """加载Pickle数据"""
        try:
            if self.compressed:
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            self.logger.error(f"加载Pickle文件失败: {e}")
            raise
    
    def get_file_extension(self) -> str:
        """获取文件扩展名"""
        return 'pkl.gz' if self.compressed else 'pkl'


class StorageEngineFactory:
    """存储引擎工厂"""
    
    _engines = {
        'json': JsonStorageEngine,
        'pickle': PickleStorageEngine
    }
    
    @classmethod
    def create_engine(cls, engine_type: str, **kwargs) -> StorageEngine:
        """
        创建存储引擎
        
        Args:
            engine_type: 引擎类型 ('json' 或 'pickle')
            **kwargs: 引擎参数
            
        Returns:
            StorageEngine: 存储引擎实例
        """
        if engine_type not in cls._engines:
            raise ValueError(f"不支持的存储引擎类型: {engine_type}")
        
        return cls._engines[engine_type](**kwargs)
    
    @classmethod
    def register_engine(cls, engine_type: str, engine_class):
        """注册新的存储引擎"""
        cls._engines[engine_type] = engine_class
