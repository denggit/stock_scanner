#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : factor_registry.py
@Description: 因子注册管理器
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

import pandas as pd
from typing import Dict, List, Callable, Optional, Any
from functools import wraps
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class FactorRegistry:
    """
    因子注册管理器
    
    负责管理所有注册的因子，提供因子的注册、查询、分类等功能
    """
    
    def __init__(self):
        """初始化因子注册管理器"""
        self._factors: Dict[str, Dict[str, Any]] = {}
        self._factor_categories: Dict[str, List[str]] = {
            'technical': [],      # 技术因子
            'fundamental': [],    # 基本面因子
            'worldquant': [],     # WorldQuant Alpha因子
            'channel': [],        # 通道分析因子
            'custom': []          # 自定义因子
        }
    
    def register_factor(self, name: str, category: str = 'custom', description: str = ""):
        """
        因子注册装饰器
        
        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 注册因子
            import inspect
            try:
                # 尝试获取函数签名
                sig = inspect.signature(func)
                signature = str(sig)
            except:
                # 如果失败，使用函数名
                signature = func.__name__
            
            self._factors[name] = {
                'function': func,
                'name': name,
                'category': category,
                'description': description,
                'signature': signature
            }
            
            # 添加到对应类别
            if category in self._factor_categories:
                if name not in self._factor_categories[category]:
                    self._factor_categories[category].append(name)
            else:
                self._factor_categories['custom'].append(name)
            
            logger.debug(f"因子 {name} 注册成功: {description}")
            return wrapper
        
        return decorator
    
    def get_factor(self, name: str) -> Optional[Callable]:
        """
        获取因子函数
        
        Args:
            name: 因子名称
            
        Returns:
            因子函数，如果不存在返回None
        """
        if name in self._factors:
            return self._factors[name]['function']
        return None
    

    
    def get_factor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取因子信息
        
        Args:
            name: 因子名称
            
        Returns:
            因子信息字典
        """
        return self._factors.get(name)
    
    def list_factors(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        列出所有因子
        
        Args:
            category: 因子类别，如果为None则列出所有因子
            
        Returns:
            因子信息DataFrame
        """
        if category:
            factor_names = self._factor_categories.get(category, [])
        else:
            factor_names = list(self._factors.keys())
        
        factor_list = []
        for name in factor_names:
            if name in self._factors:
                factor_info = self._factors[name]
                factor_list.append({
                    'name': name,
                    'category': factor_info['category'],
                    'description': factor_info['description'],
                    'signature': factor_info['signature']
                })
        
        return pd.DataFrame(factor_list)
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """
        获取指定类别的所有因子名称
        
        Args:
            category: 因子类别
            
        Returns:
            因子名称列表
        """
        return self._factor_categories.get(category, [])
    
    def get_all_categories(self) -> Dict[str, List[str]]:
        """
        获取所有因子类别
        
        Returns:
            类别到因子名称的映射
        """
        return self._factor_categories.copy()
    
    def calculate_factor(self, name: str, **kwargs) -> pd.Series:
        """
        计算指定因子
        
        Args:
            name: 因子名称
            **kwargs: 因子计算参数
            
        Returns:
            因子值序列
        """
        factor_func = self.get_factor(name)
        if factor_func is None:
            raise ValueError(f"因子 {name} 未注册")
        
        return factor_func(**kwargs)
    
    def calculate_factors(self, factor_names: List[str], **kwargs) -> pd.DataFrame:
        """
        批量计算多个因子
        
        Args:
            factor_names: 因子名称列表
            **kwargs: 因子计算参数
            
        Returns:
            因子值DataFrame
        """
        results = {}
        for name in factor_names:
            try:
                results[name] = self.calculate_factor(name, **kwargs)
            except Exception as e:
                logger.error(f"计算因子 {name} 失败: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def remove_factor(self, name: str) -> bool:
        """
        移除因子
        
        Args:
            name: 因子名称
            
        Returns:
            是否成功移除
        """
        if name in self._factors:
            category = self._factors[name]['category']
            if category in self._factor_categories and name in self._factor_categories[category]:
                self._factor_categories[category].remove(name)
            
            del self._factors[name]
            logger.info(f"因子 {name} 已移除")
            return True
        
        return False
    
    def clear_factors(self):
        """清空所有因子"""
        self._factors.clear()
        for category in self._factor_categories:
            self._factor_categories[category].clear()
        logger.info("所有因子已清空")
    
    def get_factor_count(self) -> int:
        """
        获取因子总数
        
        Returns:
            因子数量
        """
        return len(self._factors)
    
    def get_category_count(self) -> Dict[str, int]:
        """
        获取各类别因子数量
        
        Returns:
            类别到数量的映射
        """
        return {category: len(factors) for category, factors in self._factor_categories.items()}

# 全局因子注册器实例
factor_registry = FactorRegistry()

# 便捷函数
def register_factor(name: str, category: str = 'custom', description: str = ""):
    """便捷的因子注册装饰器"""
    return factor_registry.register_factor(name, category, description)

def get_factor(name: str) -> Optional[Callable]:
    """便捷的因子获取函数"""
    return factor_registry.get_factor(name)

def list_factors(category: Optional[str] = None) -> pd.DataFrame:
    """便捷的因子列表函数"""
    return factor_registry.list_factors(category)
