#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略工厂类
使用工厂模式和注册器模式统一管理策略创建
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional, List
import logging

from backend.utils.logger import setup_logger


class StrategyInterface(ABC):
    """
    策略接口
    定义所有策略必须实现的基本接口
    """
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        pass


class StrategyRegistry:
    """
    策略注册器
    使用注册器模式管理策略类型
    """
    
    def __init__(self):
        """初始化策略注册器"""
        self._strategies: Dict[str, Type] = {}
        self._strategy_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = setup_logger("strategy_factory")
        
        # 自动注册已知策略
        self._auto_register_strategies()
    
    def register_strategy(self, strategy_name: str, strategy_class: Type, 
                         default_config: Dict[str, Any] = None):
        """
        注册策略类型
        
        Args:
            strategy_name: 策略名称
            strategy_class: 策略类
            default_config: 默认配置
        """
        if not hasattr(strategy_class, '__bases__'):
            raise ValueError(f"策略类 {strategy_class} 必须是一个类")
            
        self._strategies[strategy_name] = strategy_class
        self._strategy_configs[strategy_name] = default_config or {}
        
        self.logger.info(f"注册策略: {strategy_name} -> {strategy_class.__name__}")
    
    def unregister_strategy(self, strategy_name: str):
        """
        注销策略类型
        
        Args:
            strategy_name: 策略名称
        """
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]
            del self._strategy_configs[strategy_name]
            self.logger.info(f"注销策略: {strategy_name}")
    
    def get_strategy_class(self, strategy_name: str) -> Optional[Type]:
        """
        获取策略类
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略类，未找到返回None
        """
        return self._strategies.get(strategy_name)
    
    def get_default_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        获取策略默认配置
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            默认配置字典
        """
        return self._strategy_configs.get(strategy_name, {}).copy()
    
    def list_strategies(self) -> List[str]:
        """
        列出所有已注册的策略
        
        Returns:
            策略名称列表
        """
        return list(self._strategies.keys())
    
    def is_registered(self, strategy_name: str) -> bool:
        """
        检查策略是否已注册
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            是否已注册
        """
        return strategy_name in self._strategies
    
    def _auto_register_strategies(self):
        """自动注册已知策略"""
        try:
            # 注册上升通道策略
            from ..implementations.channel import RisingChannelStrategy
            
            self.register_strategy(
                'rising_channel',
                RisingChannelStrategy,
                {
                    'max_positions': 50,
                    'min_channel_score': 60.0,
                    'min_data_points': 60,
                    'enable_logging': True
                }
            )
            
        except ImportError as e:
            self.logger.warning(f"自动注册策略失败: {e}")


class StrategyFactory:
    """
    策略工厂类
    使用工厂模式创建策略实例
    """
    
    def __init__(self, registry: StrategyRegistry = None):
        """
        初始化策略工厂
        
        Args:
            registry: 策略注册器，None表示使用默认注册器
        """
        self.registry = registry or StrategyRegistry()
        self.logger = setup_logger("strategy_factory")
    
    def create_strategy(self, strategy_name: str, **kwargs):
        """
        创建策略实例
        
        Args:
            strategy_name: 策略名称
            **kwargs: 策略参数
            
        Returns:
            策略实例
            
        Raises:
            ValueError: 策略未注册或创建失败
        """
        if not self.registry.is_registered(strategy_name):
            available = ', '.join(self.registry.list_strategies())
            raise ValueError(f"未知策略: {strategy_name}。可用策略: {available}")
        
        try:
            strategy_class = self.registry.get_strategy_class(strategy_name)
            default_config = self.registry.get_default_config(strategy_name)
            
            # 合并默认配置和用户配置
            final_config = default_config.copy()
            final_config.update(kwargs)
            
            # 创建策略实例
            strategy_instance = strategy_class(**final_config)
            
            self.logger.info(f"成功创建策略: {strategy_name}")
            return strategy_instance
            
        except Exception as e:
            self.logger.error(f"创建策略 {strategy_name} 失败: {e}")
            raise ValueError(f"创建策略失败: {e}")
    
    def create_strategy_with_config(self, strategy_name: str, config: Dict[str, Any]):
        """
        使用配置字典创建策略
        
        Args:
            strategy_name: 策略名称
            config: 完整配置字典
            
        Returns:
            策略实例
        """
        return self.create_strategy(strategy_name, **config)
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """
        获取策略信息
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略信息字典
        """
        if not self.registry.is_registered(strategy_name):
            raise ValueError(f"未知策略: {strategy_name}")
        
        strategy_class = self.registry.get_strategy_class(strategy_name)
        default_config = self.registry.get_default_config(strategy_name)
        
        return {
            'name': strategy_name,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'default_config': default_config,
            'docstring': strategy_class.__doc__
        }
    
    def list_available_strategies(self) -> List[Dict[str, Any]]:
        """
        列出所有可用策略的信息
        
        Returns:
            策略信息列表
        """
        strategies = []
        for strategy_name in self.registry.list_strategies():
            try:
                info = self.get_strategy_info(strategy_name)
                strategies.append(info)
            except Exception as e:
                self.logger.warning(f"获取策略 {strategy_name} 信息失败: {e}")
        
        return strategies
    
    def register_strategy(self, strategy_name: str, strategy_class: Type, 
                         default_config: Dict[str, Any] = None):
        """
        注册新策略（代理到注册器）
        
        Args:
            strategy_name: 策略名称
            strategy_class: 策略类
            default_config: 默认配置
        """
        self.registry.register_strategy(strategy_name, strategy_class, default_config)


# 全局策略工厂实例（单例模式的简化版本）
_global_factory = None

def get_strategy_factory() -> StrategyFactory:
    """
    获取全局策略工厂实例
    
    Returns:
        策略工厂实例
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = StrategyFactory()
    return _global_factory


def create_strategy(strategy_name: str, **kwargs):
    """
    便捷函数：创建策略实例
    
    Args:
        strategy_name: 策略名称
        **kwargs: 策略参数
        
    Returns:
        策略实例
    """
    factory = get_strategy_factory()
    return factory.create_strategy(strategy_name, **kwargs)


def list_strategies() -> List[str]:
    """
    便捷函数：列出所有可用策略
    
    Returns:
        策略名称列表
    """
    factory = get_strategy_factory()
    return factory.registry.list_strategies()


def register_strategy(strategy_name: str, strategy_class: Type, 
                     default_config: Dict[str, Any] = None):
    """
    便捷函数：注册策略
    
    Args:
        strategy_name: 策略名称
        strategy_class: 策略类
        default_config: 默认配置
    """
    factory = get_strategy_factory()
    factory.register_strategy(strategy_name, strategy_class, default_config)
