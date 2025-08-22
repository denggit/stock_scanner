#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : base_factor.py
@Description: 基础因子类
@Author     : Zijun Deng
@Date       : 2025-08-20
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import pandas as pd

from backend.utils.logger import setup_logger
from .factor_registry import factor_registry, register_factor

logger = setup_logger("backtest_factor")


class BaseFactor(ABC):
    """
    基础因子类
    
    提供因子注册和装饰器功能，所有因子都应该继承此类或使用其装饰器
    """

    @staticmethod
    def register_factor(name: str, category: str = 'custom', description: str = ""):
        """
        因子注册装饰器
        
        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
        """
        return register_factor(name, category, description)

    @classmethod
    def get_registered_factors(cls) -> pd.DataFrame:
        """
        获取所有注册的因子
        
        Returns:
            因子信息DataFrame
        """
        return factor_registry.list_factors()

    @classmethod
    def get_factor_function(cls, name: str) -> Optional[Callable]:
        """
        获取因子函数
        
        Args:
            name: 因子名称
            
        Returns:
            因子函数
        """
        return factor_registry.get_factor(name)

    @classmethod
    def calculate_factor(cls, name: str, **kwargs) -> pd.Series:
        """
        计算指定因子
        
        Args:
            name: 因子名称
            **kwargs: 因子计算参数
            
        Returns:
            因子值序列
        """
        return factor_registry.calculate_factor(name, **kwargs)

    @classmethod
    def list_factors(cls, category: Optional[str] = None) -> pd.DataFrame:
        """
        列出因子
        
        Args:
            category: 因子类别
            
        Returns:
            因子信息DataFrame
        """
        return factor_registry.list_factors(category)

    @abstractmethod
    def calculate(self, **kwargs) -> pd.Series:
        """
        计算因子值（抽象方法）
        
        Args:
            **kwargs: 计算参数
            
        Returns:
            因子值序列
        """
        pass


# 便捷函数
def register_technical_factor(name: str, description: str = ""):
    """注册技术因子"""
    return BaseFactor.register_factor(name, 'technical', description)


def register_fundamental_factor(name: str, description: str = ""):
    """注册基本面因子"""
    return BaseFactor.register_factor(name, 'fundamental', description)


def register_worldquant_factor(name: str, description: str = ""):
    """注册WorldQuant因子"""
    return BaseFactor.register_factor(name, 'worldquant', description)


def register_channel_factor(name: str, description: str = ""):
    """注册通道分析因子"""
    return BaseFactor.register_factor(name, 'channel', description)
