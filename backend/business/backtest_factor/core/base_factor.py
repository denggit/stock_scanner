#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : base_factor.py
@Description: 基础因子类，提供因子注册和装饰器功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Any, Optional, Union
from abc import ABC, abstractmethod
from functools import wraps
import inspect
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseFactor(ABC):
    """
    基础因子类，提供因子注册和装饰器功能
    
    使用示例:
    @BaseFactor.register_factor(name='alpha_8')
    @staticmethod
    def alpha_8(open_price: pd.Series, pct_chg: pd.Series) -> pd.Series:
        # 因子计算逻辑
        pass
    """
    
    # 注册的因子字典
    _registered_factors: Dict[str, Callable] = {}
    
    @classmethod
    def register_factor(cls, name: str, description: str = ""):
        """
        因子注册装饰器
        
        Args:
            name: 因子名称
            description: 因子描述
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 注册因子
            cls._registered_factors[name] = {
                'function': wrapper,
                'description': description,
                'signature': inspect.signature(func)
            }
            
            logger.info(f"因子 {name} 注册成功: {description}")
            return wrapper
        return decorator
    
    @classmethod
    def get_registered_factors(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有注册的因子
        
        Returns:
            注册的因子字典
        """
        return cls._registered_factors
    
    @classmethod
    def get_factor_function(cls, name: str) -> Optional[Callable]:
        """
        获取指定名称的因子函数
        
        Args:
            name: 因子名称
            
        Returns:
            因子函数
        """
        if name in cls._registered_factors:
            return cls._registered_factors[name]['function']
        return None
    
    @classmethod
    def calculate_factor(cls, name: str, **kwargs) -> pd.Series:
        """
        计算指定因子
        
        Args:
            name: 因子名称
            **kwargs: 因子计算所需的参数
            
        Returns:
            因子值序列
        """
        if name not in cls._registered_factors:
            raise ValueError(f"因子 {name} 未注册")
        
        factor_func = cls._registered_factors[name]['function']
        return factor_func(**kwargs)
    
    @classmethod
    def list_factors(cls) -> pd.DataFrame:
        """
        列出所有注册的因子信息
        
        Returns:
            因子信息DataFrame
        """
        factors_info = []
        for name, info in cls._registered_factors.items():
            factors_info.append({
                'name': name,
                'description': info['description'],
                'signature': str(info['signature'])
            })
        
        return pd.DataFrame(factors_info)
    
    @abstractmethod
    def calculate(self, **kwargs) -> pd.Series:
        """
        抽象方法：子类必须实现的因子计算方法
        
        Args:
            **kwargs: 计算参数
            
        Returns:
            因子值序列
        """
        pass


# 预定义一些常用因子
class CommonFactors(BaseFactor):
    """
    常用因子集合
    """
    
    @BaseFactor.register_factor(name='momentum_5d', description='5日动量因子')
    @staticmethod
    def momentum_5d(close: pd.Series) -> pd.Series:
        """
        5日动量因子：过去5日收益率
        
        Args:
            close: 收盘价序列
            
        Returns:
            5日动量因子值
        """
        return close.pct_change(5)
    
    @BaseFactor.register_factor(name='momentum_20d', description='20日动量因子')
    @staticmethod
    def momentum_20d(close: pd.Series) -> pd.Series:
        """
        20日动量因子：过去20日收益率
        
        Args:
            close: 收盘价序列
            
        Returns:
            20日动量因子值
        """
        return close.pct_change(20)
    
    @BaseFactor.register_factor(name='volatility_20d', description='20日波动率因子')
    @staticmethod
    def volatility_20d(returns: pd.Series) -> pd.Series:
        """
        20日波动率因子：过去20日收益率的标准差
        
        Args:
            returns: 收益率序列
            
        Returns:
            20日波动率因子值
        """
        return returns.rolling(20).std()
    
    @BaseFactor.register_factor(name='volume_ratio_5d', description='5日成交量比率因子')
    @staticmethod
    def volume_ratio_5d(volume: pd.Series) -> pd.Series:
        """
        5日成交量比率因子：当前成交量与过去5日平均成交量的比值
        
        Args:
            volume: 成交量序列
            
        Returns:
            5日成交量比率因子值
        """
        return volume / volume.rolling(5).mean()
    
    @BaseFactor.register_factor(name='price_position_20d', description='20日价格位置因子')
    @staticmethod
    def price_position_20d(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        20日价格位置因子：当前价格在20日高低点之间的位置
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            20日价格位置因子值 (0-1之间)
        """
        high_20d = high.rolling(20).max()
        low_20d = low.rolling(20).min()
        return (close - low_20d) / (high_20d - low_20d)
    
    @BaseFactor.register_factor(name='ma_cross_5_20', description='5日与20日均线交叉因子')
    @staticmethod
    def ma_cross_5_20(close: pd.Series) -> pd.Series:
        """
        5日与20日均线交叉因子：5日均线相对20日均线的位置
        
        Args:
            close: 收盘价序列
            
        Returns:
            均线交叉因子值
        """
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        return (ma5 - ma20) / ma20
    
    @BaseFactor.register_factor(name='rsi_14', description='14日RSI因子')
    @staticmethod
    def rsi_14(close: pd.Series) -> pd.Series:
        """
        14日RSI因子：相对强弱指数
        
        Args:
            close: 收盘价序列
            
        Returns:
            RSI因子值 (0-100之间)
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @BaseFactor.register_factor(name='bollinger_position', description='布林带位置因子')
    @staticmethod
    def bollinger_position(close: pd.Series, window: int = 20, num_std: float = 2) -> pd.Series:
        """
        布林带位置因子：价格在布林带中的相对位置
        
        Args:
            close: 收盘价序列
            window: 移动平均窗口
            num_std: 标准差倍数
            
        Returns:
            布林带位置因子值
        """
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return (close - lower) / (upper - lower)
