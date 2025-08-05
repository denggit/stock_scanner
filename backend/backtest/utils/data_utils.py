#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据工具类
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class DataUtils:
    """
    数据工具类
    提供数据处理和生成功能
    """
    
    @staticmethod
    def create_sample_data(days: int = 252, start_price: float = 100.0, 
                          volatility: float = 0.02, trend: float = 0.001) -> pd.DataFrame:
        """
        创建示例数据
        
        Args:
            days: 数据天数
            start_price: 起始价格
            volatility: 波动率
            trend: 趋势
            
        Returns:
            示例数据
        """
        # 生成日期序列
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # 生成价格数据
        np.random.seed(42)
        returns = np.random.normal(trend, volatility, days)
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
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据格式和质量
        
        Args:
            data: 待验证的数据
            
        Returns:
            验证结果字典
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            result["is_valid"] = False
            result["errors"].append(f"缺少必需列: {missing_columns}")
        
        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                result["is_valid"] = False
                result["errors"].append(f"列 {col} 必须是数值类型")
        
        # 检查数据完整性
        if not data.empty:
            null_counts = data[required_columns].isnull().sum()
            if null_counts.any():
                result["warnings"].append(f"数据包含空值: {null_counts[null_counts > 0].to_dict()}")
        
        # 检查价格逻辑
        if not data.empty and all(col in data.columns for col in ['high', 'low', 'open', 'close']):
            invalid_prices = (
                (data['high'] < data['low']) |
                (data['open'] > data['high']) |
                (data['close'] > data['high']) |
                (data['open'] < data['low']) |
                (data['close'] < data['low'])
            )
            
            if invalid_prices.any():
                invalid_count = invalid_prices.sum()
                result["warnings"].append(f"发现 {invalid_count} 条价格逻辑错误的数据")
        
        # 添加数据信息
        if not data.empty:
            result["info"] = {
                "start_date": data.index[0] if len(data) > 0 else None,
                "end_date": data.index[-1] if len(data) > 0 else None,
                "total_days": len(data),
                "price_range": {
                    "min": data['close'].min() if 'close' in data.columns else None,
                    "max": data['close'].max() if 'close' in data.columns else None,
                    "mean": data['close'].mean() if 'close' in data.columns else None
                }
            }
        
        return result
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据
        
        Args:
            data: 原始数据
            
        Returns:
            清理后的数据
        """
        # 复制数据
        cleaned_data = data.copy()
        
        # 处理空值
        cleaned_data = cleaned_data.dropna()
        
        # 处理异常值（使用3倍标准差）
        for col in ['open', 'high', 'low', 'close']:
            if col in cleaned_data.columns:
                mean = cleaned_data[col].mean()
                std = cleaned_data[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                # 将异常值替换为边界值
                cleaned_data[col] = cleaned_data[col].clip(lower_bound, upper_bound)
        
        # 确保价格逻辑正确
        if all(col in cleaned_data.columns for col in ['high', 'low', 'open', 'close']):
            cleaned_data['high'] = cleaned_data[['high', 'open', 'close']].max(axis=1)
            cleaned_data['low'] = cleaned_data[['low', 'open', 'close']].min(axis=1)
        
        return cleaned_data
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            data: 价格数据
            
        Returns:
            包含收益率的数据
        """
        if 'close' not in data.columns:
            raise ValueError("数据必须包含close列")
        
        result = data.copy()
        
        # 计算日收益率
        result['daily_return'] = result['close'].pct_change()
        
        # 计算累计收益率
        result['cumulative_return'] = (1 + result['daily_return']).cumprod() - 1
        
        # 计算对数收益率
        result['log_return'] = np.log(result['close'] / result['close'].shift(1))
        
        return result
    
    @staticmethod
    def resample_data(data: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        重采样数据
        
        Args:
            data: 原始数据
            freq: 重采样频率 ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            重采样后的数据
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是日期时间类型")
        
        # 重采样
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled.dropna() 