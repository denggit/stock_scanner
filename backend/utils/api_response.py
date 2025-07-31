#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 6:41 PM
@File       : api_response.py
@Description: 
"""

from typing import Any

import numpy as np
import pandas as pd
import datetime


def convert_to_python_types(obj: Any) -> Any:
    """将各种数据类型转换为Python原生类型
    
    Args:
        obj: 任意类型的输入对象
        
    Returns:
        转换后的Python原生类型对象
    """
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        # 处理无穷大和NaN值
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        # 处理DataFrame中的特殊值
        df_copy = obj.copy()
        for col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                # 将无穷大和NaN值替换为None
                df_copy[col] = df_copy[col].replace([np.inf, -np.inf], None)
                df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
            else:
                # 对于非数值列，也要处理可能的NaN值
                df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
        
        # 转换为字典记录
        records = df_copy.to_dict('records')
        
        # 进一步处理字典中的特殊值
        for record in records:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = None
                elif pd.isna(value):
                    record[key] = None
        
        return records
    elif isinstance(obj, pd.Series):
        # 处理Series中的特殊值
        series_copy = obj.copy()
        if pd.api.types.is_numeric_dtype(series_copy):
            series_copy = series_copy.replace([np.inf, -np.inf], None)
            series_copy = series_copy.where(pd.notna(series_copy), None)
        else:
            series_copy = series_copy.where(pd.notna(series_copy), None)
        
        # 转换为字典
        result_dict = series_copy.to_dict()
        
        # 进一步处理字典中的特殊值
        for key, value in result_dict.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                result_dict[key] = None
            elif pd.isna(value):
                result_dict[key] = None
        
        return result_dict
    elif isinstance(obj, np.generic):
        # 处理numpy标量类型
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    elif isinstance(obj, (np.ndarray, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, datetime.date):
        return obj.strftime("%Y-%m-%d")
    elif pd.isna(obj):
        return None
    elif isinstance(obj, float):
        # 处理Python float类型的无穷大和NaN
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj
