#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:59 PM
@File       : stock_service.py
@Description: 
"""
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from backend.data.stock_data_fetcher import StockDataFetcher
from backend.utils import format_info
from backend.utils.logger import setup_logger

logger = setup_logger("stock_service")


class StockService:

    def __init__(self):
        self.data_fetcher = StockDataFetcher()

    async def get_stock_data(
            self,
            code: str,
            period: str = 'daily',
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            ma_periods: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """获取股票数据，考虑均线计算需要的额外数据"""
        try:
            logger.info(f"Fetching {period} data for {code} from {start_date} to {end_date}")
            formatted_code = format_info.stock_code_dot(code)

            # 如果有均线周期，计算需要提前获取的天数
            extra_days = 0
            if ma_periods:
                extra_days = max(ma_periods)  # 取最大值作为额外天数

            # 调节开始日期
            if start_date and extra_days > 0:
                start_date_ori = pd.to_datetime(start_date)
                # 考虑节假日等因素，多取2倍的天数
                adjusted_start_date_dt = start_date_ori - pd.Timedelta(days=extra_days * 2)
                start_date = adjusted_start_date_dt.strftime('%Y-%m-%d')

            # 获取数据
            df = self.data_fetcher.fetch_stock_data(
                code=formatted_code,
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

            if df.empty:
                logger.warning(f"No data found for {formatted_code}")
                return []

            # 如果使用了调整后的开始日期，在返回前截取回原始日期范围
            if extra_days > 0:
                original_start_date = pd.to_datetime(start_date_ori)
                df = df[df.index >= original_start_date]

            # 处理无穷大和NaN值，确保JSON序列化兼容
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # 处理超出JSON范围的浮点数值
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df.columns:
                    # 将超出范围的数值替换为None
                    df[col] = df[col].apply(lambda x: None if pd.isna(x) or abs(x) > 1e308 else x)
            
            # 将NaN替换为None，这样JSON序列化时会被转换为null
            df = df.where(pd.notna(df), None)
            
            # 处理日期字段，确保JSON序列化兼容
            for col in df.columns:
                if df[col].dtype == 'object':
                    # 检查是否包含日期对象
                    sample_value = df[col].iloc[0] if not df[col].empty else None
                    if hasattr(sample_value, 'strftime'):
                        # 将日期对象转换为字符串
                        df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d') if x is not None else None)
            
            # 确保所有数值都是JSON兼容的
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(float)
                    # 处理特殊数值
                    df[col] = df[col].apply(lambda x: None if pd.isna(x) or np.isinf(x) or abs(x) > 1e308 else float(x))
            
            # 确保所有字符串字段都是字符串类型
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                    # 将'None'字符串转换为None
                    df[col] = df[col].replace('None', None)
                    df[col] = df[col].replace('nan', None)
                    df[col] = df[col].replace('', None)

            # 最终的安全处理：确保所有数据都是JSON兼容的
            def safe_json_value(value):
                """确保值是JSON兼容的"""
                if value is None:
                    return None
                elif isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value) or abs(value) > 1e308:
                        return None
                    return float(value) if isinstance(value, float) else int(value)
                elif isinstance(value, str):
                    if value.lower() in ['nan', 'none', 'inf', '-inf', '']:
                        return None
                    return value
                else:
                    return str(value)

            # 应用安全处理到所有数据
            for col in df.columns:
                df[col] = df[col].apply(safe_json_value)

            logger.info(f"Fetched stock data for {formatted_code} from {start_date} to {end_date}")
            return df.to_dict(orient='records')

        except Exception as e:
            logger.error(f"Failed to fetch stock data for {code}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to fetch stock data {code}: {e}")
