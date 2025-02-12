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

            logger.info(f"Fetched stock data for {formatted_code} from {start_date} to {end_date}")
            return df.to_dict(orient='records')

        except Exception as e:
            logger.error(f"Failed to fetch stock data for {code}: {e}")
            raise Exception(f"Failed to fetch stock data {code}: {e}")
