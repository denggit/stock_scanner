#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:56 PM
@File       : stock_interface.py
@Description: 
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from backend.services.stock_service import StockService
from backend.utils.logger import setup_logger

router = APIRouter(prefix="/api/stocks")
stock_service = StockService()

logger = setup_logger("stock_service")


@router.get("/{code}")
async def get_stock_data(
        code: str,
        period: str = Query('daily', enum=['daily', 'weekly', 'monthly']),
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ma_periods: Optional[List[int]] = None,
):
    """获取股票数据"""
    logger.info(
        f"API call: get_stock_data - code: {code}, period: {period}, start_date: {start_date}, end_date: {end_date}, ma_periods: {ma_periods}")
    try:
        # 如果没有指定日期，默认获取最近一年的数据
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # 调用服务层获取数据
        data = await stock_service.get_stock_data(
            code=code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            ma_periods=ma_periods
        )
        logger.info(f"Successfully get stock data for code: {code}")
        return data
    except Exception as e:
        logger.exception(f"Error in get_stock_data: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{code}/indicators")
async def get_stock_indicators(
        code: str,
        indicators: List[str] = Query(..., description="指标列表，如：MA, MACD, RSI"),
        date: Optional[str] = Query(None, description="日期，格式为YYYY-MM-DD")
):
    """获取股票指标数据"""
    try:
        return await stock_service.get_stock_indicators(code, indicators, date)
    except Exception as e:
        logger.exception(f"Error in get_stock_indicators: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
