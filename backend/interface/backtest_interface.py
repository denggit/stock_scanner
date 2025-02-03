#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:56 PM
@File       : backtest_interface.py
@Description: 
"""


from fastapi import APIRouter, HTTPException

from backend.services.backtest_service import BacktestService


router = APIRouter(prefix="/api/backtest")
backtest_service = BacktestService()


@router.get("/run")
async def run_backtest(
    strategy: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    params: dict = {},
):
    """运行回测"""
    try:
        return await backtest_service.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            params=params,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """获取回测结果"""
    try:
        return await backtest_service.get_backtest_results(backtest_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

