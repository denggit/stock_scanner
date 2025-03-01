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
        backtest_init_params: dict,
        params: dict = None,
):
    """运行回测
    
    Args:
        strategy: 策略名称
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        backtest_init_params: 回测初始化参数 {
            "stock_pool": str,  # 股票池名称
            "initial_capital": float,  # 初始资金
            "max_positions": int,  # 最大持仓数量
            "allocation_strategy": str,  # 资金分配策略
        }
        params: 策略参数
    """
    try:
        if params is None:
            params = {}
            
        result = await backtest_service.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            backtest_init_params=backtest_init_params,
            params=params,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="No backtest backtest_results found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/backtest_results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """获取回测结果"""
    try:
        result = await backtest_service.get_backtest_results(backtest_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"No backtest_results found for backtest ID: {backtest_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
