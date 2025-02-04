#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:56 PM
@File       : strategy_interface.py
@Description: 
"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.strategy_service import StrategyService

router = APIRouter(prefix="/api/strategy")
strategy_service = StrategyService()


class ScanRequest(BaseModel):
    """扫描请求"""
    strategy: str
    params: Dict[str, Any]


@router.post("/scan")
async def scan(request: ScanRequest):
    """使用策略扫描股票"""
    try:
        result = await strategy_service.scan_stocks(strategy=request.strategy, params=request.params)
        if result is None:
            raise HTTPException(status_code=404, detail="No results found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/list")
async def list_strategies():
    """列出所有策略"""
    try:
        return await strategy_service.list_strategies()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
