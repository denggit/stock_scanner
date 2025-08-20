#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:46 PM
@File       : app.py
@Description: 
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 修改导入路径
from backend.interface import stock_interface, strategy_interface, backtest_interface

app = FastAPI(
    title="Stock Screener API",
    description="股票筛选和分析系统的后端API",
    version="0.0.1",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(stock_interface.router, tags=["stock"])
app.include_router(strategy_interface.router, tags=["strategy"])
app.include_router(backtest_interface.router, tags=["backtest_event"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Stock Screener API"}


@app.get("/test")
async def test():
    return {"message": "Test successful"}
