#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:43 PM
@File       : data_viewer.py
@Description: 
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from backend.utils.indicators import CalIndicators


def fetch_stock_data(code: str, period: str = 'daily', start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从后端API获取股票数据，带上时间范围"""
    backend_url = os.getenv('BACKEND_URL')
    backend_port = os.getenv('BACKEND_PORT')

    params = {'period': period}

    if start_date and end_date:
        params['start_date'] = start_date
        params['end_date'] = end_date

    try:
        response = requests.get(f'http://{backend_url}:{backend_port}/api/stock/{code}', params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"获取股票数据失败: {e}")
        return pd.DataFrame()


def plot_candlestick(df: pd.DataFrame, ma_periods: list, show_volume: bool = True, show_macd: bool = False, start_date: str = None, end_date: str = None) -> go.Figure:
    """绘制K线图和副图"""
    pass


def main():
    st.title("数据查看器")



if __name__ == "__main__":
    main()

