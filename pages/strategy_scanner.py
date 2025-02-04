#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/4/2025 2:44 AM
@File       : strategy_scanner.py
@Description: 
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import requests


def scan_stocks(strategy: str, params: dict):
    """调用后端API扫描股票"""
    backend_url = os.getenv('BACKEND_URL')
    backend_port = os.getenv('BACKEND_PORT')

    try:
        response = requests.post(f'http://{backend_url}:{backend_port}/api/strategy/scan', json=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"扫描股票失败: {e}")
        return None


def main():
    st.title("策略扫描器")


if __name__ == "__main__":
    main()

