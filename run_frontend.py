#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/27/2025 4:49 PM
@File       : run_frontend.py
@Description: 
"""
from dotenv import load_dotenv

load_dotenv()

import streamlit as st


def main():
    st.title("Stock Analysis Platform")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 1. Data Viewer
        查看个股K线、技术指标等数据
        """)
        if st.button("进入数据查看", key="data_viewer"):
            st.switch_page("pages/data_viewer.py")

    with col2:
        st.markdown("""
        #### 2. Strategy Scanner
        基于交易策略筛选股票
        """)
        if st.button("进入策略筛选", key="strategy_scanner"):
            st.switch_page("pages/strategy_scanner.py")

    with col3:
        st.markdown("""
        #### 3. Backtest
        策略回测及性能分析
        """)
        if st.button("进入回测系统", key="backtest"):
            st.switch_page("pages/backtest.py")


if __name__ == "__main__":
    main()
