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
import requests
import streamlit as st

from backend.utils.file_check import ensure_dir


def scan_stocks(strategy: str, params: dict):
    """调用后端API扫描股票"""
    backend_url = os.getenv('BACKEND_URL')
    backend_port = os.getenv('BACKEND_PORT')

    try:
        response = requests.post(f'http://{backend_url}:{backend_port}/api/strategy/scan',
                                 json={"strategy": strategy, "params": params})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"扫描股票失败: {e}")
        return None


def main():
    # 初始化 session state
    if 'scanning' not in st.session_state:
        st.session_state.scanning = False
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None
    if 'last_params' not in st.session_state:
        st.session_state.last_params = {}

    st.title("策略扫描器")

    # 计算日期范围
    today = datetime.today()
    two_years_ago = today - timedelta(days=365 * 2)  # 两年前的日期

    # 设置默认日期
    default_start_date = two_years_ago
    default_end_date = today

    # 侧边栏 - 策略设置
    with st.sidebar:
        st.header("策略设置")
        strategy = st.selectbox(
            "选择策略",
            ["均线回踩策略", "突破策略", "波段交易策略", "扫描翻倍股", "长期上涨策略"]
        )
        params = {}
        if strategy == "均线回踩策略":
            st.subheader("均线回踩策略参数配置")

            col1, col2 = st.columns(2)
            with col1:
                params['ma_period'] = st.number_input(
                    "均线周期",
                    min_value=5,
                    max_value=200,
                    value=20,
                    help='计算移动平均线的周期，常用值：20、50、120等'
                )
                params['lookback_period'] = st.number_input(
                    "回溯周期",
                    min_value=5,
                    max_value=200,
                    value=20,
                    help='计算回踩的周期，常用值：10、20、30等'
                )
                params['min_pullback_count'] = st.number_input(
                    "最小回踩次数",
                    min_value=0,
                    max_value=10,
                    value=2,
                    help='回溯其内最少需要的回踩次数'
                )
            with col2:
                params['price_margin'] = st.number_input(
                    "价格误差范围",
                    min_value=0.001,
                    max_value=0.05,
                    value=0.01,
                    help='价格与均线的最大偏离比例'
                )
                params['volume_ratio'] = st.number_input(
                    "成交量放大倍数",
                    min_value=1.0,
                    max_value=10.0,
                    value=1.5,
                    help='相对于5日均量的最小放大倍数'
                )

            # 权重设置
            st.subheader("信号强度权重设置")
            weights = {}
            col1, col2, col3 = st.columns(3)
            with col1:
                weights['price'] = st.number_input(
                    "价格偏离",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    format="%.1f",
                    help='价格偏离度对信号强度的影响，计算的是价格距离均线的距离，值越小，信号强度越高'
                )
            with col2:
                weights['volume'] = st.number_input(
                    "成交量",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    format="%.1f",
                    help='成交量对信号强度的影响，计算的是成交量与5日均量的比例，值越大，信号强度越高'
                )
            with col3:
                weights['frequency'] = st.number_input(
                    "回踩频率",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    format="%.1f",
                    help='回踩频率对信号强度的影响，计算的是回溯周期内，回踩均线的频率，值越大，信号强度越高'
                )

            # 检查权重和是否为1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"权重和必须为1.0, 当前权重和为 {total_weight}")

            params['weights'] = weights

        elif strategy == "突破策略":
            pass
        elif strategy == "波段交易策略":
            st.subheader("波段交易策略参数配置")
            bullish = st.checkbox("多头排列", value=True, help="是否多头排列")
            ma_periods = []
            if bullish:
                ma_periods = st.multiselect("均线周期", [5, 10, 20, 30, 60, 120, 250], default=[5, 20, 250],
                                            help="多头排列的均线周期")

            # 时间周期参数
            col1, col2 = st.columns(2)
            with col1:
                short_ma_period = st.number_input("短期均线周期", min_value=3, value=5, max_value=250, help="短期均线")
                lookback_period = st.number_input("回溯周期", min_value=5, value=20, max_value=250, help="回溯周期")
            with col2:
                long_ma_period = st.number_input("长期均线周期", min_value=10, value=20, max_value=250, help="长期均线")

            # MACD参数
            st.subheader("MACD参数配置")
            col1, col2 = st.columns(2)
            with col1:
                macd_slow_period = st.number_input("慢速均线周期", min_value=1, value=26, max_value=250,
                                                   help="慢速均线")
                macd_signal_period = st.number_input("信号均线周期", min_value=1, value=9, max_value=250,
                                                     help="信号均线")
            with col2:
                macd_fast_period = st.number_input("快速均线周期", min_value=1, value=12, max_value=250,
                                                   help="快速均线")

            # RSI参数
            st.subheader("RSI参数配置")
            col1, col2 = st.columns(2)
            with col1:
                rsi_overbought = st.number_input("超买阈值", min_value=1, value=70, max_value=100,
                                                 help="卖出条件需满足RSI大于该值")
                rsi_period = st.number_input("RSI周期", min_value=1, value=14, max_value=250, help="RSI周期")
            with col2:
                rsi_oversold = st.number_input("超卖阈值", min_value=1, value=30, max_value=100,
                                               help="买入条件需满足RSI小于该值")

            # 振幅与波动性参数
            st.subheader("振幅与波动性参数配置")
            col1, col2 = st.columns(2)
            with col1:
                volatility_threshold = st.number_input("波动性阈值", min_value=0.02, value=0.10, max_value=0.10,
                                                       format="%.3f", help="波动性需大于该值")
                bollinger_k = st.number_input("布林带宽度", min_value=1.0, value=2.0, max_value=3.0, format="%.1f",
                                              help="布林带系数")
            with col2:
                amplitude_threshold = st.number_input("振幅阈值", min_value=0.02, value=0.05, max_value=0.20,
                                                      format="%.3f", help="回溯周期内，振幅大于该值")

            # 权重参数
            st.subheader("权重参数配置")
            col1, col2 = st.columns(2)
            with col1:
                w_price = st.number_input("价格权重", min_value=0.0, value=0.3, max_value=1.0, format="%.2f",
                                          help="价格对信号强度的影响")
                w_trend = st.number_input("趋势权重", min_value=0.0, value=0.2, max_value=1.0, format="%.2f",
                                          help="趋势对信号强度的影响")
                w_rsi = st.number_input("RSI权重", min_value=0.0, value=0.15, max_value=1.0, format="%.2f",
                                        help="RSI对信号强度的影响")
            with col2:
                w_volatility = st.number_input("波动性权重", min_value=0.0, value=0.2, max_value=1.0, format="%.2f",
                                               help="波动性对信号强度的影响")
                w_macd = st.number_input("MACD权重", min_value=0.0, value=0.15, max_value=1.0, format="%.2f",
                                         help="MACD对信号强度的影响")

            # 检查权重和是否为1
            total_weight = w_price + w_trend + w_rsi + w_volatility + w_macd
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"权重和必须为1.0, 当前权重和为 {total_weight:.2f}")

            params = {
                "bullish": bullish,
                "ma_periods": ma_periods,
                "lookback_period": lookback_period,
                "short_ma_period": short_ma_period,
                "long_ma_period": long_ma_period,
                "macd_slow_period": macd_slow_period,
                "macd_signal_period": macd_signal_period,
                "macd_fast_period": macd_fast_period,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold,
                "rsi_period": rsi_period,
                "volatility_threshold": volatility_threshold,
                "bollinger_k": bollinger_k,
                "amplitude_threshold": amplitude_threshold,
                "weights": {
                    "price": w_price,
                    "trend": w_trend,
                    "rsi": w_rsi,
                    "volatility": w_volatility,
                    "macd": w_macd
                }
            }

        elif strategy == "扫描翻倍股":
            st.subheader("扫描翻倍股参数配置")

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("开始日期", value=default_start_date).strftime("%Y-%m-%d")
                double_period = st.number_input("翻倍周期", min_value=1, value=20, max_value=250,
                                                help="翻倍周期，交易日")
            with col2:
                end_date = st.date_input("结束日期", value=default_end_date).strftime("%Y-%m-%d")
                times = st.number_input("增长倍数", min_value=1.0, value=2.0, max_value=30.0, format="%.1f",
                                        help="在翻倍周期内大于该增长倍数")

            params = {
                "start_date": start_date,
                "end_date": end_date,
                "double_period": double_period,
                "times": times
            }

        elif strategy == "长期上涨策略":
            st.subheader("长期上涨策略参数配置")
            ma_periods = st.multiselect("均线周期", [5, 10, 20, 30, 60, 120, 250], default=[20, 60, 250],
                                        help="长期上涨的均线周期")
            col1, col2 = st.columns(2)
            with col1:
                continuous_days = st.number_input("连续多头排列天数", min_value=1, value=20, max_value=500,
                                                  help="连续多头排列的天数")
            with col2:
                ma_period = st.number_input("回踩均线", min_value=5, value=20, max_value=500,
                                            help="回踩均线，对比查看回踩哪条均线")

            params = {
                "ma_periods": ma_periods,
                "ma_period": ma_period,
                "continuous_days": continuous_days
            }

    # 主界面
    col1, col2 = st.columns([1, 4])  # 创建两列，比例为1:4
    with col1:
        if st.button("开始扫描", key='start_scan', disabled=st.session_state.scanning):
            st.session_state.scanning = True
            st.session_state.last_params = {"strategy": strategy, "params": params}  # 保存当前参数
            st.rerun()

    with col2:
        # 添加取消扫描按钮
        if st.session_state.scanning:
            if st.button("取消扫描", type="secondary"):
                st.session_state.scanning = False
                st.rerun()

    # 如果正在扫描，显示进度
    if st.session_state.scanning:
        with st.spinner("正在扫描，请稍等..."):
            start_time = datetime.now()
            results = scan_stocks(st.session_state.last_params['strategy'],
                                  st.session_state.last_params['params'])
            end_time = datetime.now()

            if results:
                st.session_state.scan_results = {
                    'results': results,
                    'start_time': start_time,
                    'end_time': end_time
                }
            else:
                st.session_state.scan_results = {
                    'results': None,
                    'start_time': start_time,
                    'end_time': end_time
                }

            st.session_state.scanning = False
            st.rerun()

    # 显示结果
    if st.session_state.scan_results:
        results = st.session_state.scan_results['results']
        start_time = st.session_state.scan_results['start_time']
        end_time = st.session_state.scan_results['end_time']
        st.session_state.scan_results = None

        if results:
            # 将结果转换为DataFrame
            df = pd.DataFrame(results)

            # 显示统计信息
            st.subheader(f"扫描结果统计，耗时 {end_time - start_time}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("符合条件股票数量", len(df))
            with col2:
                avg_strength = df['signal_strength'].mean() if 'signal_strength' in df.columns else 0
                st.metric("平均信号强度", f"{avg_strength:.1f}")
            with col3:
                st.metric("扫描时间", f"{datetime.now().strftime('%m-%d %H:%M')}")

            # 显示结果表格
            st.subheader(f"扫描结果 - {st.session_state.last_params['strategy']}")

            # 设置列的显示格式
            column_config = {
                "code": "股票代码",
                "name": "股票名称",
                "trade_date": "交易日期",
                "start_date": "开始日期",
                "end_date": "结束日期",
                "times": st.column_config.NumberColumn(
                    '增长倍率',
                    format='%.2f'
                ),
                "max_return": st.column_config.NumberColumn(
                    '最大收益',
                    format='%.2f'
                ),
                "price": st.column_config.NumberColumn(
                    '当前价格',
                    format='%.2f'
                ),
                "start_price": st.column_config.NumberColumn(
                    '起始价格',
                    format='%.2f'
                ),
                "end_price": st.column_config.NumberColumn(
                    '结束价格',
                    format='%.2f'
                ),
                "pct_chg": st.column_config.NumberColumn(
                    '涨跌幅',
                    format='%.2f%%'
                ),
                "ma_price": st.column_config.NumberColumn(
                    '均线价格',
                    format='%.2f'
                ),
                "price_to_ma": st.column_config.NumberColumn(
                    '价格偏离度',
                    format='%.2f%%'
                ),
                "volume_ratio": st.column_config.NumberColumn(
                    '成交量比',
                    format='%.2f'
                ),
                "continuous_trend_days": "连续趋势天数",
                "pe_ttm": st.column_config.NumberColumn(
                    '市盈率',
                    format='%.2f'
                ),
                "ps_ttm": st.column_config.NumberColumn(
                    '市销率',
                    format='%.2f'
                ),
                "pcf_ncf_ttm": st.column_config.NumberColumn(
                    '市现率',
                    format='%.2f'
                ),
                "pb_mrq": st.column_config.NumberColumn(
                    '市净率',
                    format='%.2f'
                ),
                "signal_strength": st.column_config.NumberColumn(
                    '信号强度',
                    format='%.2f'
                ),
                "signal": "买卖信号"
            }

            st.dataframe(df, column_config=column_config, hide_index=True)

            # 添加下载按钮
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="下载选股结果",
                data=csv,
                file_name="stock_signals.csv",
                mime="text/csv",
                key="download-csv"
            )

            # 创建必要的目录
            results_dir = os.path.join(os.getcwd(), "results", datetime.today().strftime("%Y%m%d"))
            ensure_dir(results_dir)

            # 设置文件路径
            scan_time = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"{strategy}_{scan_time}.xlsx"
            final_file = os.path.join(results_dir, filename)

            # 创建一个ExcelWriter对象，先保存到临时文件
            with pd.ExcelWriter(final_file, engine='openpyxl') as writer:
                # 将参数写入第一个sheet
                params_df = pd.DataFrame([st.session_state.last_params['params']])
                params_df.to_excel(writer, sheet_name='参数设置', index=False)

                # 将扫描结果写入第二个sheet
                df.to_excel(writer, sheet_name='扫描结果', index=False)

            # 显示保存成功消息
            st.success(f"结果已自动保存到: {final_file}")

        else:
            st.info("未找到符合条件的股票")


if __name__ == "__main__":
    main()
