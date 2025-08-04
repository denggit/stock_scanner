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

from backend.utils import format_info
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
    one_year_ago = today - timedelta(days=366)  # 一年前的日期

    # 设置默认日期
    default_start_date = two_years_ago
    default_end_date = today

    # 侧边栏 - 策略设置
    with st.sidebar:
        st.header("策略设置")
        strategy = st.selectbox(
            "选择策略",
            ["均线回踩策略", "突破策略", "波段交易策略", "扫描翻倍股", "长期上涨策略", "头肩底形态策略",
             "爆发式选股策略", "放量上涨策略", "上升通道策略"]
        )

        # 股票池选择
        st.subheader("筛选股票条件")
        stock_pool = st.selectbox(
            "选择股票池",
            ["全量股票", "非ST股票", "上证50", "沪深300", "中证500"],
            index=1,  # 默认选择非ST股票
            help="选择要回测的股票池范围"
        )
        ipo_date = st.date_input("最晚上市日期", value=one_year_ago, help="默认一年前").strftime("%Y-%m-%d")
        min_amount = st.number_input("5日最小成交额", min_value=0, value=100000000,
                                     help="排除五日平均交易额低于该值股票")
        params = {"stock_pool": stock_pool, "ipo_date": ipo_date, "min_amount": min_amount}

        # 策略参数设置
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
                    value=0,
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
                ma_periods = st.multiselect("均线周期", [5, 10, 20, 30, 60, 120, 250], default=[5, 20, 60, 250],
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
                rsi_oversold = st.number_input("超卖阈值", min_value=1, value=45, max_value=100,
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
                "stock_pool": stock_pool,
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
                params["start_date"] = st.date_input("开始日期", value=default_start_date).strftime("%Y-%m-%d")
                params["target_return"] = st.number_input("目标收益率(%)", min_value=0.0, value=100.0, max_value=3000.0,
                                                          format="%.2f",
                                                          help="在翻倍周期内获得大于该收益率")
            with col2:
                params["end_date"] = st.date_input("结束日期", value=default_end_date).strftime("%Y-%m-%d")

            st.subheader("限制条件")
            col1, col2 = st.columns(2)
            with col1:
                choose_period = st.checkbox("周期内", value=True, help="周期内完成翻倍")  # 默认选中
                if choose_period:
                    params["double_period"] = st.number_input("翻倍周期", min_value=0, value=20, max_value=500,
                                                              help="该周期内完成翻倍")
                else:
                    params["double_period"] = 500
            with col2:
                choose_drawdown = st.checkbox("最大回撤", value=False, help="最大回撤出现则停止")
                if choose_drawdown:
                    params["allowed_drawdown"] = st.number_input("最大回撤", min_value=0.00, value=0.05, max_value=0.95,
                                                                 format="%.2f",
                                                                 help="最大回撤值，翻倍前遇到该回撤值则取消")
                else:
                    params["allowed_drawdown"] = 0.95

            # 检查是否至少选择了一个条件
            if not choose_period and not choose_drawdown:
                st.error("请至少选择'周期内'或'最大回撤'其中一个条件")
                st.stop()  # 停止执行后续代码

        elif strategy == "长期上涨策略":
            st.subheader("长期上涨策略参数配置")
            ma_periods = st.multiselect("均线周期", [5, 10, 20, 30, 60, 120, 250], default=[20, 60, 250],
                                        help="长期上涨的均线周期")
            col1, col2 = st.columns(2)
            with col1:
                continuous_days = st.number_input("连续多头排列天数", min_value=1, value=20, max_value=500,
                                                  help="连续多头排列的天数")
                pe_ttm_range = st.slider("市盈率范围",
                                         min_value=-1000.0,
                                         max_value=1000.0,
                                         value=(0.0, 20.0),  # 设置默认范围
                                         step=0.1,
                                         format="%.1f",
                                         help="选择市盈率(PE-TTM)的范围")
                ps_ttm_range = st.slider("市销率范围",
                                         min_value=-1000.0,
                                         max_value=1000.0,
                                         value=(0.0, 20.0),  # 设置默认范围
                                         step=0.1,
                                         format="%.1f",
                                         help="选择市销率(PS-TTM)的范围")
            with col2:
                ma_period = st.number_input("回踩均线", min_value=5, value=20, max_value=500,
                                            help="回踩均线，对比查看回踩哪条均线")
                pb_mrq_range = st.slider("市净率范围",
                                         min_value=-100.0,
                                         max_value=100.0,
                                         value=(0.0, 5.0),  # 设置默认范围
                                         step=0.1,
                                         format="%.1f",
                                         help="选择市净率(PB-MRQ)的范围")
                pcf_ncf_ttm_range = st.slider("市现率范围",
                                              min_value=-30000.0,
                                              max_value=30000.0,
                                              value=(-30000.0, 30000.0),  # 设置默认范围
                                              step=0.1,
                                              format="%.1f",
                                              help="选择市现率(PCF-NCF-TTM)的范围")

            params = {
                "stock_pool": stock_pool,
                "ma_periods": ma_periods,
                "ma_period": ma_period,
                "continuous_days": continuous_days,
                "pe_ttm_range": pe_ttm_range,
                "ps_ttm_range": ps_ttm_range,
                "pb_mrq_range": pb_mrq_range,
                "pcf_ncf_ttm_range": pcf_ncf_ttm_range
            }

        elif strategy == "头肩底形态策略":
            st.subheader("头肩底形态策略参数配置")

            # 基础参数
            col1, col2 = st.columns(2)
            with col1:
                params['lookback_period'] = st.number_input(
                    "回看天数",
                    min_value=60,
                    max_value=250,
                    value=120,
                    help='分析头肩底形态的历史数据天数'
                )
                params['min_pattern_points'] = st.number_input(
                    "最小形态点数",
                    min_value=10,
                    max_value=30,
                    value=15,
                    help='头肩底形态的最小点数要求'
                )
                params['volume_ratio'] = st.number_input(
                    "成交量放大倍数",
                    min_value=1.0,
                    max_value=5.0,
                    value=1.5,
                    format="%.1f",
                    help='突破颈线时的成交量要求（相对于平均成交量）'
                )

            with col2:
                params['shoulder_height_diff'] = st.number_input(
                    "左右肩高度差异",
                    min_value=0.01,
                    max_value=0.30,
                    value=0.10,
                    format="%.2f",
                    help='左右肩高度差异的容忍度（百分比）'
                )
                params['max_pattern_points'] = st.number_input(
                    "最大形态点数",
                    min_value=30,
                    max_value=120,
                    value=60,
                    help='头肩底形态的最大点数限制'
                )
                params['neckline_slope_range'] = st.slider(
                    "颈线最大斜率",
                    min_value=-0.50,
                    max_value=0.50,
                    value=(-0.087, 0.268),
                    format="%.3f",
                    help='颈线允许的斜率范围'
                )

            # 高级参数
            st.subheader("信号强度权重设置")
            col1, col2, col3 = st.columns(3)
            weights = {}
            with col1:
                weights['pattern'] = st.number_input(
                    "形态完整",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    format="%.1f",
                    help='形态标准度对信号强度的影响'
                )
            with col2:
                weights['volume'] = st.number_input(
                    "成交量",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    format="%.1f",
                    help='成交量表现对信号强度的影响'
                )
            with col3:
                weights['breakout'] = st.number_input(
                    "突破强度",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    format="%.1f",
                    help='颈线突破强度对信号强度的影响'
                )

            # 检查权重和是否为1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"权重和必须为1.0, 当前权重和为 {total_weight}")

            params['weights'] = weights

            # 添加过滤条件
            st.subheader("过滤条件")
            col1, col2 = st.columns(2)
            with col1:
                params['min_volume'] = st.number_input(
                    "最小成交量（手）",
                    min_value=1000,
                    max_value=1000000,
                    value=10000,
                    step=1000,
                    help='最小成交量要求（手）'
                )
                params['min_amount'] = st.number_input(
                    "最小成交额（万）",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help='最小成交额要求（万元）'
                )

            with col2:
                params['price_range'] = st.slider(
                    "股价范围",
                    min_value=0.0,
                    max_value=1000.0,
                    value=(5.0, 200.0),
                    step=0.1,
                    format="%.1f",
                    help='选股的价格范围'
                )

        elif strategy == "爆发式选股策略":
            st.subheader("爆发式选股策略参数配置")

            # 添加持仓信息
            st.subheader("持仓信息")

            # 初始化持仓列表
            if 'holdings' not in st.session_state:
                st.session_state.holdings = []

            # 使用 expander 来收起持仓信息输入区域
            with st.expander("添加持仓股票", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    stock_code = st.text_input(
                        "股票代码",
                        placeholder="例如: 000001",
                        help="输入持仓股票代码"
                    )
                    stock_code = format_info.stock_code_dot(stock_code)
                with col2:
                    cost_price = st.number_input(
                        "持仓成本",
                        min_value=0.01,
                        value=10.0,
                        format="%.2f",
                        help="输入持仓成本价"
                    )
                with col3:
                    if st.button("添加持仓"):
                        hold = {'code': stock_code, 'cost': cost_price}
                        if stock_code and cost_price > 0 and hold not in st.session_state.holdings:
                            st.session_state.holdings.append(hold)
                            st.success(f"已添加持仓: {stock_code}")

            # 显示当前持仓
            if st.session_state.holdings:
                st.write("当前持仓:")
                holdings_df = pd.DataFrame(st.session_state.holdings)
                holdings_df.columns = ['股票代码', '持仓成本']
                st.dataframe(holdings_df)

                # 添加清除持仓按钮
                if st.button("清除所有持仓"):
                    st.session_state.holdings = []
                    st.success("已清除所有持仓信息")

            # 将持仓信息添加到参数中
            params['holdings'] = st.session_state.holdings

            # 基础参数
            col1, col2 = st.columns(2)
            with col1:
                params['volume_ma'] = st.number_input(
                    "成交量均线周期",
                    min_value=5,
                    max_value=60,
                    value=20,
                    help='计算成交量均线的周期'
                )
                params['rsi_period'] = st.number_input(
                    "RSI周期",
                    min_value=5,
                    max_value=30,
                    value=14,
                    help='计算RSI指标的周期'
                )
                params['bb_period'] = st.number_input(
                    "布林带周期",
                    min_value=5,
                    max_value=60,
                    value=20,
                    help='计算布林带的周期'
                )

            with col2:
                params['bb_std'] = st.number_input(
                    "布林带标准差倍数",
                    min_value=1.0,
                    max_value=4.0,
                    value=2.0,
                    format="%.1f",
                    help='布林带的标准差倍数'
                )
                params['recent_days'] = st.number_input(
                    "近期趋势分析天数",
                    min_value=3,
                    max_value=20,
                    value=5,
                    help='分析近期趋势的天数'
                )
                params['period'] = st.number_input(
                    "获取股票数据长度",
                    min_value=60,
                    max_value=500,
                    value=100,
                    help='用于计算的股票交易天数'
                )

            st.subheader("过滤条件")
            need_filter = st.checkbox("是否过滤结果", value=False, help="通过下列条件过滤扫描结果")
            if need_filter:
                col1, col2 = st.columns(2)
                with col1:
                    params["signal"] = st.number_input(
                        "综合分数",
                        min_value=0.0,
                        max_value=100.00,
                        value=70.0,
                        format="%.2f",
                        help="返回结果signal大于该值"
                    )
                    params["explosion_probability"] = st.number_input(
                        "暴涨概率",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        format="%.2f",
                        help="返回结果暴涨概率大于该值"
                    )
                with col2:
                    params["volume_ratio"] = st.number_input(
                        "量比",
                        min_value=0.0,
                        max_value=10.0,
                        value=1.5,
                        format="%.2f",
                        help="返回结果增量比例需大于该值"
                    )
                    params["rsi_range"] = st.slider(
                        "RSI区间",
                        min_value=0.0,
                        max_value=100.0,
                        value=(45.0, 65.0),  # 设置默认范围
                        step=0.1,
                        format="%.1f",
                        help="返回结果rsi在此区间"
                    )

            # 权重设置
            st.subheader("信号强度权重设置")
            weights = {}
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                weights['volume'] = st.number_input(
                    "成交量",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.35,
                    format="%.2f",
                    help='成交量分析的权重'
                )
            with col2:
                weights['momentum'] = st.number_input(
                    "动量",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.30,
                    format="%.2f",
                    help='动量分析的权重'
                )
            with col3:
                weights['pattern'] = st.number_input(
                    "形态",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    format="%.2f",
                    help='形态分析的权重'
                )
            with col4:
                weights['volatility'] = st.number_input(
                    "波动性",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    format="%.2f",
                    help='波动性分析的权重'
                )

            # 检查权重和是否为1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"权重和必须为1.0, 当前权重和为 {total_weight}")

            params['weights'] = weights

        elif strategy == "放量上涨策略":
            st.subheader("放量上涨策略参数配置")
            params["continuous_days"] = st.number_input("量价齐升天数", min_value=1, value=3, max_value=10,
                                                        help="持续量价齐升天数")

        elif strategy == "上升通道策略":
            st.subheader("上升通道策略参数配置")
            params['start_date'] = st.date_input("开始日期", value=one_year_ago).strftime("%Y-%m-%d")

            # 基础通道参数
            col1, col2 = st.columns(2)
            with col1:
                params['k'] = st.number_input(
                    "通道宽度倍数 (k)",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    help="通道宽度倍数，影响通道的宽度 (±k·σ)"
                )
                params['L_max'] = st.number_input(
                    "最大窗口长度 (L_max)",
                    min_value=60,
                    max_value=200,
                    value=120,
                    step=10,
                    help="窗口最长天数，超出后向右滑动"
                )
                params['delta_cut'] = st.number_input(
                    "滑动剔除天数 (delta_cut)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                    help="滑动时一次剔除最早的天数"
                )
                params['pivot_m'] = st.number_input(
                    "锚点检测参数 (pivot_m)",
                    min_value=2,
                    max_value=10,
                    value=3,
                    step=1,
                    help="判断pivot low的宽度参数 (m左m右更高)"
                )

            with col2:
                params['gain_trigger'] = st.number_input(
                    "重锚涨幅触发 (gain_trigger)",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.30,
                    step=0.05,
                    help="累计涨幅触发重锚的阈值"
                )
                params['beta_delta'] = st.number_input(
                    "斜率变化阈值 (beta_delta)",
                    min_value=0.05,
                    max_value=0.3,
                    value=0.15,
                    step=0.05,
                    help="斜率变化阈值 (±15%)"
                )
                params['break_days'] = st.number_input(
                    "连续突破天数 (break_days)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help="连续n日突破上下沿视为失效"
                )
                params['reanchor_fail_max'] = st.number_input(
                    "重锚失败次数 (reanchor_fail_max)",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1,
                    help="连续n次重锚仍突破/跌破时进入极端状态"
                )

            # 质量参数
            st.subheader("质量参数设置")
            col1, col2, col3 = st.columns(3)
            with col1:
                params['min_data_points'] = st.number_input(
                    "最小数据点数 (min_data_points)",
                    min_value=30,
                    max_value=100,
                    value=60,
                    step=5,
                    help="最小有效数据点要求"
                )
                params['R2_min'] = st.number_input(
                    "最小R²值 (R2_min)",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.20,
                    step=0.05,
                    help="最小回归拟合优度，低于此视为无效通道"
                )

            with col2:
                params['width_pct_min'] = st.number_input(
                    "通道宽度下限 (width_pct_min)",
                    min_value=0.02,
                    max_value=0.10,
                    value=0.04,
                    step=0.01,
                    help="通道宽度下限，小于此视为过窄"
                )
                params['width_pct_max'] = st.number_input(
                    "通道宽度上限 (width_pct_max)",
                    min_value=0.08,
                    max_value=0.20,
                    value=0.12,
                    step=0.01,
                    help="通道宽度上限，超过此视为过宽"
                )

            with col3:
                params['min_signal_score'] = st.number_input(
                    "最小信号分数 (min_signal_score)",
                    min_value=30,
                    max_value=90,
                    value=60,
                    step=5,
                    help="最小综合信号分数，低于此不显示"
                )
                params['min_r2'] = st.number_input(
                    "最小R²值 (min_r2)",
                    min_value=0.1,
                    max_value=0.6,
                    value=0.30,
                    step=0.05,
                    help="最小R²值，用于最终筛选"
                )

            # 筛选条件
            st.subheader("筛选条件设置")
            col1, col2, col3 = st.columns(3)
            with col1:
                params['width_pct_max'] = st.number_input(
                    "最大通道宽度 (width_pct_max)",
                    min_value=0.10,
                    max_value=0.25,
                    value=0.15,
                    step=0.01,
                    help="最大通道宽度，超过此视为过宽"
                )
                params['min_slope_deg'] = st.number_input(
                    "最小斜率角度 (min_slope_deg)",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="最小斜率角度，低于此视为趋势过弱"
                )

            with col2:
                params['max_volatility'] = st.number_input(
                    "最大波动率 (max_volatility)",
                    min_value=0.03,
                    max_value=0.15,
                    value=0.08,
                    step=0.01,
                    help="最大波动率，超过此视为风险过高"
                )

            # 评分权重设置
            st.subheader("评分权重设置")
            weights = {}
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                weights['channel_quality'] = st.number_input(
                    "通道质量权重",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.35,
                    format="%.2f",
                    help='通道质量对综合得分的影响权重'
                )
            with col2:
                weights['trend_strength'] = st.number_input(
                    "趋势强度权重",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    format="%.2f",
                    help='趋势强度对综合得分的影响权重'
                )
            with col3:
                weights['risk_control'] = st.number_input(
                    "风险控制权重",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    format="%.2f",
                    help='风险控制对综合得分的影响权重'
                )
            with col4:
                weights['volume_analysis'] = st.number_input(
                    "成交量分析权重",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    format="%.2f",
                    help='成交量分析对综合得分的影响权重'
                )

            # 检查权重和是否为1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"权重总和应为1.0，当前为{total_weight:.2f}")
            else:
                st.success("权重设置正确")

            params['weights'] = weights

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
            try:
                start_time = datetime.now()
                results = scan_stocks(st.session_state.last_params['strategy'],
                                      st.session_state.last_params['params'])
                end_time = datetime.now()

                if results is not None:  # 确保results不是None
                    st.session_state.scan_results = {
                        'results': results,
                        'start_time': start_time,
                        'end_time': end_time
                    }
                    st.success(f"扫描完成！找到 {len(results)} 只符合条件的股票")
                else:
                    st.error("扫描失败，请检查网络连接或后端服务状态")
                    st.session_state.scan_results = None

            except Exception as e:
                st.error(f"扫描过程中发生错误: {e}")
                st.session_state.scan_results = None

            finally:
                st.session_state.scanning = False
                st.rerun()

    # 显示结果
    if st.session_state.scan_results:
        results = st.session_state.scan_results['results']
        start_time = st.session_state.scan_results['start_time']
        end_time = st.session_state.scan_results['end_time']

        # 不要立即清空scan_results，让用户可以看到结果
        # st.session_state.scan_results = None  # 注释掉这行

        if results:
            # 将结果转换为DataFrame
            if isinstance(results, pd.DataFrame):
                df = results
            else:
                df = pd.DataFrame(results)

            # 显示统计信息
            st.subheader(f"扫描结果统计，耗时 {end_time - start_time}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("符合条件股票数量", len(df))
            with col2:
                # 检查是否有signal字段，如果没有则使用signal_strength
                if 'signal' in df.columns:
                    avg_signal = df['signal'].mean()
                    signal_col = 'signal'
                elif 'signal_strength' in df.columns:
                    avg_signal = df['signal_strength'].mean()
                    signal_col = 'signal_strength'
                else:
                    avg_signal = 0
                    signal_col = None
                st.metric("平均信号分数", f"{avg_signal:.1f}")
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
                "max_drawdown": st.column_config.NumberColumn(
                    '最大回撤',
                    format='%.2f'
                ),
                "pullback_count": "回踩均线次数",
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
                "signal": st.column_config.NumberColumn(
                    '信号分数',
                    format='%.2f'
                ),
                "signal": "买卖信号",
                # 上升通道策略特有字段
                "channel_status": "通道状态",
                "beta": st.column_config.NumberColumn(
                    '斜率',
                    format='%.4f'
                ),
                "r2": st.column_config.NumberColumn(
                    'R²值',
                    format='%.3f'
                ),
                "width_pct": st.column_config.NumberColumn(
                    '通道宽度',
                    format='%.3f'
                ),
                "slope_deg": st.column_config.NumberColumn(
                    '斜率角度',
                    format='%.2f°'
                ),
                "volatility": st.column_config.NumberColumn(
                    '波动率',
                    format='%.3f'
                ),
                "cumulative_gain": st.column_config.NumberColumn(
                    '累计涨幅',
                    format='%.3f'
                ),
                "window_size": "窗口大小",
                "days_since_anchor": "距锚点天数",
                "mid_today": st.column_config.NumberColumn(
                    '今日中轴',
                    format='%.2f'
                ),
                "upper_today": st.column_config.NumberColumn(
                    '今日上沿',
                    format='%.2f'
                ),
                "lower_today": st.column_config.NumberColumn(
                    '今日下沿',
                    format='%.2f'
                ),
                "price_position": "价格位置",
                "anchor_date": "锚点日期",
                "anchor_price": st.column_config.NumberColumn(
                    '锚点价格',
                    format='%.2f'
                ),
                "channel_quality_score": st.column_config.NumberColumn(
                    '通道质量得分',
                    format='%.1f'
                ),
                "trend_strength_score": st.column_config.NumberColumn(
                    '趋势强度得分',
                    format='%.1f'
                ),
                "risk_control_score": st.column_config.NumberColumn(
                    '风险控制得分',
                    format='%.1f'
                ),
                "volume_analysis_score": st.column_config.NumberColumn(
                    '成交量分析得分',
                    format='%.1f'
                ),
                "buy_signal": "买入建议",
                "risk_level": "风险等级",
                "position_advice": "持仓建议"
            }

            st.dataframe(df, column_config=column_config, hide_index=True)

            # 添加股票代码点击跳转功能
            st.subheader("🔍 查看个股详情")

            # 检查DataFrame是否有必要的字段
            if 'code' in df.columns and 'name' in df.columns:
                # 生成股票选择选项
                stock_options = [f"{row['code']} - {row['name']}" for _, row in df.iterrows()]

                selected_stock = st.selectbox(
                    "选择要查看的股票",
                    options=stock_options,
                    index=0,
                    help="选择股票后点击下方按钮查看详细数据"
                )

                if selected_stock:
                    stock_code = selected_stock.split("-")[0].strip()
                    stock_name = selected_stock.split("-")[1].strip()

                    # 构建URL
                    import urllib.parse
                    query_params = {
                        'code': stock_code,
                        'name': stock_name,
                        'strategy': st.session_state.last_params['strategy']
                    }
                    query_string = urllib.parse.urlencode(query_params)
                    data_viewer_url = f"http://localhost:8501/data_viewer?{query_string}"

                    # 显示跳转按钮
                    st.link_button(
                        "🔗 直接跳转到数据查看器",
                        data_viewer_url,
                        type="secondary",
                        help="点击此按钮直接跳转到数据查看器页面"
                    )
            else:
                st.error("数据中缺少股票代码或名称字段，无法显示股票选择功能")

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

            # 添加清除结果按钮
            if st.button("清除结果", type="secondary"):
                st.session_state.scan_results = None
                st.rerun()

        else:
            st.info("未找到符合条件的股票")
            # 清除空结果
            st.session_state.scan_results = None


if __name__ == "__main__":
    main()
