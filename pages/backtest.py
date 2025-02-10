import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


def backtest_page():
    """回测页面"""
    st.title("策略回测系统")

    with st.sidebar:
        st.header("回测设置")

        # 策略选择
        strategy = st.selectbox(
            "选择策略",
            ["爆发式选股策略"],  # 可以添加更多策略

        )

        # 股票池选择
        stock_pool = st.selectbox(
            "选择股票池",
            ["全量股票", "非ST股票", "上证50", "沪深300", "中证500"],
            index=1,  # 默认选择非ST股票
            help="选择要回测的股票池范围"
        )

        # 回测参数设置
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=366)
            )
            initial_capital = st.number_input(
                "初始资金",
                value=100000.0,
                step=10000.0,
                help="设置回测的初始资金金额"
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now()
            )
            max_positions = st.number_input(
                "最大持仓数量",
                min_value=1,
                max_value=100,
                value=4,
                help="同时持有的最大股票数量"
            )

        # 资金分配策略
        allocation_strategy = st.selectbox(
            "资金分配策略",
            ["信号强度加权", "市值加权", "等权重"],
            help="选择多股票间的资金分配方式"
        )

        # 策略特定参数
        with st.expander("策略参数设置", expanded=True):
            params = {}
            if strategy == "爆发式选股策略":
                # 基础参数
                st.subheader("基础参数")
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

                # 买入卖出条件
                st.subheader("回测买入卖出条件设置")

                # 买入条件部分
                st.write("买入条件")
                col1, col2 = st.columns(2)
                with col1:
                    params['buy_conditions'] = {
                        'min_signal': st.number_input(
                            "最小信号分数",
                            min_value=0.0,
                            max_value=100.0,
                            value=70.0,
                            help='买入时要求的最小信号分数'
                        ),
                        'min_volume_ratio': st.number_input(
                            "最小量比",
                            min_value=1.0,
                            max_value=5.0,
                            value=1.5,
                            help='买入时要求的最小成交量比'
                        )
                    }

                with col2:
                    params['buy_conditions'].update({
                        'rsi_range': st.slider(
                            "RSI范围",
                            min_value=0,
                            max_value=100,
                            value=(45, 65),
                            help='买入时RSI指标的合理范围'
                        ),
                        'min_explosion_prob': st.number_input(
                            "最小暴涨概率",
                            min_value=0.0,
                            max_value=100.0,
                            value=50.0,
                            help='买入时要求的最小暴涨概率'
                        )
                    })

                # 卖出条件部分
                st.write("卖出条件")
                col1, col2, col3 = st.columns(3)
                with col1:
                    params['sell_conditions'] = {
                        'profit_target': st.number_input(
                            "止盈目标",
                            min_value=5.0,
                            max_value=100.0,
                            value=30.0,
                            help='达到目标收益后卖出（百分比）'
                        ),
                        'stop_loss': st.number_input(
                            "止损线",
                            min_value=-20.0,
                            max_value=0.0,
                            value=-7.0,
                            help='达到止损线后卖出（百分比）'
                        )
                    }

                with col2:
                    params['sell_conditions'].update({
                        'max_rsi': st.number_input(
                            "RSI超买线",
                            min_value=50,
                            max_value=100,
                            value=85,
                            help='RSI超过该值视为超买'
                        ),
                        'min_signal': st.number_input(
                            "信号转弱阈值",
                            min_value=0.0,
                            max_value=100.0,
                            value=40.0,
                            help='信号低于该值考虑卖出'
                        )
                    })

                with col3:
                    params['sell_conditions'].update({
                        'volume_shrink': st.number_input(
                            "成交量萎缩阈值",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            help='量比低于该值视为量能萎缩'
                        )
                    })

    # 开始回测按钮
    col1, col2 = st.columns([1, 4])
    with col1:
        start_backtest = st.button(
            "开始回测",
            help="点击开始运行回测",
            use_container_width=True
        )

    if start_backtest:
        try:
            with st.spinner('正在运行回测...'):
                backtest_init_params = {
                    "stock_pool": stock_pool,
                    "allocation_strategy": allocation_strategy,
                    "initial_capital": initial_capital,
                    "max_positions": max_positions
                }
                # 调用后端API
                response = run_backtest(
                    strategy=strategy,
                    params=params,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    backtest_init_params=backtest_init_params
                )

                if response.status_code == 200:
                    results = response.json()
                    st.session_state.backtest_results = results  # 保存结果到session state
                else:
                    st.error(f"回测请求失败: {response.text}")
                    return

        except Exception as e:
            st.error(f"回测失败: {str(e)}")
            return

    # 显示回测结果
    if hasattr(st.session_state, 'backtest_results'):
        results = st.session_state.backtest_results

        # 使用tabs来组织不同的结果展示
        tab1, tab2, tab3 = st.tabs(["收益分析", "交易记录", "最新信号"])

        with tab1:
            # 显示主要指标
            st.subheader("回测指标")
            metrics = results["summary"]["metrics"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总收益率", f"{metrics['total_return']:.2f}%")
            with col2:
                st.metric("最大回撤", f"{metrics['max_drawdown']:.2f}%")
            with col3:
                st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
            with col4:
                st.metric("年化收益", f"{metrics.get('annual_return', 0):.2f}%")

            # 显示收益率曲线
            st.subheader("收益率曲线")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(results["summary"]["dates"]),
                y=results["summary"]["returns"],
                mode='lines',
                name='策略收益率'
            ))
            fig.update_layout(
                title="策略收益率走势",
                xaxis_title="日期",
                yaxis_title="收益率(%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # 显示交易记录
            st.subheader("交易记录")
            if results["summary"]["trades"]:
                trades_df = pd.DataFrame(results["summary"]["trades"])
                trades_df['日期'] = pd.to_datetime(trades_df['日期'])
                trades_df = trades_df.sort_values('日期', ascending=False)
                st.dataframe(
                    trades_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("暂无交易记录")

        with tab3:
            # 显示最新信号
            st.subheader("最新信号")
            if results.get("latest_signals"):
                signals_df = pd.DataFrame(results["latest_signals"])
                st.dataframe(
                    signals_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("暂无最新信号")


def run_backtest(strategy: str, params: dict, start_date: str, end_date: str, backtest_init_params: dict):
    """运行回测"""
    backend_url = os.getenv('BACKEND_URL', 'http://localhost')
    backend_port = os.getenv('BACKEND_PORT', '8000')

    url = f"{backend_url}:{backend_port}/api/backtest/run"
    response = requests.get(url, params={
        "strategy": strategy,
        "start_date": start_date,
        "end_date": end_date,
        "backtest_init_params": backtest_init_params,
        "params": params
    })

    return response


if __name__ == "__main__":
    # 直接调用页面函数
    backtest_page()
