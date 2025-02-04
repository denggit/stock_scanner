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


def plot_candlestick(df: pd.DataFrame, ma_periods: list, show_volume: bool = True, show_macd: bool = False,
                     start_date: str = None, end_date: str = None) -> go.Figure:
    """绘制K线图和副图"""
    # 多获取数据的df，用于计算均线
    df_extra = df.copy()

    # 过滤日期范围
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # 过滤掉没有数据的日期
    df = df.dropna(subset=['open', 'close', 'high', 'low', 'volume'])

    # 创建子图，根据是否显示MACD决定子图数量
    rows = 2 if not show_macd else 3
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,  # 共享X轴
        vertical_spacing=0.03,  # 子图间距
        row_heights=[0.6, 0.2] if not show_macd else [0.6, 0.2, 0.2]
    )  # 主图附图高度比例

    # 绘制K线图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing=dict(line=dict(color='red')),
            decreasing=dict(line=dict(color='green')),
            name='K线图'
        ),
        row=1, col=1
    )

    # 计算并添加均线（使用完整数据计算，但只显示指定范围）
    for ma_period in ma_periods:
        # 使用原始数据计算均线，确保开始日期也有值
        ma_series = df_extra['close'].rolling(window=ma_period).mean()

        # 过滤日期范围
        if start_date:
            ma_series = ma_series[ma_series.index >= start_date]
        if end_date:
            ma_series = ma_series[ma_series.index <= end_date]

        # 添加均线到图表
        fig.add_trace(
            go.Scatter(
                x=ma_series.index,
                y=ma_series,
                mode='lines',
                name=f'{ma_period}日均线',
                line=dict(width=2)
            ), row=1, col=1
        )

    # 根据附图数量调整总高度和比例
    base_height = 300  # 主图基础高度
    sub_height = 150  # 附图基础高度
    total_height = base_height
    row_heights = [0.6]  # 主图高度比例

    # 添加成交量图（使用已过滤的数据）
    if show_volume:
        colors = ['red' if row['close'] >= row['open'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                marker=dict(color=colors),
                name='成交量'
            ),
            row=2, col=1
        )
        total_height += sub_height
        row_heights.append(0.2)

    # 添加MACD图（使用已过滤的数据）
    if show_macd:
        dif, dea, macd_hist = CalIndicators.macd(df, fast_period=12, slow_period=26, signal_period=9)

        # 添加MACD柱状图
        colors = ['red' if val >= 0 else 'green' for val in macd_hist]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=macd_hist,
                marker=dict(color=colors),
                name='MACD'
            ),
            row=3, col=1
        )

        # 添加DIF和DEA线
        fig.add_trace(
            go.Scatter(x=df.index, y=dif, mode='lines', name='DIF', line=dict(width=2, color='orange')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=dea, mode='lines', name='DEA', line=dict(width=2, color='blue')),
            row=3, col=1
        )

        total_height += sub_height
        row_heights.append(0.2)

    # 调整图表布局
    layout_params = {
        'title': 'K线图',
        'height': total_height,
        'xaxis_rangeslider_visible': False,
        'yaxis_title': '价格',
        'yaxis2_title': '成交量',
        'hovermode': 'x unified',  # 鼠标悬停时，所有图表的X轴都显示相同的数据
        'legend': dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        'spikedistance': 1000,  # 鼠标悬停时，光标距离图表的距离
        'hoverdistance': 100,  # 鼠标悬停时，光标距离图表的距离
        'legend_orientation': 'h',  # 图例方向
        'legend_yanchor': 'bottom',  # 图例Y轴对齐方式
        'legend_y': 1.02,  # 图例Y轴位置
        'legend_xanchor': 'right',  # 图例X轴对齐方式
        'legend_x': 1,  # 图例X轴位置
    }
    if show_macd:
        layout_params['yaxis3_title'] = 'MACD'
    fig.update_layout(**layout_params)

    # 计算要显示的日期刻度
    dates = df.index.tolist()
    n_ticks = 4
    tick_positions = list(range(0, len(dates), len(dates) // (n_ticks - 1)))
    if tick_positions[-1] != len(dates) - 1:  # 确保最后一个刻度是最后一个日期
        tick_positions.append(len(dates) - 1)
    tick_texts = [dates[i] for i in tick_positions]

    # 更新X轴和Y轴的样式
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showspikes=True,  # 显示光标
        spikecolor='grey',  # 光标颜色
        spikethickness=1,  # 光标厚度
        spikemode='across',  # 光标模式
        spikesnap='cursor',  # 光标对齐方式
        rangeslider=dict(visible=False),  # 隐藏范围选择器
        type='category',  # 设置X轴为分类轴
        showticklabels=False,  # 主图不显示日期
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showspikes=True,
        spikecolor='grey',
        spikethickness=1,
        spikemode='across',
        spikesnap='cursor',
        row=1, col=1
    )

    # 副图1
    if show_volume:
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showspikes=True,  # 显示光标
            spikecolor='grey',  # 光标颜色
            spikethickness=1,  # 光标厚度
            spikemode='across',  # 光标模式
            spikesnap='cursor',  # 光标对齐方式
            rangeslider=dict(visible=False),  # 隐藏范围选择器
            type='category',  # 设置X轴为分类轴
            tickmode='array',  # 使用自定义刻度
            tickvals=tick_positions,  # 刻度位置
            ticktext=tick_texts,  # 刻度标签
            tickangle=0,  # 刻度标签角度
            row=2, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showspikes=True,
            spikethickness=1,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='grey',
            row=2, col=1
        )

    # 副图2
    if show_macd:
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showspikes=True,  # 显示光标
            spikecolor='grey',  # 光标颜色
            spikethickness=1,  # 光标厚度
            spikemode='across',  # 光标模式
            spikesnap='cursor',  # 光标对齐方式
            rangeslider=dict(visible=False),  # 隐藏范围选择器
            type='category',  # 设置X轴为分类轴
            tickmode='array',  # 使用自定义刻度
            tickvals=tick_positions,  # 刻度位置
            ticktext=tick_texts,  # 刻度标签
            tickangle=0,  # 刻度标签角度
            row=3, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showspikes=True,
            spikethickness=1,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='grey',
            row=3, col=1
        )

    return fig


def main():
    st.title("数据查看器")

    # 计算日期范围
    today = datetime.today()
    three_months_ago = today - timedelta(days=90)

    # 设置默认日期
    default_start_date = three_months_ago
    default_end_date = today

    # 侧边栏设置
    with st.sidebar:
        st.header("数据设置")
        code = st.text_input('股票代码', value='000001')
        period = st.selectbox('数据周期', options=['daily', 'weekly', 'monthly'])

        # 日期选择(默认值为三个月前到今天)
        start_date = st.date_input("开始日期", value=default_start_date)
        end_date = st.date_input("结束日期", value=default_end_date)

        # 技术指标选择
        st.header("技术指标")
        show_ma = st.checkbox('显示均线', value=True)
        if show_ma:
            ma_periods = st.multiselect('均线周期', options=[5, 10, 20, 30, 60, 120, 250], default=[5, 20])
        show_volume = st.checkbox('显示成交量', value=True)
        show_macd = st.checkbox('显示MACD', value=False)

    # 转换日期为字符串格式
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

    # 主界面
    if st.button('获取数据', key='fetch_data'):
        with st.spinner('获取数据中...'):
            # 计算向前推的日期
            if show_ma:
                max_period = max(ma_periods)
                adjusted_start = (pd.to_datetime(start_date_str) - pd.Timedelta(days=max_period * 2)).strftime(
                    '%Y-%m-%d')
            else:
                adjusted_start = start_date_str

            # 获取数据（包括额外的历史数据）
            data = fetch_stock_data(code, period, adjusted_start, end_date_str)

            if data:
                # 转换为DataFrame
                df = pd.DataFrame(data)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df.index = df.index.strftime('%Y-%m-%d')

                # 显示K线图和成交量副图
                st.plotly_chart(plot_candlestick(
                    df,
                    ma_periods if show_ma else [],
                    show_volume=show_volume,
                    show_macd=show_macd,
                    start_date=start_date_str,
                    end_date=end_date_str
                ))

                # 显示基本信息
                st.subheader("基本信息")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("当前价格", f"￥{df.iloc[-1]['close']: .2f}")
                with info_col2:
                    st.metric("当前成交量", f"{df.iloc[-1]['volume'] / 10000: .2f}万")
                with info_col3:
                    st.metric("振幅",
                              f"{((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-2]) * 100: .2f}%")

                # 显示数据表格
                st.subheader("数据表格")
                st.dataframe(df.sort_index(ascending=False))


if __name__ == "__main__":
    main()
