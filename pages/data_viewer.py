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
        data = response.json()
        # 将返回的数据转换为DataFrame
        if isinstance(data, list) and data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"获取股票数据失败: {e}")
        return pd.DataFrame()


def plot_candlestick(df: pd.DataFrame, ma_periods: list, show_volume: bool = True, show_macd: bool = False,
                     start_date: str = None, end_date: str = None) -> go.Figure:
    """绘制K线图和副图 - 支持拖动和缩放"""
    # 多获取数据的df，用于计算均线
    df_extra = df.copy()

    # 过滤日期范围
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # 过滤掉没有数据的日期
    df = df.dropna(subset=['open', 'close', 'high', 'low', 'volume'])

    # 保存完整数据用于计算和绘制
    df_full = df.copy()

    # 创建子图，根据是否显示MACD决定子图数量
    rows = 2 if not show_macd else 3
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,  # 共享X轴
        vertical_spacing=0.08,  # 增加子图间距
        row_heights=[0.7, 0.3] if not show_macd else [0.6, 0.2, 0.2],
        subplot_titles=['K线图', '成交量'] if not show_macd else ['K线图', '成交量', 'MACD']
    )

    # 定义颜色主题
    colors = {
        'up': '#ff4444',      # 上涨红色
        'down': '#00aa00',    # 下跌绿色
        'ma5': '#ff8800',     # 5日均线橙色
        'ma10': '#0088ff',    # 10日均线蓝色
        'ma20': '#8800ff',    # 20日均线紫色
        'ma30': '#ff0088',    # 30日均线粉色
        'ma60': '#00ff88',    # 60日均线青色
        'ma120': '#888800',   # 120日均线棕色
        'ma250': '#008888',   # 250日均线青色
        'background': '#ffffff',  # 背景白色
        'grid': '#e0e0e0',    # 网格深灰色
        'text': '#000000',    # 文字黑色
        'axis': '#000000',    # 坐标轴黑色
        'spike': '#666666'    # 悬停线深灰色
    }

    # 绘制K线图 - 使用完整数据
    # 准备悬停文本
    hover_texts = []
    for i, (date, row) in enumerate(df_full.iterrows()):
        pct_chg = df_full['pct_chg'].iloc[i] if 'pct_chg' in df_full.columns else 0
        amount = df_full['amount'].iloc[i] if 'amount' in df_full.columns else 0
        turn = df_full['turn'].iloc[i] if 'turn' in df_full.columns else 0
        
        hover_text = f"<b>{date}</b><br>" + \
                    f"开盘价: {row['open']:.2f}<br>" + \
                    f"最高价: {row['high']:.2f}<br>" + \
                    f"最低价: {row['low']:.2f}<br>" + \
                    f"收盘价: {row['close']:.2f}<br>" + \
                    f"涨跌幅: {pct_chg:.2f}%<br>" + \
                    f"成交量: {row['volume']:,.0f}<br>" + \
                    f"成交额: {amount:,.0f}<br>" + \
                    f"换手率: {turn:.2f}%"
        hover_texts.append(hover_text)
    
    fig.add_trace(
        go.Candlestick(
            x=df_full.index,  # 使用完整数据
            open=df_full['open'],
            high=df_full['high'],
            low=df_full['low'],
            close=df_full['close'],
            increasing=dict(
                line=dict(color=colors['up'], width=1),
                fillcolor=colors['up']
            ),
            decreasing=dict(
                line=dict(color=colors['down'], width=1),
                fillcolor=colors['down']
            ),
            name='K线图',
            hovertext=hover_texts,
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # 计算并添加均线（使用完整数据计算）
    ma_colors = [colors['ma5'], colors['ma10'], colors['ma20'], colors['ma30'], colors['ma60'], colors['ma120'], colors['ma250']]
    for i, ma_period in enumerate(ma_periods):
        # 使用完整数据计算均线
        ma_series = df_full['close'].rolling(window=ma_period).mean()

        # 准备均线悬停文本
        ma_hover_texts = []
        for date, ma_value in zip(df_full.index, ma_series):
            ma_hover_text = f"<b>{date}</b><br>{ma_period}日均线: {ma_value:.2f}"
            ma_hover_texts.append(ma_hover_text)
        
        # 添加均线到图表
        fig.add_trace(
            go.Scatter(
                x=df_full.index,
                y=ma_series,
                mode='lines',
                name=f'{ma_period}日均线',
                line=dict(
                    width=2,
                    color=ma_colors[i % len(ma_colors)]
                ),
                hovertext=ma_hover_texts,
                hoverinfo='text'
            ), row=1, col=1
        )

    # 添加成交量图（使用完整数据）
    if show_volume:
        colors_volume = ['red' if row['close'] >= row['open'] else 'green' for _, row in df_full.iterrows()]
        
        # 准备成交量悬停文本
        volume_hover_texts = []
        for i, (date, row) in enumerate(df_full.iterrows()):
            amount = df_full['amount'].iloc[i] if 'amount' in df_full.columns else 0
            volume_hover_text = f"<b>{date}</b><br>成交量: {row['volume']:,.0f}<br>成交额: {amount:,.0f}"
            volume_hover_texts.append(volume_hover_text)
        
        fig.add_trace(
            go.Bar(
                x=df_full.index,  # 使用完整数据
                y=df_full['volume'],
                marker=dict(
                    color=colors_volume,
                    opacity=0.7,
                    line=dict(width=0)
                ),
                name='成交量',
                showlegend=False,
                hovertext=volume_hover_texts,
                hoverinfo='text'
            ),
            row=2, col=1
        )

    # 添加MACD图（使用完整数据）
    if show_macd:
        dif, dea, macd_hist = CalIndicators.macd(df_full, fast_period=12, slow_period=26, signal_period=9)

        # 准备MACD悬停文本
        macd_hover_texts = []
        for date, macd_val in zip(dif.index, macd_hist):
            macd_hover_text = f"<b>{date}</b><br>MACD: {macd_val:.4f}"
            macd_hover_texts.append(macd_hover_text)
        
        # 添加MACD柱状图
        colors_macd = ['red' if val >= 0 else 'green' for val in macd_hist]
        fig.add_trace(
            go.Bar(
                x=dif.index,
                y=macd_hist,
                marker=dict(
                    color=colors_macd,
                    opacity=0.7,
                    line=dict(width=0)
                ),
                name='MACD',
                showlegend=False,
                hovertext=macd_hover_texts,
                hoverinfo='text'
            ),
            row=3, col=1
        )

        # 准备DIF和DEA悬停文本
        dif_hover_texts = []
        dea_hover_texts = []
        for date, dif_val, dea_val in zip(dif.index, dif, dea):
            dif_hover_text = f"<b>{date}</b><br>DIF: {dif_val:.4f}"
            dea_hover_text = f"<b>{date}</b><br>DEA: {dea_val:.4f}"
            dif_hover_texts.append(dif_hover_text)
            dea_hover_texts.append(dea_hover_text)

        # 添加DIF和DEA线
        fig.add_trace(
            go.Scatter(
                x=dif.index, 
                y=dif, 
                mode='lines', 
                name='DIF', 
                line=dict(width=2, color='orange'),
                hovertext=dif_hover_texts,
                hoverinfo='text'
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dif.index, 
                y=dea, 
                mode='lines', 
                name='DEA', 
                line=dict(width=2, color='blue'),
                hovertext=dea_hover_texts,
                hoverinfo='text'
            ),
            row=3, col=1
        )

    # 调整图表布局
    fig.update_layout(
        title=dict(
            text='K线图',
            x=0.5,
            font=dict(size=20, color=colors['text'])
        ),
        height=800 if not show_macd else 1000,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=12),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(color=colors['text'])
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        # 启用拖动和缩放功能
        dragmode='pan',  # 默认拖动模式
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.8)',
            color='gray',
            activecolor='black'
        ),
        # 鼠标悬停配置
        hovermode='x unified',  # 统一X轴悬停模式
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            font=dict(color=colors['text'], size=11)
        )
    )
    
    # 设置初始视图范围，显示最后60个交易日
    initial_range = None
    if len(df_full) > 60:
        # 计算最后60个交易日的位置
        start_idx = len(df_full) - 60
        end_idx = len(df_full) - 1
        initial_range = [start_idx, end_idx]
    else:
        # 如果数据少于60天，显示全部数据
        initial_range = [0, len(df_full) - 1]

    # 计算要显示的日期刻度
    dates = df_full.index.tolist()  # 使用完整数据
    n_ticks = min(6, len(dates))  # 最多显示6个刻度
    tick_positions = list(range(0, len(dates), max(1, len(dates) // (n_ticks - 1))))
    if tick_positions and tick_positions[-1] != len(dates) - 1:  # 确保最后一个刻度是最后一个日期
        tick_positions.append(len(dates) - 1)
    tick_texts = [dates[i] for i in tick_positions if i < len(dates)]

    # 更新X轴和Y轴的样式
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showspikes=True,
        spikecolor=colors['spike'],
        spikethickness=1,
        spikemode='across',
        spikesnap='cursor',
        rangeslider=dict(visible=False),
        type='category',
        showticklabels=False,  # 主图不显示日期
        title=dict(text='日期', font=dict(color=colors['text'], size=14)),
        fixedrange=False,  # 允许X轴拖动
        constrain='domain',  # 限制在数据范围内
        range=initial_range,  # 使用初始范围
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showspikes=True,
        spikecolor=colors['spike'],
        spikethickness=1,
        spikemode='across',
        spikesnap='cursor',
        title=dict(text='价格', font=dict(color=colors['text'], size=14)),
        tickfont=dict(color=colors['text'], size=11),
        fixedrange=True,  # 禁止Y轴拖动
        constrain='domain',
        row=1, col=1
    )

    # 副图1 - 成交量
    if show_volume:
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            showspikes=True,
            spikecolor=colors['spike'],
            spikethickness=1,
            spikemode='across',
            spikesnap='cursor',
            rangeslider=dict(visible=False),
            type='category',
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_texts,
            tickangle=45,  # 倾斜45度
            title=dict(text='日期', font=dict(color=colors['text'], size=14)),
            tickfont=dict(color=colors['text'], size=10),
            fixedrange=False,  # 允许X轴拖动
            constrain='domain',  # 限制在数据范围内
            range=initial_range,  # 使用初始范围
            row=2, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            showspikes=True,
            spikethickness=1,
            spikemode='across',
            spikesnap='cursor',
            spikecolor=colors['spike'],
            title=dict(text='成交量', font=dict(color=colors['text'], size=14)),
            tickfont=dict(color=colors['text'], size=11),
            fixedrange=True,  # 禁止Y轴拖动
            constrain='domain',
            row=2, col=1
        )

    # 副图2 - MACD
    if show_macd:
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            showspikes=True,
            spikecolor=colors['spike'],
            spikethickness=1,
            spikemode='across',
            spikesnap='cursor',
            rangeslider=dict(visible=False),
            type='category',
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_texts,
            tickangle=45,
            title=dict(text='日期', font=dict(color=colors['text'], size=14)),
            tickfont=dict(color=colors['text'], size=10),
            fixedrange=False,  # 允许X轴拖动
            constrain='domain',  # 限制在数据范围内
            range=initial_range,  # 使用初始范围
            row=3, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            showspikes=True,
            spikethickness=1,
            spikemode='across',
            spikesnap='cursor',
            spikecolor=colors['spike'],
            title=dict(text='MACD', font=dict(color=colors['text'], size=14)),
            tickfont=dict(color=colors['text'], size=11),
            fixedrange=True,  # 禁止Y轴拖动
            constrain='domain',
            row=3, col=1
        )

    return fig


def main():
    st.title("数据查看器")

    # 初始化session_state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'chart_params' not in st.session_state:
        st.session_state.chart_params = None

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
        else:
            ma_periods = []
        show_volume = st.checkbox('显示成交量', value=True)
        show_macd = st.checkbox('显示MACD', value=False)

    # 转换日期为字符串格式
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

    # 主界面
    if st.button('获取数据', key='fetch_data'):
        with st.spinner('获取数据中...'):
            # 计算向前推的日期
            if show_ma and ma_periods:
                max_period = max(ma_periods)
                adjusted_start = (pd.to_datetime(start_date_str) - pd.Timedelta(days=max_period * 2)).strftime(
                    '%Y-%m-%d')
            else:
                adjusted_start = start_date_str

            # 获取数据（包括额外的历史数据）
            df = fetch_stock_data(code, period, adjusted_start, end_date_str)

            if not df.empty:
                # 设置trade_date为索引
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df.index = df.index.strftime('%Y-%m-%d')

                # 保存数据到session_state
                st.session_state.stock_data = df
                st.session_state.chart_params = {
                    'ma_periods': ma_periods if show_ma else [],
                    'show_volume': show_volume,
                    'show_macd': show_macd,
                    'start_date': start_date_str,
                    'end_date': end_date_str
                }
                
                st.success(f"成功获取 {code} 的数据，共 {len(df)} 条记录")

    # 显示图表（如果有数据）
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        params = st.session_state.chart_params
        
        # 显示K线图和成交量副图
        st.plotly_chart(plot_candlestick(
            df,
            params['ma_periods'],
            show_volume=params['show_volume'],
            show_macd=params['show_macd'],
            start_date=params['start_date'],
            end_date=params['end_date']
        ), use_container_width=True)

        # 添加使用说明
        with st.expander("📖 图表操作说明", expanded=False):
            st.markdown("""
            **🖱️ 鼠标操作：**
            - **左右拖动**：按住鼠标左键左右拖动可以平移图表，查看历史数据
            - **鼠标悬停**：将鼠标放到图表上任意位置即可显示当天详细信息
            - **滚轮缩放**：使用鼠标滚轮可以缩放图表
            - **框选放大**：按住鼠标左键框选区域可以放大到该区域
            
            **📱 工具栏功能：**
            - **🏠 重置**：点击重置按钮可以恢复到初始视图（最近60个交易日）
            - **📷 截图**：点击相机图标可以保存图表为图片
            - **🔍 缩放**：点击放大镜图标可以进入缩放模式
            - **✋ 平移**：点击手图标可以进入平移模式
            
            **📊 数据显示：**
            - 初始显示最近60个交易日的数据
            - 图表包含完整的历史数据，可以通过拖动查看更早的数据
            - 鼠标悬停显示：开盘价、收盘价、最高价、最低价、涨跌幅、成交量、成交额、换手率
            - 均线悬停显示：对应均线的价格
            - 成交量悬停显示：成交量和成交额
            - MACD悬停显示：MACD、DIF、DEA值
            - 拖动到数据边界会自动停止，防止超出范围
            - 只能左右拖动，不能上下拖动
            """)

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

        # 显示数据统计信息
        st.subheader("数据统计")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("最高价", f"￥{df['high'].max(): .2f}")
        with stats_col2:
            st.metric("最低价", f"￥{df['low'].min(): .2f}")
        with stats_col3:
            st.metric("平均价", f"￥{df['close'].mean(): .2f}")
        with stats_col4:
            st.metric("数据点数", len(df))

        # 显示数据表格
        st.subheader("数据表格")
        # 添加搜索功能
        search_term = st.text_input("🔍 搜索日期或价格", placeholder="输入日期(YYYY-MM-DD)或价格范围")
        
        # 过滤数据
        if search_term:
            try:
                # 尝试解析为日期
                search_date = pd.to_datetime(search_term).strftime('%Y-%m-%d')
                filtered_df = df[df.index.str.contains(search_date, na=False)]
            except:
                # 如果不是日期，尝试解析为价格范围
                try:
                    if '-' in search_term:
                        min_price, max_price = map(float, search_term.split('-'))
                        filtered_df = df[(df['close'] >= min_price) & (df['close'] <= max_price)]
                    else:
                        price = float(search_term)
                        filtered_df = df[df['close'] == price]
                except:
                    # 如果都不是，按字符串搜索
                    filtered_df = df[df.index.str.contains(search_term, na=False)]
        else:
            filtered_df = df
        
        st.dataframe(filtered_df.sort_index(ascending=False), use_container_width=True, height=400)
    else:
        # 如果没有数据，显示提示信息
        st.info("请在左侧设置参数后点击'获取数据'按钮来查看股票数据")


if __name__ == "__main__":
    main()
