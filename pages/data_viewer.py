#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:43 PM
@File       : data_viewer.py
@Description: 数据查看器
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
                     show_ascending_channel: bool = False, ascending_channel_info: dict = None,
                     start_date: str = None, end_date: str = None) -> go.Figure:
    """绘制K线图和副图 - 支持拖动和缩放，新增上升通道支持"""
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
        'spike': '#666666',   # 悬停线深灰色
        'channel_mid': '#ff6600',    # 上升通道中轴橙色
        'channel_upper': '#ff0000',  # 上升通道上沿红色
        'channel_lower': '#00ff00'   # 上升通道下沿绿色
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

    # 添加上升通道线（如果启用）
    if show_ascending_channel and ascending_channel_info:
        try:
            # 获取通道信息
            mid_today = ascending_channel_info.get('mid_today')
            mid_tomorrow = ascending_channel_info.get('mid_tomorrow')
            upper_today = ascending_channel_info.get('upper_today')
            lower_today = ascending_channel_info.get('lower_today')
            anchor_date = ascending_channel_info.get('anchor_date')
            anchor_price = ascending_channel_info.get('anchor_price')
            
            if all([mid_today, mid_tomorrow, upper_today, lower_today, anchor_date, anchor_price]):
                # 将anchor_date转换为datetime
                if isinstance(anchor_date, str):
                    anchor_date = pd.to_datetime(anchor_date)
                
                # 获取最新日期
                latest_date = pd.to_datetime(df_full.index[-1])
                
                # 计算通道线的日期范围（从锚点日期到最新日期）
                anchor_date_str = anchor_date.strftime('%Y-%m-%d')
                channel_dates = df_full[df_full.index >= anchor_date_str].index.tolist()
                
                if channel_dates:
                    # 计算斜率（基于mid_today和mid_tomorrow）
                    days_diff = 1  # 从今天到明天的天数差
                    beta = (mid_tomorrow - mid_today) / days_diff
                    
                    # 计算每个日期距离锚点的天数
                    days_since_anchor = []
                    for date_str in channel_dates:
                        date_obj = pd.to_datetime(date_str)
                        days = (date_obj - anchor_date).days
                        days_since_anchor.append(days)
                    
                    # 计算通道线价格
                    # 中轴：从mid_today开始，使用计算出的斜率
                    # 计算每个日期相对于今日的天数
                    days_to_today = (latest_date - anchor_date).days
                    days_relative_to_today = [days - days_to_today for days in days_since_anchor]
                    
                    # 确保今日对应的相对天数为0
                    # 如果最后一个值不是0，需要调整
                    if days_relative_to_today and days_relative_to_today[-1] != 0:
                        # 找到今日对应的索引
                        today_index = len(days_relative_to_today) - 1
                        # 重新计算相对天数，确保今日为0
                        days_relative_to_today = [i - today_index for i in range(len(days_relative_to_today))]
                    
                    mid_prices = [mid_today + beta * days_rel for days_rel in days_relative_to_today]
                    
                    # 上沿：从upper_today开始，保持相同斜率
                    upper_prices = [upper_today + beta * days_rel for days_rel in days_relative_to_today]
                    
                    # 下沿：从lower_today开始，保持相同斜率
                    lower_prices = [lower_today + beta * days_rel for days_rel in days_relative_to_today]
                    
                    # 准备通道线悬停文本
                    mid_hover_texts = []
                    upper_hover_texts = []
                    lower_hover_texts = []
                    
                    for date_str, mid_price, upper_price, lower_price in zip(channel_dates, mid_prices, upper_prices, lower_prices):
                        mid_hover_text = f"<b>{date_str}</b><br>中轴: {mid_price:.2f}<br>斜率: {beta:.4f}"
                        upper_hover_text = f"<b>{date_str}</b><br>上沿: {upper_price:.2f}<br>状态: {ascending_channel_info.get('channel_status', 'NORMAL')}"
                        lower_hover_text = f"<b>{date_str}</b><br>下沿: {lower_price:.2f}<br>累计涨幅: {ascending_channel_info.get('cumulative_gain', 0):.2%}"
                        
                        mid_hover_texts.append(mid_hover_text)
                        upper_hover_texts.append(upper_hover_text)
                        lower_hover_texts.append(lower_hover_text)
                    
                    # 添加中轴线
                    fig.add_trace(
                        go.Scatter(
                            x=channel_dates,
                            y=mid_prices,
                            mode='lines',
                            name='上升通道中轴',
                            line=dict(
                                width=3,
                                color=colors['channel_mid'],
                                dash='solid'
                            ),
                            hovertext=mid_hover_texts,
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )
                    
                    # 添加上沿线
                    fig.add_trace(
                        go.Scatter(
                            x=channel_dates,
                            y=upper_prices,
                            mode='lines',
                            name='上升通道上沿',
                            line=dict(
                                width=2,
                                color=colors['channel_upper'],
                                dash='dash'
                            ),
                            hovertext=upper_hover_texts,
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )
                    
                    # 添加下沿线
                    fig.add_trace(
                        go.Scatter(
                            x=channel_dates,
                            y=lower_prices,
                            mode='lines',
                            name='上升通道下沿',
                            line=dict(
                                width=2,
                                color=colors['channel_lower'],
                                dash='dash'
                            ),
                            hovertext=lower_hover_texts,
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )
                    
                    # 添加锚点标记
                    fig.add_trace(
                        go.Scatter(
                            x=[anchor_date_str],
                            y=[anchor_price],
                            mode='markers',
                            name='锚点',
                            marker=dict(
                                size=10,
                                color=colors['channel_mid'],
                                symbol='diamond',
                                line=dict(width=2, color='black')
                            ),
                            hovertext=f"<b>锚点</b><br>日期: {anchor_date_str}<br>价格: {anchor_price:.2f}",
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )
                    
        except Exception as e:
            st.warning(f"绘制上升通道线时出错: {e}")

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
    if 'ascending_channel_info' not in st.session_state:
        st.session_state.ascending_channel_info = None

    # 获取URL参数
    query_params = st.query_params
    
    # 添加调试信息
    with st.expander("🔧 URL参数调试", expanded=False):
        st.write("**原始query_params:**")
        st.write(query_params)
        st.write("**query_params类型:**")
        st.write(type(query_params))
    
     # 从URL参数中获取股票代码和其他设置
    default_code = query_params.get('code', ['000001']) if 'code' in query_params else '000001'
    default_name = query_params.get('name', ['']) if 'name' in query_params else ''
    auto_ascending_channel = query_params.get('auto_ascending_channel', ['false']) == 'true'
    strategy_name = query_params.get('strategy', ['']) if 'strategy' in query_params else ''
    
    # 添加解析后的参数调试信息
    with st.expander("🔧 解析后的参数", expanded=False):
        st.write(f"**default_code:** {default_code}")
        st.write(f"**default_name:** {default_name}")
        st.write(f"**auto_ascending_channel:** {auto_ascending_channel}")
        st.write(f"**strategy_name:** {strategy_name}")
    
    # 强制刷新机制 - 如果参数不完整，显示警告
    if len(default_code) < 3 or len(default_name) < 2:
        st.warning("⚠️ 检测到参数可能不完整，请检查URL或重新跳转")
        st.info("💡 建议：点击策略扫描器中的'🔗 直接跳转到数据查看器'按钮")
        
        # 提供手动输入选项
        st.subheader("手动输入股票信息")
        manual_code = st.text_input("手动输入股票代码", value=default_code if default_code != '000001' else '')
        manual_name = st.text_input("手动输入股票名称", value=default_name)
        
        if manual_code:
            default_code = manual_code
        if manual_name:
            default_name = manual_name

    # 计算日期范围
    today = datetime.today()
    three_months_ago = today - timedelta(days=90)
    one_year_ago = today - timedelta(days=365)  # 一年前

    # 设置默认日期
    default_start_date = one_year_ago  # 改为一年前
    default_end_date = today

    # 侧边栏设置
    with st.sidebar:
        st.header("数据设置")
        
        # 显示股票信息（如果从策略扫描器跳转过来）
        if default_name and strategy_name:
            st.info(f"**股票**: {default_code} {default_name}")
            st.info(f"**来源策略**: {strategy_name}")
        
        # 股票代码输入框 - 显示"代码-名称"格式
        if default_name:
            code_display = f"{default_code} - {default_name}"
        else:
            code_display = default_code
        
        code = st.text_input('股票代码', value=code_display)
        period = st.selectbox('数据周期', options=['daily', 'weekly', 'monthly'])

        # 日期选择(默认值为一年前到今天)
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
        show_ascending_channel = st.checkbox('显示上升通道', value=auto_ascending_channel)
        
        # 上升通道参数配置
        if show_ascending_channel:
            st.header("上升通道参数")
            
            # 使用expander来组织参数，避免侧边栏过长
            with st.expander("⚙️ 通道参数设置", expanded=False):
                # 基础参数
                st.subheader("基础参数")
                k = st.slider("通道宽度倍数 (k)", min_value=1.0, max_value=5.0, value=2.0, step=0.1, 
                             help="通道宽度倍数，影响通道的宽度 (±k·σ)")
                L_max = st.slider("最大窗口长度 (L_max)", min_value=60, max_value=200, value=120, step=10,
                                 help="窗口最长天数，超出后向右滑动")
                delta_cut = st.slider("滑动剔除天数 (delta_cut)", min_value=1, max_value=10, value=5, step=1,
                                     help="滑动时一次剔除最早的天数")
                pivot_m = st.slider("锚点检测参数 (pivot_m)", min_value=2, max_value=10, value=3, step=1,
                                   help="判断pivot low的宽度参数 (m左m右更高)")
                
                # 触发参数
                st.subheader("触发参数")
                gain_trigger = st.slider("重锚涨幅触发 (gain_trigger)", min_value=0.1, max_value=0.5, value=0.30, step=0.05,
                                        help="累计涨幅触发重锚的阈值")
                beta_delta = st.slider("斜率变化阈值 (beta_delta)", min_value=0.05, max_value=0.3, value=0.15, step=0.05,
                                      help="斜率变化阈值 (±15%)")
                break_days = st.slider("连续突破天数 (break_days)", min_value=1, max_value=10, value=3, step=1,
                                      help="连续n日突破上下沿视为失效")
                reanchor_fail_max = st.slider("重锚失败次数 (reanchor_fail_max)", min_value=1, max_value=5, value=2, step=1,
                                             help="连续n次重锚仍突破/跌破时进入极端状态")
                
                # 质量参数
                st.subheader("质量参数")
                min_data_points = st.slider("最小数据点数 (min_data_points)", min_value=30, max_value=100, value=60, step=5,
                                           help="最小有效数据点要求")
                R2_min = st.slider("最小R²值 (R2_min)", min_value=0.1, max_value=0.5, value=0.20, step=0.05,
                                  help="最小回归拟合优度，低于此视为无效通道")
                width_pct_min = st.slider("通道宽度下限 (width_pct_min)", min_value=0.02, max_value=0.10, value=0.04, step=0.01,
                                         help="通道宽度下限，小于此视为过窄")
                width_pct_max = st.slider("通道宽度上限 (width_pct_max)", min_value=0.08, max_value=0.20, value=0.12, step=0.01,
                                         help="通道宽度上限，超过此视为过宽")
            
            # 参数说明
            with st.expander("📖 参数说明", expanded=False):
                st.markdown("""
                **基础参数：**
                - **k**: 通道宽度倍数，影响通道的宽度范围
                - **L_max**: 最大窗口长度，控制计算窗口大小
                - **delta_cut**: 滑动剔除天数，影响窗口滑动速度
                - **pivot_m**: 锚点检测参数，影响锚点选择的敏感度
                
                **触发参数：**
                - **gain_trigger**: 重锚涨幅触发阈值，影响重锚频率
                - **beta_delta**: 斜率变化阈值，影响趋势判断
                - **break_days**: 连续突破天数，影响通道失效判断
                - **reanchor_fail_max**: 重锚失败次数，影响极端状态判断
                
                **质量参数：**
                - **min_data_points**: 最小数据点数，确保计算可靠性
                - **R2_min**: 最小R²值，确保回归质量
                - **width_pct_min/max**: 通道宽度范围，避免过窄或过宽
                """)

    # 转换日期为字符串格式
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

    # 主界面
    # 如果是从策略扫描器跳转过来的，自动获取数据
    auto_fetch = auto_ascending_channel and default_code != '000001'
    
    if st.button('获取数据', key='fetch_data') or auto_fetch:
        with st.spinner('获取数据中...'):
            # 从输入框中提取股票代码（如果格式是"代码-名称"）
            if ' - ' in code:
                actual_code = code.split(' - ')[0]
            else:
                actual_code = code
            
            # 计算向前推的日期
            if show_ma and ma_periods:
                max_period = max(ma_periods)
                adjusted_start = (pd.to_datetime(start_date_str) - pd.Timedelta(days=max_period * 2)).strftime(
                    '%Y-%m-%d')
            else:
                adjusted_start = start_date_str

            # 获取数据（包括额外的历史数据）
            df = fetch_stock_data(actual_code, period, adjusted_start, end_date_str)

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
                    'show_ascending_channel': show_ascending_channel,
                    'start_date': start_date_str,
                    'end_date': end_date_str
                }
                
                # 如果启用了上升通道，计算上升通道信息
                if show_ascending_channel:
                    try:
                        with st.spinner('计算上升通道中...'):
                            # 准备数据格式（重置索引以便计算）
                            df_for_calc = df.reset_index()
                            df_for_calc['trade_date'] = pd.to_datetime(df_for_calc['trade_date'])
                            
                            # 构建上升通道参数
                            channel_params = {
                                'k': k,
                                'L_max': L_max,
                                'delta_cut': delta_cut,
                                'pivot_m': pivot_m,
                                'gain_trigger': gain_trigger,
                                'beta_delta': beta_delta,
                                'break_days': break_days,
                                'reanchor_fail_max': reanchor_fail_max,
                                'min_data_points': min_data_points,
                                'R2_min': R2_min,
                                'width_pct_min': width_pct_min,
                                'width_pct_max': width_pct_max
                            }
                            
                            # 计算上升通道，传递自定义参数
                            channel_info = CalIndicators.ascending_channel(df_for_calc, **channel_params)
                            st.session_state.ascending_channel_info = channel_info
                            
                            st.success("上升通道计算完成")
                    except Exception as e:
                        st.error(f"上升通道计算失败: {e}")
                        st.session_state.ascending_channel_info = None
                else:
                    st.session_state.ascending_channel_info = None
                
                if auto_fetch:
                    st.success(f"自动获取 {actual_code} {default_name} 的数据，共 {len(df)} 条记录")
                else:
                    st.success(f"成功获取 {actual_code} 的数据，共 {len(df)} 条记录")

    # 显示图表（如果有数据）
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        params = st.session_state.chart_params
        ascending_channel_info = st.session_state.ascending_channel_info
        
        # 显示K线图和成交量副图
        st.plotly_chart(plot_candlestick(
            df,
            params['ma_periods'],
            show_volume=params['show_volume'],
            show_macd=params['show_macd'],
            show_ascending_channel=params['show_ascending_channel'],
            ascending_channel_info=ascending_channel_info,
            start_date=params['start_date'],
            end_date=params['end_date']
        ), use_container_width=True)

        # 显示上升通道信息（如果启用）
        if params['show_ascending_channel'] and ascending_channel_info:
            st.subheader("📈 上升通道信息")
            
            # 创建列布局显示通道信息
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                beta_value = ascending_channel_info.get('beta', 0)
                st.metric("斜率", f"{beta_value:.4f}" if beta_value is not None else "N/A")
                st.metric("通道状态", ascending_channel_info.get('channel_status', 'NORMAL'))
                r2_value = ascending_channel_info.get('r2', 0)
                st.metric("R²值", f"{r2_value:.3f}" if r2_value is not None else "N/A")
            
            with col2:
                mid_today = ascending_channel_info.get('mid_today', 0)
                st.metric("今日中轴", f"￥{mid_today:.2f}" if mid_today is not None else "N/A")
                upper_today = ascending_channel_info.get('upper_today', 0)
                st.metric("今日上沿", f"￥{upper_today:.2f}" if upper_today is not None else "N/A")
                width_pct = ascending_channel_info.get('width_pct', 0)
                st.metric("通道宽度", f"{width_pct:.2%}" if width_pct is not None else "N/A")
            
            with col3:
                lower_today = ascending_channel_info.get('lower_today', 0)
                st.metric("今日下沿", f"￥{lower_today:.2f}" if lower_today is not None else "N/A")
                cumulative_gain = ascending_channel_info.get('cumulative_gain', 0)
                st.metric("累计涨幅", f"{cumulative_gain:.2%}" if cumulative_gain is not None else "N/A")
                slope_deg = ascending_channel_info.get('slope_deg', 0)
                st.metric("斜率角度", f"{slope_deg:.2f}°" if slope_deg is not None else "N/A")
            
            with col4:
                anchor_price = ascending_channel_info.get('anchor_price', 0)
                st.metric("锚点价格", f"￥{anchor_price:.2f}" if anchor_price is not None else "N/A")
                anchor_date = ascending_channel_info.get('anchor_date', 'N/A')
                st.metric("锚点日期", anchor_date[:10] if anchor_date and anchor_date != 'N/A' else 'N/A')
                volatility = ascending_channel_info.get('volatility', 0)
                st.metric("波动率", f"{volatility:.3f}" if volatility is not None else "N/A")
            
            # 显示通道质量评估
            st.subheader("📊 通道质量评估")
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            
            with quality_col1:
                r2_value = ascending_channel_info.get('r2', 0)
                if r2_value is not None:
                    if r2_value > 0.7:
                        st.success(f"拟合质量: 优秀 ({r2_value:.3f})")
                    elif r2_value > 0.5:
                        st.info(f"拟合质量: 良好 ({r2_value:.3f})")
                    else:
                        st.warning(f"拟合质量: 一般 ({r2_value:.3f})")
                else:
                    st.warning("拟合质量: 未知")
            
            with quality_col2:
                width_pct = ascending_channel_info.get('width_pct', 0)
                if width_pct is not None:
                    if width_pct < 0.05:
                        st.warning(f"通道宽度: 过窄 ({width_pct:.2%})")
                    elif width_pct > 0.15:
                        st.warning(f"通道宽度: 过宽 ({width_pct:.2%})")
                    else:
                        st.success(f"通道宽度: 适中 ({width_pct:.2%})")
                else:
                    st.warning("通道宽度: 未知")
            
            with quality_col3:
                slope_deg = ascending_channel_info.get('slope_deg', 0)
                if slope_deg is not None:
                    if slope_deg > 5:
                        st.info(f"趋势强度: 强 ({slope_deg:.2f}°)")
                    elif slope_deg > 1:
                        st.success(f"趋势强度: 中 ({slope_deg:.2f}°)")
                    else:
                        st.warning(f"趋势强度: 弱 ({slope_deg:.2f}°)")
                else:
                    st.warning("趋势强度: 未知")
            
            with quality_col4:
                volatility = ascending_channel_info.get('volatility', 0)
                if volatility is not None:
                    if volatility < 0.02:
                        st.success(f"波动率: 低 ({volatility:.3f})")
                    elif volatility < 0.05:
                        st.info(f"波动率: 中 ({volatility:.3f})")
                    else:
                        st.warning(f"波动率: 高 ({volatility:.3f})")
                else:
                    st.warning("波动率: 未知")
            
            # 显示详细通道信息
            with st.expander("📊 详细通道信息", expanded=False):
                st.json(ascending_channel_info)

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
            - 上升通道悬停显示：中轴、上沿、下沿价格和通道状态
            - 拖动到数据边界会自动停止，防止超出范围
            - 只能左右拖动，不能上下拖动
            
            **📈 上升通道说明：**
            - **中轴线**：橙色实线，表示通道的中心趋势线
            - **上沿线**：红色虚线，表示通道的上边界
            - **下沿线**：绿色虚线，表示通道的下边界
            - **锚点**：橙色菱形标记，表示通道的起始点
            - 通道状态包括：NORMAL（正常）、ACCEL_BREAKOUT（加速突破）、BREAKDOWN（跌破）、BROKEN（失效）
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
