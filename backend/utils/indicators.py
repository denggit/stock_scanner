#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 1:22 AM
@File       : indicators.py
@Description: 
"""
from typing import Tuple, Dict, Any, Optional
import sys
import os

import numpy as np
import pandas as pd
import logging

# 添加项目根目录到路径，以便导入上升通道模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from backend.quant.core.factor_engine.factor_library.channel_analysis import AscendingChannelRegression
    ASCENDING_CHANNEL_AVAILABLE = True
except ImportError:
    ASCENDING_CHANNEL_AVAILABLE = False
    logging.warning("上升通道回归模块导入失败，相关功能将不可用")


class CalIndicators:
    def __init__(self):
        pass

    @staticmethod
    def macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean().round(2)
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean().round(2)
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean().round(2)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    @staticmethod
    def ema(df: pd.DataFrame, period: int, cal_value: str = 'close') -> pd.Series:
        """计算EMA指标"""
        return df[cal_value].ewm(span=period, min_periods=period, adjust=False).mean().round(2)

    @staticmethod
    def sma(df: pd.DataFrame, period: int, cal_value: str = 'close') -> pd.Series:
        """计算SMA指标"""
        return df[cal_value].rolling(window=period, min_periods=period, center=False).mean().round(2)

    @staticmethod
    def amplitude(df: pd.DataFrame, lookback_period: int = 14) -> pd.Series:
        """计算振幅
        振幅 = (当日最高价 - 当日最低价) / 前收盘价 × 100%
        """
        amplitude = ((df['high'] - df['low']) / df['close'].shift(1) * 100).round(2)
        if lookback_period > 1:
            amplitude = amplitude.rolling(window=lookback_period).mean().round(2)
        return amplitude

    @staticmethod
    def volatility(df: pd.DataFrame, lookback_period: int = 14, annualized: bool = True) -> pd.Series:
        """计算波动率
        Args:
            df: 数据框
            lookback_period: 回看周期
            annualized: 是否年化，默认为True
        Returns:
            波动率序列
        """
        # 计算对数收益率
        log_returns = np.log(df['close'] / df['close'].shift(1))
        # 计算滚动标准差
        vol = log_returns.rolling(window=lookback_period).std()

        if annualized:
            # 假设一年250个交易日，年化处理
            vol = vol * np.sqrt(250)

        return vol.round(4) * 100  # 转换为百分比并保留4位小数

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, ma_period: int = 20, bollinger_k: int = 2) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        bb_mid = CalIndicators.ema(df, ma_period)
        std = df['close'].rolling(window=ma_period).std()
        bb_upper = (bb_mid + std * bollinger_k).round(2)
        bb_lower = (bb_mid - std * bollinger_k).round(2)
        return bb_mid, bb_upper, bb_lower

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - 100 / (1 + rs)).round(2)

    @staticmethod
    def support(df: pd.DataFrame, window: int = 20) -> float:
        """计算支撑位"""
        try:
            lows = df['low'].iloc[-window:]
            # 找到局部最低点
            local_mins = []
            for i in range(1, len(lows)-1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    local_mins.append(lows.iloc[i])
            
            if local_mins:
                return max(local_mins)  # 返回最高的支撑位
            return df['low'].min()
            
        except Exception as e:
            logging.warning(f"支撑位计算失败: {e}")
            return None

    @staticmethod
    def resistance(df: pd.DataFrame, window: int = 20) -> float:
        """计算阻力位"""
        try:
            highs = df['high'].iloc[-window:]
            # 找到局部最高点
            local_maxs = []
            for i in range(1, len(highs)-1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    local_maxs.append(highs.iloc[i])
            
            if local_maxs:
                return min(local_maxs)  # 返回最低的阻力位
            return df['high'].max()
            
        except Exception as e:
            logging.warning(f"阻力位计算失败: {e}")
            return None

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
        """计算ROC (Rate of Change) 指标
        ROC = (当前收盘价 - n日前收盘价) / n日前收盘价 × 100
        """
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100).round(2)
        return roc

    @staticmethod
    def kdj(df: pd.DataFrame, window: int = 9, smooth: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算KDJ指标
        Args:
            df: 数据框
            window: RSV计算窗口期，默认9
            smooth: K值和D值的平滑周期，默认3
        Returns:
            K, D, J值序列
        """
        df = df.copy()
        df.close = df.close.astype(float)
        # 计算RSV
        low_list = df['low'].rolling(window=window, min_periods=1).min()
        high_list = df['high'].rolling(window=window, min_periods=1).max()
        
        rsv = pd.Series(0.0, index=df.index)
        # 添加除零保护
        denominator = high_list - low_list
        rsv = np.where(denominator != 0,
                      (df['close'] - low_list) / denominator * 100,
                      0)
        
        # 计算K值，使用EMA平滑
        k = pd.Series(50.0, index=df.index)  # 初始值设为50
        k = pd.Series(rsv).ewm(alpha=2/(smooth+1), adjust=False).mean()
        
        # 计算D值，使用相同的平滑系数
        d = pd.Series(50.0, index=df.index)  # 初始值设为50
        d = k.ewm(alpha=2/(smooth+1), adjust=False).mean()
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 限制范围在0-100之间
        k = k.clip(0, 100).round(2)
        d = d.clip(0, 100).round(2)
        j = j.clip(0, 100).round(2)
        
        return k, d, j

    @staticmethod
    def dmi(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算DMI (Directional Movement Index) 指标
        Returns:
            PDI(+DI), MDI(-DI), ADX
        """
        # 计算真实波幅（TR）
        tr = pd.DataFrame(index=df.index)
        tr['hl'] = df['high'] - df['low']
        tr['hc'] = abs(df['high'] - df['close'].shift(1))
        tr['lc'] = abs(df['low'] - df['close'].shift(1))
        tr = tr.max(axis=1)
        
        # 计算方向变动（DM）
        pdm = df['high'] - df['high'].shift(1)
        mdm = df['low'].shift(1) - df['low']
        
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
        
        # 计算平滑值
        tr14 = tr.ewm(alpha=1/period, adjust=False).mean()
        pdm14 = pdm.ewm(alpha=1/period, adjust=False).mean()
        mdm14 = mdm.ewm(alpha=1/period, adjust=False).mean()
        
        # 计算DI
        pdi = (pdm14 / tr14 * 100).round(2)
        mdi = (mdm14 / tr14 * 100).round(2)
        
        # 计算DX和ADX
        dx = (abs(pdi - mdi) / (pdi + mdi) * 100).round(2)
        adx = dx.ewm(alpha=1/period, adjust=False).mean().round(2)
        
        return pdi, mdi, adx

    @staticmethod
    def ascending_channel(df: pd.DataFrame, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        计算上升通道回归分析
        
        Args:
            df (pd.DataFrame): 价格数据，必须包含 trade_date, open, high, low, close, volume 列
            config_path (Optional[str]): 配置文件路径，如果为None则使用默认配置
            
        Returns:
            Dict[str, Any]: 上升通道信息字典，包含以下字段：
                - beta: 斜率
                - mid_today: 今日中轴价
                - upper_today: 今日上沿价
                - lower_today: 今日下沿价
                - mid_tomorrow: 明日预测中轴价
                - upper_tomorrow: 明日预测上沿价
                - lower_tomorrow: 明日预测下沿价
                - channel_status: 通道状态 (NORMAL/ACCEL_BREAKOUT/BREAKDOWN/BROKEN)
                - anchor_date: 锚点日期
                - anchor_price: 锚点价格
                - break_cnt_up: 连续突破上沿次数
                - break_cnt_down: 连续突破下沿次数
                - cumulative_gain: 累计涨幅
                - last_update: 最后更新时间
                
        Raises:
            ImportError: 如果上升通道模块不可用
            ValueError: 如果数据格式不正确
            Exception: 其他计算错误
        """
        if not ASCENDING_CHANNEL_AVAILABLE:
            raise ImportError("上升通道回归模块不可用，请检查模块安装")
        
        # 数据验证
        required_columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需列: {missing_columns}")
        
        if len(df) < 60:
            raise ValueError("数据不足，至少需要60个交易日的数据")
        
        try:
            # 初始化上升通道分析器
            analyzer = AscendingChannelRegression(config_path=config_path)
            
            # 拟合上升通道
            state = analyzer.fit_channel(df)
            
            # 返回通道信息
            return state.to_dict()
            
        except Exception as e:
            logging.error(f"上升通道计算失败: {e}")
            raise
    
    @staticmethod
    def ascending_channel_update(state_dict: Dict[str, Any], new_bar: Dict[str, Any], 
                               config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        更新上升通道状态
        
        Args:
            state_dict (Dict[str, Any]): 当前通道状态字典
            new_bar (Dict[str, Any]): 新的K线数据，包含 trade_date, open, high, low, close, volume
            config_path (Optional[str]): 配置文件路径
            
        Returns:
            Dict[str, Any]: 更新后的通道信息字典
        """
        if not ASCENDING_CHANNEL_AVAILABLE:
            raise ImportError("上升通道回归模块不可用，请检查模块安装")
        
        try:
            # 初始化分析器
            analyzer = AscendingChannelRegression(config_path=config_path)
            
            # 从字典重建状态对象（简化实现）
            # 注意：这里需要从state_dict重建ChannelState对象
            # 为了简化，我们直接重新拟合整个数据集
            
            # 获取原始数据（这里需要调用方提供完整的历史数据）
            # 这是一个简化的实现，实际使用时可能需要更复杂的状态管理
            logging.warning("ascending_channel_update方法需要完整的历史数据，建议使用ascending_channel方法重新计算")
            
            return state_dict
            
        except Exception as e:
            logging.error(f"上升通道更新失败: {e}")
            raise
    
    @staticmethod
    def ascending_channel_batch(df_list: list, config_path: Optional[str] = None) -> list:
        """
        批量计算上升通道回归分析
        
        Args:
            df_list (list): 价格数据列表，每个元素是一个DataFrame，必须包含 trade_date, open, high, low, close, volume 列
            config_path (Optional[str]): 配置文件路径
            
        Returns:
            list: 上升通道信息列表，每个元素对应一个DataFrame的结果
        """
        if not ASCENDING_CHANNEL_AVAILABLE:
            raise ImportError("上升通道回归模块不可用，请检查模块安装")
        
        results = []
        
        for i, df in enumerate(df_list):
            try:
                # 验证数据格式
                required_columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logging.error(f"第 {i+1} 个数据缺少必需列: {missing_columns}")
                    results.append(None)
                    continue
                
                if len(df) < 60:
                    logging.error(f"第 {i+1} 个数据不足，至少需要60个交易日的数据")
                    results.append(None)
                    continue
                
                result = CalIndicators.ascending_channel(df, config_path)
                results.append(result)
                logging.info(f"第 {i+1}/{len(df_list)} 个数据计算完成")
            except Exception as e:
                logging.error(f"第 {i+1} 个数据计算失败: {e}")
                results.append(None)
        
        return results
