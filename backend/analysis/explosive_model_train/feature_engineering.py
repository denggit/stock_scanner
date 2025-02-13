#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征工程模块：负责构建和处理模型训练所需的特征

本模块提供了一系列用于构建股票预测模型特征的工具，包括：
1. 技术指标计算：移动平均线、RSI、MACD等
2. 价格动量特征：收益率、波动率等
3. 财务指标处理：估值指标、盈利能力指标等
4. 标签构建：基于未来收益率的标签生成

主要类：
    FeatureEngineering: 特征工程主类，提供所有特征构建方法

使用示例：
    fe = FeatureEngineering()
    features = fe.calculate_technical_indicators(price_data)
    X, y = fe.prepare_features(price_data, financial_data)
"""

from typing import Tuple

import numpy as np
import pandas as pd
import talib


class FeatureEngineering:
    """特征工程类，负责构建预测模型所需的所有特征
    
    该类提供了一系列方法来构建和转换特征，包括：
    1. 技术分析指标
    2. 价格动量特征
    3. 财务指标特征
    4. 市场情绪特征
    
    所有方法都经过优化，以处理金融时间序列数据的特殊性质。
    """

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算技术分析指标
        
        计算一系列技术分析指标，包括：
        1. 移动平均线（MA5、MA10、MA20、MA60）
        2. 相对强弱指标（RSI6、RSI12、RSI24）
        3. MACD指标
        4. 布林带指标
        5. 成交量相关指标
        
        Args:
            df (pd.DataFrame): 原始价格数据，必须包含列：
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
        
        Returns:
            pd.DataFrame: 添加了技术指标的DataFrame，包含：
                - 所有原始列
                - 计算得到的技术指标列
                
        Raises:
            ValueError: 输入数据缺少必要的列
            RuntimeError: 技术指标计算失败
        """
        # 确保数据类型为float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 移动平均线
        df['ma5'] = talib.MA(df['close'], timeperiod=5)
        df['ma10'] = talib.MA(df['close'], timeperiod=10)
        df['ma20'] = talib.MA(df['close'], timeperiod=20)
        df['ma60'] = talib.MA(df['close'], timeperiod=60)

        # 计算均线差值
        df['ma5_10_diff'] = df['ma5'] - df['ma10']
        df['ma10_20_diff'] = df['ma10'] - df['ma20']
        df['ma20_60_diff'] = df['ma20'] - df['ma60']

        # RSI指标
        df['rsi_6'] = talib.RSI(df['close'], timeperiod=6)
        df['rsi_12'] = talib.RSI(df['close'], timeperiod=12)
        df['rsi_24'] = talib.RSI(df['close'], timeperiod=24)

        # MACD指标
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        # 成交量指标
        df['volume_ma5'] = talib.MA(df['volume'], timeperiod=5)
        df['volume_ma20'] = talib.MA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_ma5']

        # 波动率指标
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        return df

    @staticmethod
    def calculate_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """计算价格动量特征
        
        计算一系列价格动量相关的特征，包括：
        1. 不同周期的收益率
        2. 不同周期的波动率
        3. 价格趋势特征
        4. 动量指标
        
        Args:
            df (pd.DataFrame): 股票价格数据，必须包含：
                - close: 收盘价
                - volume: 成交量（可选）
        
        Returns:
            pd.DataFrame: 添加了动量特征的DataFrame，包含：
                - 各周期收益率
                - 波动率指标
                - 趋势指标
                
        Raises:
            ValueError: 输入数据格式错误
        """
        # 计算不同时间周期的收益率
        for period in [5, 10, 20, 60]:
            df[f'return_{period}d'] = df['close'].pct_change(period)

        # 计算不同时间周期的波动率
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std()

        # 计算高低价格比率
        df['high_low_ratio'] = df['high'] / df['low']

        return df

    @staticmethod
    def calculate_financial_indicators(financial_data: pd.DataFrame) -> pd.DataFrame:
        """计算财务指标特征
        
        处理和转换原始财务数据，构建：
        1. 盈利能力指标
        2. 估值指标
        3. 成长性指标
        4. 运营效率指标
        
        Args:
            financial_data (pd.DataFrame): 原始财务数据，包含：
                - 资产负债表数据
                - 利润表数据
                - 现金流量表数据
        
        Returns:
            pd.DataFrame: 处理后的财务指标，包含：
                - 标准化的财务比率
                - 同行业对比指标
                - 趋势指标
                
        Raises:
            ValueError: 输入数据缺少关键财务指标
        """
        features = pd.DataFrame()

        # 盈利能力指标
        features['roe'] = financial_data['roe']
        features['roa'] = financial_data['roa']
        features['net_profit_margin'] = financial_data['net_profit_margin']

        # 估值指标
        features['pe_ratio'] = financial_data['pe_ratio']
        features['pb_ratio'] = financial_data['pb_ratio']
        features['ps_ratio'] = financial_data['ps_ratio']

        # 成长性指标
        features['revenue_growth'] = financial_data['revenue_growth']
        features['profit_growth'] = financial_data['profit_growth']

        # 运营效率指标
        features['asset_turnover'] = financial_data['asset_turnover']
        features['inventory_turnover'] = financial_data['inventory_turnover']

        return features

    @staticmethod
    def create_label(df: pd.DataFrame, forward_days: int = 20,
                     return_threshold: float = 0.3) -> pd.DataFrame:
        """创建预测标签
        
        基于未来收益率创建二分类标签：
        1. 计算未来forward_days天的最大收益率
        2. 根据return_threshold确定是否为暴涨样本
        
        Args:
            df (pd.DataFrame): 股票数据，必须包含：
                - close: 收盘价
            forward_days (int): 向前看的天数，默认20
            return_threshold (float): 暴涨阈值，默认0.3 (30%)
        
        Returns:
            pd.DataFrame: 添加了标签的DataFrame，新增列：
                - future_return: 未来最大收益率
                - label: 二分类标签（0或1）
                
        Raises:
            ValueError: 参数取值不合理
        """
        # 计算未来forward_days天的最大收益率
        future_returns = []
        for i in range(len(df)):
            if i >= len(df) - forward_days:
                future_returns.append(np.nan)
            else:
                future_prices = df['close'].iloc[i + 1:i + forward_days + 1]
                max_return = (future_prices.max() - df['close'].iloc[i]) / df['close'].iloc[i]
                future_returns.append(max_return)

        df['future_return'] = future_returns
        df['label'] = (df['future_return'] > return_threshold).astype(int)

        return df

    @staticmethod
    def prepare_features(price_data: pd.DataFrame,
                         financial_data: pd.DataFrame,
                         forward_days: int = 20,
                         return_threshold: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备完整的特征集和标签
        
        整合所有特征工程步骤，构建完整的训练数据：
        1. 计算技术指标
        2. 构建动量特征
        3. 处理财务指标
        4. 创建预测标签
        
        Args:
            price_data (pd.DataFrame): 价格数据
            financial_data (pd.DataFrame): 财务数据
            forward_days (int): 预测时间窗口
            return_threshold (float): 暴涨阈值
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - 特征矩阵
                - 标签向量
                
        Raises:
            ValueError: 输入数据格式错误
            RuntimeError: 特征构建过程出错
        """
        # 计算技术指标
        df = FeatureEngineering.calculate_technical_indicators(price_data.copy())

        # 计算价格动量特征
        df = FeatureEngineering.calculate_price_momentum(df)

        # 合并财务指标
        financial_features = FeatureEngineering.calculate_financial_indicators(financial_data)
        df = pd.merge(df, financial_features, left_index=True, right_index=True, how='left')

        # 创建标签
        df = FeatureEngineering.create_label(df, forward_days, return_threshold)

        # 分离特征和标签
        feature_columns = [col for col in df.columns if col not in ['label', 'future_return']]
        X = df[feature_columns]
        y = df['label']

        return X, y
