#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Your Name
@Date       : 2025-02-04
@File       : strategy_template.py
@Description: 策略开发模板 - 复制此文件并修改以创建新策略
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class StrategyTemplate(BaseStrategy):
    """
    策略模板 - 请修改类名和描述
    
    策略逻辑描述：
    1. 第一步：数据预处理
    2. 第二步：计算技术指标
    3. 第三步：生成交易信号
    4. 第四步：返回结果
    """

    def __init__(self):
        super().__init__(name="策略模板", description="请修改为您的策略描述")
        self._init_params()

    def _init_params(self):
        """初始化策略参数"""
        self._params = {
            # 基础参数
            "signal": 70.0,  # 信号阈值
            "volume_ratio": 1.5,  # 成交量比率阈值
            "rsi_range": (30, 70),  # RSI范围
            
            # 技术指标参数
            "ma_period": 20,  # 移动平均线周期
            "rsi_period": 14,  # RSI周期
            "bb_period": 20,  # 布林带周期
            "bb_std": 2.0,  # 布林带标准差
            
            # 权重配置
            "price_weight": 0.3,  # 价格权重
            "volume_weight": 0.3,  # 成交量权重
            "technical_weight": 0.4,  # 技术指标权重
            
            # 风险控制
            "max_risk": 0.1,  # 最大风险
            "stop_loss": 0.05,  # 止损比例
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            data: 股票数据DataFrame
            
        Returns:
            pd.Series: 包含信号信息的Series
        """
        try:
            # 1. 数据验证
            if not self.validate_data(data):
                return pd.Series({'signal': 0})

            # 2. 数据预处理
            df = self._preprocess_data(data)

            # 3. 计算技术指标
            df = self._calculate_indicators(df)

            # 4. 生成信号
            signal = self._generate_signal_logic(df)

            # 5. 应用过滤条件
            if not self._apply_filters(signal):
                return pd.Series({'signal': 0})

            return signal

        except Exception as e:
            logging.exception(f"生成信号时发生错误: {e}")
            return pd.Series({'signal': 0})

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        df = data.copy()
        
        # 确保数据按日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算基本变化率
        df['pct_chg'] = df['close'].pct_change()
        df['volume_pct'] = df['volume'].pct_change()
        
        # 移除无效数据
        df = df.dropna()
        
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 预处理后的数据
            
        Returns:
            pd.DataFrame: 包含技术指标的数据
        """
        # 移动平均线
        df['ma'] = CalIndicators.ema(df, self._params['ma_period'], 'close')
        
        # RSI
        df['rsi'] = CalIndicators.rsi(df, self._params['rsi_period'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = CalIndicators.macd(df)
        
        # 布林带
        df['bb_mid'], df['bb_upper'], df['bb_lower'] = CalIndicators.bollinger_bands(
            df, self._params['bb_period'], self._params['bb_std']
        )
        
        # 成交量均线
        df['volume_ma'] = CalIndicators.ema(df, self._params['ma_period'], 'volume')
        
        # 添加您的自定义指标
        # df['custom_indicator'] = self._calculate_custom_indicator(df)
        
        return df

    def _generate_signal_logic(self, df: pd.DataFrame) -> pd.Series:
        """
        生成信号逻辑
        
        Args:
            df: 包含技术指标的数据
            
        Returns:
            pd.Series: 信号信息
        """
        # 获取最新数据
        latest = df.iloc[-1]
        
        # 计算各维度得分
        price_score = self._calculate_price_score(df)
        volume_score = self._calculate_volume_score(df)
        technical_score = self._calculate_technical_score(df)
        
        # 计算综合得分
        final_score = (
            price_score * self._params['price_weight'] +
            volume_score * self._params['volume_weight'] +
            technical_score * self._params['technical_weight']
        )
        
        # 生成信号
        signal = pd.Series({
            'signal': round(final_score * 100, 2),
            'trade_date': latest['trade_date'],
            'price': round(float(latest['close']), 2),
            'volume_ratio': round(float(latest['volume'] / latest['volume_ma']), 2),
            'rsi': round(float(latest['rsi']), 2),
            'macd_hist': round(float(latest['macd_hist']), 4),
            'price_score': round(price_score * 100, 2),
            'volume_score': round(volume_score * 100, 2),
            'technical_score': round(technical_score * 100, 2),
            'buy_signal': self._generate_buy_signal(final_score),
            'risk_level': self._assess_risk(df)
        })
        
        return signal

    def _calculate_price_score(self, df: pd.DataFrame) -> float:
        """计算价格得分"""
        latest = df.iloc[-1]
        score = 0.0
        
        # 价格趋势
        if latest['close'] > latest['ma']:
            score += 0.3
        
        # 价格位置（布林带）
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        if 0.2 <= bb_position <= 0.8:
            score += 0.4
        
        # 价格动量
        if latest['pct_chg'] > 0:
            score += 0.3
        
        return min(score, 1.0)

    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """计算成交量得分"""
        latest = df.iloc[-1]
        score = 0.0
        
        # 成交量比率
        volume_ratio = latest['volume'] / latest['volume_ma']
        if volume_ratio > 1.5:
            score += 0.5
        elif volume_ratio > 1.0:
            score += 0.3
        
        # 量价配合
        if latest['pct_chg'] > 0 and latest['volume_pct'] > 0:
            score += 0.5
        
        return min(score, 1.0)

    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """计算技术指标得分"""
        latest = df.iloc[-1]
        score = 0.0
        
        # RSI
        rsi = latest['rsi']
        if 30 <= rsi <= 70:
            score += 0.3
        elif 40 <= rsi <= 60:
            score += 0.5
        
        # MACD
        if latest['macd_hist'] > 0:
            score += 0.3
        
        # MACD金叉
        if len(df) >= 2:
            if df['macd_hist'].iloc[-1] > 0 and df['macd_hist'].iloc[-2] < 0:
                score += 0.4
        
        return min(score, 1.0)

    def _apply_filters(self, signal: pd.Series) -> bool:
        """
        应用过滤条件
        
        Args:
            signal: 信号信息
            
        Returns:
            bool: 是否通过过滤
        """
        # 信号强度过滤
        if signal['signal'] < self._params['signal']:
            return False
        
        # 成交量比率过滤
        if signal['volume_ratio'] < self._params['volume_ratio']:
            return False
        
        # RSI范围过滤
        rsi_min, rsi_max = self._params['rsi_range']
        if not (rsi_min <= signal['rsi'] <= rsi_max):
            return False
        
        return True

    def _generate_buy_signal(self, score: float) -> str:
        """生成买入建议"""
        if score > 0.8:
            return "强烈建议买入"
        elif score > 0.6:
            return "建议买入"
        elif score > 0.4:
            return "观察"
        else:
            return "暂不建议"

    def _assess_risk(self, df: pd.DataFrame) -> str:
        """评估风险等级"""
        volatility = df['close'].pct_change().std() * np.sqrt(252)
        
        if volatility > 0.5:
            return "高风险"
        elif volatility > 0.3:
            return "中等风险"
        else:
            return "低风险"

    def _calculate_custom_indicator(self, df: pd.DataFrame) -> pd.Series:
        """
        计算自定义指标
        
        Args:
            df: 数据
            
        Returns:
            pd.Series: 自定义指标
        """
        # 在这里实现您的自定义指标
        # 例如：自定义动量指标、波动率指标等
        
        # 示例：计算价格动量
        momentum = df['close'].pct_change(5)  # 5日动量
        
        return momentum

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据是否满足基本条件"""
        try:
            if len(data) < 60:  # 至少需要60个交易日的数据
                logging.warning(f"数据不足60天，跳过")
                return False

            # 检查必要的列是否存在
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_date']
            if not all(col in data.columns for col in required_columns):
                logging.warning(f"缺少必要的列，跳过")
                return False

            return True
        except Exception as e:
            logging.exception(f"验证数据时发生错误: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 创建策略实例
    strategy = StrategyTemplate()
    
    # 设置参数
    strategy.set_parameters({
        "signal": 75.0,
        "volume_ratio": 2.0,
        "rsi_range": (35, 65)
    })
    
    print("策略模板创建成功！")
    print("请复制此文件并修改以创建您的新策略。") 