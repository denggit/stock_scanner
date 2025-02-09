import pandas as pd
import numpy as np
from typing import Tuple

from backend.utils.indicators import CalIndicators
from backend.utils.logger import setup_logger

logger = setup_logger("train_model")


class ExplosiveStockDataCollector:
    """
    爆发式股票数据收集器：用于收集和处理训练数据
    目标：预测未来20天内可能涨幅超过30%的股票
    """
    
    def __init__(self, 
                 price_increase_threshold=0.3,   # 涨幅阈值30%
                 forward_window=20,              # 向前看20个交易日
                 volume_multiplier=3             # 成交量倍数阈值
                 ):
        self.price_threshold = price_increase_threshold
        self.forward_window = forward_window
        self.volume_multiplier = volume_multiplier
    
    def collect_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        收集训练数据的主方法
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            features: 特征DataFrame
            labels: 标签Series (1表示未来20天内会涨30%，0表示不会)
        """
        # 1. 生成特征
        features = self._generate_features(df)
        
        # 2. 生成标签
        labels = self._generate_labels(df)
        
        # 3. 清理数据
        return self._clean_data(features, labels)
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成预测特征"""
        features = pd.DataFrame(index=df.index)
        
        # 使用 CalIndicators 计算均线
        features['price_ma5'] = CalIndicators.ema(df, 5, 'close')
        features['price_ma10'] = CalIndicators.ema(df, 10, 'close')
        features['price_ma20'] = CalIndicators.ema(df, 20, 'close')
        features['ma5_ma10_cross'] = (features['ma5'] > features['ma10']).astype(int)

        # 计算MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = CalIndicators.macd(df)
        
        # 计算RSI
        features['rsi'] = CalIndicators.rsi(df, 14)
        
        # 计算布林带
        features['mid_band'], features['upper_band'], features['lower_band'] = CalIndicators.bollinger_bands(
            df, ma_period=20, bollinger_k=2
        )
        
        # 成交量指标
        features['volume_ma5'] = CalIndicators.sma(df, 5, 'volume')
        features['volume_ma10'] = CalIndicators.sma(df, 10, 'volume')
        features['volume_ratio'] = df['volume'] / features['volume_ma5']
        
        # 动量指标
        features['roc'] = CalIndicators.roc(df, 12)
        features['kdj_k'], features['kdj_d'], features['kdj_j'] = CalIndicators.kdj(df)
        
        # 趋势指标
        features['dmi_pdi'], features['dmi_mdi'], features['dmi_adx'] = CalIndicators.dmi(df)
        
        # 价格位置特征
        features['bb_position'] = (df['close'] - features['mid_band']) / (features['upper_band'] - features['lower_band'])
        features['ma_position'] = (df['close'] - features['ma20']) / features['ma20']
        
        return features
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        生成标签：标记未来20天内是否会出现30%以上涨幅的时间点
        """
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - self.forward_window):
            current_price = df['close'].iloc[i]
            future_window = df.iloc[i:i + self.forward_window]
            
            # 计算未来区间内的最大涨幅
            future_max_price = future_window['high'].max()
            price_increase = (future_max_price - current_price) / current_price
            
            # 计算未来成交量是否会放大
            current_volume = df['volume'].iloc[i-5:i].mean()  # 当前5日平均成交量
            future_max_volume = future_window['volume'].max()
            volume_increase = future_max_volume / current_volume
            
            # 如果未来会出现涨幅超过阈值且成交量放大，则当前时间点标记为1
            if (price_increase >= self.price_threshold and 
                volume_increase >= self.volume_multiplier):
                labels.iloc[i] = 1
        
        # 对于最后forward_window天的数据标记为-1（因为无法知道未来）
        labels.iloc[-self.forward_window:] = -1
        
        return labels
    
    def _clean_data(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """清理数据：删除无效样本"""
        # 删除标签为-1的数据和包含NaN的行
        valid_mask = (labels != -1) & (~features.isnull().any(axis=1))
        return features[valid_mask], labels[valid_mask]
