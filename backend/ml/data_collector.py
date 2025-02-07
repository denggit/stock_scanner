import numpy as np
import pandas as pd

from backend.utils.indicators import CalIndicators
from backend.utils.logger import setup_logger

logger = setup_logger("train_model")


class ExplosiveStockDataCollector:
    """收集爆发式股票的训练数据"""

    def __init__(self):
        self.features = []
        self.labels = []

    def collect_training_data(self, stock_data: pd.DataFrame) -> tuple:
        """
        收集单只股票的训练数据
        
        Args:
            stock_data: 股票历史数据，包含OHLCV等信息
            
        Returns:
            tuple: (特征DataFrame, 标签Series)
        """
        try:
            df = stock_data.copy()

            # 生成标签：未来20个交易日内是否出现30%以上涨幅
            df['future_return'] = self._calculate_future_returns(df, days=20)
            df['is_explosive'] = (df['future_return'] >= 0.3).astype(int)

            # 生成特征
            features_df = self._generate_features(df)

            # 去除包含NaN的行
            valid_mask = ~features_df.isnull().any(axis=1)
            features_df = features_df[valid_mask]
            labels = df['is_explosive'][valid_mask]

            return features_df, labels

        except Exception as e:
            logger.exception(f"收集训练数据时发生错误: {e}")
            return pd.DataFrame(), pd.Series()

    @staticmethod
    def _calculate_future_returns(df: pd.DataFrame, days: int) -> pd.Series:
        """计算未来N个交易日的最大收益率"""
        future_returns = []
        for i in range(len(df) - days):
            current_price = df['close'].iloc[i]
            future_max = df['high'].iloc[i:i + days].max()
            future_return = (future_max - current_price) / current_price
            future_returns.append(future_return)

        # 对于最后days天，填充NaN
        future_returns.extend([np.nan] * days)
        return pd.Series(future_returns, index=df.index)

    @staticmethod
    def _generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """生成特征"""
        # 只将需要进行数值计算的列转换为float类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)

        features = pd.DataFrame(index=df.index)

        # 1. 价格特征
        features['price_ma5'] = CalIndicators.sma(df, period=5)
        features['price_ma10'] = CalIndicators.sma(df, period=10)
        features['price_ma20'] = CalIndicators.sma(df, period=20)

        # 2. 成交量特征
        features['volume_ma5'] = CalIndicators.sma(df, period=5, cal_value='volume')
        features['volume_ma10'] = CalIndicators.sma(df, period=10, cal_value='volume')
        features['volume_ma20'] = CalIndicators.sma(df, period=20, cal_value='volume')

        # 3. 波动率特征
        features['volatility_5'] = df['close'].pct_change().rolling(5).std()
        features['volatility_10'] = df['close'].pct_change().rolling(10).std()

        # 4. 动量特征
        features['momentum_5'] = df['close'].pct_change(5)
        features['momentum_10'] = df['close'].pct_change(10)
        features['momentum_20'] = df['close'].pct_change(20)

        # 5. RSI特征
        features['rsi_14'] = CalIndicators.rsi(df, 14)

        # 6. 布林带特征
        bb_mid, bb_upper, bb_lower = CalIndicators.bollinger_bands(df, ma_period=20, bollinger_k=2)
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_mid

        return features
