import decimal
from typing import Tuple

import numpy as np
import pandas as pd

from backend.utils.indicators import CalIndicators
from backend.utils.logger import setup_logger

logger = setup_logger("train_model")


class ExplosiveStockDataCollector:
    """
    爆发式股票数据收集器：用于收集和处理训练数据
    目标：预测未来20天内可能涨幅超过30%的股票
    """

    def __init__(self,
                 price_increase_threshold=0.3,  # 涨幅阈值30%
                 forward_window=20,  # 向前看20个交易日
                 volume_multiplier=3  # 成交量倍数阈值
                 ):
        self.price_threshold = price_increase_threshold
        self.forward_window = forward_window
        self.volume_multiplier = volume_multiplier

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据框，根据字段类型进行适当的转换"""
        # 需要转换为float的decimal字段
        float_columns = [
            'open', 'high', 'low', 'close', 'preclose',
            'amount', 'turn', 'pct_chg',
            'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm'
        ]

        # 需要保持为整数的字段
        int_columns = ['volume', 'tradestatus', 'is_st']

        for column in df.columns:
            if column in float_columns and df[column].dtype == object:
                try:
                    df[column] = df[column].apply(
                        lambda x: float(x) if isinstance(x, decimal.Decimal) else x
                    )
                except Exception as e:
                    logger.warning(f"转换列 {column} 时出错: {e}")
            elif column in int_columns and df[column].dtype == object:
                try:
                    df[column] = df[column].apply(
                        lambda x: int(x) if isinstance(x, decimal.Decimal) else x
                    )
                except Exception as e:
                    logger.warning(f"转换列 {column} 时出错: {e}")

        return df

    def collect_training_data(self, stock_data: pd.DataFrame) -> tuple:
        """收集训练数据"""
        try:
            # 预处理数据
            df = self._preprocess_dataframe(stock_data.copy())

            # 生成特征
            features = self._generate_features(df)

            # 生成标签
            labels = self._generate_labels(df)

            return features, labels

        except Exception as e:
            logger.exception(f"收集训练数据时出错: {e}")
            return pd.DataFrame(), pd.Series()

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成预测特征"""
        try:
            # 确保所有数值都是float类型
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].astype(float)

            features = pd.DataFrame(index=df.index)

            # 1. 基础价格特征
            features['price_ma5'] = CalIndicators.ema(df, 5, 'close')
            features['price_ma10'] = CalIndicators.ema(df, 10, 'close')
            features['price_ma20'] = CalIndicators.ema(df, 20, 'close')
            features['price_ma60'] = CalIndicators.ema(df, 60, 'close')

            # 计算均线多空排列
            features['ma_trend'] = self._calculate_ma_trend(features)

            # 2. 成交量特征
            features['volume_ma5'] = CalIndicators.sma(df, 5, 'volume')
            features['volume_ma10'] = CalIndicators.sma(df, 10, 'volume')
            features['volume_ratio'] = df['volume'] / features['volume_ma5']
            features['volume_trend'] = self._calculate_volume_trend(df)

            # 新增：成交额特征
            features['amount_ma5'] = df['amount'].rolling(5).mean()
            features['amount_ratio'] = df['amount'] / features['amount_ma5']

            # 新增：换手率特征
            features['turn_ma5'] = df['turn'].rolling(5).mean()
            features['turn_ma20'] = df['turn'].rolling(20).mean()
            features['turn_ratio'] = df['turn'] / features['turn_ma20']

            # 3. 动量指标
            features['roc'] = CalIndicators.roc(df, 12)
            features['kdj_k'], features['kdj_d'], features['kdj_j'] = CalIndicators.kdj(df)
            features['rsi'] = CalIndicators.rsi(df, 14)
            features['macd'], features['macd_signal'], features['macd_hist'] = CalIndicators.macd(df)

            # 新增：涨跌幅特征
            features['pct_chg_ma5'] = df['pct_chg'].rolling(5).mean()
            features['pct_chg_std'] = df['pct_chg'].rolling(20).std()
            features['up_down_streak'] = self._calculate_streak(df['pct_chg'])

            # 4. 波动率特征
            features['volatility'] = self._calculate_volatility(df)
            features['atr'] = self._calculate_atr(df)
            features['bb_width'] = self._calculate_bb_width(df)

            # 新增：日内波动特征
            features['daily_range'] = (df['high'] - df['low']) / df['preclose'] * 100
            features['gap'] = (df['open'] - df['preclose']) / df['preclose'] * 100

            # 5. 估值特征（新增）
            for col in ['pe_ttm', 'pb_mrq', 'ps_ttm']:
                if col in df.columns:
                    features[f'{col}_percentile'] = self._calculate_percentile(df[col], 250)
                    features[f'{col}_change'] = df[col].pct_change(5)

            # 6. 趋势强度特征
            features['dmi_pdi'], features['dmi_mdi'], features['dmi_adx'] = CalIndicators.dmi(df)
            features['trend_strength'] = self._calculate_trend_strength(df)

            # 7. 资金流向特征
            features['mfi'] = self._calculate_mfi(df)
            features['obv'] = self._calculate_obv(df)

            # 7. 交易状态特征（新增）
            if 'is_st' in df.columns:
                features['is_st'] = df['is_st'].astype(float)
            if 'tradestatus' in df.columns:
                features['tradestatus'] = df['tradestatus'].astype(float)

            # 新增：趋势持续性特征
            features['sideways_days'] = self._calculate_sideways_days(df)
            features['volatility_days'] = self._calculate_volatility_days(df)
            features['trend_duration'] = self._calculate_trend_duration(df)

            # 新增：历史统计特征
            features['price_momentum'] = self._calculate_price_momentum(df)
            features['historical_volatility'] = self._calculate_historical_volatility(df)
            features['volume_pattern'] = self._calculate_volume_pattern(df)

            # 新增：周期性特征
            features['seasonal_pattern'] = self._calculate_seasonal_pattern(df)
            features['price_cycle'] = self._detect_price_cycle(df)

            return features

        except Exception as e:
            logger.exception(f"生成特征时出错: {e}")
            return pd.DataFrame()

    def _calculate_ma_trend(self, features: pd.DataFrame) -> pd.Series:
        """计算均线多空排列趋势
        1: 多头排列
        0: 盘整
        -1: 空头排列
        """
        ma_trend = pd.Series(0, index=features.index)

        bullish = (features['price_ma5'] > features['price_ma10']) & \
                  (features['price_ma10'] > features['price_ma20']) & \
                  (features['price_ma20'] > features['price_ma60'])

        bearish = (features['price_ma5'] < features['price_ma10']) & \
                  (features['price_ma10'] < features['price_ma20']) & \
                  (features['price_ma20'] < features['price_ma60'])

        ma_trend[bullish] = 1
        ma_trend[bearish] = -1
        return ma_trend

    def _calculate_volume_trend(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量趋势"""
        volume = df['volume']
        vol_ma5 = volume.rolling(window=5).mean()
        vol_ma20 = volume.rolling(window=20).mean()

        trend = pd.Series(0, index=df.index)
        trend[volume > vol_ma5 * 1.5] = 1  # 放量
        trend[volume < vol_ma5 * 0.5] = -1  # 缩量
        return trend

    def _calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算波动率"""
        returns = df['close'].pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_bb_width(self, df: pd.DataFrame) -> pd.Series:
        """计算布林带宽度"""
        mid, upper, lower = CalIndicators.bollinger_bands(df)
        return (upper - lower) / mid

    def _detect_double_bottom(self, df: pd.DataFrame) -> pd.Series:
        """检测双底形态"""
        window = 20
        result = pd.Series(0, index=df.index)

        for i in range(window, len(df)):
            section = df['low'].iloc[i - window:i]
            bottoms = self._find_bottoms(section)
            if len(bottoms) >= 2:
                last_two = bottoms[-2:]
                if abs(last_two[0] - last_two[1]) / last_two[0] < 0.02:  # 两个底部价格相差不超过2%
                    result.iloc[i] = 1

        return result

    def _detect_flag_pattern(self, df: pd.DataFrame) -> pd.Series:
        """检测旗形形态"""
        result = pd.Series(0, index=df.index)
        window = 20

        for i in range(window, len(df)):
            section = df.iloc[i - window:i]
            if self._is_flag_pattern(section):
                result.iloc[i] = 1

        return result

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.Series:
        """计算支撑阻力位强度"""
        window = 20
        levels = pd.Series(0, index=df.index)

        for i in range(window, len(df)):
            section = df.iloc[i - window:i]
            price = df['close'].iloc[i]
            support = CalIndicators.support(section)
            resistance = CalIndicators.resistance(section)

            # 计算当前价格在支撑位和阻力位之间的位置
            if support and resistance:
                position = (price - support) / (resistance - support)
                levels.iloc[i] = position

        return levels

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算趋势强度"""
        # 使用ADX和价格趋势计算综合趋势强度
        pdi, mdi, adx = CalIndicators.dmi(df)
        price_trend = df['close'].pct_change().rolling(window=20).mean()

        trend_strength = pd.Series(0, index=df.index)
        trend_strength = np.where(price_trend > 0, adx * 1, adx * -1)
        return pd.Series(trend_strength, index=df.index)

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算资金流向指标(Money Flow Index)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume'].astype(float)  # 显式转换volume为float用于计算

        positive_flow = pd.Series(0.0, index=df.index)  # 明确指定dtype为float
        negative_flow = pd.Series(0.0, index=df.index)  # 明确指定dtype为float

        # 计算正向和负向资金流
        price_change = typical_price.diff()
        positive_flow[price_change > 0] = money_flow[price_change > 0]
        negative_flow[price_change < 0] = money_flow[price_change < 0]

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """计算能量潮指标(On Balance Volume)"""
        obv = pd.Series(0, index=df.index)
        obv.iloc[0] = df['volume'].iloc[0]

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

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
            current_volume = df['volume'].iloc[i - 5:i].mean()  # 当前5日平均成交量
            future_max_volume = future_window['volume'].max()

            # 添加除零保护
            if current_volume > 0:
                volume_increase = future_max_volume / current_volume
            else:
                volume_increase = 0  # 如果当前成交量为0，则设置volume_increase为0

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

    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """计算连续涨跌天数"""
        streak = pd.Series(0, index=series.index)
        streak[1:] = np.where(
            series[1:] * series[:-1].values > 0,
            streak[:-1].values + np.sign(series[1:]),
            np.sign(series[1:])
        )
        return streak

    def _calculate_percentile(self, series: pd.Series, window: int) -> pd.Series:
        """计算历史分位数"""
        return series.rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

    def analyze_feature_correlation(self, features: pd.DataFrame):
        """分析特征相关性"""
        try:
            # 计算相关性矩阵
            corr_matrix = features.corr()

            # 找出高度相关的特征对
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:  # 相关系数阈值
                        high_corr.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })

            # 输出结果
            if high_corr:
                logger.info("\n高度相关的特征对：")
                for pair in high_corr:
                    logger.info(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")

            return high_corr

        except Exception as e:
            logger.exception(f"特征相关性分析失败: {e}")
            return None

    def _find_bottoms(self, series: pd.Series) -> list:
        """找出序列中的底部点"""
        bottoms = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] < series.iloc[i - 1] and series.iloc[i] < series.iloc[i + 1]:
                bottoms.append(series.iloc[i])
        return bottoms

    def _is_flag_pattern(self, section: pd.DataFrame) -> bool:
        """检测旗形形态
        旗形特征：
        1. 前期有明显上涨
        2. 横盘整理，形成平行通道
        3. 成交量逐渐萎缩
        """
        try:
            # 计算前期涨幅
            price_change = (section['close'].iloc[-1] - section['close'].iloc[0]) / section['close'].iloc[0]

            # 计算通道的上下轨道线斜率
            highs = section['high']
            lows = section['low']
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]

            # 检查成交量趋势
            volume_trend = np.polyfit(range(len(section)), section['volume'], 1)[0]

            # 判断是否符合旗形特征
            is_flag = (
                    price_change > 0.1 and  # 前期涨幅超过10%
                    abs(high_slope - low_slope) < 0.01 and  # 上下轨道基本平行
                    volume_trend < 0  # 成交量萎缩
            )

            return is_flag

        except Exception as e:
            logger.warning(f"旗形形态检测失败: {e}")
            return False

    def _calculate_sideways_days(self, df: pd.DataFrame, threshold: float = 0.03) -> pd.Series:
        """计算横盘天数
        threshold: 价格波动阈值，默认3%
        """
        sideways_days = pd.Series(0, index=df.index)
        price = df['close']

        for i in range(1, len(df)):
            # 计算前一天的横盘天数
            prev_days = sideways_days.iloc[i - 1]

            # 计算当天价格相对前一天的变化幅度
            price_change = abs((price.iloc[i] - price.iloc[i - 1]) / price.iloc[i - 1])

            # 如果价格变化在阈值内，累加横盘天数
            if price_change <= threshold:
                sideways_days.iloc[i] = prev_days + 1

        return sideways_days

    def _calculate_volatility_days(self, df: pd.DataFrame,
                                   window: int = 20,
                                   threshold: float = 0.02) -> pd.Series:
        """计算高波动持续天数
        window: 计算波动率的窗口期
        threshold: 高波动阈值
        """
        volatility = df['close'].pct_change().rolling(window).std()
        volatility_days = pd.Series(0, index=df.index)

        for i in range(window, len(df)):
            if volatility.iloc[i] > threshold:
                volatility_days.iloc[i] = volatility_days.iloc[i - 1] + 1

        return volatility_days

    def _calculate_trend_duration(self, df: pd.DataFrame,
                                  ma_period: int = 20) -> pd.Series:
        """计算趋势持续时间
        1: 上升趋势持续天数
        -1: 下降趋势持续天数
        """
        ma = df['close'].rolling(ma_period).mean()
        trend_duration = pd.Series(0, index=df.index)

        for i in range(ma_period, len(df)):
            if df['close'].iloc[i] > ma.iloc[i]:
                trend_duration.iloc[i] = (
                    trend_duration.iloc[i - 1] + 1 if trend_duration.iloc[i - 1] >= 0
                    else 1
                )
            else:
                trend_duration.iloc[i] = (
                    trend_duration.iloc[i - 1] - 1 if trend_duration.iloc[i - 1] <= 0
                    else -1
                )

        return trend_duration

    def _calculate_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算价格动量
        使用多个时间窗口的收益率来衡量动量
        """
        momentum = pd.Series(0, index=df.index)
        windows = [5, 10, 20, 60]  # 多个时间窗口
        weights = [0.4, 0.3, 0.2, 0.1]  # 对应权重

        for window, weight in zip(windows, weights):
            returns = df['close'].pct_change(window)
            momentum += returns * weight

        return momentum

    def _calculate_historical_volatility(self, df: pd.DataFrame) -> pd.Series:
        """计算历史波动率特征"""
        windows = [5, 10, 20, 60]
        vol_features = pd.Series(0, index=df.index)

        for window in windows:
            # 计算不同周期的波动率
            vol = df['close'].pct_change().rolling(window).std() * np.sqrt(252)
            vol_features += vol / len(windows)

        return vol_features

    def _calculate_volume_pattern(self, df: pd.DataFrame) -> pd.Series:
        """分析成交量模式
        1: 放量
        0: 正常
        -1: 缩量
        """
        volume = df['volume']
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()

        pattern = pd.Series(0, index=df.index)
        pattern[volume > vol_ma5 * 1.5] = 1  # 放量
        pattern[volume < vol_ma20 * 0.5] = -1  # 缩量

        return pattern

    def _calculate_seasonal_pattern(self, df: pd.DataFrame) -> pd.Series:
        """计算季节性模式"""
        # 提取日期特征
        dates = pd.to_datetime(df.index)
        month = dates.month

        # 计算每个月的历史表现
        monthly_returns = pd.Series(0, index=df.index)

        for m in range(1, 13):
            # 计算该月份的历史平均收益
            month_mask = month == m
            if month_mask.any():
                monthly_returns[month_mask] = df['pct_chg'][month_mask].mean()

        return monthly_returns

    def _detect_price_cycle(self, df: pd.DataFrame,
                            window: int = 120) -> pd.Series:
        """检测价格周期
        使用傅里叶变换检测主要周期
        """
        try:
            from scipy import fft

            cycles = pd.Series(0, index=df.index)
            price = df['close'].values

            for i in range(window, len(df)):
                # 获取滑动窗口的价格数据
                window_prices = price[i - window:i]

                # 执行傅里叶变换
                fft_values = fft.fft(window_prices)
                frequencies = fft.fftfreq(len(window_prices))

                # 找出主要周期
                main_freq_idx = np.argmax(np.abs(fft_values)[1:]) + 1
                main_cycle = 1 / frequencies[main_freq_idx]

                cycles.iloc[i] = main_cycle

            return cycles

        except Exception as e:
            logger.warning(f"周期检测失败: {e}")
            return pd.Series(0, index=df.index)
