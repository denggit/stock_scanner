import decimal
import gc
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.utils.indicators import CalIndicators
from backend.utils.logger import setup_logger

logger = setup_logger("train_model", set_root_logger=True)


class ExplosiveStockDataCollector:
    """
    爆发式股票数据收集器：用于收集和处理训练数据
    目标：预测未来20天内可能涨幅超过30%的股票
    """

    def __init__(self,
                 price_increase_threshold=0.3,  # 涨幅阈值30%
                 forward_window=20,  # 向前看20个交易日
                 volume_multiplier=3,  # 成交量倍数阈值
                 feature_params=None,
                 cache_dir="temp_results"):  # 添加缓存目录参数
        self.price_threshold = price_increase_threshold
        self.forward_window = forward_window
        self.volume_multiplier = volume_multiplier
        self.cache_dir = Path(cache_dir)
        # 使用默认参数或传入的自定义参数
        self.feature_params = {
            # 均线参数
            'ma_periods': [5, 10, 20, 60],

            # 波动率参数
            'volatility_window': 20,
            'volatility_threshold': 0.02,

            # 趋势参数
            'sideways_threshold': 0.03,
            'trend_ma_period': 20,

            # 动量参数
            'momentum_windows': [5, 10, 20, 60],
            'momentum_weights': [0.4, 0.3, 0.2, 0.1],

            # 周期参数
            'cycle_window': 120,

            # KDJ参数
            'kdj_window': 9,
            'kdj_smooth': 3,

            # MACD参数
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,

            # RSI参数
            'rsi_window': 14,

            # 布林带参数
            'bb_window': 20,
            'bb_std': 2,

            # DMI参数
            'dmi_window': 14,
            'dmi_smooth': 14,

            # 成交量参数
            'volume_ma_windows': [5, 10],
            'volume_ratio_threshold': 1.5,

            # 历史统计参数
            'historical_windows': [5, 10, 20, 60],
            'percentile_window': 250,

            # 趋势强度参数
            'trend_strength_window': 20,

            # 资金流向参数
            'mfi_period': 14,

            # 季节性参数
            'seasonal_period': 20
        } if feature_params is None else feature_params

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
            for period in self.feature_params['ma_periods']:
                features[f'price_ma{period}'] = CalIndicators.ema(df, period, 'close')

            features['volatility'] = self._calculate_volatility(df, 
                window=self.feature_params['volatility_window'])

            features['ma_trend'] = self._calculate_ma_trend(features)

            # 2. 成交量特征
            for period in self.feature_params['volume_ma_windows'][0]:  # 使用第一组参数
                features[f'volume_ma{period}'] = CalIndicators.sma(df, period, 'volume')
            
            features['volume_ratio'] = df['volume'] / features['volume_ma5']
            features['volume_trend'] = self._calculate_volume_trend(df)

            # 3. 动量指标
            momentum_result = 0
            for window, weight in zip(
                self.feature_params['momentum_windows'][0],
                self.feature_params['momentum_weights'][0]
            ):
                momentum_result += weight * CalIndicators.roc(df, window)
            features['momentum'] = momentum_result

            features['kdj_k'], features['kdj_d'], features['kdj_j'] = CalIndicators.kdj(
                df, window=self.feature_params['kdj_window']
            )

            features['rsi'] = CalIndicators.rsi(
                df, window=self.feature_params['rsi_window']
            )

            features['macd'], features['macd_signal'], features['macd_hist'] = CalIndicators.macd(
                df,
                fast=self.feature_params['macd_fast'],
                slow=self.feature_params['macd_slow'],
                signal=self.feature_params['macd_signal']
            )

            bb_mid, bb_upper, bb_lower = CalIndicators.bollinger_bands(
                df,
                window=self.feature_params['bb_window'],
                std=self.feature_params['bb_std']
            )
            features['bb_width'] = (bb_upper - bb_lower) / bb_mid

            features['dmi_pdi'], features['dmi_mdi'], features['dmi_adx'] = CalIndicators.dmi(
                df, window=self.feature_params['dmi_window']
            )

            features['trend_strength'] = self._calculate_trend_strength(
                df, window=self.feature_params['trend_strength_window']
            )

            features['mfi'] = self._calculate_mfi(
                df, period=self.feature_params['mfi_period']
            )

            features['price_cycle'] = self._detect_price_cycle(
                df, window=self.feature_params['cycle_window']
            )

            features['seasonal_pattern'] = self._calculate_seasonal_pattern(
                df, period=self.feature_params['seasonal_period']
            )

            for window in self.feature_params['historical_windows'][0]:
                features[f'historical_vol_{window}'] = self._calculate_historical_volatility(
                    df, window=window
                )

            features['sideways_days'] = self._calculate_sideways_days(df)

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

    def _calculate_trend_strength(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算趋势强度"""
        # 使用ADX和价格趋势计算综合趋势强度
        pdi, mdi, adx = CalIndicators.dmi(df, window=window)
        price_trend = df['close'].pct_change().rolling(window).mean()

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

    def _calculate_sideways_days(self, df: pd.DataFrame) -> pd.Series:
        """计算横盘天数"""
        threshold = self.feature_params['sideways_threshold']
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

    def _calculate_volatility_days(self, df: pd.DataFrame) -> pd.Series:
        """计算高波动持续天数"""
        window = self.feature_params['volatility_window']
        threshold = self.feature_params['volatility_threshold']
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
        """计算价格动量"""
        windows = self.feature_params['momentum_windows']
        weights = self.feature_params['momentum_weights']
        momentum = pd.Series(0, index=df.index)

        for window, weight in zip(windows, weights):
            returns = df['close'].pct_change(window)
            momentum += returns * weight

        return momentum

    def _calculate_historical_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算历史波动率特征"""
        returns = df['close'].pct_change()
        vol = returns.rolling(window).std() * np.sqrt(252)
        return vol

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

    def _calculate_seasonal_pattern(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
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

    def _detect_price_cycle(self, df: pd.DataFrame, window: int = 120) -> pd.Series:
        """检测价格周期"""
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

    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(exist_ok=True)

    def _clean_cache(self):
        """清理缓存文件"""
        try:
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            self.cache_dir.rmdir()
        except Exception as e:
            logger.warning(f"清理缓存文件失败: {e}")

    def analyze_best_params(self, best_params: dict, all_results: list) -> dict:
        """分析最优参数组合并生成报告"""
        try:
            # 创建分析报告
            analysis = {
                'best_params': best_params,
                'param_importance': {},
                'param_stability': {},
                'recommendations': {}
            }

            # 计算每个参数的重要性（基于得分变化）
            for param_name in best_params.keys():
                param_scores = []
                for result in all_results:
                    if param_name in result['params']:
                        param_scores.append({
                            'value': result['params'][param_name],
                            'score': result['score']
                        })
                
                if param_scores:
                    # 计算参数重要性（得分方差）
                    score_variance = np.var([s['score'] for s in param_scores])
                    analysis['param_importance'][param_name] = score_variance
                    
                    # 计算参数稳定性
                    top_scores = sorted(param_scores, key=lambda x: x['score'], reverse=True)[:5]
                    value_stability = len(set(str(s['value']) for s in top_scores))
                    analysis['param_stability'][param_name] = {
                        'stability_score': 1.0 / value_stability,  # 值越少越稳定
                        'top_values': [s['value'] for s in top_scores]
                    }
                    
                    # 生成建议
                    if score_variance < 0.001:  # 参数影响很小
                        analysis['recommendations'][param_name] = {
                            'importance': 'LOW',
                            'suggestion': '可以使用默认值，影响较小'
                        }
                    elif analysis['param_stability'][param_name]['stability_score'] > 0.5:
                        analysis['recommendations'][param_name] = {
                            'importance': 'HIGH',
                            'suggestion': '建议固定使用最优值'
                        }
                    else:
                        analysis['recommendations'][param_name] = {
                            'importance': 'MEDIUM',
                            'suggestion': '建议在每次训练时重新优化'
                        }

            # 生成总结报告
            logger.info("\n=== 参数优化分析报告 ===")
            logger.info("\n1. 最优参数组合:")
            for param, value in best_params.items():
                logger.info(f"   {param}: {value}")
            
            logger.info("\n2. 参数重要性排名:")
            sorted_importance = sorted(
                analysis['param_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for param, importance in sorted_importance:
                logger.info(f"   {param}: {importance:.6f}")
            
            logger.info("\n3. 参数优化建议:")
            for param, rec in analysis['recommendations'].items():
                logger.info(f"   {param}:")
                logger.info(f"      重要性: {rec['importance']}")
                logger.info(f"      建议: {rec['suggestion']}")

            # 保存分析结果
            analysis_file = self.cache_dir / "param_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

            return analysis

        except Exception as e:
            logger.exception(f"参数分析失败: {e}")
            return None

    def optimize_feature_params(self, df: pd.DataFrame, test_params: dict) -> dict:
        """优化特征工程参数"""
        self._ensure_cache_dir()

        try:
            # 检查是否存在中断的训练
            checkpoint_file = self.cache_dir / "checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    best_params = checkpoint['best_params']
                    best_score = checkpoint['best_score']
                    completed_groups = set(checkpoint.get('completed_groups', []))
                    current_group = checkpoint.get('current_group', '')
                    current_batch = checkpoint.get('current_batch', 0)
                    logger.info(f"发现中断的训练，从 {current_group} 组的第 {current_batch} 批次继续")
            else:
                best_params = {}
                best_score = 0
                completed_groups = set()
                current_group = ''
                current_batch = 0

            # 将参数分组
            param_groups = {
                '基础指标': {
                    'ma_periods': test_params['ma_periods'],
                    'volatility_window': test_params['volatility_window'],
                    'volatility_threshold': test_params['volatility_threshold']
                },
                '趋势指标': {
                    'sideways_threshold': test_params['sideways_threshold'],
                    'trend_ma_period': test_params['trend_ma_period'],
                    'trend_strength_window': test_params['trend_strength_window']
                },
                '技术指标1': {
                    'kdj_window': test_params['kdj_window'],
                    'kdj_smooth': test_params.get('kdj_smooth', [3]),
                    'rsi_window': test_params['rsi_window']
                },
                '技术指标2': {
                    'macd_fast': test_params['macd_fast'],
                    'macd_slow': test_params['macd_slow'],
                    'macd_signal': test_params['macd_signal']
                },
                '量价指标': {
                    'volume_ma_windows': test_params['volume_ma_windows'],
                    'volume_ratio_threshold': test_params['volume_ratio_threshold'],
                    'mfi_period': test_params['mfi_period']
                },
                '波动指标': {
                    'bb_window': test_params['bb_window'],
                    'bb_std': test_params['bb_std'],
                    'dmi_window': test_params['dmi_window']
                },
                # 添加缺失的参数组
                '周期指标': {
                    'cycle_window': test_params['cycle_window'],
                    'seasonal_period': test_params['seasonal_period']
                },
                '历史统计': {
                    'historical_windows': test_params['historical_windows'],
                    'trend_strength_window': test_params['trend_strength_window']
                }
            }

            # 逐组优化参数
            for group_name, group_params in param_groups.items():
                # 跳过已完成的组
                if group_name in completed_groups:
                    logger.info(f"跳过已完成的参数组: {group_name}")
                    continue

                logger.info(f"\n开始优化 {group_name} 参数组...")

                # 生成当前组的参数组合
                current_combinations = self._generate_param_combinations(group_params)

                # 分批处理参数组合
                batch_size = 20  # 每批处理的参数组合数量
                start_batch = current_batch if group_name == current_group else 0

                for batch_idx in range(start_batch, len(current_combinations), batch_size):
                    # 更新检查点
                    checkpoint = {
                        'best_params': best_params,
                        'best_score': best_score,
                        'completed_groups': list(completed_groups),
                        'current_group': group_name,
                        'current_batch': batch_idx
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f)

                    batch_combinations = current_combinations[batch_idx:batch_idx + batch_size]
                    batch_results = []

                    # 处理当前批次的参数组合
                    for params in batch_combinations:
                        try:
                            # 合并当前组参数和之前的最优参数
                            test_params = {**best_params, **params}
                            self.feature_params = test_params

                            # 生成特征并评估
                            features, labels = self.collect_training_data(df)
                            if len(features) == 0 or len(labels) == 0:
                                continue

                            score = self._evaluate_features(features, labels)

                            # 保存结果
                            batch_results.append({
                                'params': test_params,
                                'score': score
                            })

                            if score > best_score:
                                best_score = score
                                best_params = test_params.copy()
                                logger.info(f"找到更好的参数组合，得分: {score:.4f}")
                                logger.info(f"参数: {params}")

                        except Exception as e:
                            logger.warning(f"参数组合 {params} 评估失败: {e}")
                            continue

                    # 保存当前批次结果到临时文件
                    result_file = self.cache_dir / f"{group_name}_batch_{batch_idx}.json"
                    with open(result_file, 'w') as f:
                        json.dump(batch_results, f)

                    # 清理内存
                    del batch_results
                    gc.collect()

                # 处理完当前组后，整合该组的所有结果
                group_results = []
                for result_file in self.cache_dir.glob(f"{group_name}_batch_*.json"):
                    with open(result_file, 'r') as f:
                        group_results.extend(json.load(f))
                    result_file.unlink()  # 删除临时文件

                # 更新最优参数
                if group_results:
                    best_group_result = max(group_results, key=lambda x: x['score'])
                    if best_group_result['score'] > best_score:
                        best_score = best_group_result['score']
                        best_params = best_group_result['params']

                # 标记当前组为已完成
                completed_groups.add(group_name)
                current_batch = 0  # 重置批次计数

                # 更新检查点
                checkpoint = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'completed_groups': list(completed_groups),
                    'current_group': '',
                    'current_batch': 0
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)

            # 训练完成后删除检查点文件
            checkpoint_file.unlink()

            # 清理临时文件夹
            try:
                self.cache_dir.rmdir()
            except:
                logger.warning("无法删除临时文件夹，可能还有文件存在")

            # 在完成所有优化后进行分析
            analysis = self.analyze_best_params(best_params, group_results)
            
            # 根据分析结果，将参数分为三类
            fixed_params = {}  # 可以固定的参数
            optional_params = {}  # 可选优化的参数
            required_params = {}  # 必须优化的参数
            
            if analysis:
                for param, rec in analysis['recommendations'].items():
                    if rec['importance'] == 'LOW':
                        fixed_params[param] = best_params[param]
                    elif rec['importance'] == 'HIGH':
                        required_params[param] = best_params[param]
                    else:
                        optional_params[param] = best_params[param]
                
                # 保存参数分类结果
                param_categories = {
                    'fixed_params': fixed_params,
                    'optional_params': optional_params,
                    'required_params': required_params
                }
                
                with open(self.cache_dir / "param_categories.json", 'w', encoding='utf-8') as f:
                    json.dump(param_categories, f, ensure_ascii=False, indent=2)
            
            return best_params

        except Exception as e:
            # 保存当前状态到检查点文件
            checkpoint = {
                'best_params': best_params,
                'best_score': best_score,
                'completed_groups': list(completed_groups),
                'current_group': current_group,
                'current_batch': current_batch
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

            logger.exception(f"优化过程中断: {e}")
            raise
        finally:
            # 确保在结束时尝试清理缓存
            self._clean_cache()

    def _generate_param_combinations(self, test_params: dict) -> list:
        """生成参数组合"""
        from itertools import product

        # 示例参数范围
        default_test_params = {
            # 均线参数
            'ma_periods': [
                [5, 10, 20, 60],
                [3, 7, 15, 30],
                [7, 14, 30, 90]
            ],

            # 波动率参数
            'volatility_window': [10, 20, 30],
            'volatility_threshold': [0.01, 0.02, 0.03],

            # 趋势参数
            'sideways_threshold': [0.02, 0.03, 0.04],
            'trend_ma_period': [10, 20, 30],

            # 动量参数
            'momentum_windows': [
                [5, 10, 20],
                [3, 7, 15],
                [7, 14, 30]
            ],
            'momentum_weights': [
                [0.4, 0.3, 0.3],
                [0.33, 0.33, 0.34],
                [0.5, 0.3, 0.2]
            ],

            # 周期参数
            'cycle_window': [60, 120, 180],

            # KDJ参数
            'kdj_window': [7, 9, 14],
            'kdj_smooth': [2, 3, 4],

            # MACD参数
            'macd_fast': [8, 12, 15],
            'macd_slow': [21, 26, 30],
            'macd_signal': [7, 9, 11],

            # RSI参数
            'rsi_window': [10, 14, 20],

            # 布林带参数
            'bb_window': [15, 20, 25],
            'bb_std': [1.5, 2, 2.5],

            # DMI参数
            'dmi_window': [10, 14, 20],
            'dmi_smooth': [10, 14, 20],

            # 成交量参数
            'volume_ma_windows': [
                [3, 7],
                [5, 10],
                [7, 14]
            ],
            'volume_ratio_threshold': [1.3, 1.5, 2.0],

            # 历史统计参数
            'historical_windows': [
                [5, 10, 20],
                [7, 14, 30],
                [10, 20, 60]
            ],
            'percentile_window': [200, 250, 300],

            # 趋势强度参数
            'trend_strength_window': [15, 20, 25],

            # 资金流向参数
            'mfi_period': [10, 14, 20],

            # 季节性参数
            'seasonal_period': [15, 20, 25]
        }

        # 使用传入的参数范围或默认范围
        test_params = test_params or default_test_params

        # 生成所有可能的参数组合
        keys = test_params.keys()
        values = test_params.values()
        combinations = list(product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]

    def _evaluate_features(self, features: pd.DataFrame, labels: pd.Series) -> float:
        """评估特征质量"""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier  # 使用随机森林进行评估

            # 使用较小的样本进行评估,可以根据机器配置调整 max_samples
            max_samples = 20000  # 增加样本量以提高评估质量
            if len(features) > max_samples:
                from sklearn.model_selection import train_test_split
                features, _, labels, _ = train_test_split(
                    features, labels,
                    train_size=max_samples,
                    random_state=42
                )

            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            # 使用随机森林进行评估
            clf = RandomForestClassifier(
                n_estimators=100,  # 增加树的数量
                max_depth=5,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )

            # 使用5折交叉验证
            scores = cross_val_score(
                clf, X_scaled, labels,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )

            return scores.mean()

        except Exception as e:
            logger.warning(f"特征评估失败: {e}")
            return 0.0

    def _save_checkpoint(self, checkpoint_data: dict):
        """保存检查点"""
        checkpoint_file = self.cache_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

    def _load_checkpoint(self) -> dict:
        """加载检查点"""
        checkpoint_file = self.cache_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return None

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """验证和清理特征数据"""
        try:
            # 1. 替换无穷值
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # 2. 处理过大的值
            for col in features.columns:
                # 计算每列的分位数
                q1 = features[col].quantile(0.25)
                q3 = features[col].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 3 * iqr
                lower_bound = q1 - 3 * iqr
                
                # 将超出范围的值替换为边界值
                features.loc[features[col] > upper_bound, col] = upper_bound
                features.loc[features[col] < lower_bound, col] = lower_bound
            
            # 3. 检查是否存在全为NaN的列
            null_columns = features.columns[features.isnull().all()].tolist()
            if null_columns:
                logger.warning(f"删除全为空值的特征列: {null_columns}")
                features = features.drop(columns=null_columns)
            
            # 4. 填充剩余的NaN值
            features = features.fillna(features.mean())
            
            # 5. 检查是否存在常量列
            constant_columns = features.columns[features.nunique() == 1].tolist()
            if constant_columns:
                logger.warning(f"删除常量特征列: {constant_columns}")
                features = features.drop(columns=constant_columns)
            
            # 6. 最终检查确保没有无穷值和NaN
            if not np.isfinite(features.values).all():
                problematic_cols = features.columns[~np.isfinite(features).all()].tolist()
                logger.warning(f"以下特征列仍包含无效值: {problematic_cols}")
                # 最后的安全检查：将所有剩余的无效值替换为0
                features = features.replace([np.inf, -np.inf], 0)
                features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.exception(f"特征验证清理失败: {e}")
            # 如果清理过程出错，返回一个空的DataFrame
            return pd.DataFrame()

    def _log_optimization_progress(self, group_name: str, batch_idx: int, total_batches: int):
        """记录优化进度"""
        progress = (batch_idx + 1) / total_batches * 100
        logger.info(f"参数组 {group_name} 优化进度: {progress:.2f}%")
