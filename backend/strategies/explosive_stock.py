# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description: 爆发式选股策略 - 寻找20个交易日内可能暴涨30%的股票
"""

import logging

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class ExplosiveStockStrategy(BaseStrategy):
    """
    爆发式选股策略
    寻找短期爆发潜力的股票

    建议买入条件：
        1. 综合信号分数 > 70
        2. 成交量比率 > 1.5
        3. RSI在45-65之间
        4. 价格位于"低于均线"或"接近均线"位置
        5. 暴涨概率 > 50%
        6. 风险等级可接受
    """

    def __init__(self):
        super().__init__(name="爆发式选股策略", description="寻找20个交易日内可能暴涨30%的股票")
        self._params = {
            "volume_ma": 20,  # 成交量均线周期
            "rsi_period": 14,  # RSI周期
            "bb_period": 20,  # 布林带周期
            "bb_std": 2,  # 布林带标准差倍数
            "recent_days": 5,  # 近期趋势分析天数
            "volume_weight": 0.35,  # 成交量分析权重
            "momentum_weight": 0.30,  # 动量分析权重
            "pattern_weight": 0.20,  # 形态分析权重
            "volatility_weight": 0.15  # 波动性分析权重
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        try:
            if not self.validate_data(data):
                return pd.Series({'signal': 0})

            # 数据预处理
            df = data.copy()
            for column in ['open', 'high', 'low', 'close', 'volume']:
                df[column] = df[column].astype(float)

            # 股票筛选条件
            if self._should_filter_stock(df):
                return pd.Series({'signal': 0})

            # 1. 计算技术指标
            df = self._calculate_indicators(df)

            # 2. 计算各维度得分
            scores = self._calculate_dimension_scores(df)

            # 3. 计算综合得分
            final_score = self._calculate_final_score(scores)

            # 4. 生成详细的信号信息
            return self._generate_detailed_signal(df, scores, final_score)

        except Exception as e:
            logging.exception("生成信号时发生错误")
            return pd.Series({'signal': 0})

    def _should_filter_stock(self, df: pd.DataFrame) -> bool:
        """
        检查是否应该过滤掉该股票
        
        Args:
            df: 股票数据
            
        Returns:
            bool: True 表示应该过滤，False 表示保留
        """
        try:
            # 成交量过小
            if df['volume'].mean() < 10000:
                logging.info("成交量过小，被过滤")
                return True

            # 股价不在合理范围
            if df['close'].iloc[-1] < 2 or df['close'].iloc[-1] > 200:
                logging.info("股价不在合理范围，被过滤")
                return True

            return False
        except Exception as e:
            logging.exception("检查股票过滤条件时发生错误")
            return True

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算变化率
        df['pct_chg'] = df['close'].pct_change()
        df['volume_pct'] = df['volume'].pct_change()

        # 计算均线
        df['ma20'] = CalIndicators.ema(df, self._params['bb_period'], 'close')
        df['volume_ma20'] = CalIndicators.ema(df, self._params['volume_ma'], 'volume')

        # 计算RSI
        df['rsi'] = CalIndicators.rsi(df, self._params['rsi_period'])

        # 计算MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = CalIndicators.macd(df)

        # 计算布林带
        df['mid_band'], df['upper_band'], df['lower_band'] = CalIndicators.bollinger_bands(
            df,
            ma_period=self._params['bb_period'],
            bollinger_k=self._params['bb_std']
        )

        return df

    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """
        分析成交量异动
        
        通过分析成交量的变化和量价关系来评估成交量异动情况。
        
        Args:
            df: 包含成交量数据的DataFrame
            
        Returns:
            float: 0-1之间的成交量异动得分
        """
        try:
            # 确保数据不含 NaN
            recent_volume = df['volume'].iloc[-self._params['recent_days']:]
            recent_volume_ma = df['volume_ma20'].iloc[-self._params['recent_days']:]

            # 处理分母为0的情况
            mask = recent_volume_ma != 0
            if not mask.any():
                logging.warning("所有成交量均线为0，无法计算成交量比率")
                return 0.0

            # 计算最近5天的成交量相对于20日均量的比值
            recent_volume_ratio = (recent_volume[mask] / recent_volume_ma[mask]).mean()

            # 计算量价配合度
            # 使用dropna()去除缺失值后再计算相关系数
            price_changes = df['pct_chg'].iloc[-self._params['recent_days']:].dropna()
            volume_changes = df['volume_pct'].iloc[-self._params['recent_days']:].dropna()

            # 确保两个序列长度相同且不为空
            if len(price_changes) > 0 and len(volume_changes) > 0:
                # 取两个序列的交集
                common_index = price_changes.index.intersection(volume_changes.index)
                if len(common_index) > 0:
                    price_changes = price_changes[common_index]
                    volume_changes = volume_changes[common_index]
                    price_volume_coord = np.corrcoef(price_changes, volume_changes)[0, 1]
                else:
                    price_volume_coord = 0
            else:
                price_volume_coord = 0

            # 处理 nan 和 inf 值
            if np.isnan(price_volume_coord) or np.isinf(price_volume_coord):
                logging.warning("量价相关系数计算结果无效")
                price_volume_coord = 0

            if np.isnan(recent_volume_ratio) or np.isinf(recent_volume_ratio):
                logging.warning("成交量比率计算结果无效")
                recent_volume_ratio = 1

            # 归一化处理
            volume_score = (
                    min(recent_volume_ratio / 3, 1) * 0.6 +
                    (price_volume_coord + 1) / 2 * 0.4
            )

            logging.debug(f"成交量分析得分: {volume_score:.2f}, "
                          f"成交量比率: {recent_volume_ratio:.2f}, "
                          f"量价相关: {price_volume_coord:.2f}")

            return float(min(max(volume_score, 0), 1))  # 确保返回值在 0-1 之间

        except Exception as e:
            logging.exception(f"分析成交量异动时发生错误: {e}")
            return 0.0

    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """分析动量指标"""
        try:
            # 1. RSI指标评分
            rsi = df['rsi'].iloc[-1]
            if np.isnan(rsi) or np.isinf(rsi):
                rsi = 50
            rsi_score = 1 - abs(rsi - 55) / 45  # RSI在55附近最优

            # 2. MACD金叉判断
            macd_cross = (df['macd_hist'].iloc[-1] > 0 and df['macd_hist'].iloc[-2] < 0)

            # 3. 历史暴涨概率分析
            hist_explosive = self._analyze_historical_explosive(df)

            # 4. 近期动能
            recent_returns = df['pct_chg'].iloc[-self._params['recent_days']:].mean()
            if np.isnan(recent_returns) or np.isinf(recent_returns):
                recent_returns = 0

            momentum_score = (
                    max(min(rsi_score, 1), 0) * 0.3 +  # RSI权重降低
                    (1 if macd_cross else 0) * 0.2 +  # MACD金叉权重降低
                    hist_explosive * 0.3 +  # 历史暴涨概率分析
                    min(max(recent_returns * 20, 0), 1) * 0.2
            )

            return float(min(max(momentum_score, 0), 1))
        except Exception as e:
            logging.exception(f"分析动量指标时发生错误: {e}")
            return 0.0

    def _analyze_historical_explosive(self, df: pd.DataFrame) -> float:
        """分析历史暴涨概率"""
        try:
            # 计算历史上20个交易日的滚动收益率
            returns_20d = []
            for i in range(len(df) - 20):
                start_price = df['close'].iloc[i]
                end_price = df['close'].iloc[i:i + 20].max()  # 20日内最高价
                returns_20d.append((end_price - start_price) / start_price)

            if not returns_20d:
                return 0.0

            # 计算历史上出现30%以上涨幅的频率
            explosive_freq = sum(1 for r in returns_20d if r >= 0.3) / len(returns_20d)

            # 分析当前是否具备暴涨条件
            current_conditions = self._check_explosive_conditions(df)

            return min(explosive_freq + current_conditions, 1.0)
        except Exception as e:
            logging.exception(f"分析历史暴涨概率时发生错误: {e}")
            return 0.0

    @staticmethod
    def _check_explosive_conditions(df: pd.DataFrame) -> float:
        """检查当前是否具备暴涨条件"""
        score = 0.0
        try:
            # 1. 成交量突破
            recent_vol = df['volume'].iloc[-5:].mean()
            vol_ma = df['volume_ma20'].iloc[-1]
            if recent_vol > vol_ma * 1.5:
                score += 0.2

            # 2. 股价处于相对低位
            if df['close'].iloc[-1] < df['ma20'].iloc[-1]:
                score += 0.15

            # 3. 布林带收敛（可能即将突破）
            bb_width = (df['upper_band'] - df['lower_band']) / df['ma20']
            if bb_width.iloc[-1] < bb_width.iloc[-20:].mean() * 0.8:
                score += 0.15

            # 4. MACD底部背离
            if (df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2] and
                    df['close'].iloc[-1] <= df['close'].iloc[-2]):
                score += 0.2

            # 5. RSI超卖反弹
            if 30 <= df['rsi'].iloc[-1] <= 45:
                score += 0.15

            # 6. 检查行业板块整体趋势（需要额外数据支持）
            # TODO: 添加行业板块分析

            return score
        except Exception as e:
            logging.exception(f"检查当前是否具备暴涨条件时发生错误: {e}")
            return 0.0

    def _analyze_pattern(self, df: pd.DataFrame) -> float:
        """分析价格形态"""
        try:
            # 突破布林带上轨
            resistance_break = (df['close'].iloc[-1] > df['upper_band'].iloc[-1])

            # 底部企稳（价格站上20日均线）
            low_stable = (df['close'].iloc[-1] > df['ma20'].iloc[-1] and
                          df['close'].iloc[-self._params['recent_days']:].min() >
                          df['ma20'].iloc[-self._params['recent_days']:].min())

            pattern_score = (
                    (1 if resistance_break else 0) * 0.5 +
                    (1 if low_stable else 0) * 0.5
            )

            return float(min(max(pattern_score, 0), 1))
        except Exception as e:
            logging.exception(f"分析价格形态时发生错误: {e}")
            return 0.0

    @staticmethod
    def _analyze_volatility(df: pd.DataFrame) -> float:
        """分析波动性"""
        try:
            # 布林带宽度变化
            bb_width = (df['upper_band'] - df['lower_band']) / df['ma20']
            bb_width_ratio = bb_width.iloc[-1] / bb_width.iloc[-10:].mean()

            if np.isnan(bb_width_ratio) or np.isinf(bb_width_ratio):
                bb_width_ratio = 1

            # 价格回撤程度
            drawdown = (df['close'].iloc[-1] - df['close'].iloc[-20:].max()) / df['close'].iloc[-20:].max()
            if np.isnan(drawdown) or np.isinf(drawdown):
                drawdown = 0

            volatility_score = (
                    min(bb_width_ratio, 1) * 0.5 +
                    min(max(1 + drawdown * 2, 0), 1) * 0.5
            )

            return float(min(max(volatility_score, 0), 1))
        except Exception as e:
            logging.exception(f"分析波动性时发生错误: {e}")
            return 0.0

    def _predict_with_ml(self, df: pd.DataFrame) -> float:
        """使用机器学习模型预测暴涨概率"""
        try:
            # 特征工程
            features = self._extract_features(df)

            # TODO: 加载预训练模型并预测
            # 这里需要实际训练好的模型
            # prediction = model.predict_proba(features)[0][1]

            # 临时返回基于规则的预测分数
            return self._rule_based_prediction(df)
        except Exception as e:
            logging.exception(f"使用机器学习模型预测暴涨概率时发生错误: {e}")
            return 0.0

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取用于机器学习的特征"""
        features = pd.DataFrame()

        # 价格特征
        features['price_ma_ratio'] = df['close'] / df['ma20']
        features['price_bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

        # 成交量特征
        features['volume_ma_ratio'] = df['volume'] / df['volume_ma20']

        # 动量特征
        features['rsi'] = df['rsi']
        features['macd_hist'] = df['macd_hist']

        return features.iloc[-1:]  # 只返回最新一行

    def _get_trend_strength(self, df: pd.DataFrame) -> str:
        """获取趋势强度描述"""
        recent_trend = df['close'].iloc[-5:].pct_change().mean()
        if recent_trend > 0.02:
            return "强势上涨"
        elif recent_trend > 0.005:
            return "温和上涨"
        elif recent_trend < -0.02:
            return "强势下跌"
        elif recent_trend < -0.005:
            return "温和下跌"
        return "横盘整理"

    def _get_price_position(self, df: pd.DataFrame) -> str:
        """获取价格位置描述"""
        last_close = df['close'].iloc[-1]
        ma20 = df['ma20'].iloc[-1]

        if last_close > ma20 * 1.1:
            return "显著高于均线"
        elif last_close > ma20:
            return "高于均线"
        elif last_close < ma20 * 0.9:
            return "显著低于均线"
        elif last_close < ma20:
            return "低于均线"
        return "接近均线"

    def _get_bb_position(self, df: pd.DataFrame) -> str:
        """获取布林带位置描述"""
        last_close = df['close'].iloc[-1]
        upper = df['upper_band'].iloc[-1]
        lower = df['lower_band'].iloc[-1]
        mid = df['ma20'].iloc[-1]

        if last_close > upper:
            return "突破上轨"
        elif last_close > mid:
            return "上轨区间"
        elif last_close < lower:
            return "突破下轨"
        elif last_close < mid:
            return "下轨区间"
        return "中轨区间"

    def _generate_buy_signal(self, score: float) -> str:
        """生成买入建议"""
        if score > 0.8:
            return "强烈建议买入"
        elif score > 0.6:
            return "建议买入"
        elif score > 0.4:
            return "观察"
        return "暂不建议"

    def _assess_risk(self, df: pd.DataFrame) -> str:
        """评估风险等级"""
        volatility = df['close'].pct_change().std() * np.sqrt(252)  # 年化波动率

        if volatility > 0.5:
            return "高风险"
        elif volatility > 0.3:
            return "中等风险"
        return "低风险"

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据是否满足基本条件"""
        try:
            if len(data) < 60:  # 至少需要60个交易日的数据
                logging.warning(f"股票{data['code'].iloc[-1]}数据不足60天，跳过该股票")
                return False

            # 检查必要的列是否存在
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_date']
            if not all(col in data.columns for col in required_columns):
                logging.warning(f"股票{data['code'].iloc[-1]}缺少必要的列，跳过该股票")
                return False

            return True
        except Exception as e:
            logging.exception(f"验证数据时发生错误: {e}")
            return False

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """计算价格波动率"""
        try:
            # 计算20日滚动波动率
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 年化波动率
            current_volatility = volatility.iloc[-1]

            if np.isnan(current_volatility) or np.isinf(current_volatility):
                return 0.0

            # 归一化处理，假设50%年化波动率为中等水平
            normalized_volatility = min(current_volatility / 0.5, 1.0)
            return float(normalized_volatility)
        except Exception as e:
            logging.exception(f"计算价格波动率时发生错误: {e}")
            return 0.0

    def _get_explosion_probability(self, df: pd.DataFrame) -> float:
        """计算暴涨概率"""
        try:
            # 1. 历史暴涨概率
            hist_prob = self._analyze_historical_explosive(df)

            # 2. 技术指标组合评分
            tech_score = self._calculate_technical_score(df)

            # 3. 成交量异动评分
            volume_score = self._analyze_volume(df)

            # 综合评分
            final_prob = (
                    hist_prob * 0.4 +  # 历史暴涨概率权重
                    tech_score * 0.35 +  # 技术指标权重
                    volume_score * 0.25  # 成交量权重
            )

            return float(min(max(final_prob, 0), 1))
        except Exception as e:
            logging.exception(f"计算暴涨概率时发生错误: {e}")
            return 0.0

    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """计算技术指标综合评分"""
        try:
            score = 0.0

            # 1. RSI指标评分
            rsi = df['rsi'].iloc[-1]
            if 40 <= rsi <= 60:
                score += 0.3
            elif 30 <= rsi <= 70:
                score += 0.2

            # 2. MACD指标评分
            if df['macd_hist'].iloc[-1] > 0 and df['macd_hist'].iloc[-2] < 0:  # 金叉
                score += 0.3
            elif df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2]:  # 柱状图向上
                score += 0.2

            # 3. 布林带评分
            bb_position = (df['close'].iloc[-1] - df['lower_band'].iloc[-1]) / \
                          (df['upper_band'].iloc[-1] - df['lower_band'].iloc[-1])
            if 0.3 <= bb_position <= 0.7:  # 布林带中轨附近
                score += 0.2
            elif bb_position < 0.3:  # 接近下轨
                score += 0.3

            # 4. 均线系统评分
            if df['close'].iloc[-1] > df['ma20'].iloc[-1] and \
                    df['ma20'].iloc[-1] > df['ma20'].iloc[-5]:  # 均线向上
                score += 0.2

            return float(min(score, 1.0))
        except Exception as e:
            logging.exception(f"计算技术指标综合评分时发生错误: {e}")
            return 0.0

    def _rule_based_prediction(self, df: pd.DataFrame) -> float:
        """基于规则的预测分数（在没有ML模型时使用）"""
        try:
            score = 0.0

            # 1. 价格趋势
            recent_trend = df['close'].iloc[-5:].pct_change().mean()
            if 0.005 <= recent_trend <= 0.02:  # 温和上涨
                score += 0.3
            elif recent_trend > 0.02:  # 强势上涨
                score += 0.2  # 反而降低分数，因为可能已经涨过了

            # 2. 成交量趋势
            volume_ratio = df['volume'].iloc[-1] / df['volume_ma20'].iloc[-1]
            if 1.5 <= volume_ratio <= 3:
                score += 0.2
            elif volume_ratio > 3:
                score += 0.1  # 过度放量可能风险较大

            # 3. 技术指标组合
            tech_score = self._calculate_technical_score(df)
            score += tech_score * 0.3

            # 4. 历史暴涨概率
            hist_prob = self._analyze_historical_explosive(df)
            score += hist_prob * 0.2

            return float(min(max(score, 0), 1))
        except Exception as e:
            logging.exception(f"基于规则的预测分数时发生错误: {e}")
            return 0.0

    def _calculate_dimension_scores(self, df: pd.DataFrame) -> dict:
        """
        计算各维度得分
        
        Args:
            df: 包含技术指标的数据
            
        Returns:
            dict: 包含各维度得分的字典
        """
        try:
            return {
                'volume': self._analyze_volume(df),
                'momentum': self._analyze_momentum(df),
                'pattern': self._analyze_pattern(df),
                'volatility': self._analyze_volatility(df),
                'ml': self._predict_with_ml(df)
            }
        except Exception as e:
            logging.exception(f"计算维度得分时发生错误: {e}")
            return {k: 0.0 for k in ['volume', 'momentum', 'pattern', 'volatility', 'ml']}

    def _calculate_final_score(self, scores: dict) -> float:
        """
        计算综合得分
        
        Args:
            scores: 各维度得分字典
            
        Returns:
            float: 0-1之间的综合得分
        """
        try:
            final_score = (
                    scores['volume'] * self._params['volume_weight'] +
                    scores['momentum'] * self._params['momentum_weight'] +
                    scores['pattern'] * self._params['pattern_weight'] +
                    scores['volatility'] * self._params['volatility_weight'] +
                    scores['ml'] * 0.2  # 机器学习预测权重
            )
            return float(min(max(final_score, 0), 1))
        except Exception as e:
            logging.exception(f"计算综合得分时发生错误: {e}")
            return 0.0

    def _generate_detailed_signal(self, df: pd.DataFrame,
                                  scores: dict, final_score: float) -> pd.Series:
        """
        生成详细的信号信息
        
        Args:
            df: 股票数据
            scores: 各维度得分
            final_score: 综合得分
            
        Returns:
            pd.Series: 包含所有分析结果的Series
        """
        try:
            return pd.Series({
                'signal': round(final_score * 100, 2),
                'trade_date': df['trade_date'].iloc[-1],
                'price': round(float(df['close'].iloc[-1]), 2),

                # 成交量分析
                'volume_ratio': round(float(df['volume'].iloc[-1] / df['volume_ma20'].iloc[-1]), 2),
                'volume_score': round(scores['volume'] * 100, 2),
                'volume_trend': '放量' if df['volume'].iloc[-1] > df['volume_ma20'].iloc[-1] * 1.5 else '正常',

                # 动量分析
                'momentum_score': round(scores['momentum'] * 100, 2),
                'rsi': round(float(df['rsi'].iloc[-1]), 2),
                'macd_hist': round(float(df['macd_hist'].iloc[-1]), 2),
                'trend_strength': self._get_trend_strength(df),

                # 形态分析
                'pattern_score': round(scores['pattern'] * 100, 2),
                'price_position': self._get_price_position(df),
                'bb_position': self._get_bb_position(df),

                # 波动性分析
                'volatility_score': round(scores['volatility'] * 100, 2),
                'price_volatility': round(self._calculate_volatility(df), 2),

                # 机器学习预测
                'ml_prediction': round(scores['ml'] * 100, 2),
                'explosion_probability': round(self._get_explosion_probability(df) * 100, 2),

                # 买入建议
                'buy_signal': self._generate_buy_signal(final_score),
                'risk_level': self._assess_risk(df)
            })
        except Exception as e:
            logging.exception(f"生成详细信号时发生错误: {e}")
            return pd.Series({'signal': 0})
