# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description: 爆发式选股策略 - 寻找20个交易日内可能暴涨30%的股票
"""

import logging

import numpy as np
import pandas as pd

from backend.ml.data_collector import ExplosiveStockDataCollector
from backend.ml.model_trainer import ExplosiveStockModelTrainer
from backend.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators

# MODEL_BASE_PATH = "backend/ml/models/explosive_stock_model"
MODEL_BASE_PATH = "模型没训练好先不用"


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
        self._init_params()

    def _init_params(self):
        """初始化策略参数"""
        self._params = {
            "volume_ma": 20,  # 成交量均线周期
            "rsi_period": 14,  # RSI周期
            "bb_period": 20,  # 布林带周期
            "bb_std": 2,  # 布林带标准差倍数
            "recent_days": 5,  # 近期趋势分析天数
            "signal": 0.0,  # 返回结果signal大于该值
            "rsi_range": (0.0, 100.0),  # 返回结果rsi在此区间
            "volume_ratio": 0.0,  # 返回结果增量比例需大于该值
            "explosion_probability": 0.0,  # 返回结果暴涨概率大于该值
            "volume_weight": 0.35,  # 成交量分析权重
            "momentum_weight": 0.30,  # 动量分析权重
            "pattern_weight": 0.20,  # 形态分析权重
            "volatility_weight": 0.15,  # 波动性分析权重
            "holdings": [],  # 持仓股票代码
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        try:
            if not self.validate_data(data):
                return pd.Series({'signal': 0})

            # 数据预处理
            df = self._preprocess_data(data)

            # 获取当前股票代码
            stock_code = data['code'].iloc[-1] if 'code' in data.columns else None

            # 检查是否是持仓股票
            holdings = self._params.get('holdings', [])
            holding_info = next(
                (item for item in holdings if item['code'] == stock_code),
                None
            )
            is_holding = holding_info is not None

            # 股票筛选条件
            if self._should_filter_stock(df):
                # 如果是持仓股票，即使不满足筛选条件也继续生成信号
                if not is_holding:
                    return pd.Series({'signal': 0})

            # 计算技术指标和得分
            df = self._calculate_indicators(df)
            scores = self._calculate_dimension_scores(df)
            final_score = self._calculate_final_score(scores)

            # 生成详细信号
            signal = self._generate_detailed_signal(df, scores, final_score)

            # 如果是持仓股票，添加持仓建议
            if is_holding:
                entry_price = holding_info['cost']
                current_price = df['close'].iloc[-1]
                returns = (current_price - entry_price) / entry_price * 100

                # 生成持仓建议
                should_sell = any([
                    returns >= 30,  # 达到预期目标
                    returns <= -7,  # 止损线
                    signal['signal'] < 40,  # 信号显著转弱
                    signal['rsi'] > 85,  # RSI严重超买
                    signal['volume_ratio'] < 0.5  # 成交量萎缩
                ])

                should_reduce = (
                        15 <= returns < 30 and  # 收益在目标区间
                        signal['signal'] < 60  # 信号转弱
                )

                signal = pd.concat([signal, pd.Series({
                    'is_holding': True,
                    'entry_price': entry_price,
                    'returns': round(returns, 2),
                    'position_advice': (
                        "建议卖出" if should_sell else
                        "建议减仓" if should_reduce else
                        "继续持有" if signal['signal'] > 50 else
                        "密切关注"
                    ),
                    'advice_reason': self._generate_advice_reason(
                        signal, returns, should_sell, should_reduce
                    )
                })])

            # 对非持仓股票进行信号过滤
            elif (signal['signal'] < self._params['signal'] or  # 信号分数过滤
                  not (self._params['rsi_range'][0] <= signal['rsi'] <= self._params['rsi_range'][1]) or  # RSI范围过滤
                  signal['volume_ratio'] < self._params['volume_ratio'] or  # 成交量比率过滤
                  signal['explosion_probability'] < self._params['explosion_probability']):  # 暴涨概率过滤
                return pd.Series({'signal': 0})

            return signal

        except Exception as e:
            logging.exception(f"生成信号时发生错误: {e}")
            return pd.Series({'signal': 0})

    @staticmethod
    def _preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df = data.copy()
        return df

    @staticmethod
    def _should_filter_stock(df: pd.DataFrame) -> bool:
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
                logging.info(f"股价不在合理范围，被过滤: {df['code'].iloc[-1]}")
                return True

            return False
        except Exception as e:
            logging.exception(f"检查股票过滤条件时发生错误: {e}")
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
            # 1. 计算成交量比率
            recent_volume = df['volume'].iloc[-self._params['recent_days']:]
            recent_volume_ma = df['volume_ma20'].iloc[-self._params['recent_days']:]

            # 处理分母为0的情况
            mask = recent_volume_ma > 0  # 确保分母大于0
            if not mask.any():
                logging.warning("所有成交量均线为0或无效，无法计算成交量比率")
                return 0.0

            recent_volume_ratio = (recent_volume[mask] / recent_volume_ma[mask]).mean()

            # 2. 计算量价相关性
            try:
                # 确保计算百分比变化时没有0值
                price_changes = df['close'].pct_change().iloc[-self._params['recent_days']:]
                volume_changes = df['volume'].pct_change().iloc[-self._params['recent_days']:]

                # 移除无效值
                valid_mask = (~np.isnan(price_changes)) & (~np.isnan(volume_changes)) & \
                             (~np.isinf(price_changes)) & (~np.isinf(volume_changes))

                if valid_mask.sum() >= 3:  # 至少需要3个有效数据点
                    price_changes = price_changes[valid_mask]
                    volume_changes = volume_changes[valid_mask]

                    # 使用 pandas 的 corr 方法计算相关系数
                    price_volume_coord = price_changes.corr(volume_changes)

                    if np.isnan(price_volume_coord) or np.isinf(price_volume_coord):
                        logging.warning("量价相关系数计算结果无效，使用默认值0")
                        price_volume_coord = 0
                else:
                    logging.warning("有效数据点不足，无法计算量价相关性")
                    price_volume_coord = 0

            except Exception as e:
                logging.warning(f"计算量价相关性时出错: {str(e)}")
                price_volume_coord = 0

            # 3. 处理异常值
            if np.isnan(recent_volume_ratio) or np.isinf(recent_volume_ratio):
                logging.warning("成交量比率计算结果无效，使用默认值1")
                recent_volume_ratio = 1

            # 4. 计算最终得分
            volume_score = (
                    min(recent_volume_ratio / 3, 1) * 0.6 +  # 成交量比率得分
                    (price_volume_coord + 1) / 2 * 0.4  # 量价相关性得分
            )

            logging.debug(
                f"成交量分析结果:\n"
                f"- 成交量比率: {recent_volume_ratio:.2f}\n"
                f"- 量价相关性: {price_volume_coord:.2f}\n"
                f"- 最终得分: {volume_score:.2f}"
            )

            return float(min(max(volume_score, 0), 1))

        except Exception as e:
            logging.exception("分析成交量异动时发生错误")
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

    def _is_model_ready(self) -> bool:
        """检查模型是否准备就绪"""
        return (hasattr(self, 'ml_trainer') and
                self.ml_trainer is not None and
                hasattr(self.ml_trainer, 'trained_models') and
                len(self.ml_trainer.trained_models) > 0)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备预测特征"""
        collector = ExplosiveStockDataCollector()
        features = collector._generate_features(df)
        return features.iloc[-1:].copy()

    def _make_prediction(self, features: pd.DataFrame) -> float:
        """使用集成模型进行预测"""
        try:
            # 标准化特征
            features_scaled = self.ml_trainer.scaler.transform(features)

            # 获取每个模型的预测概率
            model_predictions = {}
            for name, model in self.ml_trainer.trained_models.items():
                # 获取上涨类（类别1）的预测概率
                pred = model.predict_proba(features_scaled)[0][1]
                # pred = model.predict_proba(features_scaled)[0][2]
                model_predictions[name] = pred

            # 使用权重计算加权平均
            weighted_sum = 0
            total_weight = sum(self.ml_trainer.weights.values())

            for name, pred in model_predictions.items():
                weight = self.ml_trainer.weights[name]
                weighted_sum += pred * (weight / total_weight)

            # 对最终预测值进行校准
            # 如果任一模型预测概率超过0.7，提升整体预测值
            max_pred = max(model_predictions.values())
            if max_pred > 0.7:
                weighted_sum = (weighted_sum + max_pred) / 2

            return float(min(max(weighted_sum, 0), 1))

        except Exception as e:
            logging.exception(f"模型预测失败: {e}")
            return 0.0

    def _predict_with_ml(self, df: pd.DataFrame) -> float:
        """使用机器学习模型预测暴涨概率"""
        try:
            self._init_ml_model()
            if not self._is_model_ready():
                logging.warning("机器学习模型未就绪，使用规则基预测")
                return self._rule_based_prediction(df)

            features = self._prepare_features(df)
            if features.empty:
                logging.warning("特征生成失败，使用规则基预测")
                return self._rule_based_prediction(df)

            prediction = self._make_prediction(features)
            logging.info(f"机器学习模型预测概率: {prediction:.4f} - 股票：{df.code.iloc[-1]}")
            return prediction

        except Exception as e:
            logging.exception(f"使用机器学习模型预测暴涨概率时发生错误: {e}")
            return self._rule_based_prediction(df)

    @staticmethod
    def _get_trend_strength(df: pd.DataFrame) -> str:
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

    @staticmethod
    def _generate_buy_signal(score: float) -> str:
        """生成买入建议"""
        if score > 0.8:
            return "强烈建议买入"
        elif score > 0.6:
            return "建议买入"
        elif score > 0.4:
            return "观察"
        return "暂不建议"

    @staticmethod
    def _assess_risk(df: pd.DataFrame) -> str:
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
            if df['close'].iloc[-1] > df['ma20'].iloc[-1] > df['ma20'].iloc[-5]:  # 均线向上
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
        """计算各维度得分"""
        return {
            'volume': self._analyze_volume(df),
            'momentum': self._analyze_momentum(df),
            'pattern': self._analyze_pattern(df),
            'volatility': self._analyze_volatility(df),
            'ml': self._predict_with_ml(df)
        }

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

    @staticmethod
    def _generate_advice_reason(signal: pd.Series, returns: float,
                                should_sell: bool, should_reduce: bool) -> str:
        """
        生成持仓建议原因
        
        Args:
            signal: 信号Series，包含各项技术指标
            returns: 当前收益率（百分比）
            should_sell: 是否建议卖出 (已通过其他条件判断)
            should_reduce: 是否建议减仓
            
        Returns:
            str: 建议原因说明
        """
        # 按优先级顺序检查各种情况
        if returns >= 30:
            return "已达预期目标(30%)，建议获利了结"
        elif returns <= -7:
            return "已触及止损线(-7%)，建议止损出局"
        elif signal['rsi'] > 85:
            return f"RSI超买({signal['rsi']:.1f})，建议获利了结"
        elif signal['signal'] < 40:
            return f"技术指标显著转弱(信号:{signal['signal']:.1f})，建议清仓观望"
        elif signal['volume_ratio'] < 0.5:
            return f"成交量明显萎缩(量比:{signal['volume_ratio']:.2f})，建议注意风险"
        elif should_reduce:
            return f"已获利{returns:.1f}%且信号转弱，建议适当减仓"
        elif signal['signal'] > 70:
            return f"信号强度良好({signal['signal']:.1f})，可继续持有"
        else:
            return f"信号一般({signal['signal']:.1f})，需密切关注"

    def _init_ml_model(self):
        """初始化机器学习模型"""
        self.ml_trainer = None
        try:
            # 初始化模型训练器
            self.ml_trainer = ExplosiveStockModelTrainer()

            stock_pool = self._params.get("stock_pool", "full")
            # 修改为正确的模型文件路径
            base_path = MODEL_BASE_PATH
            model_files = {}
            for model_name in self.ml_trainer.models.keys():
                model_files[model_name] = f"{base_path}_{stock_pool}_{model_name}.joblib"
            scaler_path = f'backend/ml/models/explosive_stock_model_{stock_pool}_scaler.joblib'

            # 使用新的加载方法,传入具体的模型文件路径
            self.ml_trainer.load_models(model_files, scaler_path)

            logging.debug("成功加载机器学习模型")

            # 验证模型加载状态
            if not self._is_model_ready():
                raise ValueError("模型加载不完整")

        except Exception as e:
            logging.warning(f"加载机器学习模型失败: {e}")
            self.ml_trainer = None
