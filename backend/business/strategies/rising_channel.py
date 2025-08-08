#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : AI Assistant
@Date       : 2025-01-31
@File       : rising_channel_strategy.py
@Description: 上升通道策略 - 扫描所有符合上升通道条件的股票

该策略基于上升通道回归分析，识别出所有当前处于有效上升通道的股票。
策略逻辑：
1. 计算每只股票的上升通道
2. 筛选出通道状态为NORMAL的股票
3. 根据通道质量、趋势强度等指标进行排序
4. 生成买入信号和风险提示
"""

import logging
from typing import Dict, Any

import pandas as pd

from backend.business.strategies.base import BaseStrategy
from backend.utils.indicators import CalIndicators


class RisingChannelStrategy(BaseStrategy):
    """
    上升通道策略
    
    策略逻辑：
    1. 计算每只股票的上升通道回归分析
    2. 筛选出通道状态为NORMAL的股票
    3. 根据通道质量、趋势强度、风险指标进行综合评分
    4. 生成买入信号和持仓建议
    """

    def __init__(self):
        super().__init__(name="上升通道策略", description="扫描所有符合上升通道条件的股票")
        self._init_params()

    def _init_params(self):
        """初始化策略参数"""
        self._params = {
            # 上升通道参数
            "k": 2.0,  # 通道宽度倍数
            "L_max": 120,  # 最大窗口长度
            "delta_cut": 5,  # 滑动窗口删除天数
            "pivot_m": 3,  # 锚点检测参数
            "gain_trigger": 0.30,  # 重锚涨幅触发阈值
            "beta_delta": 0.15,  # 斜率变化阈值
            "break_days": 3,  # 连续突破天数
            "reanchor_fail_max": 2,  # 重锚失败最大次数
            "min_data_points": 60,  # 最小数据点数
            "R2_min": 0.20,  # 最小回归拟合优度
            "width_pct_min": 0.04,  # 通道宽度下限
            "width_pct_max": 0.20,  # 通道宽度上限 - 调整为更宽松的值

            # 技术指标参数
            "bb_period": 20,  # 布林带周期
            "bb_std": 2,  # 布林带标准差倍数

            # 筛选条件
            "min_signal_score": 60,  # 最小信号分数
            "min_r2": 0.30,  # 最小R²值
            "min_slope_deg": 0.5,  # 最小斜率角度
            "max_volatility": 0.08,  # 最大波动率

            # 评分权重
            "weights": {
                "channel_quality": 0.35,  # 通道质量权重
                "trend_strength": 0.25,  # 趋势强度权重
                "risk_control": 0.20,  # 风险控制权重
                "volume_analysis": 0.20  # 成交量分析权重
            }
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            data: pd.DataFrame, 股票数据
            
        Returns:
            pd.Series: 交易信号
        """
        if not self.validate_data(data):
            return pd.Series({'signal': 0})

        try:
            # 数据预处理
            df = self._preprocess_data(data)

            # 计算上升通道 - 使用用户设置的参数
            channel_info = self._calculate_ascending_channel(df)

            # 如果通道无效，返回无信号
            if channel_info is None or channel_info.get('channel_status') != 'NORMAL':
                return pd.Series({'signal': 0})

            # 计算技术指标
            df = self._calculate_indicators(df)

            # 计算各维度得分
            scores = self._calculate_dimension_scores(df, channel_info)

            # 计算综合得分
            final_score = self._calculate_final_score(scores)

            # 应用筛选条件 - 使用用户设置的参数
            if not self._apply_filters(channel_info, final_score):
                return pd.Series({'signal': 0})

            # 生成详细信号
            signal = self._generate_detailed_signal(df, channel_info, scores, final_score)

            return signal

        except Exception as e:
            logging.exception(f"生成上升通道信号时发生错误: {e}")
            return pd.Series({'signal': 0})

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df = data.copy()

        # 确保数据按时间排序
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)

        # 确保价格数据为float类型
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        return df

    def _calculate_ascending_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算上升通道"""
        try:
            # 确保数据格式正确
            if len(df) < self._params['min_data_points']:
                return None

            # 计算上升通道
            channel_info = CalIndicators.ascending_channel(df, **{
                'k': self._params['k'],
                'L_max': self._params['L_max'],
                'delta_cut': self._params['delta_cut'],
                'pivot_m': self._params['pivot_m'],
                'gain_trigger': self._params['gain_trigger'],
                'beta_delta': self._params['beta_delta'],
                'break_days': self._params['break_days'],
                'reanchor_fail_max': self._params['reanchor_fail_max'],
                'min_data_points': self._params['min_data_points'],
                'R2_min': self._params['R2_min'],
                'width_pct_min': self._params['width_pct_min'],
                'width_pct_max': self._params['width_pct_max']
            })

            return channel_info

        except Exception as e:
            logging.warning(f"计算上升通道失败: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算移动平均线
        df['ma5'] = CalIndicators.sma(df, period=5, cal_value='close')
        df['ma10'] = CalIndicators.sma(df, period=10, cal_value='close')
        df['ma20'] = CalIndicators.sma(df, period=20, cal_value='close')
        df['ma60'] = CalIndicators.sma(df, period=60, cal_value='close')

        # 计算成交量均线
        df['volume_ma5'] = CalIndicators.sma(df, period=5, cal_value='volume')
        df['volume_ma20'] = CalIndicators.sma(df, period=20, cal_value='volume')

        # 计算RSI
        df['rsi'] = CalIndicators.rsi(df, period=14)

        # 计算MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = CalIndicators.macd(
            df, fast_period=12, slow_period=26, signal_period=9
        )

        # 计算布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = CalIndicators.bollinger_bands(
            df, ma_period=self._params['bb_period'], bollinger_k=self._params['bb_std']
        )

        # 计算价格变化率
        df['pct_chg'] = df['close'].pct_change()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']

        return df

    def _calculate_dimension_scores(self, df: pd.DataFrame, channel_info: Dict[str, Any]) -> Dict[str, float]:
        """计算各维度得分"""
        return {
            'channel_quality': self._calculate_channel_quality_score(channel_info),
            'trend_strength': self._calculate_trend_strength_score(channel_info),
            'risk_control': self._calculate_risk_control_score(df, channel_info),
            'volume_analysis': self._calculate_volume_analysis_score(df)
        }

    def _calculate_channel_quality_score(self, channel_info: Dict[str, Any]) -> float:
        """计算通道质量得分"""
        try:
            score = 0.0

            # R²值评分 (0-40分)
            r2 = channel_info.get('r2', 0)
            if r2 > 0.7:
                score += 40
            elif r2 > 0.5:
                score += 30
            elif r2 > 0.3:
                score += 20
            else:
                score += 10

            # 通道宽度评分 (0-30分)
            width_pct = channel_info.get('width_pct', 0)
            if 0.05 <= width_pct <= 0.10:
                score += 30
            elif 0.04 <= width_pct <= 0.12:
                score += 20
            else:
                score += 10

            # 斜率角度评分 (0-30分)
            slope_deg = channel_info.get('slope_deg', 0)
            if 1.0 <= slope_deg <= 3.0:
                score += 30
            elif 0.5 <= slope_deg <= 5.0:
                score += 20
            else:
                score += 10

            return min(score / 100, 1.0)

        except Exception as e:
            logging.warning(f"计算通道质量得分失败: {e}")
            return 0.0

    def _calculate_trend_strength_score(self, channel_info: Dict[str, Any]) -> float:
        """计算趋势强度得分"""
        try:
            score = 0.0

            # 斜率评分 (0-40分)
            beta = channel_info.get('beta', 0)
            if beta > 0.03:
                score += 40
            elif beta > 0.02:
                score += 30
            elif beta > 0.01:
                score += 20
            else:
                score += 10

            # 累计涨幅评分 (0-30分)
            cumulative_gain = channel_info.get('cumulative_gain', 0)
            if 0.1 <= cumulative_gain <= 0.3:
                score += 30
            elif 0.05 <= cumulative_gain <= 0.5:
                score += 20
            else:
                score += 10

            # 窗口大小评分 (0-30分)
            window_size = channel_info.get('window_size', 0)
            if 60 <= window_size <= 120:
                score += 30
            elif 40 <= window_size <= 150:
                score += 20
            else:
                score += 10

            return min(score / 100, 1.0)

        except Exception as e:
            logging.warning(f"计算趋势强度得分失败: {e}")
            return 0.0

    def _calculate_risk_control_score(self, df: pd.DataFrame, channel_info: Dict[str, Any]) -> float:
        """计算风险控制得分"""
        try:
            score = 0.0

            # 波动率评分 (0-40分)
            volatility = channel_info.get('volatility', 0)
            if volatility < 0.03:
                score += 40
            elif volatility < 0.05:
                score += 30
            elif volatility < 0.08:
                score += 20
            else:
                score += 10

            # 价格位置评分 (0-30分)
            current_price = df['close'].iloc[-1]
            mid_today = channel_info.get('mid_today', current_price)
            upper_today = channel_info.get('upper_today', current_price)

            if current_price <= mid_today:
                score += 30  # 价格在中轴下方，风险较低
            elif current_price <= upper_today:
                score += 20  # 价格在通道内
            else:
                score += 10  # 价格突破上沿，风险较高

            # RSI评分 (0-30分)
            rsi = df['rsi'].iloc[-1]
            if 40 <= rsi <= 70:
                score += 30
            elif 30 <= rsi <= 80:
                score += 20
            else:
                score += 10

            return min(score / 100, 1.0)

        except Exception as e:
            logging.warning(f"计算风险控制得分失败: {e}")
            return 0.0

    def _calculate_volume_analysis_score(self, df: pd.DataFrame) -> float:
        """计算成交量分析得分"""
        try:
            score = 0.0

            # 成交量比率评分 (0-40分)
            volume_ratio = df['volume_ratio'].iloc[-1]
            if 1.0 <= volume_ratio <= 2.0:
                score += 40
            elif 0.8 <= volume_ratio <= 3.0:
                score += 30
            else:
                score += 20

            # 成交量趋势评分 (0-30分)
            recent_volume = df['volume'].iloc[-5:].mean()
            volume_ma20 = df['volume_ma20'].iloc[-1]
            if recent_volume > volume_ma20:
                score += 30
            else:
                score += 15

            # 量价配合评分 (0-30分)
            price_change = df['pct_chg'].iloc[-1]
            volume_change = df['volume'].pct_change().iloc[-1]

            if (price_change > 0 and volume_change > 0) or (price_change < 0 and volume_change < 0):
                score += 30  # 量价配合
            else:
                score += 15  # 量价背离

            return min(score / 100, 1.0)

        except Exception as e:
            logging.warning(f"计算成交量分析得分失败: {e}")
            return 0.0

    def _calculate_final_score(self, scores: Dict[str, float]) -> float:
        """计算综合得分"""
        try:
            weights = self._params['weights']
            final_score = sum(scores[key] * weights[key] for key in weights.keys())
            return min(max(final_score, 0), 1)
        except Exception as e:
            logging.warning(f"计算综合得分失败: {e}")
            return 0.0

    def _apply_filters(self, channel_info: Dict[str, Any], final_score: float) -> bool:
        """应用筛选条件"""
        try:
            # 信号分数过滤
            if final_score * 100 < self._params['min_signal_score']:
                return False

            # R²值过滤
            if channel_info.get('r2', 0) < self._params['min_r2']:
                return False

            # 通道宽度过滤
            if channel_info.get('width_pct', 0) > self._params['width_pct_max']:
                return False

            # 斜率角度过滤
            if channel_info.get('slope_deg', 0) < self._params['min_slope_deg']:
                return False

            # 波动率过滤
            if channel_info.get('volatility', 0) > self._params['max_volatility']:
                return False

            return True

        except Exception as e:
            logging.warning(f"应用筛选条件失败: {e}")
            return False

    def _generate_detailed_signal(self, df: pd.DataFrame, channel_info: Dict[str, Any],
                                  scores: Dict[str, float], final_score: float) -> pd.Series:
        """生成详细信号"""
        try:
            current_price = df['close'].iloc[-1]

            # 计算价格位置
            mid_today = channel_info.get('mid_today', current_price)
            upper_today = channel_info.get('upper_today', current_price)
            lower_today = channel_info.get('lower_today', current_price)

            if current_price > upper_today:
                price_position = "突破上沿"
            elif current_price < lower_today:
                price_position = "跌破下沿"
            elif current_price > mid_today:
                price_position = "中轴上方"
            else:
                price_position = "中轴下方"

            # 生成买入建议
            if final_score > 0.8:
                buy_signal = "强烈建议买入"
            elif final_score > 0.6:
                buy_signal = "建议买入"
            elif final_score > 0.4:
                buy_signal = "观察"
            else:
                buy_signal = "暂不建议"

            # 风险评估
            volatility = channel_info.get('volatility', 0)
            if volatility > 0.06:
                risk_level = "高风险"
            elif volatility > 0.04:
                risk_level = "中等风险"
            else:
                risk_level = "低风险"

            return pd.Series({
                'signal': round(final_score * 100, 2),
                'trade_date': df['trade_date'].iloc[-1] if 'trade_date' in df.columns else None,
                'price': round(float(current_price), 2),

                # 通道信息
                'channel_status': channel_info.get('channel_status', 'UNKNOWN'),
                'beta': round(float(channel_info.get('beta', 0)), 4),
                'r2': round(float(channel_info.get('r2', 0)), 3),
                'width_pct': round(float(channel_info.get('width_pct', 0)), 3),
                'slope_deg': round(float(channel_info.get('slope_deg', 0)), 2),
                'volatility': round(float(channel_info.get('volatility', 0)), 3),
                'cumulative_gain': round(float(channel_info.get('cumulative_gain', 0)), 3),
                'window_size': int(channel_info.get('window_size', 0)),
                'days_since_anchor': int(channel_info.get('days_since_anchor', 0)),

                # 通道边界
                'mid_today': round(float(mid_today), 2),
                'upper_today': round(float(upper_today), 2),
                'lower_today': round(float(lower_today), 2),
                'price_position': price_position,

                # 锚点信息
                'anchor_date': channel_info.get('anchor_date', ''),
                'anchor_price': round(float(channel_info.get('anchor_price', 0)), 2),

                # 技术指标
                'rsi': round(float(df['rsi'].iloc[-1]), 2),
                'macd': round(float(df['macd'].iloc[-1]), 4),
                'volume_ratio': round(float(df['volume_ratio'].iloc[-1]), 2),

                # 评分信息
                'channel_quality_score': round(scores['channel_quality'] * 100, 2),
                'trend_strength_score': round(scores['trend_strength'] * 100, 2),
                'risk_control_score': round(scores['risk_control'] * 100, 2),
                'volume_analysis_score': round(scores['volume_analysis'] * 100, 2),

                # 建议和风险
                'buy_signal': buy_signal,
                'risk_level': risk_level,
                'position_advice': self._generate_position_advice(channel_info, final_score)
            })

        except Exception as e:
            logging.exception(f"生成详细信号失败: {e}")
            return pd.Series({'signal': 0})

    def _generate_position_advice(self, channel_info: Dict[str, Any], final_score: float) -> str:
        """生成持仓建议"""
        try:
            cumulative_gain = channel_info.get('cumulative_gain', 0)
            r2 = channel_info.get('r2', 0)
            width_pct = channel_info.get('width_pct', 0)

            if final_score > 0.8 and r2 > 0.6:
                return "通道质量优秀，趋势明确，建议买入"
            elif cumulative_gain > 0.3:
                return "累计涨幅较大，注意获利了结"
            elif width_pct > 0.10:
                return "通道较宽，波动较大，注意风险控制"
            elif r2 < 0.4:
                return "通道拟合度较低，建议谨慎"
            else:
                return "通道状态良好，可考虑买入"

        except Exception as e:
            logging.warning(f"生成持仓建议失败: {e}")
            return "建议谨慎"

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据是否符合策略要求"""
        try:
            if len(data) < self._params['min_data_points']:
                return False

            # 检查必要的列是否存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                return False

            # 检查数据质量
            if data['close'].isnull().sum() > len(data) * 0.1:
                return False

            return True

        except Exception as e:
            logging.exception(f"验证数据时发生错误: {e}")
            return False
