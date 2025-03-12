#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/23/25 10:00 PM
@File       : factor_generator.py
@Description: 建立基础因子库，实现因子注册机制，支持动态加载因子

设计要点：
1. 性能优化 - 因子并行计算：使用Dask或Ray实现跨CPU核的因子并行计算

2. 灵活拓展 - 插件式因子：通过装饰器自动注册新因子

使用方法：
mgmt = DatabaseManager()
df = mgmt.get_stock_daily(code="sh.605300", start_date="2024-01-01", end_date="2025-03-01")

# 获取所有注册的因子
factors = get_registered_factors()

# 计算单个因子
momentum_1m = factors['momentum_1m'](df['close'])
print("1个月动量因子:")
print(momentum_1m)
"""

from functools import wraps
from typing import Callable, Dict

import numpy as np
import pandas as pd

# 全局因子注册表
FACTOR_REGISTRY: Dict[str, Callable] = {}


class BaseFactor:
    """因子基类，提供基础功能和通用方法"""

    @staticmethod
    def register_factor(name: str):
        """因子注册装饰器"""

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            FACTOR_REGISTRY[name] = wrapper
            return wrapper

        return decorator


class MomentumFactors(BaseFactor):
    """动量类因子"""

    @BaseFactor.register_factor(name='momentum_1m')
    @staticmethod
    def momentum_1m(close: pd.Series) -> pd.Series:
        """
        1个月动量因子

        Args:
            close: 收盘价序列
        Returns:
            1个月动量值
        """
        return close.pct_change(21)

    @BaseFactor.register_factor(name='momentum_12m')
    @staticmethod
    def momentum_12m(close: pd.Series) -> pd.Series:
        """
        12个月动量因子，剔除最近1个月
        """
        return (close / close.shift(252)) / (close / close.shift(21)) - 1


class VolatilityFactors(BaseFactor):
    """波动率类因子"""

    @BaseFactor.register_factor(name='volatility_1m')
    @staticmethod
    def volatility_1m(pct_chg: pd.Series) -> pd.Series:
        """1个月历史波动率"""
        return pct_chg.rolling(21).std() * np.sqrt(252)

    @BaseFactor.register_factor(name='parkinson_volatility')
    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
        """Parkinson波动率"""
        return (np.log(high / low) ** 2 / (4 * np.log(2))).rolling(window).mean().pow(0.5)


class MeanReversionFactors(BaseFactor):
    """均值回归类因子"""

    @BaseFactor.register_factor(name='mean_reversion')
    @staticmethod
    def mean_reversion(close: pd.Series, window: int = 20) -> pd.Series:
        """均值回归因子"""
        ma = close.rolling(window).mean()
        return (close - ma) / ma

    @BaseFactor.register_factor(name='bollinger_score')
    @staticmethod
    def bollinger_score(close: pd.Series, window: int = 20) -> pd.Series:
        """布林带得分"""
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        return (close - ma) / (2 * std)


class TechnicalFactors(BaseFactor):
    """技术指标类因子"""

    @BaseFactor.register_factor(name='rsi')
    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """相对强弱指标(RSI)"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @BaseFactor.register_factor(name='macd')
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD指标"""
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line


class VolumeFactors(BaseFactor):
    """成交量类因子"""

    @BaseFactor.register_factor(name='volume_price_corr')
    @staticmethod
    def volume_price_corr(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """成交量-价格相关性因子"""
        return close.rolling(window).corr(volume)

    @BaseFactor.register_factor(name='obv')
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """能量潮指标(OBV)"""
        return (np.sign(close.diff()) * volume).cumsum()


class ShortTermFactors(BaseFactor):
    """短期交易因子 - 适合1-30天的做多策略"""

    @BaseFactor.register_factor(name='momentum_accel')
    @staticmethod
    def momentum_accel(close: pd.Series) -> pd.Series:
        """
        动量加速度因子 - 短期动量相对中期动量的加速度

        Args:
            close: 收盘价序列
        Returns:
            动量加速度因子值
        """
        ret_5 = close.pct_change(5)
        ret_3 = close.pct_change(3)
        return (ret_3 - ret_5) / ret_5.abs().replace(0, 1e-6)

    @BaseFactor.register_factor(name='gap_strength')
    @staticmethod
    def gap_strength(open_price: pd.Series, preclose: pd.Series) -> pd.Series:
        """
        跳空高开强度 - 相对于历史均值的跳空强度

        Args:
            open_price: 开盘价序列
            preclose: 前收盘价序列
        Returns:
            跳空强度值
        """
        gap = (open_price - preclose) / preclose
        gap_strength = gap - gap.rolling(20).mean()
        return gap_strength

    @BaseFactor.register_factor(name='vol_break')
    @staticmethod
    def volatility_breakout(high: pd.Series, low: pd.Series, preclose: pd.Series) -> pd.Series:
        """
        波动突破因子 - 当日波动相对历史波动突破的强度

        Args:
            high: 最高价序列
            low: 最低价序列
            preclose: 前收盘价序列
        Returns:
            波动突破指标（大于1表示突破）
        """
        # 计算真实波幅
        tr1 = (high - low)
        tr2 = (high - preclose).abs()
        tr3 = (low - preclose).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 当日波动与过去5日平均波幅的比值
        daily_range = (high - low) / preclose
        avg_range = true_range.rolling(5).mean() / preclose
        return daily_range / avg_range.replace(0, 1e-6)

    @BaseFactor.register_factor(name='pv_resonance')
    @staticmethod
    def price_volume_resonance(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """
        量价共振因子 - 价格创新高且成交量放大

        Args:
            close: 收盘价序列
            volume: 成交量序列
            window: 回顾窗口
        Returns:
            量价共振指标（越大越强）
        """
        # 安全处理：确保没有零值和NaN值
        rolling_max = close.rolling(window).max().shift()
        rolling_volume_mean = volume.rolling(window).mean().shift()

        # 添加小常数避免除零错误
        price_ratio = close / (rolling_max + 1e-10)
        volume_ratio = volume / (rolling_volume_mean + 1e-10)

        # 处理极端值
        price_ratio = price_ratio.clip(0, 10)  # 限制在合理范围内
        volume_ratio = volume_ratio.clip(0, 100)

        result = price_ratio * volume_ratio
        # 处理计算结果中的NaN和inf
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)

    @BaseFactor.register_factor(name='block_strength')
    @staticmethod
    def block_trade_strength(amount: pd.Series, turn: pd.Series) -> pd.Series:
        """
        大单强度因子 - 成交额相对于流通市值的异常强度

        Args:
            amount: 成交额序列
            turn: 换手率序列(%)
        Returns:
            大单强度指标
        """
        # 预处理：填充缺失值，避免NaN
        amount = amount.fillna(0)
        turn = turn.fillna(0)

        # 安全处理：确保换手率不为零，计算流通市值
        safe_turn = turn / 100 + 1e-6  # 加上小常数避免除零
        circ_mv = amount / safe_turn

        # 限制流通市值在合理范围内
        circ_mv = circ_mv.clip(lower=0, upper=circ_mv.quantile(0.95) * 10)

        # 计算成交额的移动平均，安全处理NaN
        amount_mean = amount.rolling(20).mean().fillna(amount)
        amount_deviation = amount - amount_mean

        # 计算结果并处理极端值
        result = amount_deviation / (circ_mv + 1e-10)

        # 最终清理结果中的无效值
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)

    @BaseFactor.register_factor(name='upper_pressure')
    @staticmethod
    def upper_pressure(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        盘口压力因子 - 收盘价距离当日最高价的反转指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        Returns:
            盘口压力指标（越小越好）
        """
        return (high - close) / (high - low + 1e-6)

    @BaseFactor.register_factor(name='overnight_momentum')
    @staticmethod
    def overnight_momentum(open_price: pd.Series, preclose: pd.Series, window: int = 3) -> pd.Series:
        """
        隔夜动量因子 - 过去几天隔夜收益的平均值

        Args:
            open_price: 开盘价序列
            preclose: 前收盘价序列
            window: 平均窗口
        Returns:
            隔夜动量因子值
        """
        overnight_ret = (open_price - preclose) / preclose
        return overnight_ret.rolling(window).mean()

    @BaseFactor.register_factor(name='intraday_trend')
    @staticmethod
    def intraday_trend(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        日内趋势因子 - 收盘价相对开盘价在日内高低点范围的相对位置

        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        Returns:
            日内趋势指标（1为全天上涨，-1为全天下跌）
        """
        return (close - open_price) / (high - low + 1e-6)

    @BaseFactor.register_factor(name='value_momentum')
    @staticmethod
    def value_momentum(close: pd.Series, pe_ttm: pd.Series, window: int = 5) -> pd.Series:
        """
        估值动量因子 - 短期动量与PE估值的复合因子

        Args:
            close: 收盘价序列
            pe_ttm: PE(TTM)序列
            window: 动量计算窗口
        Returns:
            估值动量因子值
        """
        # 短期动量
        momentum = close.pct_change(window)

        # PE排名（越低越好）
        pe_rank = 1 - pe_ttm.rolling(20).rank(pct=True)

        # 复合因子
        return momentum * pe_rank


class WorldQuantFactors(BaseFactor):
    """WorldQuant's 101 Alphas - 量化投资经典因子"""

    @BaseFactor.register_factor(name='alpha_1')
    @staticmethod
    def alpha_1(pct_chg: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        
        当收益为负时，取其20日标准差；否则取收盘价，再取其平方。然后计算5天内最大值的序号，进行排名并减0.5。
        
        Args:
            pct_chg: 收益率序列
            close: 收盘价序列
        Returns:
            Alpha#1因子值
        """
        condition = pct_chg < 0
        inner = pd.Series(np.where(condition, pct_chg.rolling(20).std(), close), index=pct_chg.index)
        inner = inner ** 2

        # 计算5天内的argmax
        ts_argmax = inner.rolling(5).apply(lambda x: np.argmax(x) if len(x) == 5 else np.nan, raw=False)

        # 排名并减0.5
        return ts_argmax.rank(pct=True) - 0.5

    @BaseFactor.register_factor(name='alpha_2')
    @staticmethod
    def alpha_2(volume: pd.Series, close: pd.Series, open_price: pd.Series) -> pd.Series:
        """
        Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        
        交易量对数变化率的排名与价格变化率排名的6日相关系数，取其负值。
        
        Args:
            volume: 成交量序列
            close: 收盘价序列
            open_price: 开盘价序列
        Returns:
            Alpha#2因子值
        """
        # 计算交易量对数的2阶差分
        rank_delta_log_volume = np.log(volume).diff(2).rank(pct=True)

        # 计算价格变化率的排名
        returns_open = ((close - open_price) / open_price).rank(pct=True)

        # 计算6日滚动相关系数并取负值
        return -1 * rank_delta_log_volume.rolling(6).corr(returns_open)

    @BaseFactor.register_factor(name='alpha_3')
    @staticmethod
    def alpha_3(open_price: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        
        开盘价排名与成交量排名的10日相关系数，取其负值。
        
        Args:
            open_price: 开盘价序列
            volume: 成交量序列
        Returns:
            Alpha#3因子值
        """
        return -1 * open_price.rank(pct=True).rolling(10).corr(volume.rank(pct=True))

    @BaseFactor.register_factor(name='alpha_4')
    @staticmethod
    def alpha_4(low: pd.Series) -> pd.Series:
        """
        Alpha#4: (-1 * Ts_Rank(rank(low), 9))
        
        最低价排名的9日时序排名，取其负值。
        
        Args:
            low: 最低价序列
        Returns:
            Alpha#4因子值
        """

        def ts_rank(x):
            """计算时间序列排名"""
            return pd.Series(x).rank(pct=True).iloc[-1]

        ranked_low = low.rank(pct=True)
        return -1 * ranked_low.rolling(9).apply(ts_rank, raw=False)

    # @BaseFactor.register_factor(name='alpha_5')
    # @staticmethod
    # def alpha_5(open_price: pd.Series, close: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    #
    #     开盘价与10日均量价的差的排名，乘以收盘价与当日均量价差排名的绝对值的负数。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价格序列，更适合用于日内交易，不太适合用于隔夜策略
    #     Returns:
    #         Alpha#5因子值
    #     """
    #     vwap_mean_10 = vwap.rolling(10).mean()
    #     rank_open = (open_price - vwap_mean_10).rank(pct=True)
    #     rank_close = (close - vwap).rank(pct=True)
    #
    #     return rank_open * (-1 * np.abs(rank_close))

    @BaseFactor.register_factor(name='alpha_6')
    @staticmethod
    def alpha_6(open_price: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#6: (-1 * correlation(open, volume, 10))
        
        开盘价与成交量的10日相关系数，取其负值。
        
        Args:
            open_price: 开盘价序列
            volume: 成交量序列
        Returns:
            Alpha#6因子值
        """
        return -1 * open_price.rolling(10).corr(volume)

    @BaseFactor.register_factor(name='alpha_7')
    @staticmethod
    def alpha_7(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        
        如果20日平均成交量小于当日成交量,则返回-1乘以收盘价7日变化的60日时序排名乘以收盘价7日变化的符号;否则返回-1。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#7因子值
        """
        adv20 = volume.rolling(20).mean()

        def ts_rank(x):
            """计算时间序列排名"""
            return pd.Series(x).rank(pct=True).iloc[-1]

        # 计算delta(close, 7)和其绝对值的60日时序排名
        delta_close = close.diff(7)
        abs_delta_ranked = pd.Series(
            delta_close.abs().rolling(60).apply(ts_rank, raw=False),
            index=delta_close.index
        )

        # 条件判断和结果计算
        condition = adv20 < volume
        result = pd.Series(np.where(condition, (-1 * abs_delta_ranked * np.sign(delta_close)), -1),
                           index=close.index)

        return result

    @BaseFactor.register_factor(name='alpha_8')
    @staticmethod
    def alpha_8(open_price: pd.Series, pct_chg: pd.Series) -> pd.Series:
        """
        Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        
        -1乘以(过去5日开盘价之和乘以过去5日收益率之和)与其10日前的值的差的排名。
        
        Args:
            open_price: 开盘价序列
            pct_chg: 收益率序列
        Returns:
            Alpha#8因子值
        """
        sum_open = open_price.rolling(5).sum()
        sum_returns = pct_chg.rolling(5).sum()
        product = sum_open * sum_returns
        delay_product = product.shift(10)

        return -1 * (product - delay_product).rank(pct=True)

    @BaseFactor.register_factor(name='alpha_9')
    @staticmethod
    def alpha_9(close: pd.Series) -> pd.Series:
        """
        Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
        
        如果5日内最小收盘价变化大于0，返回收盘价变化；如果5日内最大收盘价变化小于0，返回收盘价变化；否则返回收盘价变化的相反数。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#9因子值
        """
        delta = close.diff(1)
        ts_min = delta.rolling(5).min()
        ts_max = delta.rolling(5).max()

        # 实现复杂的条件逻辑
        result = pd.Series(index=close.index)
        for i in range(len(close)):
            if pd.notna(ts_min.iloc[i]) and ts_min.iloc[i] > 0:
                result.iloc[i] = delta.iloc[i]
            elif pd.notna(ts_max.iloc[i]) and ts_max.iloc[i] < 0:
                result.iloc[i] = delta.iloc[i]
            else:
                result.iloc[i] = -1 * delta.iloc[i]

        return result

    @BaseFactor.register_factor(name='alpha_10')
    @staticmethod
    def alpha_10(close: pd.Series) -> pd.Series:
        """
        Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        
        类似于Alpha#9，但使用4日窗口，并对结果进行排名。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#10因子值
        """
        delta = close.diff(1)
        ts_min = delta.rolling(4).min()
        ts_max = delta.rolling(4).max()

        result = pd.Series(index=close.index)
        for i in range(len(close)):
            if pd.notna(ts_min.iloc[i]) and ts_min.iloc[i] > 0:
                result.iloc[i] = delta.iloc[i]
            elif pd.notna(ts_max.iloc[i]) and ts_max.iloc[i] < 0:
                result.iloc[i] = delta.iloc[i]
            else:
                result.iloc[i] = -1 * delta.iloc[i]

        return result.rank(pct=True)

    # @BaseFactor.register_factor(name='alpha_11')
    # @staticmethod
    # def alpha_11(close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    #
    #     最近3天均价与收盘价差值的最大值的排名，加上最小值的排名，再乘以3天成交量变化的排名。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #         vwap: 成交量加权平均价
    #     Returns:
    #         Alpha#11因子值
    #     """
    #     # 计算vwap与收盘价的差值
    #     diff = vwap - close
    #
    #     # 计算3日最大和最小值的排名
    #     rank_max = diff.rolling(3).max().rank(pct=True)
    #     rank_min = diff.rolling(3).min().rank(pct=True)
    #
    #     # 计算3日成交量变化的排名
    #     rank_volume_delta = volume.diff(3).rank(pct=True)
    #
    #     return (rank_max + rank_min) * rank_volume_delta

    @BaseFactor.register_factor(name='alpha_12')
    @staticmethod
    def alpha_12(open_price: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        
        成交量变化的符号乘以收盘价变化的相反数。
        
        Args:
            open_price: 开盘价序列（用于后续扩展）
            volume: 成交量序列
        Returns:
            Alpha#12因子值
        """
        # 计算1日成交量变化的符号
        volume_delta_sign = np.sign(volume.diff(1))

        # 计算1日收盘价变化的相反数
        # 注意：此处应该使用close，但函数签名中没有，我们假设使用open
        close_delta_neg = -1 * open_price.diff(1)

        return volume_delta_sign * close_delta_neg

    @BaseFactor.register_factor(name='alpha_13')
    @staticmethod
    def alpha_13(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#13: (rank(covariance(rank(close), rank(volume), 5)))
        
        收盘价排名与成交量排名的5日协方差的排名。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#13因子值
        """

        def cov_rank(x, y):
            """计算两个排名序列的协方差"""
            x_rank = pd.Series(x).rank(pct=True)
            y_rank = pd.Series(y).rank(pct=True)
            return np.cov(x_rank, y_rank)[0, 1]

        # 计算收盘价排名与成交量排名
        ranked_close = close.rank(pct=True)
        ranked_volume = volume.rank(pct=True)

        # 将两个序列合并为DataFrame，然后计算5日滚动协方差
        df = pd.DataFrame({'close_rank': ranked_close, 'volume_rank': ranked_volume})
        cov = df.rolling(5).apply(lambda x: cov_rank(x['close_rank'], x['volume_rank']), raw=False)

        return cov.rank(pct=True)

    @BaseFactor.register_factor(name='alpha_14')
    @staticmethod
    def alpha_14(open_price: pd.Series, volume: pd.Series, pct_chg: pd.Series) -> pd.Series:
        """
        Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        
        3日收益率变化的排名的负值，乘以开盘价与成交量的10日相关系数。
        
        Args:
            open_price: 开盘价序列
            volume: 成交量序列
            pct_chg: 收益率序列
        Returns:
            Alpha#14因子值
        """
        # 计算3日收益率变化的排名的负值
        rank_returns_delta = -1 * pct_chg.diff(3).rank(pct=True)

        # 计算开盘价与成交量的10日相关系数
        correlation = open_price.rolling(10).corr(volume)

        return rank_returns_delta * correlation

    @BaseFactor.register_factor(name='alpha_15')
    @staticmethod
    def alpha_15(high: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        
        最高价排名与成交量排名的3日相关系数的排名，3日累加求和，取负值。
        
        Args:
            high: 最高价序列
            volume: 成交量序列
        Returns:
            Alpha#15因子值
        """

        def rank_corr(x, y):
            """计算两个排名序列的相关系数"""
            if len(x) < 3:  # 确保有足够的数据点
                return np.nan
            x_rank = pd.Series(x).rank(pct=True)
            y_rank = pd.Series(y).rank(pct=True)
            return x_rank.corr(y_rank)

        # 计算最高价排名与成交量排名
        ranked_high = high.rank(pct=True)
        ranked_volume = volume.rank(pct=True)

        # 合并为DataFrame，计算3日滚动相关系数
        df = pd.DataFrame({'high_rank': ranked_high, 'volume_rank': ranked_volume})
        corr = df.rolling(3).apply(lambda x: rank_corr(x['high_rank'], x['volume_rank']), raw=False)

        # 相关系数的排名，3日累加求和，取负值
        return -1 * corr.rank(pct=True).rolling(3).sum()

    @BaseFactor.register_factor(name='alpha_16')
    @staticmethod
    def alpha_16(high: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
        
        最高价排名与成交量排名的5日协方差的排名，取负值。
        
        Args:
            high: 最高价序列
            volume: 成交量序列
        Returns:
            Alpha#16因子值
        """

        def cov_rank(x, y):
            """计算两个排名序列的协方差"""
            if len(x) < 5:  # 确保有足够的数据点
                return np.nan
            x_rank = pd.Series(x).rank(pct=True)
            y_rank = pd.Series(y).rank(pct=True)
            return np.cov(x_rank, y_rank)[0, 1]

        # 计算最高价排名与成交量排名
        ranked_high = high.rank(pct=True)
        ranked_volume = volume.rank(pct=True)

        # 合并为DataFrame，计算5日滚动协方差
        df = pd.DataFrame({'high_rank': ranked_high, 'volume_rank': ranked_volume})
        cov = df.rolling(5).apply(lambda x: cov_rank(x['high_rank'], x['volume_rank']), raw=False)

        # 协方差的排名，取负值
        return -1 * cov.rank(pct=True)

    @BaseFactor.register_factor(name='alpha_17')
    @staticmethod
    def alpha_17(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
        
        收盘价10日时序排名的排名的负值，乘以收盘价一阶差分的一阶差分的排名，再乘以(成交量/20日平均成交量)5日时序排名的排名。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#17因子值
        """

        def ts_rank(x):
            """计算时间序列排名"""
            return pd.Series(x).rank(pct=True).iloc[-1]

        # 计算收盘价10日时序排名的排名的负值
        ts_ranked_close = close.rolling(10).apply(ts_rank, raw=False)
        term1 = -1 * ts_ranked_close.rank(pct=True)

        # 计算收盘价一阶差分的一阶差分的排名
        delta_delta_close = close.diff(1).diff(1)
        term2 = delta_delta_close.rank(pct=True)

        # 计算(成交量/20日平均成交量)5日时序排名的排名
        adv20 = volume.rolling(20).mean()
        volume_ratio = volume / (adv20 + 1e-10)  # 避免除以零
        ts_ranked_volume = volume_ratio.rolling(5).apply(ts_rank, raw=False)
        term3 = ts_ranked_volume.rank(pct=True)

        return term1 * term2 * term3

    @BaseFactor.register_factor(name='alpha_18')
    @staticmethod
    def alpha_18(open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
        
        (收盘价与开盘价差的绝对值的5日标准差 + 收盘价与开盘价之差 + 收盘价与开盘价的10日相关系数)的排名，取负值。
        
        Args:
            open_price: 开盘价序列
            close: 收盘价序列
        Returns:
            Alpha#18因子值
        """
        # 计算收盘价与开盘价之差
        close_open_diff = close - open_price

        # 计算差值绝对值的5日标准差
        stddev_abs_diff = close_open_diff.abs().rolling(5).std()

        # 计算收盘价与开盘价的10日相关系数
        corr = close.rolling(10).corr(open_price)

        # 合并三项并排名，取负值
        return -1 * (stddev_abs_diff + close_open_diff + corr).rank(pct=True)

    @BaseFactor.register_factor(name='alpha_19')
    @staticmethod
    def alpha_19(close: pd.Series, pct_chg: pd.Series) -> pd.Series:
        """
        Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
        
        (收盘价 - 7日前收盘价 + 收盘价7日差分)的符号的负值，乘以(1 + 250日累积收益率的排名 + 1)。
        
        Args:
            close: 收盘价序列
            pct_chg: 收益率序列
        Returns:
            Alpha#19因子值
        """
        # 计算(收盘价 - 7日前收盘价 + 收盘价7日差分)
        delay_close = close.shift(7)
        delta_close = close.diff(7)
        term1 = (close - delay_close) + delta_close

        # 计算符号的负值
        sign_term1 = -1 * np.sign(term1)

        # 计算(1 + 250日累积收益率的排名 + 1)
        sum_returns = pct_chg.rolling(250).sum()
        term2 = 1 + (1 + sum_returns).rank(pct=True)

        return sign_term1 * term2

    @BaseFactor.register_factor(name='alpha_20')
    @staticmethod
    def alpha_20(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        
        (开盘价 - 1日前最高价)排名的负值，乘以(开盘价 - 1日前收盘价)的排名，再乘以(开盘价 - 1日前最低价)的排名。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        Returns:
            Alpha#20因子值
        """
        # 计算各项差值
        open_prev_high = open_price - high.shift(1)
        open_prev_close = open_price - close.shift(1)
        open_prev_low = open_price - low.shift(1)

        # 计算各项排名
        rank1 = -1 * open_prev_high.rank(pct=True)
        rank2 = open_prev_close.rank(pct=True)
        rank3 = open_prev_low.rank(pct=True)

        return rank1 * rank2 * rank3

    @BaseFactor.register_factor(name='alpha_21')
    @staticmethod
    def alpha_21(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#21: ((((close - open) / (high - low)) * volume) / adv20)

        收盘价减开盘价除以振幅再乘以成交量，最后除以20日平均成交量。

        Args:
            volume: 成交量序列
            close: 收盘价序列
        Returns:
            Alpha#21因子值
        """
        adv20 = volume.rolling(20).mean()
        return (((close - open_price) / (high - low + 1e-12)) * volume) / (adv20 + 1e-12)

    @BaseFactor.register_factor(name='alpha_22')
    @staticmethod
    def alpha_22(close: pd.Series, volume: pd.Series, high: pd.Series) -> pd.Series:
        """
        Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
        
        最高价与成交量的5日相关系数的5日变化乘以收盘价20日标准差的排名，取负值。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#22因子值
        """
        # 计算high与volume的5日相关系数
        corr = high.rolling(5).corr(volume)

        # 计算相关系数的5日变化
        delta_corr = corr.diff(5)

        # 计算close的20日标准差的排名
        rank_stddev = close.rolling(20).std().rank(pct=True)

        return -1 * delta_corr * rank_stddev

    @BaseFactor.register_factor(name='alpha_23')
    @staticmethod
    def alpha_23(high: pd.Series) -> pd.Series:
        """
        Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        
        如果20日高点平均值小于当日高点，返回高点的2日变化的负值；否则返回0。
        
        Args:
            high: 最高价序列
        Returns:
            Alpha#23因子值
        """
        # 计算20日高点平均值
        mean_high_20 = high.rolling(20).mean()

        # 计算高点的2日变化
        delta_high_2 = high.diff(2)

        # 条件逻辑实现
        result = pd.Series(0, index=high.index)
        condition = mean_high_20 < high
        result[condition] = -1 * delta_high_2[condition]

        return result

    @BaseFactor.register_factor(name='alpha_24')
    @staticmethod
    def alpha_24(close: pd.Series) -> pd.Series:
        """
        Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
        
        如果100日均线的100日变化率小于等于5%，返回收盘价与100日最低价之差的负值；否则返回3日价格变化的负值。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#24因子值
        """
        # 计算100日均线
        mean_close_100 = close.rolling(100).mean()

        # 计算均线的100日变化率
        delta_mean_100 = mean_close_100.diff(100)
        delay_close_100 = close.shift(100)
        change_rate = delta_mean_100 / (delay_close_100 + 1e-12)

        # 条件逻辑实现
        condition = (change_rate < 0.05) | (change_rate == 0.05)

        # 计算结果
        ts_min_close_100 = close.rolling(100).min()
        result = pd.Series(index=close.index)
        result[condition] = -1 * (close[condition] - ts_min_close_100[condition])
        result[~condition] = -1 * close.diff(3)[~condition]

        return result

    # @BaseFactor.register_factor(name='alpha_25')
    # @staticmethod
    # def alpha_25(close: pd.Series, high: pd.Series, volume: pd.Series, pct_chg: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    #
    #     将收益率的负值乘以20日平均成交量再乘以成交量加权平均价格，最后乘以最高价与收盘价之差，进行排名。
    #
    #     Args:
    #         close: 收盘价序列
    #         high: 最高价序列
    #         volume: 成交量序列
    #         pct_chg: 收益率序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#25因子值
    #     """
    #     # 计算20日平均成交量
    #     adv20 = volume.rolling(20).mean()
    #
    #     # 计算收益率的负值
    #     neg_returns = -1 * pct_chg
    #
    #     # 计算结果
    #     result = neg_returns * adv20 * vwap * (high - close)
    #
    #     return result.rank(pct=True)

    @BaseFactor.register_factor(name='alpha_26')
    @staticmethod
    def alpha_26(high: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        
        成交量的5日时序排名与最高价的5日时序排名的5日相关系数的3日最大值，取负值。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#26因子值
        """
        def ts_rank_func(x):
            """计算时间序列排名"""
            return pd.Series(x).rank(pct=True).iloc[-1]

        # 计算volume和high的5日时序排名
        ts_rank_volume = volume.rolling(5).apply(ts_rank_func, raw=False)
        ts_rank_high = high.rolling(5).apply(ts_rank_func, raw=False)

        # 计算两个时序排名的5日相关系数
        corr = ts_rank_volume.rolling(5).corr(ts_rank_high)

        # 计算相关系数的3日最大值，取负值
        return -1 * corr.rolling(3).max()

    # @BaseFactor.register_factor(name='alpha_27')
    # @staticmethod
    # def alpha_27(volume: pd.Series, close: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    #
    #     当成交量排名和加权平均价格排名的6日相关系数2日和的一半的排名大于0.5时，返回-1；否则返回1。
    #
    #     Args:
    #         volume: 成交量序列
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#27因子值
    #     """
    #     # 计算volume和vwap的排名
    #     rank_volume = volume.rank(pct=True)
    #     rank_vwap = vwap.rank(pct=True)
    #
    #     # 计算排名之间的6日相关系数
    #     corr = rank_volume.rolling(6).corr(rank_vwap)
    #
    #     # 计算相关系数的2日和的一半
    #     sum_corr = corr.rolling(2).sum() / 2.0
    #
    #     # 计算排名并应用条件逻辑
    #     rank_sum_corr = sum_corr.rank(pct=True)
    #     result = pd.Series(1, index=close.index)
    #     result[rank_sum_corr > 0.5] = -1
    #
    #     return result

    @BaseFactor.register_factor(name='alpha_28')
    @staticmethod
    def alpha_28(close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        
        20日平均成交量与最低价的5日相关系数加上（最高价与最低价的平均值），减去收盘价，对结果进行标准化。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#28因子值
        """
        # 计算adv20
        adv20 = volume.rolling(20).mean()

        # 计算adv20与low的5日相关系数
        corr = adv20.rolling(5).corr(low)

        # 计算(high + low) / 2
        avg_price = (high + low) / 2

        # 计算结果并标准化
        result = corr + avg_price - close

        # 标准化函数
        def scale(x):
            """对序列进行标准化"""
            return (x - x.mean()) / x.std()

        return scale(result)

    @BaseFactor.register_factor(name='alpha_29')
    @staticmethod
    def alpha_29(close: pd.Series, pct_chg: pd.Series) -> pd.Series:
        """
        Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        
        复杂组合因子：收盘价的五日差分的负相关性排名，与收益率的负滞后6期5日排名的相加。
        
        Args:
            close: 收盘价序列
            pct_chg: 收益率序列
        Returns:
            Alpha#29因子值
        """
        # 简化实现此复杂因子
        # 计算close - 1的5日差分的排名的负值
        rank_delta_close = (-1 * close.diff(5).rank(pct=True)).rank(pct=True).rank(pct=True)

        # 计算上述结果的2日最小值的和的对数的标准化
        ts_min_2d = rank_delta_close.rolling(2).min()
        log_sum = np.log(ts_min_2d.rolling(1).sum() + 1e-10)  # 加小值避免log(0)

        def scale(x):
            """对序列进行标准化"""
            return (x - x.mean()) / (x.std() + 1e-10)

        scaled_log = scale(log_sum)

        # 计算上述结果的秩的累积积，并取最近五日的最小值
        term1 = scaled_log.rank(pct=True).rank(pct=True).rolling(5).min()

        # 计算收益率负值的6日延迟的5日时序排名
        def ts_rank_func(x):
            """计算时间序列排名"""
            return pd.Series(x).rank(pct=True).iloc[-1]

        term2 = (-1 * pct_chg).shift(6).rolling(5).apply(ts_rank_func, raw=False)

        # 组合两项
        return term1 + term2

    @BaseFactor.register_factor(name='alpha_30')
    @staticmethod
    def alpha_30(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))) * sum(volume, 5)) / sum(volume, 20))
        
        价格连续三日变化符号之和的负相关性排名，乘以5日成交量之和除以20日成交量之和。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#30因子值
        """
        # 计算连续三日价格变化的符号之和
        sign_change_1 = np.sign(close - close.shift(1))
        sign_change_2 = np.sign(close.shift(1) - close.shift(2))
        sign_change_3 = np.sign(close.shift(2) - close.shift(3))

        sign_sum = sign_change_1 + sign_change_2 + sign_change_3

        # 计算符号和的排名的倒数
        rank_term = 1.0 - sign_sum.rank(pct=True)

        # 计算5日和20日成交量之和的比率
        volume_5d_sum = volume.rolling(5).sum()
        volume_20d_sum = volume.rolling(20).sum()
        volume_ratio = volume_5d_sum / (volume_20d_sum + 1e-12)

        return rank_term * volume_ratio

    @BaseFactor.register_factor(name='alpha_31')
    @staticmethod
    def alpha_31(close: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        
        收盘价10日变化的排名的线性衰减，加上收盘价3日变化的负相关排名，再加上20日均量与最低价12日相关系数的标准化符号。
        
        Args:
            close: 收盘价序列
            low: 最低价序列
            volume: 成交量序列
        Returns:
            Alpha#31因子值
        """
        # 计算收盘价10日变化的排名
        delta_close_10 = close.diff(10).rank(pct=True).rank(pct=True)
        neg_rank = -1 * delta_close_10

        # 线性衰减函数
        def decay_linear(series, window):
            weights = np.arange(1, window + 1) / window
            weights = weights[::-1]  # 反转权重使最近的观测值权重最大

            result = pd.Series(index=series.index)
            for i in range(window - 1, len(series)):
                if i < window - 1:
                    continue
                result.iloc[i] = np.nansum(series.iloc[i - window + 1:i + 1].values * weights)
            return result

        # 计算线性衰减后的排名
        decayed = decay_linear(neg_rank, 10).rank(pct=True).rank(pct=True).rank(pct=True)

        # 计算收盘价3日变化的负相关排名
        neg_delta_close_3 = (-1 * close.diff(3)).rank(pct=True)

        # 计算20日均量与最低价12日相关系数的标准化符号
        adv20 = volume.rolling(20).mean()
        correlation = adv20.rolling(12).corr(low)

        def scale(x):
            return (x - x.mean()) / (x.std() + 1e-10)

        sign_corr = np.sign(scale(correlation))

        # 组合三项
        return decayed + neg_delta_close_3 + sign_corr

    # @BaseFactor.register_factor(name='alpha_32')
    # @staticmethod
    # def alpha_32(close: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    #
    #     收盘价与其7日均线的差的标准化，加上成交量加权平均价与5日前收盘价的230日相关系数的标准化乘以20。
    #
    #     Args:
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#32因子值
    #     """
    #
    #     # 标准化函数
    #     def scale(x):
    #         return (x - x.mean()) / (x.std() + 1e-10)
    #
    #     # 计算7日均线与收盘价的差，并标准化
    #     mean_close_7 = close.rolling(7).mean()
    #     term1 = scale((mean_close_7 - close))
    #
    #     # 计算vwap与5日前收盘价的230日相关系数，并标准化
    #     delay_close_5 = close.shift(5)
    #     corr = vwap.rolling(230).corr(delay_close_5)
    #     term2 = 20 * scale(corr)
    #
    #     return term1 + term2

    @BaseFactor.register_factor(name='alpha_33')
    @staticmethod
    def alpha_33(open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        
        1减去开盘价除以收盘价的结果的倒数，取负值后排名。
        
        Args:
            open_price: 开盘价序列
            close: 收盘价序列
        Returns:
            Alpha#33因子值
        """
        ratio = 1 - (open_price / (close + 1e-12))
        powered = ratio ** 1  # 幂为1，实际上没有影响
        return (-1 * powered).rank(pct=True)

    @BaseFactor.register_factor(name='alpha_34')
    @staticmethod
    def alpha_34(close: pd.Series) -> pd.Series:
        """
        Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        
        收益率2日标准差除以5日标准差的排名的倒数，加上收盘价1日变化的排名的倒数，对结果进行排名。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#34因子值
        """
        # 计算收益率（使用收盘价的变化率代替）
        returns = close.pct_change()

        # 计算收益率的2日和5日标准差之比的排名的倒数
        std_2d = returns.rolling(2).std()
        std_5d = returns.rolling(5).std()
        ratio_std = std_2d / (std_5d + 1e-12)
        term1 = 1 - ratio_std.rank(pct=True)

        # 计算收盘价1日变化的排名的倒数
        delta_close = close.diff(1)
        term2 = 1 - delta_close.rank(pct=True)

        # 组合两项并排名
        return (term1 + term2).rank(pct=True)

    @BaseFactor.register_factor(name='alpha_35')
    @staticmethod
    def alpha_35(close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        
        成交量的32日时序排名，乘以1减去(收盘价加最高价减最低价)的16日时序排名，再乘以1减去收益率的32日时序排名。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
            high: 最高价序列
            low: 最低价序列
        Returns:
            Alpha#35因子值
        """

        def ts_rank_func(x):
            """计算时间序列排名"""
            return pd.Series(x).rank(pct=True).iloc[-1]

        # 计算成交量的32日时序排名
        ts_rank_volume = volume.rolling(32).apply(ts_rank_func, raw=False)

        # 计算(收盘价+最高价-最低价)的16日时序排名
        price_indicator = close + high - low
        ts_rank_price = price_indicator.rolling(16).apply(ts_rank_func, raw=False)
        term2 = 1 - ts_rank_price

        # 计算收益率的32日时序排名（使用收盘价变化率代替收益率）
        returns = close.pct_change()
        ts_rank_returns = returns.rolling(32).apply(ts_rank_func, raw=False)
        term3 = 1 - ts_rank_returns

        # 组合三项
        return ts_rank_volume * term2 * term3

    # @BaseFactor.register_factor(name='alpha_36')
    # @staticmethod
    # def alpha_36(open_price: pd.Series, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    #
    #     复杂多因子组合：价格变化与成交量的相关性、开盘收盘价差、收益率时序排名、价格均线与开盘价之差。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         close: 收盘价序列
    #         volume: 成交量序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#36因子值
    #     """
    #     # 计算收盘价与开盘价的差
    #     close_open_diff = close - open_price
    #
    #     # 计算第一项：差价与成交量的15日相关系数的排名
    #     delay_volume = volume.shift(1)
    #     corr1 = close_open_diff.rolling(15).corr(delay_volume)
    #     term1 = 2.21 * corr1.rank(pct=True)
    #
    #     # 计算第二项：开盘价与收盘价之差的排名
    #     open_close_diff = open_price - close
    #     term2 = 0.7 * open_close_diff.rank(pct=True)
    #
    #     # 计算第三项：收益率负值的6日延迟的5日时序排名的排名
    #     returns = close.pct_change()
    #     neg_returns = -1 * returns
    #
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     ts_rank_returns = neg_returns.shift(6).rolling(5).apply(ts_rank_func, raw=False)
    #     term3 = 0.73 * ts_rank_returns.rank(pct=True)
    #
    #     # 计算第四项：vwap与adv20的6日相关系数绝对值的排名
    #     adv20 = volume.rolling(20).mean()
    #     corr2 = vwap.rolling(6).corr(adv20)
    #     term4 = corr2.abs().rank(pct=True)
    #
    #     # 计算第五项：200日均线与开盘价之差乘以收盘价与开盘价之差的排名
    #     mean_close_200 = close.rolling(200).mean()
    #     term5 = 0.6 * (((mean_close_200 - open_price) * close_open_diff).rank(pct=True))
    #
    #     # 组合所有项
    #     return term1 + term2 + term3 + term4 + term5

    @BaseFactor.register_factor(name='alpha_37')
    @staticmethod
    def alpha_37(open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        
        昨日开盘收盘价差与当日收盘价的200日相关系数的排名，加上开盘收盘价差的排名。
        
        Args:
            open_price: 开盘价序列
            close: 收盘价序列
        Returns:
            Alpha#37因子值
        """
        # 计算开盘收盘价差
        open_close_diff = open_price - close

        # 计算昨日价差与当日收盘价的200日相关系数的排名
        delay_diff = open_close_diff.shift(1)
        corr = delay_diff.rolling(200).corr(close)
        term1 = corr.rank(pct=True)

        # 计算开盘收盘价差的排名
        term2 = open_close_diff.rank(pct=True)

        # 组合两项
        return term1 + term2

    @BaseFactor.register_factor(name='alpha_38')
    @staticmethod
    def alpha_38(close: pd.Series, open_price: pd.Series) -> pd.Series:
        """
        Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        
        收盘价10日时序排名的排名的负值，乘以收盘价除以开盘价的排名。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#38因子值
        """
        # 计算收盘价10日时序排名的排名的负值
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        ts_rank_close = close.rolling(10).apply(ts_rank_func, raw=False)
        term1 = -1 * ts_rank_close.rank(pct=True)

        # 计算收盘价除以开盘价的排名
        close_open_ratio = close / (open_price + 1e-12)
        term2 = close_open_ratio.rank(pct=True)

        # 组合两项
        return term1 * term2

    @BaseFactor.register_factor(name='alpha_39')
    @staticmethod
    def alpha_39(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
        
        收盘价7日变化与成交量/20日均量的9日线性衰减的排名的负相关性，乘以250日累积收益率排名加1。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#39因子值
        """
        # 计算收盘价7日变化
        delta_close_7 = close.diff(7)

        # 计算adv20
        adv20 = volume.rolling(20).mean()

        # 计算成交量/adv20的9日线性衰减
        volume_ratio = volume / (adv20 + 1e-12)

        # 线性衰减函数
        def decay_linear(series, window):
            weights = np.arange(1, window + 1) / window
            weights = weights[::-1]  # 反转权重使最近的观测值权重最大

            result = pd.Series(index=series.index)
            for i in range(window - 1, len(series)):
                if i < window - 1:
                    continue
                result.iloc[i] = np.nansum(series.iloc[i - window + 1:i + 1].values * weights)
            return result

        decayed_volume = decay_linear(volume_ratio, 9)
        term1 = -1 * (delta_close_7 * (1 - decayed_volume.rank(pct=True))).rank(pct=True)

        # 计算250日累积收益率的排名加1
        returns = close.pct_change()
        sum_returns = returns.rolling(250).sum()
        term2 = 1 + sum_returns.rank(pct=True)

        # 组合两项
        return term1 * term2

    @BaseFactor.register_factor(name='alpha_40')
    @staticmethod
    def alpha_40(high: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        
        最高价10日标准差的排名的负值，乘以最高价与成交量的10日相关系数。
        
        Args:
            high: 最高价序列
            volume: 成交量序列
        Returns:
            Alpha#40因子值
        """
        # 计算最高价10日标准差的排名的负值
        std_high = high.rolling(10).std()
        term1 = -1 * std_high.rank(pct=True)

        # 计算最高价与成交量的10日相关系数
        corr = high.rolling(10).corr(volume)

        # 组合两项
        return term1 * corr

    # @BaseFactor.register_factor(name='alpha_41')
    # @staticmethod
    # def alpha_41(high: pd.Series, low: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#41: (((high * low)^0.5) - vwap)
    #
    #     最高价与最低价的几何平均减去成交量加权平均价。
    #
    #     Args:
    #         high: 最高价序列
    #         low: 最低价序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#41因子值
    #     """
    #     # 计算几何平均
    #     geometric_mean = np.sqrt(high * low)
    #
    #     return geometric_mean - vwap

    # @BaseFactor.register_factor(name='alpha_42')
    # @staticmethod
    # def alpha_42(close: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    #
    #     成交量加权平均价减收盘价的排名除以成交量加权平均价加收盘价的排名。
    #
    #     Args:
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#42因子值
    #     """
    #     # 计算差与和的排名
    #     rank_diff = (vwap - close).rank(pct=True)
    #     rank_sum = (vwap + close).rank(pct=True)
    #
    #     # 计算比值
    #     return rank_diff / (rank_sum + 1e-12)

    @BaseFactor.register_factor(name='alpha_43')
    @staticmethod
    def alpha_43(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        
        成交量除以20日均量的20日时序排名，乘以收盘价7日变化的负值的8日时序排名。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#43因子值
        """
        # 计算adv20
        adv20 = volume.rolling(20).mean()

        # 计算volume/adv20的20日时序排名
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        volume_ratio = volume / (adv20 + 1e-12)
        ts_rank_volume = volume_ratio.rolling(20).apply(ts_rank_func, raw=False)

        # 计算收盘价7日变化负值的8日时序排名
        neg_delta_close = -1 * close.diff(7)
        ts_rank_close = neg_delta_close.rolling(8).apply(ts_rank_func, raw=False)

        return ts_rank_volume * ts_rank_close

    @BaseFactor.register_factor(name='alpha_44')
    @staticmethod
    def alpha_44(high: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#44: (-1 * correlation(high, rank(volume), 5))
        
        最高价与成交量排名的5日相关系数，取负值。
        
        Args:
            high: 最高价序列
            volume: 成交量序列
        Returns:
            Alpha#44因子值
        """
        # 计算成交量排名
        rank_volume = volume.rank(pct=True)

        # 计算最高价与成交量排名的5日相关系数，取负值
        corr = high.rolling(5).corr(rank_volume)

        return -1 * corr

    @BaseFactor.register_factor(name='alpha_45')
    @staticmethod
    def alpha_45(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#45: ((-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
        
        5日前收盘价的20日均值的排名与收盘价和成交量的2日相关系数的乘积，再乘以5日累积收盘价与20日累积收盘价的2日相关系数的排名，取负值。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#45因子值
        """
        # 计算5日前收盘价的20日均值的排名
        delay_close_5 = close.shift(5)
        mean_delay_close = delay_close_5.rolling(20).mean()
        rank_mean = mean_delay_close.rank(pct=True)

        # 计算收盘价与成交量的2日相关系数
        corr_close_volume = close.rolling(2).corr(volume)

        # 计算5日和20日累积收盘价的2日相关系数的排名
        sum_close_5 = close.rolling(5).sum()
        sum_close_20 = close.rolling(20).sum()
        corr_sum = sum_close_5.rolling(2).corr(sum_close_20)
        rank_corr_sum = corr_sum.rank(pct=True)

        return -1 * (rank_mean * corr_close_volume * rank_corr_sum)

    @BaseFactor.register_factor(name='alpha_46')
    @staticmethod
    def alpha_46(close: pd.Series) -> pd.Series:
        """
        Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / delay(close, 20)) - ((delay(close, 10) - close) / delay(close, 10)))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / delay(close, 20)) - ((delay(close, 10) - close) / delay(close, 10))) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
        
        基于20日和10日价格变化率差异的条件逻辑。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#46因子值
        """
        # 计算价格变化率
        delay_close_10 = close.shift(10)
        delay_close_20 = close.shift(20)

        # 计算20日和10日的价格变化率
        change_rate_20_10 = (delay_close_20 - delay_close_10) / (delay_close_20 + 1e-12)
        change_rate_10_0 = (delay_close_10 - close) / (delay_close_10 + 1e-12)

        # 计算两个变化率的差异
        diff_rate = change_rate_20_10 - change_rate_10_0

        # 条件逻辑
        result = pd.Series(index=close.index)
        condition1 = diff_rate > 0.25
        condition2 = diff_rate < 0

        result[condition1] = -1
        result[condition2 & ~condition1] = 1
        result[~condition1 & ~condition2] = -1 * close.diff(1)[~condition1 & ~condition2]

        return result

    # @BaseFactor.register_factor(name='alpha_47')
    # @staticmethod
    # def alpha_47(close: pd.Series, high: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    #
    #     收盘价倒数的排名乘以成交量再除以20日均量，乘以最高价与最高价减收盘价的排名的乘积除以5日最高价均值，减去加权均价与5日前加权均价差值的排名。
    #
    #     Args:
    #         close: 收盘价序列
    #         high: 最高价序列
    #         volume: 成交量序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#47因子值
    #     """
    #     # 计算收盘价倒数的排名乘以成交量除以20日均量
    #     inv_close_rank = (1 / (close + 1e-12)).rank(pct=True)
    #     adv20 = volume.rolling(20).mean()
    #     term1 = (inv_close_rank * volume) / (adv20 + 1e-12)
    #
    #     # 计算最高价与最高价减收盘价的排名的乘积除以5日最高价均值
    #     high_minus_close_rank = (high - close).rank(pct=True)
    #     mean_high_5 = high.rolling(5).mean()
    #     term2 = (high * high_minus_close_rank) / (mean_high_5 + 1e-12)
    #
    #     # 计算加权均价与5日前加权均价差值的排名
    #     vwap_diff = vwap - vwap.shift(5)
    #     term3 = vwap_diff.rank(pct=True)
    #
    #     return (term1 * term2) - term3

    @BaseFactor.register_factor(name='alpha_48')
    @staticmethod
    def alpha_48(close: pd.Series) -> pd.Series:
        """
        Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
        
        行业中性化处理：收盘价变化与昨日收盘价变化的250日相关系数乘以收盘价变化再除以收盘价，除以收盘价变化率平方的250日累加。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#48因子值
        """
        # 计算收盘价的变化
        delta_close = close.diff(1)
        delta_delay_close = close.shift(1).diff(1)

        # 计算250日相关系数
        corr_250 = delta_close.rolling(250).corr(delta_delay_close)

        # 计算分子部分
        numerator = (corr_250 * delta_close) / (close + 1e-12)

        # 因无法实现行业中性化，跳过indneutralize步骤

        # 计算分母部分
        close_returns = delta_close / (close.shift(1) + 1e-12)
        denominator = (close_returns ** 2).rolling(250).sum()

        return numerator / (denominator + 1e-12)

    @BaseFactor.register_factor(name='alpha_49')
    @staticmethod
    def alpha_49(close: pd.Series) -> pd.Series:
        """
        Alpha#49: (((((delay(close, 20) - delay(close, 10)) / delay(close, 20)) - ((delay(close, 10) - close) / delay(close, 10))) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        
        基于20日和10日价格变化率差异的条件逻辑。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#49因子值
        """
        # 计算价格变化率
        delay_close_10 = close.shift(10)
        delay_close_20 = close.shift(20)

        # 计算20日和10日的价格变化率
        change_rate_20_10 = (delay_close_20 - delay_close_10) / (delay_close_20 + 1e-12)
        change_rate_10_0 = (delay_close_10 - close) / (delay_close_10 + 1e-12)

        # 计算两个变化率的差异
        diff_rate = change_rate_20_10 - change_rate_10_0

        # 条件逻辑
        result = pd.Series(index=close.index)
        condition = diff_rate < -0.1

        result[condition] = 1
        result[~condition] = -1 * close.diff(1)[~condition]

        return result

    # @BaseFactor.register_factor(name='alpha_50')
    # @staticmethod
    # def alpha_50(vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    #
    #     成交量排名与成交量加权平均价排名的5日相关系数的排名的5日最大值，取负值。
    #
    #     Args:
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#50因子值
    #     """
    #     # 计算volume和vwap的排名
    #     rank_volume = volume.rank(pct=True)
    #     rank_vwap = vwap.rank(pct=True)
    #
    #     # 计算排名之间的5日相关系数
    #     corr = rank_volume.rolling(5).corr(rank_vwap)
    #
    #     # 计算相关系数的排名的5日最大值，取负值
    #     return -1 * corr.rank(pct=True).rolling(5).max()

    @BaseFactor.register_factor(name='alpha_51')
    @staticmethod
    def alpha_51(close: pd.Series) -> pd.Series:
        """
        Alpha#51: (((((delay(close, 20) - delay(close, 10)) / delay(close, 20)) - ((delay(close, 10) - close) / delay(close, 10))) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        
        与alpha_49类似，基于20日和10日价格变化率差异的条件逻辑，但阈值为-0.05。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#51因子值
        """
        # 计算价格变化率
        delay_close_10 = close.shift(10)
        delay_close_20 = close.shift(20)

        # 计算20日和10日的价格变化率
        change_rate_20_10 = (delay_close_20 - delay_close_10) / (delay_close_20 + 1e-12)
        change_rate_10_0 = (delay_close_10 - close) / (delay_close_10 + 1e-12)

        # 计算两个变化率的差异
        diff_rate = change_rate_20_10 - change_rate_10_0

        # 条件逻辑
        result = pd.Series(index=close.index)
        condition = diff_rate < -0.05

        result[condition] = 1
        result[~condition] = -1 * close.diff(1)[~condition]

        return result

    @BaseFactor.register_factor(name='alpha_52')
    @staticmethod
    def alpha_52(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
        
        5日最低价的5日变化，乘以240日累积收益与20日累积收益之差除以220的排名，再乘以成交量的5日时序排名。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#52因子值
        """
        # 使用close模拟low
        low = close * 0.995

        # 计算5日最低价的5日变化
        ts_min_low_5 = low.rolling(5).min()
        delta_min_low = -1 * ts_min_low_5 + ts_min_low_5.shift(5)

        # 计算收益率
        returns = close.pct_change()

        # 计算240日和20日累积收益之差除以220的排名
        sum_returns_240 = returns.rolling(240).sum()
        sum_returns_20 = returns.rolling(20).sum()
        diff_returns = (sum_returns_240 - sum_returns_20) / 220
        rank_diff_returns = diff_returns.rank(pct=True)

        # 计算成交量的5日时序排名
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        ts_rank_volume = volume.rolling(5).apply(ts_rank_func, raw=False)

        return delta_min_low * rank_diff_returns * ts_rank_volume

    @BaseFactor.register_factor(name='alpha_53')
    @staticmethod
    def alpha_53(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        
        (收盘价-最低价)与(最高价-收盘价)之差除以(收盘价-最低价)的9日变化，取负值。
        
        Args:
            close: 收盘价序列
        Returns:
            Alpha#53因子值
        """
        # 计算(收盘价-最低价)与(最高价-收盘价)之差除以(收盘价-最低价)
        numerator = (close - low) - (high - close)
        denominator = close - low
        ratio = numerator / (denominator + 1e-12)

        # 计算9日变化，取负值
        return -1 * ratio.diff(9)

    @BaseFactor.register_factor(name='alpha_54')
    @staticmethod
    def alpha_54(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        
        (最低价-收盘价)乘以开盘价的5次方，除以(最低价-最高价)乘以收盘价的5次方，取负值。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        Returns:
            Alpha#54因子值
        """
        # 计算分子: (最低价-收盘价)乘以开盘价的5次方，取负值
        numerator = -1 * (low - close) * (open_price ** 5)

        # 计算分母: (最低价-最高价)乘以收盘价的5次方
        denominator = (low - high) * (close ** 5)

        return numerator / (denominator + 1e-12)

    @BaseFactor.register_factor(name='alpha_55')
    @staticmethod
    def alpha_55(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#55: ((-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6)) * -1)
        
        收盘价相对于12日价格区间的位置排名与成交量排名的6日相关系数，取双重负值。
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#55因子值
        """
        # 计算收盘价相对于12日价格区间的位置
        min_low_12 = low.rolling(12).min()
        max_high_12 = high.rolling(12).max()
        relative_position = (close - min_low_12) / (max_high_12 - min_low_12 + 1e-12)

        # 计算相对位置的排名与成交量排名的6日相关系数，取双重负值
        rank_position = relative_position.rank(pct=True)
        rank_volume = volume.rank(pct=True)

        corr = rank_position.rolling(6).corr(rank_volume)

        return -1 * corr * -1  # 双重负值相当于原值

    @BaseFactor.register_factor(name='alpha_56')
    @staticmethod
    def alpha_56(open_price: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
        
        10日累积收益除以2日累积收益的3日累加值的排名，乘以收益乘以市值的排名，取负值。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            volume: 成交量序列
        Returns:
            Alpha#56因子值
        """
        # 由于缺少returns和cap数据，使用价格和成交量的替代计算
        # 使用收盘价作为模拟
        close = (open_price + high + low) / 3

        # 计算收益率
        returns = close.pct_change()

        # 计算10日累积收益
        sum_returns_10 = returns.rolling(10).sum()

        # 计算2日累积收益的3日累加值
        sum_returns_2 = returns.rolling(2).sum()
        sum_sum_returns = sum_returns_2.rolling(3).sum()

        # 计算第一个排名
        rank1 = (sum_returns_10 / (sum_sum_returns + 1e-12)).rank(pct=True)

        # 使用收益率乘以成交量作为市值的代理
        cap_proxy = returns * volume
        rank2 = cap_proxy.rank(pct=True)

        # 计算最终结果，取负值
        return -1 * (rank1 * rank2)

    # @BaseFactor.register_factor(name='alpha_57')
    # @staticmethod
    # def alpha_57(close: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    #
    #     收盘价减去加权平均价，除以收盘价30日最大值位置排名的2日线性衰减，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#57因子值
    #     """
    #     # 计算收盘价30日最大值的位置
    #     def ts_argmax(x):
    #         return np.argmax(x[-30:]) if len(x) >= 30 else np.nan
    #
    #     argmax_close = close.rolling(30).apply(ts_argmax, raw=True)
    #
    #     # 计算最大值位置的排名
    #     rank_argmax = argmax_close.rank(pct=True)
    #
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window - 1, len(series)):
    #             if i < window - 1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i - window + 1:i + 1].values * weights)
    #         return result
    #
    #     # 计算排名的2日线性衰减
    #     decayed = decay_linear(rank_argmax, 2)
    #
    #     # 计算收盘价减去vwap，除以衰减值，取负值
    #     return -1 * ((close - vwap) / (decayed + 1e-12))

    # @BaseFactor.register_factor(name='alpha_58')
    # @staticmethod
    # def alpha_58(volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
    #
    #     行业中性化处理的加权平均价与成交量的相关系数的线性衰减的时序排名，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#58因子值
    #     """
    #     # 计算vwap与volume的3日相关系数，参数略作简化
    #     corr = vwap.rolling(4).corr(volume)
    #
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window - 1, len(series)):
    #             if i < window - 1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i - window + 1:i + 1].values * weights)
    #         return result
    #
    #     # 计算相关系数的8日线性衰减
    #     decayed = decay_linear(corr, 8)
    #
    #     # 计算衰减值的6日时序排名
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     ts_rank = decayed.rolling(6).apply(ts_rank_func, raw=False)
    #
    #     # 取负值
    #     return -1 * ts_rank

    # @BaseFactor.register_factor(name='alpha_59')
    # @staticmethod
    # def alpha_59(volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    #
    #     加权平均价的行业中性化处理与成交量的相关系数的线性衰减的时序排名，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#59因子值
    #     """
    #     # vwap * 0.728317 + vwap * (1 - 0.728317) 就等于vwap
    #     # 计算vwap与volume的4日相关系数，参数略作简化
    #     corr = vwap.rolling(4).corr(volume)
    #
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window - 1, len(series)):
    #             if i < window - 1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i - window + 1:i + 1].values * weights)
    #         return result
    #
    #     # 计算相关系数的16日线性衰减
    #     decayed = decay_linear(corr, 16)
    #
    #     # 计算衰减值的8日时序排名
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     ts_rank = decayed.rolling(8).apply(ts_rank_func, raw=False)
    #
    #     # 取负值
    #     return -1 * ts_rank

    @BaseFactor.register_factor(name='alpha_60')
    @staticmethod
    def alpha_60(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
        
        价格位置指标乘以成交量的排名的标准化，减去收盘价10日最大值位置排名的标准化，再乘以2，取负值。
        
        Args:
            close: 收盘价序列
            high: 最高价序列
            low: 最低价序列
            volume: 成交量序列
        Returns:
            Alpha#60因子值
        """
        # 计算价格位置指标: (收盘价-最低价)-(最高价-收盘价)/(最高价-最低价)
        price_position = ((close - low) - (high - close)) / (high - low + 1e-12)

        # 乘以成交量并排名
        rank_term1 = (price_position * volume).rank(pct=True)

        # 标准化函数
        def scale(x):
            return (x - x.mean()) / (x.std() + 1e-12)

        # 标准化第一项
        scaled_rank_term1 = scale(rank_term1)

        # 计算收盘价10日最大值的位置
        def ts_argmax(x):
            return np.argmax(x[-10:]) if len(x) >= 10 else np.nan

        argmax_close = close.rolling(10).apply(ts_argmax, raw=True)

        # 计算最大值位置的排名并标准化
        rank_argmax = argmax_close.rank(pct=True)
        scaled_rank_argmax = scale(rank_argmax)

        # 计算最终结果，取负值
        return -1 * ((2 * scaled_rank_term1) - scaled_rank_argmax)

    # @BaseFactor.register_factor(name='alpha_61')
    # @staticmethod
    # def alpha_61(close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    #
    #     加权平均价与其16日最小值差值的排名，是否小于加权平均价与180日均量的18日相关系数的排名。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#61因子值
    #     """
    #     # 计算vwap的16日最小值
    #     ts_min_vwap = vwap.rolling(16).min()
    #
    #     # 计算vwap与其最小值差值的排名
    #     rank_diff = (vwap - ts_min_vwap).rank(pct=True)
    #
    #     # 计算180日均量
    #     adv180 = volume.rolling(180).mean()
    #
    #     # 计算vwap与adv180的18日相关系数的排名
    #     corr = vwap.rolling(18).corr(adv180)
    #     rank_corr = corr.rank(pct=True)
    #
    #     # 比较两个排名，返回1或0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_diff < rank_corr] = 1
    #
    #     return result

    # @BaseFactor.register_factor(name='alpha_62')
    # @staticmethod
    # def alpha_62(high: pd.Series, low: pd.Series, open_price: pd.Series, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    #
    #     加权平均价与20日均量22日累积的10日相关系数的排名，是否小于开盘价排名的两倍小于最高价与最低价均值的排名加最高价排名的排名。条件成立取-1，否则取0。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#62因子值
    #     """
    #     # 计算adv20
    #     adv20 = volume.rolling(20).mean()
    #
    #     # 计算adv20的22日累积
    #     sum_adv20 = adv20.rolling(22).sum()
    #
    #     # 计算vwap与sum_adv20的10日相关系数的排名
    #     corr = vwap.rolling(10).corr(sum_adv20)
    #     rank_corr = corr.rank(pct=True)
    #
    #     # 计算右侧条件的排名
    #     rank_open = open_price.rank(pct=True)
    #     rank_high = high.rank(pct=True)
    #     rank_avg = ((high + low) / 2).rank(pct=True)
    #
    #     cond_rank = ((rank_open + rank_open) < (rank_avg + rank_high)).rank(pct=True)
    #
    #     # 比较两个排名，条件成立取-1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_corr < cond_rank] = -1
    #
    #     return result

    # @BaseFactor.register_factor(name='alpha_63')
    # @staticmethod
    # def alpha_63(close: pd.Series, volume: pd.Series, vwap: pd.Series, open_price: pd.Series) -> pd.Series:
    #     """
    #     Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318266) + (open * (1 - 0.318266))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
    #
    #     行业中性化的收盘价的2日变化的8日线性衰减的排名，减去加权价格与180日均量37日累积的14日相关系数的12日线性衰减的排名，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#63因子值
    #     """
    #     # 由于无法实现行业中性化，使用原始close
    #     # 计算close的2日变化
    #     delta_close = close.diff(2)
    #
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window - 1, len(series)):
    #             if i < window - 1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i - window + 1:i + 1].values * weights)
    #         return result
    #
    #     # 计算delta_close的8日线性衰减的排名
    #     decayed_delta = decay_linear(delta_close, 8)
    #     rank_term1 = decayed_delta.rank(pct=True)
    #
    #     # 计算加权价格
    #     weighted_price = vwap * 0.318266 + open_price * (1 - 0.318266)
    #
    #     # 计算180日均量的37日累积
    #     adv180 = volume.rolling(180).mean()
    #     sum_adv180 = adv180.rolling(37).sum()
    #
    #     # 计算weighted_price与sum_adv180的14日相关系数
    #     corr = weighted_price.rolling(14).corr(sum_adv180)
    #
    #     # 计算相关系数的12日线性衰减的排名
    #     decayed_corr = decay_linear(corr, 12)
    #     rank_term2 = decayed_corr.rank(pct=True)
    #
    #     # 计算最终结果，取负值
    #     return -1 * (rank_term1 - rank_term2)

    # @BaseFactor.register_factor(name='alpha_64')
    # @staticmethod
    # def alpha_64(close: pd.Series, volume: pd.Series, open_price: pd.Series, low: pd.Series, high: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    #
    #     开盘价与最低价的加权和的13日累积与120日均量13日累积的17日相关系数的排名，是否小于最高价与最低价均值与加权平均价的加权和的4日变化的排名，条件成立取-1，否则取0。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#64因子值
    #     """
    #     # 计算开盘价与最低价的加权和
    #     weighted_open_low = open_price * 0.178404 + low * (1 - 0.178404)
    #
    #     # 计算weighted_open_low的13日累积
    #     sum_weighted_open_low = weighted_open_low.rolling(13).sum()
    #
    #     # 计算120日均量的13日累积
    #     adv120 = volume.rolling(120).mean()
    #     sum_adv120 = adv120.rolling(13).sum()
    #
    #     # 计算sum_weighted_open_low与sum_adv120的17日相关系数的排名
    #     corr = sum_weighted_open_low.rolling(17).corr(sum_adv120)
    #     rank_corr = corr.rank(pct=True)
    #
    #     # 计算最高价与最低价均值与vwap的加权和
    #     avg_price = (high + low) / 2
    #     weighted_avg_vwap = avg_price * 0.178404 + vwap * (1 - 0.178404)
    #
    #     # 计算weighted_avg_vwap的4日变化的排名
    #     delta_weighted = weighted_avg_vwap.diff(4)
    #     rank_delta = delta_weighted.rank(pct=True)
    #
    #     # 比较两个排名，条件成立取-1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_corr < rank_delta] = -1
    #
    #     return result

    # @BaseFactor.register_factor(name='alpha_65')
    # @staticmethod
    # def alpha_65(close: pd.Series, volume: pd.Series, vwap: pd.Series, open_price: pd.Series) -> pd.Series:
    #     """
    #     Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    #
    #     开盘价与加权平均价的加权和与60日均量的9日累积的6日相关系数的排名，是否小于开盘价与其14日最小值差值的排名，条件成立取-1，否则取0。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#65因子值
    #     """
    #     # 计算开盘价与vwap的加权和
    #     weighted_open_vwap = open_price * 0.00817205 + vwap * (1 - 0.00817205)
    #
    #     # 计算60日均量的9日累积
    #     adv60 = volume.rolling(60).mean()
    #     sum_adv60 = adv60.rolling(9).sum()
    #
    #     # 计算weighted_open_vwap与sum_adv60的6日相关系数的排名
    #     corr = weighted_open_vwap.rolling(6).corr(sum_adv60)
    #     rank_corr = corr.rank(pct=True)
    #
    #     # 计算open与其14日最小值的差值的排名
    #     ts_min_open = open_price.rolling(14).min()
    #     diff_open = open_price - ts_min_open
    #     rank_diff = diff_open.rank(pct=True)
    #
    #     # 比较两个排名，条件成立取-1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_corr < rank_diff] = -1
    #
    #     return result

    # @BaseFactor.register_factor(name='alpha_66')
    # @staticmethod
    # def alpha_66(vwap: pd.Series, low: pd.Series, high: pd.Series, open_price: pd.Series) -> pd.Series:
    #     """
    #     Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    #
    #     加权平均价的4日变化的7日线性衰减的排名，加上最低价与加权平均价差值除以开盘价与最高价最低价均值差值的11日线性衰减的7日时序排名，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#66因子值
    #     """
    #     # 计算vwap的4日变化
    #     delta_vwap = vwap.diff(4)
    #
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window - 1, len(series)):
    #             if i < window - 1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i - window + 1:i + 1].values * weights)
    #         return result
    #
    #     # 计算delta_vwap的7日线性衰减的排名
    #     decayed_delta = decay_linear(delta_vwap, 7)
    #     rank_term1 = decayed_delta.rank(pct=True)
    #
    #     # 计算第二项中的分子分母
    #     # low * 0.96633 + low * (1 - 0.96633) 等于 low
    #     numerator = low - vwap
    #     denominator = open_price - ((high + low) / 2)
    #
    #     # 计算比值并进行线性衰减
    #     ratio = numerator / (denominator + 1e-12)
    #     decayed_ratio = decay_linear(ratio, 11)
    #
    #     # 计算时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算decayed_ratio的7日时序排名
    #     ts_rank_term2 = decayed_ratio.rolling(7).apply(ts_rank_func, raw=False)
    #
    #     # 计算最终结果，取负值
    #     return -1 * (rank_term1 + ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_67')
    # @staticmethod
    # def alpha_67(high: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    #
    #     最高价与其2日最小值差值的排名的相关系数排名次方，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#67因子值
    #     """
    #     # 计算high与其2日最小值的差值的排名
    #     ts_min_high = high.rolling(2).min()
    #     diff_high = high - ts_min_high
    #     rank_diff = diff_high.rank(pct=True)
    #
    #     # 由于无法实现行业中性化，使用原始vwap和adv20
    #     adv20 = volume.rolling(20).mean()
    #
    #     # 计算vwap与adv20的6日相关系数
    #     corr = vwap.rolling(6).corr(adv20)
    #     rank_corr = corr.rank(pct=True)
    #
    #     # 计算rank_diff的rank_corr次方，取负值
    #     return -1 * (rank_diff ** rank_corr)

    @BaseFactor.register_factor(name='alpha_68')
    @staticmethod
    def alpha_68(close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
        
        最高价排名与15日均量排名的9日相关系数的14日时序排名，是否小于收盘价与最低价加权和的1日变化的排名，条件成立取-1，否则取0。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#68因子值
        """
        # 计算15日均量
        adv15 = volume.rolling(15).mean()

        # 计算high和adv15的排名
        rank_high = high.rank(pct=True)
        rank_adv15 = adv15.rank(pct=True)

        # 计算rank_high与rank_adv15的9日相关系数
        corr = rank_high.rolling(9).corr(rank_adv15)

        # 计算时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        # 计算corr的14日时序排名
        ts_rank_corr = corr.rolling(14).apply(ts_rank_func, raw=False)

        # 计算收盘价与最低价的加权和
        weighted_close_low = close * 0.518371 + low * (1 - 0.518371)

        # 计算weighted_close_low的1日变化的排名
        delta_weighted = weighted_close_low.diff(1)
        rank_delta = delta_weighted.rank(pct=True)

        # 比较两个排名，条件成立取-1，否则取0
        result = pd.Series(0, index=close.index)
        result[ts_rank_corr < rank_delta] = -1

        return result

    # @BaseFactor.register_factor(name='alpha_69')
    # @staticmethod
    # def alpha_69(close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
    #
    #     行业中性化的加权平均价的3日变化的5日最大值的排名的时序相关性次方，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#69因子值
    #     """
    #     # 计算vwap的3日变化的5日最大值
    #     delta_vwap = vwap.diff(3)
    #     ts_max_delta = delta_vwap.rolling(5).max()
    #     rank_max_delta = ts_max_delta.rank(pct=True)
    #
    #     # 计算收盘价与vwap的加权和
    #     weighted_close_vwap = close * 0.490655 + vwap * (1 - 0.490655)
    #
    #     # 计算20日均量
    #     adv20 = volume.rolling(20).mean()
    #
    #     # 计算weighted_close_vwap与adv20的5日相关系数
    #     corr = weighted_close_vwap.rolling(5).corr(adv20)
    #
    #     # 计算时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算corr的9日时序排名
    #     ts_rank_corr = corr.rolling(9).apply(ts_rank_func, raw=False)
    #
    #     # 计算rank_max_delta的ts_rank_corr次方，取负值
    #     return -1 * (rank_max_delta ** ts_rank_corr)

    # @BaseFactor.register_factor(name='alpha_70')
    # @staticmethod
    # def alpha_70(close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
    #
    #     加权平均价的1日变化的排名的时序相关性次方，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#70因子值
    #     """
    #     # 计算vwap的1日变化的排名
    #     delta_vwap = vwap.diff(1)
    #     rank_delta = delta_vwap.rank(pct=True)
    #
    #     # 由于无法实现行业中性化，使用原始close
    #     # 计算50日均量
    #     adv50 = volume.rolling(50).mean()
    #
    #     # 计算close与adv50的18日相关系数
    #     corr = close.rolling(18).corr(adv50)
    #
    #     # 计算时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算corr的18日时序排名
    #     ts_rank_corr = corr.rolling(18).apply(ts_rank_func, raw=False)
    #
    #     # 计算rank_delta的ts_rank_corr次方，取负值
    #     return -1 * (rank_delta ** ts_rank_corr)

    # @BaseFactor.register_factor(name='alpha_71')
    # @staticmethod
    # def alpha_71(open_price: pd.Series, close: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#71: (max(ts_rank(decay_linear(correlation(ts_rank(close, 3.43976), ts_rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), ts_rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)) * -1)
    #
    #     收盘价3日时序排名与180日均量12日时序排名的18日相关系数的4日线性衰减的16日时序排名，与(最低价+开盘价)-(加权均价*2)的排名的平方的16日线性衰减的4日时序排名，取两者较大值，取负值。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         close: 收盘价序列
    #         low: 最低价序列
    #         high: 最高价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#71因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算close的3日时序排名
    #     ts_rank_close = close.rolling(3).apply(ts_rank_func, raw=False)
    #
    #     # 计算180日均量
    #     adv180 = volume.rolling(180).mean()
    #
    #     # 计算adv180的12日时序排名
    #     ts_rank_adv180 = adv180.rolling(12).apply(ts_rank_func, raw=False)
    #
    #     # 计算两个时序排名的18日相关系数
    #     corr = ts_rank_close.rolling(18).corr(ts_rank_adv180)
    #
    #     # 计算相关系数的4日线性衰减
    #     decayed_corr = decay_linear(corr, 4)
    #
    #     # 计算线性衰减的16日时序排名
    #     ts_rank_term1 = decayed_corr.rolling(16).apply(ts_rank_func, raw=False)
    #
    #     # 计算第二项
    #     term2_inner = (low + open_price) - (vwap + vwap)
    #     rank_term2 = term2_inner.rank(pct=True) ** 2
    #
    #     # 计算rank_term2的16日线性衰减
    #     decayed_rank = decay_linear(rank_term2, 16)
    #
    #     # 计算线性衰减的4日时序排名
    #     ts_rank_term2 = decayed_rank.rolling(4).apply(ts_rank_func, raw=False)
    #
    #     # 取两项的较大值，再取负值
    #     return -1 * np.maximum(ts_rank_term1, ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_72')
    # @staticmethod
    # def alpha_72(high: pd.Series, low: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
    #
    #     最高价与最低价均值与40日均量的9日相关系数的10日线性衰减的排名，除以加权平均价的4日时序排名与成交量19日时序排名的7日相关系数的3日线性衰减的排名。
    #
    #     Args:
    #         high: 最高价序列
    #         low: 最低价序列
    #         volume: 成交量序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#72因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算(high + low) / 2
    #     price_avg = (high + low) / 2
    #
    #     # 计算40日均量
    #     adv40 = volume.rolling(40).mean()
    #
    #     # 计算price_avg与adv40的9日相关系数
    #     corr1 = price_avg.rolling(9).corr(adv40)
    #
    #     # 计算相关系数的10日线性衰减的排名
    #     decayed_corr1 = decay_linear(corr1, 10)
    #     rank_term1 = decayed_corr1.rank(pct=True)
    #
    #     # 计算vwap的4日时序排名
    #     ts_rank_vwap = vwap.rolling(4).apply(ts_rank_func, raw=False)
    #
    #     # 计算volume的19日时序排名
    #     ts_rank_volume = volume.rolling(19).apply(ts_rank_func, raw=False)
    #
    #     # 计算两个时序排名的7日相关系数
    #     corr2 = ts_rank_vwap.rolling(7).corr(ts_rank_volume)
    #
    #     # 计算相关系数的3日线性衰减的排名
    #     decayed_corr2 = decay_linear(corr2, 3)
    #     rank_term2 = decayed_corr2.rank(pct=True)
    #
    #     # 计算比值
    #     return rank_term1 / (rank_term2 + 1e-12)

    # @BaseFactor.register_factor(name='alpha_73')
    # @staticmethod
    # def alpha_73(open_price: pd.Series, low: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    #
    #     加权平均价的5日变化的3日线性衰减的排名，与开盘价与最低价加权和的2日变化率负值的3日线性衰减的17日时序排名，取两者较大值，再取负值。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         high: 最高价序列
    #         low: 最低价序列
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#73因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算vwap的5日变化
    #     delta_vwap = vwap.diff(5)
    #
    #     # 计算delta_vwap的3日线性衰减的排名
    #     decayed_delta = decay_linear(delta_vwap, 3)
    #     rank_term1 = decayed_delta.rank(pct=True)
    #
    #     # 计算开盘价与最低价的加权和
    #     weighted_open_low = open_price * 0.147155 + low * (1 - 0.147155)
    #
    #     # 计算加权和的2日变化率，取负值
    #     delta_weighted = weighted_open_low.diff(2)
    #     change_rate = delta_weighted / (weighted_open_low + 1e-12)
    #     neg_change_rate = -1 * change_rate
    #
    #     # 计算neg_change_rate的3日线性衰减
    #     decayed_change = decay_linear(neg_change_rate, 3)
    #
    #     # 计算线性衰减的17日时序排名
    #     ts_rank_term2 = decayed_change.rolling(17).apply(ts_rank_func, raw=False)
    #
    #     # 取两项的较大值，再取负值
    #     return -1 * np.maximum(rank_term1, ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_74')
    # @staticmethod
    # def alpha_74(high: pd.Series, close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
    #
    #     收盘价与30日均量37日累积的15日相关系数的排名，是否小于最高价与加权平均价的加权和的排名与成交量排名的11日相关系数的排名，条件成立取-1，否则取0。
    #
    #     Args:
    #         high: 最高价序列
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#74因子值
    #     """
    #     # 计算30日均量的37日累积
    #     adv30 = volume.rolling(30).mean()
    #     sum_adv30 = adv30.rolling(37).sum()
    #
    #     # 计算close与sum_adv30的15日相关系数的排名
    #     corr1 = close.rolling(15).corr(sum_adv30)
    #     rank_corr1 = corr1.rank(pct=True)
    #
    #     # 计算最高价与加权平均价的加权和
    #     weighted_high_vwap = high * 0.0261661 + vwap * (1 - 0.0261661)
    #
    #     # 计算weighted_high_vwap的排名与volume排名的11日相关系数的排名
    #     rank_weighted = weighted_high_vwap.rank(pct=True)
    #     rank_volume = volume.rank(pct=True)
    #
    #     corr2 = rank_weighted.rolling(11).corr(rank_volume)
    #     rank_corr2 = corr2.rank(pct=True)
    #
    #     # 比较两个排名，条件成立取-1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_corr1 < rank_corr2] = -1
    #
    #     return result

    # @BaseFactor.register_factor(name='alpha_75')
    # @staticmethod
    # def alpha_75(low: pd.Series, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
    #
    #     加权平均价与成交量的4日相关系数的排名，是否小于最低价排名与50日均量排名的12日相关系数的排名，条件成立取1，否则取0。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         high: 最高价序列
    #         low: 最低价序列
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#75因子值
    #     """
    #     # 计算vwap与volume的4日相关系数的排名
    #     corr1 = vwap.rolling(4).corr(volume)
    #     rank_corr1 = corr1.rank(pct=True)
    #
    #     # 计算50日均量
    #     adv50 = volume.rolling(50).mean()
    #
    #     # 计算low排名与adv50排名的12日相关系数的排名
    #     rank_low = low.rank(pct=True)
    #     rank_adv50 = adv50.rank(pct=True)
    #
    #     corr2 = rank_low.rolling(12).corr(rank_adv50)
    #     rank_corr2 = corr2.rank(pct=True)
    #
    #     # 比较两个排名，条件成立取1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_corr1 < rank_corr2] = 1
    #
    #     return result

    # @BaseFactor.register_factor(name='alpha_76')
    # @staticmethod
    # def alpha_76(low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
    #
    #     加权平均价的1日变化的12日线性衰减的排名，与行业中性化的最低价与81日均量的8日相关系数的20日时序排名的17日线性衰减的19日时序排名，取两者较大值，再取负值。
    #
    #     Args:
    #         low: 最低价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#76因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算vwap的1日变化
    #     delta_vwap = vwap.diff(1)
    #
    #     # 计算delta_vwap的12日线性衰减的排名
    #     decayed_delta = decay_linear(delta_vwap, 12)
    #     rank_term1 = decayed_delta.rank(pct=True)
    #
    #     # 计算81日均量
    #     adv81 = volume.rolling(81).mean()
    #
    #     # 由于无法实现行业中性化，使用原始low
    #
    #     # 计算low与adv81的8日相关系数
    #     corr = low.rolling(8).corr(adv81)
    #
    #     # 计算相关系数的20日时序排名
    #     ts_rank_corr = corr.rolling(20).apply(ts_rank_func, raw=False)
    #
    #     # 计算ts_rank_corr的17日线性衰减
    #     decayed_ts_rank = decay_linear(ts_rank_corr, 17)
    #
    #     # 计算线性衰减的19日时序排名
    #     ts_rank_term2 = decayed_ts_rank.rolling(19).apply(ts_rank_func, raw=False)
    #
    #     # 取两项的较大值，再取负值
    #     return -1 * np.maximum(rank_term1, ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_77')
    # @staticmethod
    # def alpha_77(high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#77: (min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125))) * -1)
    #
    #     价格因子的20日线性衰减的排名，与最高价最低价均值与40日均量的3日相关系数的6日线性衰减的排名，取两者较小值，再取负值。
    #
    #     Args:
    #         high: 最高价序列
    #         low: 最低价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#77因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 计算(high + low) / 2
    #     price_avg = (high + low) / 2
    #
    #     # 计算价格因子
    #     price_factor = ((price_avg + high) - (vwap + high))
    #
    #     # 计算price_factor的20日线性衰减的排名
    #     decayed_price = decay_linear(price_factor, 20)
    #     rank_term1 = decayed_price.rank(pct=True)
    #
    #     # 计算40日均量
    #     adv40 = volume.rolling(40).mean()
    #
    #     # 计算price_avg与adv40的3日相关系数
    #     corr = price_avg.rolling(3).corr(adv40)
    #
    #     # 计算corr的6日线性衰减的排名
    #     decayed_corr = decay_linear(corr, 6)
    #     rank_term2 = decayed_corr.rank(pct=True)
    #
    #     # 取两项的较小值，再取负值
    #     return -1 * np.minimum(rank_term1, rank_term2)

    # @BaseFactor.register_factor(name='alpha_78')
    # @staticmethod
    # def alpha_78(low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    #
    #     最低价与加权平均价的加权和的20日累积与40日均量20日累积的7日相关系数的排名，的加权平均价排名与成交量排名的6日相关系数的排名次方。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         high: 最高价序列
    #         low: 最低价序列
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#78因子值
    #     """
    #     # 计算最低价与加权平均价的加权和
    #     weighted_low_vwap = low * 0.352233 + vwap * (1 - 0.352233)
    #
    #     # 计算weighted_low_vwap的20日累积
    #     sum_weighted = weighted_low_vwap.rolling(20).sum()
    #
    #     # 计算40日均量的20日累积
    #     adv40 = volume.rolling(40).mean()
    #     sum_adv40 = adv40.rolling(20).sum()
    #
    #     # 计算sum_weighted与sum_adv40的7日相关系数的排名
    #     corr1 = sum_weighted.rolling(7).corr(sum_adv40)
    #     rank_corr1 = corr1.rank(pct=True)
    #
    #     # 计算vwap排名与volume排名的6日相关系数的排名
    #     rank_vwap = vwap.rank(pct=True)
    #     rank_volume = volume.rank(pct=True)
    #
    #     corr2 = rank_vwap.rolling(6).corr(rank_volume)
    #     rank_corr2 = corr2.rank(pct=True)
    #
    #     # 计算rank_corr1的rank_corr2次方
    #     return rank_corr1 ** rank_corr2

    # @BaseFactor.register_factor(name='alpha_79')
    # @staticmethod
    # def alpha_79(open_price: pd.Series, close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#79: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
    #
    #     收盘价与开盘价加权和的行业中性化的1日变化的排名，是否小于加权平均价的4日时序排名与150日均量的9日时序排名的15日相关系数的排名，条件成立取1，否则取0。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         high: 最高价序列
    #         low: 最低价序列
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#79因子值
    #     """
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算收盘价与开盘价的加权和
    #     weighted_close_open = close * 0.60733 + open_price * (1 - 0.60733)
    #
    #     # 由于无法实现行业中性化，使用原始weighted_close_open
    #
    #     # 计算weighted_close_open的1日变化的排名
    #     delta_weighted = weighted_close_open.diff(1)
    #     rank_delta = delta_weighted.rank(pct=True)
    #
    #     # 计算vwap的4日时序排名
    #     ts_rank_vwap = vwap.rolling(4).apply(ts_rank_func, raw=False)
    #
    #     # 计算150日均量
    #     adv150 = volume.rolling(150).mean()
    #
    #     # 计算adv150的9日时序排名
    #     ts_rank_adv150 = adv150.rolling(9).apply(ts_rank_func, raw=False)
    #
    #     # 计算两个时序排名的15日相关系数的排名
    #     corr = ts_rank_vwap.rolling(15).corr(ts_rank_adv150)
    #     rank_corr = corr.rank(pct=True)
    #
    #     # 比较两个排名，条件成立取1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_delta < rank_corr] = 1
    #
    #     return result

    @BaseFactor.register_factor(name='alpha_80')
    @staticmethod
    def alpha_80(open_price: pd.Series, high: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#80: ((rank(sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
        
        开盘价与最高价加权和的行业中性化的4日变化符号的排名的时序相关性次方，取负值。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#80因子值
        """
        # 时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        # 计算开盘价与最高价的加权和
        weighted_open_high = open_price * 0.868128 + high * (1 - 0.868128)
        
        # 由于无法实现行业中性化，使用原始weighted_open_high
        
        # 计算weighted_open_high的4日变化符号的排名
        delta_weighted = weighted_open_high.diff(4)
        sign_delta = np.sign(delta_weighted)
        rank_sign = sign_delta.rank(pct=True)
        
        # 计算10日均量
        adv10 = volume.rolling(10).mean()
        
        # 计算high与adv10的5日相关系数
        corr = high.rolling(5).corr(adv10)
        
        # 计算相关系数的6日时序排名
        ts_rank_corr = corr.rolling(6).apply(ts_rank_func, raw=False)
        
        # 计算rank_sign的ts_rank_corr次方，取负值
        return -1 * (rank_sign ** ts_rank_corr)

    # @BaseFactor.register_factor(name='alpha_81')
    # @staticmethod
    # def alpha_81(close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    #
    #     加权平均价与10日均量50日累积的8日相关系数的排名的4次方的15日乘积取对数的排名，是否小于加权平均价排名与成交量排名的5日相关系数的排名，条件成立取-1，否则取0。
    #
    #     Args:
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#81因子值
    #     """
    #     # 计算10日均量
    #     adv10 = volume.rolling(10).mean()
    #
    #     # 计算adv10的50日累积
    #     sum_adv10 = adv10.rolling(50).sum()
    #
    #     # 计算vwap与sum_adv10的8日相关系数
    #     corr1 = vwap.rolling(8).corr(sum_adv10)
    #
    #     # 计算相关系数的排名的4次方
    #     rank_corr1 = corr1.rank(pct=True) ** 4
    #
    #     # 计算rank_corr1的15日乘积的对数的排名
    #     def product(x):
    #         return np.prod(x)
    #
    #     prod_rank = rank_corr1.rolling(15).apply(product, raw=True)
    #     log_prod = np.log(prod_rank + 1e-12)  # 加小值避免log(0)
    #     rank_log_prod = log_prod.rank(pct=True)
    #
    #     # 计算vwap排名与volume排名的5日相关系数的排名
    #     rank_vwap = vwap.rank(pct=True)
    #     rank_volume = volume.rank(pct=True)
    #
    #     corr2 = rank_vwap.rolling(5).corr(rank_volume)
    #     rank_corr2 = corr2.rank(pct=True)
    #
    #     # 比较两个排名，条件成立取-1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[rank_log_prod < rank_corr2] = -1
    #
    #     return result

    @BaseFactor.register_factor(name='alpha_82')
    @staticmethod
    def alpha_82(open_price: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
        
        开盘价的1日变化的15日线性衰减的排名，与行业中性化的成交量与开盘价的17日相关系数的7日线性衰减的13日时序排名，取两者较小值，再取负值。
        
        Args:
            open_price: 开盘价序列
            volume: 成交量序列
        Returns:
            Alpha#82因子值
        """
        # 线性衰减函数
        def decay_linear(series, window):
            weights = np.arange(1, window + 1) / window
            weights = weights[::-1]  # 反转权重使最近的观测值权重最大
            
            result = pd.Series(index=series.index)
            for i in range(window-1, len(series)):
                if i < window-1:
                    continue
                result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
            return result
        
        # 时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        # 计算开盘价的1日变化
        delta_open = open_price.diff(1)
        
        # 计算delta_open的15日线性衰减的排名
        decayed_delta = decay_linear(delta_open, 15)
        rank_term1 = decayed_delta.rank(pct=True)
        
        # 由于无法实现行业中性化，使用原始volume
        
        # 计算volume与open的17日相关系数
        # 注意：(open * 0.634196) + (open * (1 - 0.634196)) 就等于open
        corr = volume.rolling(17).corr(open_price)
        
        # 计算相关系数的7日线性衰减
        decayed_corr = decay_linear(corr, 7)
        
        # 计算线性衰减的13日时序排名
        ts_rank_term2 = decayed_corr.rolling(13).apply(ts_rank_func, raw=False)
        
        # 取两项的较小值，再取负值
        return -1 * np.minimum(rank_term1, ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_83')
    # @staticmethod
    # def alpha_83(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    #
    #     振幅与5日均价的比值的2日延迟的排名乘以成交量排名的排名，除以振幅与5日均价的比值除以加权平均价与收盘价之差。
    #
    #     Args:
    #         high: 最高价序列
    #         low: 最低价序列
    #         close: 收盘价序列
    #         volume: 成交量序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#83因子值
    #     """
    #     # 计算振幅
    #     range_hl = high - low
    #
    #     # 计算5日均价
    #     mean_close_5 = close.rolling(5).mean()
    #
    #     # 计算振幅与5日均价的比值
    #     ratio = range_hl / mean_close_5
    #
    #     # 计算ratio的2日延迟的排名
    #     delay_ratio = ratio.shift(2)
    #     rank_delay = delay_ratio.rank(pct=True)
    #
    #     # 计算成交量排名的排名
    #     rank_volume = volume.rank(pct=True).rank(pct=True)
    #
    #     # 计算分子
    #     numerator = rank_delay * rank_volume
    #
    #     # 计算分母
    #     denominator = ratio / (vwap - close + 1e-12)
    #
    #     # 计算比值
    #     return numerator / (denominator + 1e-12)

    # @BaseFactor.register_factor(name='alpha_84')
    # @staticmethod
    # def alpha_84(close: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
    #
    #     加权平均价与其15日最大值之差的21日时序排名，的收盘价5日变化次方。
    #
    #     Args:
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#84因子值
    #     """
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算vwap的15日最大值
    #     ts_max_vwap = vwap.rolling(15).max()
    #
    #     # 计算vwap与其最大值之差
    #     diff_vwap = vwap - ts_max_vwap
    #
    #     # 计算diff_vwap的21日时序排名
    #     ts_rank_diff = diff_vwap.rolling(21).apply(ts_rank_func, raw=False)
    #
    #     # 计算收盘价的5日变化
    #     delta_close = close.diff(5)
    #
    #     # 计算ts_rank_diff的delta_close次方
    #     # SignedPower函数保持基数的符号，并将指数应用于其绝对值
    #     def signed_power(base, exponent):
    #         return np.sign(base) * (np.abs(base) ** exponent)
    #
    #     return signed_power(ts_rank_diff, delta_close)

    @BaseFactor.register_factor(name='alpha_85')
    @staticmethod
    def alpha_85(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
        
        最高价与收盘价加权和与30日均量的10日相关系数的排名的收盘价位置的4日时序排名与成交量的10日时序排名的7日相关系数的排名次方。
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#85因子值
        """
        # 时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        # 计算最高价与收盘价的加权和
        weighted_high_close = high * 0.876703 + close * (1 - 0.876703)
        
        # 计算30日均量
        adv30 = volume.rolling(30).mean()
        
        # 计算weighted_high_close与adv30的10日相关系数的排名
        corr1 = weighted_high_close.rolling(10).corr(adv30)
        rank_corr1 = corr1.rank(pct=True)
        
        # 计算(high + low) / 2
        price_avg = (high + low) / 2
        
        # 计算price_avg的4日时序排名
        ts_rank_price = price_avg.rolling(4).apply(ts_rank_func, raw=False)
        
        # 计算volume的10日时序排名
        ts_rank_volume = volume.rolling(10).apply(ts_rank_func, raw=False)
        
        # 计算两个时序排名的7日相关系数的排名
        corr2 = ts_rank_price.rolling(7).corr(ts_rank_volume)
        rank_corr2 = corr2.rank(pct=True)
        
        # 计算rank_corr1的rank_corr2次方
        return rank_corr1 ** rank_corr2

    # @BaseFactor.register_factor(name='alpha_86')
    # @staticmethod
    # def alpha_86(open_price: pd.Series, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
    #
    #     收盘价与20日均量15日累积的6日相关系数的20日时序排名，是否小于(开盘价+收盘价)-(加权平均价+开盘价)的排名，条件成立取-1，否则取0。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         close: 收盘价序列
    #         volume: 成交量序列
    #         vwap: 成交量加权平均价序列
    #     Returns:
    #         Alpha#86因子值
    #     """
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算20日均量
    #     adv20 = volume.rolling(20).mean()
    #
    #     # 计算adv20的15日累积
    #     sum_adv20 = adv20.rolling(15).sum()
    #
    #     # 计算close与sum_adv20的6日相关系数
    #     corr = close.rolling(6).corr(sum_adv20)
    #
    #     # 计算相关系数的20日时序排名
    #     ts_rank_corr = corr.rolling(20).apply(ts_rank_func, raw=False)
    #
    #     # 计算(open + close) - (vwap + open)
    #     term = (open_price + close) - (vwap + open_price)
    #
    #     # 计算term的排名
    #     rank_term = term.rank(pct=True)
    #
    #     # 比较两个排名，条件成立取-1，否则取0
    #     result = pd.Series(0, index=close.index)
    #     result[ts_rank_corr < rank_term] = -1
    #
    #     return result

    # @BaseFactor.register_factor(name='alpha_87')
    # @staticmethod
    # def alpha_87(open_price: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
    #     """
    #     Alpha#87: ((rank(decay_linear(delta(vwap, 4.72775), 5.87851)) + Ts_Rank(decay_linear(((((close * 0.485) + (vwap * (1 - 0.485))) - close) / close), 3.67975), 15.3522)) / (rank(decay_linear(correlation(IndNeutralize(adv20, IndClass.industry), low, 4.87219), 10.332)) * rank(decay_linear(delta(((vwap * 0.369701) + (open * (1 - 0.369701))), 2.15146), 3.33666))))
    #
    #     加权平均价的5日变化的6日线性衰减的排名，加上收盘价与加权平均价加权和的相对偏差的3日线性衰减的15日时序排名，除以行业中性化的20日均量与最低价的5日相关系数的10日线性衰减的排名乘以加权平均价与开盘价加权和的2日变化的3日线性衰减的排名。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         high: 最高价序列
    #         low: 最低价序列
    #         close: 收盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#87因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算vwap的5日变化的6日线性衰减的排名
    #     delta_vwap = vwap.diff(5)
    #     decayed_delta = decay_linear(delta_vwap, 6)
    #     rank_term1 = decayed_delta.rank(pct=True)
    #
    #     # 计算收盘价与加权平均价的加权和
    #     weighted_close_vwap = close * 0.485 + vwap * (1 - 0.485)
    #
    #     # 计算相对偏差
    #     deviation = (weighted_close_vwap - close) / (close + 1e-12)
    #
    #     # 计算deviation的3日线性衰减的15日时序排名
    #     decayed_dev = decay_linear(deviation, 4)
    #     ts_rank_dev = decayed_dev.rolling(15).apply(ts_rank_func, raw=False)
    #
    #     # 计算20日均量
    #     adv20 = volume.rolling(20).mean()
    #
    #     # 由于无法实现行业中性化，使用原始adv20
    #
    #     # 计算adv20与low的5日相关系数的10日线性衰减的排名
    #     corr = adv20.rolling(5).corr(low)
    #     decayed_corr = decay_linear(corr, 10)
    #     rank_term3 = decayed_corr.rank(pct=True)
    #
    #     # 计算vwap与open的加权和
    #     weighted_vwap_open = vwap * 0.369701 + open_price * (1 - 0.369701)
    #
    #     # 计算weighted_vwap_open的2日变化的3日线性衰减的排名
    #     delta_weighted = weighted_vwap_open.diff(2)
    #     decayed_delta2 = decay_linear(delta_weighted, 3)
    #     rank_term4 = decayed_delta2.rank(pct=True)
    #
    #     # 计算分子和分母
    #     numerator = rank_term1 + ts_rank_dev
    #     denominator = rank_term3 * rank_term4
    #
    #     # 计算比值
    #     return numerator / (denominator + 1e-12)

    @BaseFactor.register_factor(name='alpha_88')
    @staticmethod
    def alpha_88(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
        
        开盘价排名加最低价排名减最高价排名加收盘价排名的8日线性衰减的排名，与收盘价的8日时序排名与60日均量的21日时序排名的8日相关系数的7日线性衰减的3日时序排名，取两者较小值。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#88因子值
        """
        # 线性衰减函数
        def decay_linear(series, window):
            weights = np.arange(1, window + 1) / window
            weights = weights[::-1]  # 反转权重使最近的观测值权重最大
            
            result = pd.Series(index=series.index)
            for i in range(window-1, len(series)):
                if i < window-1:
                    continue
                result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
            return result
        
        # 时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        # 计算价格因子
        rank_open = open_price.rank(pct=True)
        rank_low = low.rank(pct=True)
        rank_high = high.rank(pct=True)
        rank_close = close.rank(pct=True)
        
        price_factor = (rank_open + rank_low) - (rank_high + rank_close)
        
        # 计算price_factor的8日线性衰减的排名
        decayed_price = decay_linear(price_factor, 8)
        rank_term1 = decayed_price.rank(pct=True)
        
        # 计算close的8日时序排名
        ts_rank_close = close.rolling(8).apply(ts_rank_func, raw=False)
        
        # 计算60日均量
        adv60 = volume.rolling(60).mean()
        
        # 计算adv60的21日时序排名
        ts_rank_adv60 = adv60.rolling(21).apply(ts_rank_func, raw=False)
        
        # 计算两个时序排名的8日相关系数
        corr = ts_rank_close.rolling(8).corr(ts_rank_adv60)
        
        # 计算相关系数的7日线性衰减
        decayed_corr = decay_linear(corr, 7)
        
        # 计算线性衰减的3日时序排名
        ts_rank_term2 = decayed_corr.rolling(3).apply(ts_rank_func, raw=False)
        
        # 取两项的较小值
        return np.minimum(rank_term1, ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_89')
    # @staticmethod
    # def alpha_89(low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
    #
    #     最低价与10日均量的7日相关系数的6日线性衰减的4日时序排名，减去行业中性化的加权平均价的3日变化的10日线性衰减的15日时序排名。
    #
    #     Args:
    #         low: 最低价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#89因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算10日均量
    #     adv10 = volume.rolling(10).mean()
    #
    #     # 注意：(low * 0.967285) + (low * (1 - 0.967285)) 就等于low
    #
    #     # 计算low与adv10的7日相关系数
    #     corr = low.rolling(7).corr(adv10)
    #
    #     # 计算相关系数的6日线性衰减的4日时序排名
    #     decayed_corr = decay_linear(corr, 6)
    #     ts_rank_term1 = decayed_corr.rolling(4).apply(ts_rank_func, raw=False)
    #
    #     # 由于无法实现行业中性化，使用原始vwap
    #
    #     # 计算vwap的3日变化
    #     delta_vwap = vwap.diff(3)
    #
    #     # 计算delta_vwap的10日线性衰减的15日时序排名
    #     decayed_delta = decay_linear(delta_vwap, 10)
    #     ts_rank_term2 = decayed_delta.rolling(15).apply(ts_rank_func, raw=False)
    #
    #     # 计算差值
    #     return ts_rank_term1 - ts_rank_term2

    @BaseFactor.register_factor(name='alpha_90')
    @staticmethod
    def alpha_90(low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
        
        收盘价与其5日最大值之差的排名的时序相关性次方，取负值。
        
        Args:
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#90因子值
        """
        # 时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        # 计算close的5日最大值
        ts_max_close = close.rolling(5).max()
        
        # 计算close与其最大值之差的排名
        rank_term1 = (close - ts_max_close).rank(pct=True)
        
        # 计算40日均量
        adv40 = volume.rolling(40).mean()
        
        # 由于无法实现行业中性化，使用原始adv40
        
        # 计算adv40与low的5日相关系数
        corr = adv40.rolling(5).corr(low)
        
        # 计算相关系数的3日时序排名
        ts_rank_term2 = corr.rolling(3).apply(ts_rank_func, raw=False)
        
        # 计算rank_term1的ts_rank_term2次方，取负值
        return -1 * (rank_term1 ** ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_91')
    # @staticmethod
    # def alpha_91(close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    #
    #     行业中性化的收盘价与成交量的10日相关系数的16日线性衰减的4日线性衰减的5日时序排名，减去加权平均价与30日均量的4日相关系数的3日线性衰减的排名，取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#91因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 由于无法实现行业中性化，使用原始close
    #
    #     # 计算close与volume的10日相关系数
    #     corr1 = close.rolling(10).corr(volume)
    #
    #     # 计算相关系数的16日线性衰减
    #     decayed_corr1 = decay_linear(corr1, 16)
    #
    #     # 计算decayed_corr1的4日线性衰减
    #     decayed_twice = decay_linear(decayed_corr1, 4)
    #
    #     # 计算decayed_twice的5日时序排名
    #     ts_rank_term1 = decayed_twice.rolling(5).apply(ts_rank_func, raw=False)
    #
    #     # 计算30日均量
    #     adv30 = volume.rolling(30).mean()
    #
    #     # 计算vwap与adv30的4日相关系数
    #     corr2 = vwap.rolling(4).corr(adv30)
    #
    #     # 计算相关系数的3日线性衰减的排名
    #     decayed_corr2 = decay_linear(corr2, 3)
    #     rank_term2 = decayed_corr2.rank(pct=True)
    #
    #     # 计算差值，取负值
    #     return -1 * (ts_rank_term1 - rank_term2)

    @BaseFactor.register_factor(name='alpha_92')
    @staticmethod
    def alpha_92(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
        
        (最高价与最低价均值加收盘价)小于(最低价加开盘价)的14日线性衰减的19日时序排名，与最低价排名与30日均量排名的8日相关系数的7日线性衰减的7日时序排名，取两者较小值。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#92因子值
        """
        # 线性衰减函数
        def decay_linear(series, window):
            weights = np.arange(1, window + 1) / window
            weights = weights[::-1]  # 反转权重使最近的观测值权重最大
            
            result = pd.Series(index=series.index)
            for i in range(window-1, len(series)):
                if i < window-1:
                    continue
                result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
            return result
        
        # 时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        # 计算价格条件
        condition = (((high + low) / 2) + close) < (low + open_price)
        condition = condition.astype(int)
        
        # 计算condition的15日线性衰减的19日时序排名
        decayed_cond = decay_linear(condition, 15)
        ts_rank_term1 = decayed_cond.rolling(19).apply(ts_rank_func, raw=False)
        
        # 计算30日均量
        adv30 = volume.rolling(30).mean()
        
        # 计算low排名与adv30排名的8日相关系数
        rank_low = low.rank(pct=True)
        rank_adv30 = adv30.rank(pct=True)
        
        corr = rank_low.rolling(8).corr(rank_adv30)
        
        # 计算相关系数的7日线性衰减的7日时序排名
        decayed_corr = decay_linear(corr, 7)
        ts_rank_term2 = decayed_corr.rolling(7).apply(ts_rank_func, raw=False)
        
        # 取两项的较小值
        return np.minimum(ts_rank_term1, ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_93')
    # @staticmethod
    # def alpha_93(close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
    #
    #     行业中性化的加权平均价与81日均量的17日相关系数的20日线性衰减的8日时序排名，除以收盘价与加权平均价加权和的3日变化的16日线性衰减的排名。
    #
    #     Args:
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#93因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 由于无法实现行业中性化，使用原始vwap
    #
    #     # 计算81日均量
    #     adv81 = volume.rolling(81).mean()
    #
    #     # 计算vwap与adv81的17日相关系数
    #     corr = vwap.rolling(17).corr(adv81)
    #
    #     # 计算相关系数的20日线性衰减的8日时序排名
    #     decayed_corr = decay_linear(corr, 20)
    #     ts_rank_term1 = decayed_corr.rolling(8).apply(ts_rank_func, raw=False)
    #
    #     # 计算收盘价与vwap的加权和
    #     weighted_close_vwap = close * 0.524434 + vwap * (1 - 0.524434)
    #
    #     # 计算weighted_close_vwap的3日变化
    #     delta_weighted = weighted_close_vwap.diff(3)
    #
    #     # 计算delta_weighted的16日线性衰减的排名
    #     decayed_delta = decay_linear(delta_weighted, 16)
    #     rank_term2 = decayed_delta.rank(pct=True)
    #
    #     # 计算比值
    #     return ts_rank_term1 / (rank_term2 + 1e-12)

    # @BaseFactor.register_factor(name='alpha_94')
    # @staticmethod
    # def alpha_94(vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    #
    #     加权平均价与其12日最小值之差的排名的时序相关性次方，取负值。
    #
    #     Args:
    #         high: 最高价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#94因子值
    #     """
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算vwap的12日最小值
    #     ts_min_vwap = vwap.rolling(12).min()
    #
    #     # 计算vwap与其最小值之差的排名
    #     rank_term1 = (vwap - ts_min_vwap).rank(pct=True)
    #
    #     # 计算vwap的20日时序排名
    #     ts_rank_vwap = vwap.rolling(20).apply(ts_rank_func, raw=False)
    #
    #     # 计算60日均量
    #     adv60 = volume.rolling(60).mean()
    #
    #     # 计算adv60的4日时序排名
    #     ts_rank_adv60 = adv60.rolling(4).apply(ts_rank_func, raw=False)
    #
    #     # 计算两个时序排名的18日相关系数
    #     corr = ts_rank_vwap.rolling(18).corr(ts_rank_adv60)
    #
    #     # 计算相关系数的3日时序排名
    #     ts_rank_corr = corr.rolling(3).apply(ts_rank_func, raw=False)
    #
    #     # 计算rank_term1的ts_rank_corr次方，取负值
    #     return -1 * (rank_term1 ** ts_rank_corr)

    @BaseFactor.register_factor(name='alpha_95')
    @staticmethod
    def alpha_95(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank(rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742)), 11.7584))
        
        开盘价与其12日最小值之差的排名，是否小于最高价与最低价均值的19日累积与40日均量19日累积的13日相关系数的排名的12日时序排名，条件成立取1，否则取0。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#95因子值
        """
        # 时序排名函数
        def ts_rank_func(x):
            return pd.Series(x).rank(pct=True).iloc[-1]
        
        # 计算open的12日最小值
        ts_min_open = open_price.rolling(12).min()
        
        # 计算open与其最小值之差的排名
        rank_term1 = (open_price - ts_min_open).rank(pct=True)
        
        # 计算(high + low) / 2的19日累积
        price_avg = (high + low) / 2
        sum_price = price_avg.rolling(19).sum()
        
        # 计算40日均量的19日累积
        adv40 = volume.rolling(40).mean()
        sum_adv40 = adv40.rolling(19).sum()
        
        # 计算sum_price与sum_adv40的13日相关系数
        corr = sum_price.rolling(13).corr(sum_adv40)
        
        # 计算相关系数的排名的12日时序排名
        rank_corr = corr.rank(pct=True)
        ts_rank_term2 = rank_corr.rolling(12).apply(ts_rank_func, raw=False)
        
        # 比较两个排名，条件成立取1，否则取0
        result = pd.Series(0, index=close.index)
        result[rank_term1 < ts_rank_term2] = 1
        
        return result

    # @BaseFactor.register_factor(name='alpha_96')
    # @staticmethod
    # def alpha_96(close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    #
    #     加权平均价排名与成交量排名的4日相关系数的4日线性衰减的8日时序排名，与收盘价7日时序排名与60日均量4日时序排名的4日相关系数的13日最大值位置的14日线性衰减的13日时序排名，取两者较大值，再取负值。
    #
    #     Args:
    #         close: 收盘价序列
    #         vwap: 成交量加权平均价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#96因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 时序最大值位置函数
    #     def ts_argmax(x):
    #         return np.argmax(x[-13:]) if len(x) >= 13 else np.nan
    #
    #     # 计算vwap排名与volume排名的4日相关系数
    #     rank_vwap = vwap.rank(pct=True)
    #     rank_volume = volume.rank(pct=True)
    #
    #     corr1 = rank_vwap.rolling(4).corr(rank_volume)
    #
    #     # 计算相关系数的4日线性衰减的8日时序排名
    #     decayed_corr1 = decay_linear(corr1, 4)
    #     ts_rank_term1 = decayed_corr1.rolling(8).apply(ts_rank_func, raw=False)
    #
    #     # 计算close的7日时序排名
    #     ts_rank_close = close.rolling(7).apply(ts_rank_func, raw=False)
    #
    #     # 计算60日均量
    #     adv60 = volume.rolling(60).mean()
    #
    #     # 计算adv60的4日时序排名
    #     ts_rank_adv60 = adv60.rolling(4).apply(ts_rank_func, raw=False)
    #
    #     # 计算两个时序排名的4日相关系数
    #     corr2 = ts_rank_close.rolling(4).corr(ts_rank_adv60)
    #
    #     # 计算相关系数的13日最大值位置
    #     argmax_corr = corr2.rolling(13).apply(ts_argmax, raw=True)
    #
    #     # 计算argmax_corr的14日线性衰减的13日时序排名
    #     decayed_argmax = decay_linear(argmax_corr, 14)
    #     ts_rank_term2 = decayed_argmax.rolling(13).apply(ts_rank_func, raw=False)
    #
    #     # 取两项的较大值，再取负值
    #     return -1 * np.maximum(ts_rank_term1, ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_97')
    # @staticmethod
    # def alpha_97(low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    #
    #     最低价与加权平均价加权和的行业中性化的3日变化的20日线性衰减的排名，减去最低价8日时序排名与60日均量17日时序排名的5日相关系数的19日时序排名的16日线性衰减的7日时序排名，取负值。
    #
    #     Args:
    #         low: 最低价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#97因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 计算最低价与vwap的加权和
    #     weighted_low_vwap = low * 0.721001 + vwap * (1 - 0.721001)
    #
    #     # 由于无法实现行业中性化，使用原始weighted_low_vwap
    #
    #     # 计算weighted_low_vwap的3日变化
    #     delta_weighted = weighted_low_vwap.diff(3)
    #
    #     # 计算delta_weighted的20日线性衰减的排名
    #     decayed_delta = decay_linear(delta_weighted, 20)
    #     rank_term1 = decayed_delta.rank(pct=True)
    #
    #     # 计算low的8日时序排名
    #     ts_rank_low = low.rolling(8).apply(ts_rank_func, raw=False)
    #
    #     # 计算60日均量
    #     adv60 = volume.rolling(60).mean()
    #
    #     # 计算adv60的17日时序排名
    #     ts_rank_adv60 = adv60.rolling(17).apply(ts_rank_func, raw=False)
    #
    #     # 计算两个时序排名的5日相关系数
    #     corr = ts_rank_low.rolling(5).corr(ts_rank_adv60)
    #
    #     # 计算相关系数的19日时序排名
    #     ts_rank_corr = corr.rolling(19).apply(ts_rank_func, raw=False)
    #
    #     # 计算ts_rank_corr的16日线性衰减的7日时序排名
    #     decayed_ts_rank = decay_linear(ts_rank_corr, 16)
    #     ts_rank_term2 = decayed_ts_rank.rolling(7).apply(ts_rank_func, raw=False)
    #
    #     # 计算差值，取负值
    #     return -1 * (rank_term1 - ts_rank_term2)

    # @BaseFactor.register_factor(name='alpha_98')
    # @staticmethod
    # def alpha_98(open_price: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
    #     """
    #     Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
    #
    #     加权平均价与5日均量26日累积的5日相关系数的7日线性衰减的排名，减去开盘价排名与15日均量排名的21日相关系数的9日最小值位置的7日时序排名的8日线性衰减的排名。
    #
    #     Args:
    #         open_price: 开盘价序列
    #         volume: 成交量序列
    #     Returns:
    #         Alpha#98因子值
    #     """
    #     # 线性衰减函数
    #     def decay_linear(series, window):
    #         weights = np.arange(1, window + 1) / window
    #         weights = weights[::-1]  # 反转权重使最近的观测值权重最大
    #
    #         result = pd.Series(index=series.index)
    #         for i in range(window-1, len(series)):
    #             if i < window-1:
    #                 continue
    #             result.iloc[i] = np.nansum(series.iloc[i-window+1:i+1].values * weights)
    #         return result
    #
    #     # 时序排名函数
    #     def ts_rank_func(x):
    #         return pd.Series(x).rank(pct=True).iloc[-1]
    #
    #     # 时序最小值位置函数
    #     def ts_argmin(x):
    #         return np.argmin(x[-9:]) if len(x) >= 9 else np.nan
    #
    #     # 计算5日均量
    #     adv5 = volume.rolling(5).mean()
    #
    #     # 计算adv5的26日累积
    #     sum_adv5 = adv5.rolling(26).sum()
    #
    #     # 计算vwap与sum_adv5的5日相关系数
    #     corr1 = vwap.rolling(5).corr(sum_adv5)
    #
    #     # 计算相关系数的7日线性衰减的排名
    #     decayed_corr1 = decay_linear(corr1, 7)
    #     rank_term1 = decayed_corr1.rank(pct=True)
    #
    #     # 计算open排名与adv15排名的21日相关系数
    #     rank_open = open_price.rank(pct=True)
    #
    #     # 计算15日均量
    #     adv15 = volume.rolling(15).mean()
    #
    #     rank_adv15 = adv15.rank(pct=True)
    #
    #     corr2 = rank_open.rolling(21).corr(rank_adv15)
    #
    #     # 计算相关系数的9日最小值位置
    #     argmin_corr = corr2.rolling(9).apply(ts_argmin, raw=True)
    #
    #     # 计算argmin_corr的7日时序排名
    #     ts_rank_argmin = argmin_corr.rolling(7).apply(ts_rank_func, raw=False)
    #
    #     # 计算ts_rank_argmin的8日线性衰减的排名
    #     decayed_ts_rank = decay_linear(ts_rank_argmin, 8)
    #     rank_term2 = decayed_ts_rank.rank(pct=True)
    #
    #     # 计算差值
    #     return rank_term1 - rank_term2

    @BaseFactor.register_factor(name='alpha_99')
    @staticmethod
    def alpha_99(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
        
        最高价与最低价均值的20日累积与60日均量20日累积的9日相关系数的排名，是否小于最低价与成交量的6日相关系数的排名，条件成立取-1，否则取0。
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#99因子值
        """
        # 计算(high + low) / 2的20日累积
        price_avg = (high + low) / 2
        sum_price = price_avg.rolling(20).sum()
        
        # 计算60日均量
        adv60 = volume.rolling(60).mean()
        
        # 计算adv60的20日累积
        sum_adv60 = adv60.rolling(20).sum()
        
        # 计算sum_price与sum_adv60的9日相关系数的排名
        corr1 = sum_price.rolling(9).corr(sum_adv60)
        rank_corr1 = corr1.rank(pct=True)
        
        # 计算low与volume的6日相关系数的排名
        corr2 = low.rolling(6).corr(volume)
        rank_corr2 = corr2.rank(pct=True)
        
        # 比较两个排名，条件成立取-1，否则取0
        result = pd.Series(0, index=close.index)
        result[rank_corr1 < rank_corr2] = -1
        
        return result

    @BaseFactor.register_factor(name='alpha_100')
    @staticmethod
    def alpha_100(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
        
        价格位置指标乘以成交量的行业中性化的1.5倍，减去收盘价与20日均量排名的5日相关系数与收盘价30日最小值位置排名之差的行业中性化，再乘以成交量与20日均量之比。
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#100因子值
        """
        # 标准化函数
        def scale(x):
            return (x - x.mean()) / (x.std() + 1e-12)
        
        # 时序最小值位置函数
        def ts_argmin(x):
            return np.argmin(x[-30:]) if len(x) >= 30 else np.nan
        
        # 计算价格位置指标
        price_position = ((close - low) - (high - close)) / (high - low + 1e-12)
        price_vol = price_position * volume
        
        # 计算price_vol的排名
        rank_price_vol = price_vol.rank(pct=True)
        
        # 由于无法实现行业中性化，使用原始rank_price_vol并进行标准化
        scale_term1 = scale(rank_price_vol) * 1.5
        
        # 计算20日均量
        adv20 = volume.rolling(20).mean()
        
        # 计算adv20的排名
        rank_adv20 = adv20.rank(pct=True)
        
        # 计算close与rank_adv20的5日相关系数
        corr = close.rolling(5).corr(rank_adv20)
        
        # 计算close的30日最小值位置的排名
        argmin_close = close.rolling(30).apply(ts_argmin, raw=True)
        rank_argmin = argmin_close.rank(pct=True)
        
        # 计算corr与rank_argmin的差
        diff = corr - rank_argmin
        
        # 由于无法实现行业中性化，使用原始diff并进行标准化
        scale_term2 = scale(diff)
        
        # 计算成交量与20日均量之比
        volume_ratio = volume / (adv20 + 1e-12)
        
        # 计算最终结果
        return -1 * ((scale_term1 - scale_term2) * volume_ratio)

    @BaseFactor.register_factor(name='alpha_101')
    @staticmethod
    def alpha_101(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#101: ((close - open) / ((high - low) + .001))
        
        收盘价减开盘价，除以振幅加0.001，即日间收益率相对于日内波动的比值。
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
        Returns:
            Alpha#101因子值
        """
        return (close - open_price) / ((high - low) + 0.001)


def get_registered_factors() -> Dict[str, Callable]:
    """获取所有已注册的因子"""
    return FACTOR_REGISTRY


def get_factor_by_type(factor_type: str) -> Dict[str, Callable]:
    """
    按类型获取因子

    Args:
        factor_type: 因子类型名称（如 'momentum', 'volatility' 等）
    Returns:
        该类型的所有因子字典
    """
    return {name: func for name, func in FACTOR_REGISTRY.items()
            if name.startswith(factor_type.lower())}
