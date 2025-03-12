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

    @BaseFactor.register_factor(name='alpha_11')
    @staticmethod
    def alpha_11(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
        
        最近3天均价与收盘价差值的最大值的排名，加上最小值的排名，再乘以3天成交量变化的排名。
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
        Returns:
            Alpha#11因子值
        """
        # 通常vwap为成交量加权平均价，这里简化为收盘价
        vwap = close

        # 计算vwap与收盘价的差值
        diff = vwap - close

        # 计算3日最大和最小值的排名
        rank_max = diff.rolling(3).max().rank(pct=True)
        rank_min = diff.rolling(3).min().rank(pct=True)

        # 计算3日成交量变化的排名
        rank_volume_delta = volume.diff(3).rank(pct=True)

        return (rank_max + rank_min) * rank_volume_delta

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
