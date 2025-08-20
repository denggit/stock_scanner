#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据工具类
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from backend.utils.logger import setup_logger

# 导入数据获取器
try:
    from backend.business.data.data_fetcher import StockDataFetcher
except ImportError:
    # 如果导入失败，设置为None，在方法中处理
    StockDataFetcher = None


class DataUtils:
    """
    数据工具类
    提供数据处理和生成功能
    """

    @staticmethod
    def calculate_moving_average(data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        计算移动平均线
        
        Args:
            data: 股票数据DataFrame
            period: 移动平均周期
            column: 计算列名，默认为'close'
            
        Returns:
            pd.Series: 移动平均线序列
        """
        if len(data) < period:
            return pd.Series([np.nan] * len(data), index=data.index)
        
        return data[column].rolling(window=period, min_periods=period).mean()

    @staticmethod
    def check_price_above_ma(data: pd.DataFrame, ma_period: int = 60, price_column: str = 'close') -> bool:
        """
        检查当前价格是否高于移动平均线
        
        Args:
            data: 股票数据DataFrame
            ma_period: 移动平均周期
            price_column: 价格列名，默认为'close'
            
        Returns:
            bool: 当前价格是否高于移动平均线
        """
        if len(data) < ma_period:
            return False
        
        ma = DataUtils.calculate_moving_average(data, ma_period, price_column)
        if ma.empty or pd.isna(ma.iloc[-1]):
            return False
        
        current_price = data[price_column].iloc[-1]
        return bool(current_price > ma.iloc[-1])

    @staticmethod
    def check_recent_high(data: pd.DataFrame, lookback_days: int = 20, price_column: str = 'close') -> bool:
        """
        检查最近是否创下阶段性新高
        
        Args:
            data: 股票数据DataFrame
            lookback_days: 回看天数
            price_column: 价格列名，默认为'close'
            
        Returns:
            bool: 是否在最近期间创下新高
        """
        if len(data) < lookback_days:
            return False
        
        recent_data = data[price_column].iloc[-lookback_days:]
        current_price = recent_data.iloc[-1]
        
        # 检查当前价格是否在最近期间的最高价附近（允许1%的误差）
        max_price = recent_data.max()
        return bool(current_price >= max_price * 0.99)

    @staticmethod
    def check_uptrend_strength(data: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> bool:
        """
        检查上升趋势强度
        
        Args:
            data: 股票数据DataFrame
            short_period: 短期移动平均周期
            long_period: 长期移动平均周期
            
        Returns:
            bool: 是否处于强上升趋势
        """
        if len(data) < long_period:
            return False
        
        short_ma = DataUtils.calculate_moving_average(data, short_period)
        long_ma = DataUtils.calculate_moving_average(data, long_period)
        
        if short_ma.empty or long_ma.empty:
            return False
        
        # 短期均线在长期均线之上，且短期均线向上倾斜
        short_ma_current = short_ma.iloc[-1]
        long_ma_current = long_ma.iloc[-1]
        
        if pd.isna(short_ma_current) or pd.isna(long_ma_current):
            return False
        
        # 检查短期均线是否在长期均线之上
        if short_ma_current <= long_ma_current:
            return False
        
        # 检查短期均线是否向上倾斜（最近3个值递增）
        if len(short_ma) >= 3:
            recent_short_ma = short_ma.iloc[-3:].dropna()
            if len(recent_short_ma) >= 3:
                return bool(recent_short_ma.iloc[-1] > recent_short_ma.iloc[-2] > recent_short_ma.iloc[-3])
        
        return False

    @staticmethod
    def check_volume_surge(data: pd.DataFrame, lookback_days: int = 20, volume_threshold: float = 1.5) -> bool:
        """
        检查成交量是否放大
        
        Args:
            data: 股票数据DataFrame
            lookback_days: 回看天数
            volume_threshold: 成交量放大倍数阈值
            
        Returns:
            bool: 成交量是否显著放大
        """
        if len(data) < lookback_days:
            return False
        
        if 'volume' not in data.columns:
            return False
        
        recent_volume = data['volume'].iloc[-lookback_days:]
        current_volume = recent_volume.iloc[-1]
        avg_volume = recent_volume.mean()
        
        if avg_volume <= 0:
            return False
        
        return bool(current_volume >= avg_volume * volume_threshold)

    @staticmethod
    def prefilter_stocks(
            stock_data_dict: Dict[str, pd.DataFrame],
            min_data_points: int = 60,
            ma_period: int = 60,
            lookback_days: int = 20,
            volume_threshold: float = 1.5,
            min_conditions_met: int = 2,
            enable_volume_check: bool = False
    ) -> List[str]:
        """
        预筛选股票，过滤出满足基本趋势条件的股票
        
        Args:
            stock_data_dict: 股票数据字典 {stock_code: DataFrame}
            min_data_points: 最小数据点数
            ma_period: 移动平均周期
            lookback_days: 回看天数
            volume_threshold: 成交量放大倍数阈值
            min_conditions_met: 至少满足的条件数量
            enable_volume_check: 是否启用成交量检查（可选条件）
            
        Returns:
            List[str]: 通过预筛选的股票代码列表
        """
        filtered_stocks = []
        
        for stock_code, data in stock_data_dict.items():
            try:
                # 检查数据完整性
                if len(data) < min_data_points:
                    continue
                
                # 检查是否有必要的列
                required_columns = ['close', 'high', 'low']
                if not all(col in data.columns for col in required_columns):
                    continue
                
                # 应用预筛选条件
                conditions_met = 0
                
                # 条件1: 价格高于移动平均线
                if DataUtils.check_price_above_ma(data, ma_period):
                    conditions_met += 1
                
                # 条件2: 最近创下阶段性新高
                if DataUtils.check_recent_high(data, lookback_days):
                    conditions_met += 1
                
                # 条件3: 上升趋势强度
                if DataUtils.check_uptrend_strength(data):
                    conditions_met += 1
                
                # 条件4: 成交量放大（可选条件）
                if enable_volume_check and DataUtils.check_volume_surge(data, lookback_days, volume_threshold):
                    conditions_met += 1
                
                # 检查是否满足最小条件数量
                if conditions_met >= min_conditions_met:
                    filtered_stocks.append(stock_code)
                    
            except Exception as e:
                logging.debug(f"预筛选股票 {stock_code} 时出错: {e}")
                continue
        
        return filtered_stocks

    @staticmethod
    def get_stock_data_for_backtest(
            stock_pool: str = "no_st",
            start_date: str = None,
            end_date: str = None,
            period: str = "daily",
            adjust: str = "3",
            min_data_days: int = 60,
            max_stocks: Optional[int] = None,
            progress_callback: Optional[callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        获取回测所需的股票数据
        
        Args:
            stock_pool: 股票池名称，可选值：
                - "full": 全量股票
                - "no_st": 非ST股票（推荐）
                - "st": ST股票
                - "sz50": 上证50
                - "hs300": 沪深300
                - "zz500": 中证500
            start_date: 开始日期 (YYYY-MM-DD)，如果为None则自动计算
            end_date: 结束日期 (YYYY-MM-DD)，如果为None则使用今天
            period: 数据周期，可选值：daily, 5min
            adjust: 复权类型，1:后复权，2:前复权，3:不复权
            min_data_days: 最小数据天数，少于这个天数的股票会被过滤
            max_stocks: 最大股票数量，用于限制回测规模
            progress_callback: 进度回调函数，用于显示获取进度
            
        Returns:
            Dict[str, pd.DataFrame]: 股票代码到数据的映射
                {
                    '000001.SZ': DataFrame(...),
                    '000002.SZ': DataFrame(...),
                    ...
                }
        """
        if StockDataFetcher is None:
            raise ImportError("无法导入StockDataFetcher，请确保数据模块可用")

        # 统一使用backtest主日志记录器，便于全局日志管理和追踪
        logger = setup_logger("backtest_event")
        data_fetcher = StockDataFetcher()

        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            # 默认获取一年的数据，确保有足够的历史数据用于计算指标
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        logger.info(f"开始获取股票数据: 股票池={stock_pool}, 时间范围={start_date} 到 {end_date}")

        # 1. 获取股票列表
        try:
            stock_list_df = data_fetcher.get_stock_list(pool_name=stock_pool)
            logger.info(f"获取到 {len(stock_list_df)} 只股票")
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise

        # 限制股票数量（如果指定了max_stocks）
        if max_stocks and len(stock_list_df) > max_stocks:
            stock_list_df = stock_list_df.head(max_stocks)
            logger.info(f"限制股票数量为 {max_stocks} 只")

        # 2. 获取每只股票的历史数据
        all_stock_data = {}
        total_stocks = len(stock_list_df)
        success_count = 0
        failed_count = 0

        for idx, (_, stock_info) in enumerate(stock_list_df.iterrows()):
            stock_code = stock_info['code']

            try:
                # 获取股票数据
                stock_data = data_fetcher.fetch_stock_data(
                    code=stock_code,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )

                # 检查数据质量
                if stock_data.empty:
                    logger.warning(f"股票 {stock_code} 数据为空")
                    failed_count += 1
                    continue

                # 检查数据天数
                if len(stock_data) < min_data_days:
                    logger.warning(f"股票 {stock_code} 数据天数不足: {len(stock_data)} < {min_data_days}")
                    failed_count += 1
                    continue

                # 数据验证
                validation_result = DataUtils.validate_data(stock_data)
                if not validation_result["is_valid"]:
                    logger.warning(f"股票 {stock_code} 数据验证失败: {validation_result['errors']}")
                    failed_count += 1
                    continue

                # 数据清理
                cleaned_data = DataUtils.clean_data(stock_data)

                # 保存数据
                all_stock_data[stock_code] = cleaned_data
                success_count += 1

                logger.debug(f"成功获取股票 {stock_code} 数据: {len(cleaned_data)} 条记录")

            except Exception as e:
                logger.warning(f"获取股票 {stock_code} 数据失败: {e}")
                failed_count += 1
                continue

            # 进度回调
            if progress_callback:
                progress = (idx + 1) / total_stocks
                progress_callback(progress, stock_code, success_count, failed_count)

        logger.info(f"数据获取完成: 成功 {success_count} 只，失败 {failed_count} 只")

        if not all_stock_data:
            raise ValueError("没有获取到任何有效的股票数据")

        return all_stock_data

    @staticmethod
    def get_stock_list_info(stock_pool: str = "no_st") -> pd.DataFrame:
        """
        获取股票列表信息
        
        Args:
            stock_pool: 股票池名称
            
        Returns:
            DataFrame: 股票列表信息
        """
        if StockDataFetcher is None:
            raise ImportError("无法导入StockDataFetcher，请确保数据模块可用")

        data_fetcher = StockDataFetcher()
        return data_fetcher.get_stock_list(pool_name=stock_pool)

    @staticmethod
    def get_single_stock_data(
            stock_code: str,
            start_date: str = None,
            end_date: str = None,
            period: str = "daily",
            adjust: str = "3"
    ) -> pd.DataFrame:
        """
        获取单只股票的数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 数据周期
            adjust: 复权类型
            
        Returns:
            DataFrame: 股票数据
        """
        if StockDataFetcher is None:
            raise ImportError("无法导入StockDataFetcher，请确保数据模块可用")

        data_fetcher = StockDataFetcher()

        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # 获取数据
        stock_data = data_fetcher.fetch_stock_data(
            code=stock_code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        # 数据验证和清理
        if not stock_data.empty:
            validation_result = DataUtils.validate_data(stock_data)
            if validation_result["is_valid"]:
                return DataUtils.clean_data(stock_data)
            else:
                logging.warning(f"股票 {stock_code} 数据验证失败: {validation_result['errors']}")

        return stock_data

    @staticmethod
    def filter_stocks_by_condition(
            stock_list: pd.DataFrame,
            min_amount: Optional[float] = None,
            min_volume: Optional[float] = None,
            start_date: str = None,
            end_date: str = None
    ) -> pd.DataFrame:
        """
        根据条件过滤股票
        
        Args:
            stock_list: 股票列表
            min_amount: 最小成交额
            min_volume: 最小成交量
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 过滤后的股票列表
        """
        if StockDataFetcher is None:
            raise ImportError("无法导入StockDataFetcher，请确保数据模块可用")

        if stock_list.empty:
            return stock_list

        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")

        data_fetcher = StockDataFetcher()
        filtered_stocks = []

        for _, stock_info in stock_list.iterrows():
            stock_code = stock_info['code']

            try:
                # 获取最近的数据
                stock_data = data_fetcher.fetch_stock_data(
                    code=stock_code,
                    start_date=start_date,
                    end_date=end_date
                )

                if stock_data.empty:
                    continue

                # 检查成交额条件
                if min_amount is not None:
                    avg_amount = stock_data['amount'].mean()
                    if avg_amount < min_amount:
                        continue

                # 检查成交量条件
                if min_volume is not None:
                    avg_volume = stock_data['volume'].mean()
                    if avg_volume < min_volume:
                        continue

                filtered_stocks.append(stock_info)

            except Exception as e:
                logging.warning(f"检查股票 {stock_code} 条件失败: {e}")
                continue

        return pd.DataFrame(filtered_stocks)

    @staticmethod
    def create_sample_data(days: int = 252, start_price: float = 100.0,
                           volatility: float = 0.02, trend: float = 0.001) -> pd.DataFrame:
        """
        创建示例数据
        
        Args:
            days: 数据天数
            start_price: 起始价格
            volatility: 波动率
            trend: 趋势
            
        Returns:
            示例数据
        """
        # 生成日期序列
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(days)]

        # 生成价格数据
        np.random.seed(42)
        returns = np.random.normal(trend, volatility, days)
        prices = [start_price]

        for i in range(1, days):
            price = prices[-1] * (1 + returns[i])
            prices.append(price)

        # 生成OHLCV数据
        data = []
        for date, close in zip(dates, prices):
            open_price = close * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 10000000)

            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data, index=dates)

    @staticmethod
    def validate_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据格式和质量
        
        Args:
            data: 待验证的数据
            
        Returns:
            验证结果字典
        """
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }

        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            result["is_valid"] = False
            result["errors"].append(f"缺少必需列: {missing_columns}")

        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                result["is_valid"] = False
                result["errors"].append(f"列 {col} 必须是数值类型")

        # 检查数据完整性
        if not data.empty:
            null_counts = data[required_columns].isnull().sum()
            if null_counts.any():
                result["warnings"].append(f"数据包含空值: {null_counts[null_counts > 0].to_dict()}")

        # 检查价格逻辑
        if not data.empty and all(col in data.columns for col in ['high', 'low', 'open', 'close']):
            invalid_prices = (
                    (data['high'] < data['low']) |
                    (data['open'] > data['high']) |
                    (data['close'] > data['high']) |
                    (data['open'] < data['low']) |
                    (data['close'] < data['low'])
            )

            if invalid_prices.any():
                invalid_count = invalid_prices.sum()
                result["warnings"].append(f"发现 {invalid_count} 条价格逻辑错误的数据")

        # 添加数据信息
        if not data.empty:
            result["info"] = {
                "start_date": data.index[0] if len(data) > 0 else None,
                "end_date": data.index[-1] if len(data) > 0 else None,
                "total_days": len(data),
                "price_range": {
                    "min": data['close'].min() if 'close' in data.columns else None,
                    "max": data['close'].max() if 'close' in data.columns else None,
                    "mean": data['close'].mean() if 'close' in data.columns else None
                }
            }

        return result

    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据
        
        Args:
            data: 原始数据
            
        Returns:
            清理后的数据
        """
        # 复制数据
        cleaned_data = data.copy()

        # 确保索引是日期类型
        if not isinstance(cleaned_data.index, pd.DatetimeIndex):
            # 如果索引不是日期类型，尝试从date列创建索引
            if 'date' in cleaned_data.columns:
                cleaned_data.set_index('date', inplace=True)
                cleaned_data.index = pd.to_datetime(cleaned_data.index)
            else:
                # 如果没有date列，尝试从第一列创建索引
                first_col = cleaned_data.columns[0]
                if cleaned_data[first_col].dtype == 'object':
                    try:
                        cleaned_data.set_index(first_col, inplace=True)
                        cleaned_data.index = pd.to_datetime(cleaned_data.index)
                    except:
                        pass

        # 处理空值，但保留索引
        cleaned_data = cleaned_data.dropna()

        # 处理异常值（使用3倍标准差）
        for col in ['open', 'high', 'low', 'close']:
            if col in cleaned_data.columns:
                mean = cleaned_data[col].mean()
                std = cleaned_data[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std

                # 将异常值替换为边界值
                cleaned_data[col] = cleaned_data[col].clip(lower_bound, upper_bound)

        # 确保价格逻辑正确
        if all(col in cleaned_data.columns for col in ['high', 'low', 'open', 'close']):
            cleaned_data['high'] = cleaned_data[['high', 'open', 'close']].max(axis=1)
            cleaned_data['low'] = cleaned_data[['low', 'open', 'close']].min(axis=1)

        return cleaned_data

    @staticmethod
    def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            data: 价格数据
            
        Returns:
            包含收益率的数据
        """
        if 'close' not in data.columns:
            raise ValueError("数据必须包含close列")

        result = data.copy()

        # 计算日收益率
        result['daily_return'] = result['close'].pct_change()

        # 计算累计收益率
        result['cumulative_return'] = (1 + result['daily_return']).cumprod() - 1

        # 计算对数收益率
        result['log_return'] = np.log(result['close'] / result['close'].shift(1))

        return result

    @staticmethod
    def resample_data(data: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        重采样数据
        
        Args:
            data: 原始数据
            freq: 重采样频率 ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            重采样后的数据
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是日期时间类型")

        # 重采样
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        return resampled.dropna()
