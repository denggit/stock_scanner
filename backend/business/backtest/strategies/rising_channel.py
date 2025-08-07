#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道回测策略 - 多股票组合策略
基于 ascending_channel.py 中的上升通道回归分析算法

策略逻辑：
1. 找出所有上升通道为NORMAL的股票
2. 按股价距离下沿的百分比距离排序（从小到大）
3. 平均买入前50只股票至满仓
4. 每天检查持仓股票状态，非NORMAL状态则卖出
5. 当未满50只股票时，重新选股并买入至50只
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

import backtrader as bt
import pandas as pd

# 导入配置
from backend.business.backtest.configs.rising_channel_config import RisingChannelConfig
from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus


class MockChannelAnalyzer:
    """模拟通道分析器，用于测试"""

    def analyze(self, data):
        """模拟分析方法"""
        return MockChannelState()


class MockChannelState:
    """模拟通道状态"""

    def __init__(self):
        self.channel_status = MockChannelStatus()


class MockChannelStatus:
    """模拟通道状态枚举"""

    def __init__(self):
        self.value = ChannelStatus.NORMAL

    def set_value(self, value):
        self.value = value

    def __str__(self):
        return self.value


class RisingChannelBacktestStrategy(bt.Strategy):
    """
    上升通道回测策略 - 多股票组合策略
    
    策略特点：
    - 多股票组合管理，目标持仓50只股票
    - 基于上升通道NORMAL状态选股
    - 按距离下沿百分比排序选股
    - 动态调仓：非NORMAL状态立即卖出
    - 平均分配资金
    """

    params = (
        ('max_positions', 50),  # 最大持仓数量
        ('min_channel_score', 60.0),  # 最小通道评分
        ('k', 2.0),  # 通道斜率参数
        ('L_max', 120),  # 最大回看天数
        ('delta_cut', 5),  # 切割参数
        ('pivot_m', 3),  # 枢轴参数
        ('gain_trigger', 0.30),  # 收益触发阈值
        ('beta_delta', 0.15),  # beta增量
        ('break_days', 3),  # 突破天数
        ('reanchor_fail_max', 2),  # 重锚定失败最大次数
        ('min_data_points', 60),  # 最小数据点数
        ('R2_min', 0.20),  # 最小R2值
        ('width_pct_min', 0.04),  # 最小宽度百分比
        ('width_pct_max', 0.20),  # 最大宽度百分比 - 调整为更宽松的值
    )

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame] = None):
        """
        初始化策略
        
        Args:
            stock_data_dict: 股票数据字典 {股票代码: DataFrame}
        """
        # 必须调用父类初始化
        super().__init__()

        # 策略状态
        self.current_positions = {}  # 当前持仓 {股票代码: 持仓数量}
        self.buy_prices = {}  # 买入价格 {股票代码: 买入价格}
        self.buy_dates = {}  # 买入日期 {股票代码: 买入日期}
        self.last_trade_date = None  # 上次交易日期

        # 通道状态管理
        self.channel_states = {}  # 当前通道状态 {股票代码: 通道状态对象}
        self.channel_analyzer = None  # 通道分析器
        self.channel_scores = {}  # 当前通道评分 {股票代码: 评分}

        # 交易记录
        self.trades = []
        self.trade_count = 0

        # 通道分析记录
        self.channel_analysis_records = []

        # 历史数据缓存
        self.historical_data = {}  # {股票代码: [历史数据]}

        # 多股票数据管理
        self.all_stock_data = {}  # 所有股票数据 {股票代码: DataFrame}
        self.stock_codes = []  # 股票代码列表
        self.current_date = None  # 当前日期

        # 统一使用backtest主日志记录器
        self.logger = logging.getLogger("backtest")

        # 延迟初始化通道分析器，避免在backtrader初始化时出现问题
        self._channel_analyzer_initialized = False

        # 如果传入了股票数据，立即设置
        if stock_data_dict is not None:
            self.set_stock_data(stock_data_dict)

        # 记录策略初始化
        self.logger.info("成功初始化上升通道策略")
        self.logger.info(f"将跳过前 {self.params.min_data_points} 天的数据，等待足够的历史数据")
        self.logger.info(f"最大持仓数量: {self.params.max_positions}")
        self.logger.info(f"最小通道评分: {self.params.min_channel_score}")

    def _init_channel_analyzer(self):
        """初始化通道分析器"""
        try:
            # 尝试导入真实的通道分析器
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
                from backend.business.factor.core.engine.library.channel_analysis.ascending_channel import \
                    AscendingChannelRegression

                # 构建通道分析器参数
                channel_params = {
                    "k": self.params.k,
                    "L_max": self.params.L_max,
                    "delta_cut": self.params.delta_cut,
                    "pivot_m": self.params.pivot_m,
                    "gain_trigger": self.params.gain_trigger,
                    "beta_delta": self.params.beta_delta,
                    "break_days": self.params.break_days,
                    "reanchor_fail_max": self.params.reanchor_fail_max,
                    "min_data_points": self.params.min_data_points,
                    "R2_min": self.params.R2_min,
                    "width_pct_min": self.params.width_pct_min,
                    "width_pct_max": self.params.width_pct_max
                }

                self.logger.info(f"初始化通道分析器，参数: {channel_params}")
                self.channel_analyzer = AscendingChannelRegression(**channel_params)
                self.logger.info("成功初始化上升通道分析器")
            except ImportError as e:
                # 如果导入失败，创建一个模拟的分析器
                self.logger.warning(f"无法导入上升通道分析器: {e}，使用模拟分析器")
                self.channel_analyzer = MockChannelAnalyzer()

        except Exception as e:
            self.logger.error(f"初始化上升通道分析器失败: {e}")
            self.channel_analyzer = MockChannelAnalyzer()

    def _ensure_channel_analyzer_initialized(self):
        """确保通道分析器已初始化"""
        if not self._channel_analyzer_initialized:
            self._init_channel_analyzer()
            self._channel_analyzer_initialized = True

    def set_stock_data(self, stock_data_dict: Dict[str, pd.DataFrame]):
        """
        设置所有股票数据
        
        Args:
            stock_data_dict: 股票数据字典 {股票代码: DataFrame}
        """
        self.all_stock_data = stock_data_dict
        self.stock_codes = list(stock_data_dict.keys())
        self.logger.info(f"设置股票数据: {len(self.stock_codes)} 只股票")

        # 初始化历史数据缓存
        for stock_code in self.stock_codes:
            self.historical_data[stock_code] = []

    def next(self):
        """
        策略主逻辑 - 每个交易日执行

        每次回测交易日会打印当前日期，便于追踪每日操作。
        """
        # 确保通道分析器已初始化
        self._ensure_channel_analyzer_initialized()

        # 如果没有设置多股票数据，使用单股票模式
        if not self.all_stock_data:
            self._run_single_stock_mode()
            return

        # 获取当前日期
        current_date = self.data.datetime.date(0)
        self.current_date = current_date

        # 跳过前min_data_points天的数据，等待足够的历史数据
        if len(self.data) < self.params.min_data_points:
            self.logger.debug(f"跳过第 {len(self.data)} 天，等待足够的历史数据 (需要 {self.params.min_data_points} 天)")
            return

        # 打印当前回测交易日日期日志
        self.logger.info(f"========回测交易日: {current_date}========")

        # 记录当前数据
        self._record_current_data(current_date)

        # 更新所有股票的通道状态
        self._update_all_channel_states(current_date)

        # 检查持仓股票状态，卖出非NORMAL状态的股票
        self._check_and_sell_positions(current_date)

        # 如果持仓不足max_positions只，重新选股并买入
        if len([p for p in self.current_positions.values() if p > 0]) < self.params.max_positions:
            self._select_and_buy_stocks(current_date)

    def _run_single_stock_mode(self):
        """
        单股票模式 - 用于简化测试
        """
        # 获取当前日期
        current_date = self.data.datetime.date(0)
        self.current_date = current_date

        # 跳过前min_data_points天的数据，等待足够的历史数据
        if len(self.data) < self.params.min_data_points:
            self.logger.debug(
                f"单股票模式：跳过第 {len(self.data)} 天，等待足够的历史数据 (需要 {self.params.min_data_points} 天)")
            return

        # 记录当前数据
        self._record_current_data(current_date)

        # 简单的买入逻辑：如果当前没有持仓，买入
        if not self.current_positions:
            current_price = self.data.close[0]
            if current_price > 0:
                # 买入当前股票
                self._buy_stock("current_stock", current_price, current_date)

    def _record_current_data(self, current_date):
        """记录当前数据"""
        # 获取当前股票代码
        stock_code = self._get_current_stock_code()

        # 确保时间格式一致
        if isinstance(current_date, datetime):
            trade_date = current_date
        else:
            trade_date = datetime.combine(current_date, datetime.min.time())

        data_record = {
            'trade_date': trade_date,
            'open': self.data.open[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'close': self.data.close[0],
            'volume': self.data.volume[0] if hasattr(self.data, 'volume') else 0
        }

        if stock_code not in self.historical_data:
            self.historical_data[stock_code] = []

        self.historical_data[stock_code].append(data_record)

        # 保持数据量在合理范围内
        max_data_points = self.params.L_max * 2
        if len(self.historical_data[stock_code]) > max_data_points:
            self.historical_data[stock_code] = self.historical_data[stock_code][-max_data_points:]

    def _get_current_stock_code(self) -> str:
        """获取当前股票代码"""
        # 从数据源名称获取股票代码
        data_name = getattr(self.data, '_name', 'data')
        if data_name != 'data':
            return data_name
        else:
            # 如果没有明确的数据名称，使用默认值
            return "current_stock"

    def _update_all_channel_states(self, current_date):
        """更新所有股票的通道状态"""
        self._ensure_channel_analyzer_initialized()

        self.logger.info(f"开始更新 {len(self.stock_codes)} 只股票的通道状态...")

        normal_count = 0
        broken_count = 0
        error_count = 0

        for stock_code in self.stock_codes:
            try:
                # 获取股票历史数据
                stock_data = self._get_stock_historical_data(stock_code, current_date)
                if stock_data is None or len(stock_data) < self.params.min_data_points:
                    self.logger.debug(f"股票 {stock_code} 数据不足: {len(stock_data) if stock_data is not None else 0} < {self.params.min_data_points}")
                    continue

                # 计算上升通道
                channel_state = self.channel_analyzer.fit_channel(stock_data)

                # 计算通道评分
                channel_score = self._calculate_channel_score(channel_state, stock_data)

                # 更新状态
                self.channel_states[stock_code] = channel_state
                self.channel_scores[stock_code] = channel_score

                # 记录分析结果
                self._record_channel_analysis(stock_code, channel_state, channel_score, current_date)

                # 统计状态
                if (channel_state and channel_state.channel_status and
                        channel_state.channel_status == ChannelStatus.NORMAL):
                    normal_count += 1
                    self.logger.debug(
                        f"股票 {stock_code} 通道分析: "
                        f"状态={channel_state.channel_status.value}, "
                        f"评分={channel_score:.1f}, "
                        f"R²={channel_state.r2:.3f}"
                    )
                else:
                    broken_count += 1

            except Exception as e:
                error_count += 1
                self.logger.error(f"更新股票 {stock_code} 通道状态失败: {e}")
                continue

        # 输出统计信息
        self.logger.info(f"通道状态统计: NORMAL={normal_count}, BROKEN={broken_count}, ERROR={error_count}")
        
        # 如果没有找到NORMAL状态的股票，输出一些调试信息
        if normal_count == 0:
            self.logger.warning("没有找到NORMAL状态的股票，可能的原因:")
            self.logger.warning("1. 通道参数设置过于严格")
            self.logger.warning("2. 数据质量不足")
            self.logger.warning("3. 通道分析器配置问题")

    def _get_stock_historical_data(self, stock_code: str, current_date) -> Optional[pd.DataFrame]:
        """获取指定股票的历史数据"""
        if stock_code not in self.all_stock_data:
            return None

        stock_df = self.all_stock_data[stock_code].copy()

        # 确保current_date是datetime类型
        if isinstance(current_date, datetime):
            current_datetime = current_date
        elif isinstance(current_date, pd.Timestamp):
            current_datetime = current_date.to_pydatetime()
        else:
            current_datetime = pd.to_datetime(current_date).to_pydatetime()

        # 使用trade_date列进行过滤，而不是索引
        # 过滤到当前日期之前的数据
        filtered_df = stock_df[stock_df['trade_date'] <= current_datetime.date()].copy()

        if len(filtered_df) < self.params.min_data_points:
            self.logger.debug(f"股票 {stock_code} 过滤后数据不足: {len(filtered_df)} < {self.params.min_data_points}")
            return None

        # 确保数据格式符合通道分析器的要求
        # 通道分析器期望的列名：trade_date, open, high, low, close, volume
        formatted_df = filtered_df.copy()

        # 确保所有必需的列都存在
        required_columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = []
        for col in required_columns:
            if col not in formatted_df.columns:
                missing_columns.append(col)
                if col == 'volume':
                    formatted_df[col] = 1000000  # 默认成交量
                else:
                    formatted_df[col] = formatted_df['close']  # 使用收盘价作为默认值

        if missing_columns:
            self.logger.debug(f"股票 {stock_code} 缺少列: {missing_columns}，已使用默认值填充")

        # 检查数据质量
        if formatted_df['close'].isnull().any():
            self.logger.debug(f"股票 {stock_code} 收盘价存在空值")
        
        if formatted_df['volume'].isnull().any():
            self.logger.debug(f"股票 {stock_code} 成交量存在空值")

        return formatted_df

    def _calculate_channel_score(self, channel_state, stock_data: pd.DataFrame) -> float:
        """计算通道评分"""
        if not channel_state or not channel_state.channel_status:
            return 0.0

        # 基础评分：通道状态
        base_score = 0.0
        if channel_state.channel_status == ChannelStatus.NORMAL:
            base_score = 60.0  # 提高NORMAL状态的基础评分
        elif channel_state.channel_status == ChannelStatus.ACCEL_BREAKOUT:
            base_score = 40.0
        elif channel_state.channel_status == ChannelStatus.BREAKDOWN:
            base_score = 20.0
        else:
            base_score = 0.0

        # TODO: 完善评分机制
        # 数据质量评分（模拟）
        # data_quality_score = min(len(stock_data) / 100.0 * 20.0, 20.0)
        data_quality_score = 0

        # 通道稳定性评分（模拟）
        stability_score = 0

        total_score = base_score + data_quality_score + stability_score
        return min(total_score, 100.0)

    def _record_channel_analysis(self, stock_code: str, channel_state, channel_score: float, current_date):
        """记录通道分析结果"""
        record = {
            'date': current_date,
            'stock_code': stock_code,
            'channel_status': channel_state.channel_status.value if channel_state and channel_state.channel_status else None,
            'channel_score': channel_score,
            'data_points': len(self.historical_data.get(stock_code, []))
        }
        self.channel_analysis_records.append(record)

    def _check_and_sell_positions(self, current_date):
        """检查持仓股票状态，卖出非NORMAL状态的股票"""
        stocks_to_sell = []

        for stock_code, position_size in self.current_positions.items():
            if position_size <= 0:
                continue

            # 检查通道状态
            if stock_code in self.channel_states:
                channel_state = self.channel_states[stock_code]
                if (channel_state and channel_state.channel_status and
                        channel_state.channel_status != ChannelStatus.NORMAL):
                    stocks_to_sell.append(stock_code)

        # 卖出非NORMAL状态的股票
        for stock_code in stocks_to_sell:
            self._sell_stock(stock_code, current_date)

    def _buy_stock(self, stock_code: str, current_price: float, current_date):
        """买入指定股票"""
        if current_price <= 0:
            return

        # 计算每只股票的资金分配
        total_cash = self.broker.getcash()
        commission = 0.0003  # 默认手续费率
        available_cash = total_cash * (1 - commission)

        # 平均分配资金给每只股票
        cash_per_stock = available_cash / self.params.max_positions
        shares = int(cash_per_stock / current_price)

        if shares > 0:
            # 执行买入 - 使用当前数据源
            self.buy(size=shares)

            # 更新持仓状态
            self.current_positions[stock_code] = shares
            self.buy_prices[stock_code] = current_price
            self.buy_dates[stock_code] = current_date

            # 记录交易
            self._log_trade('BUY', stock_code, shares, current_price, current_date)

            self.logger.info(f"买入股票 {stock_code}: {shares} 股，价格: {current_price:.2f}，"
                             f"日期: {current_date}")

    def _sell_stock(self, stock_code: str, current_date):
        """卖出指定股票"""
        if stock_code not in self.current_positions or self.current_positions[stock_code] <= 0:
            return

        shares = self.current_positions[stock_code]
        current_price = self.data.close[0]  # 使用当前数据源的价格

        if current_price > 0:
            # 执行卖出
            self.sell(size=shares)

            # 计算收益
            buy_price = self.buy_prices.get(stock_code, 0)
            returns = (current_price - buy_price) / buy_price * 100 if buy_price > 0 else 0

            # 更新持仓状态
            self.current_positions[stock_code] = 0
            del self.buy_prices[stock_code]
            del self.buy_dates[stock_code]

            # 记录交易
            self._log_trade('SELL', stock_code, shares, current_price, current_date, returns)

            self.logger.info(f"卖出股票 {stock_code}: {shares} 股，价格: {current_price:.2f}，"
                             f"收益: {returns:.2f}%，日期: {current_date}")

    def _select_and_buy_stocks(self, current_date):
        """选择并买入股票"""
        self._ensure_channel_analyzer_initialized()

        # 找出所有NORMAL状态的股票
        normal_stocks = []

        for stock_code in self.stock_codes:
            if (stock_code in self.channel_states and
                    self.channel_states[stock_code] and
                    self.channel_states[stock_code].channel_status and
                    self.channel_states[stock_code].channel_status == ChannelStatus.NORMAL and
                    self.channel_scores.get(stock_code, 0) >= self.params.min_channel_score):

                # 获取当前价格
                current_price = self._get_stock_price(stock_code, current_date)
                if current_price <= 0:
                    continue

                # 计算距离下沿的百分比距离
                if stock_code in self.channel_states and self.channel_states[stock_code]:
                    channel_state = self.channel_states[stock_code]
                    # 使用真实的通道下沿价格
                    if (hasattr(channel_state, 'lower_today') and
                            channel_state.lower_today is not None and
                            channel_state.lower_today > 0):
                        lower_price = channel_state.lower_today
                        distance_to_lower = self._calculate_distance_to_lower(current_price, lower_price)
                    else:
                        # 如果下沿价格无效，使用配置中的备用值
                        distance_config = RisingChannelConfig.get_distance_config()
                        lower_price = current_price * distance_config['lower_price_ratio_invalid']
                        distance_to_lower = distance_config['fallback_distance_invalid']
                else:
                    # 如果没有通道状态，使用配置中的备用值
                    distance_config = RisingChannelConfig.get_distance_config()
                    lower_price = current_price * distance_config['lower_price_ratio_no_state']
                    distance_to_lower = distance_config['fallback_distance_no_state']

                normal_stocks.append({
                    'stock_code': stock_code,
                    'current_price': current_price,
                    'distance_to_lower': distance_to_lower,
                    'channel_score': self.channel_scores.get(stock_code, 0.0),
                    'lower_today': lower_price,  # 添加下沿价格
                    'mid_today': channel_state.mid_today if hasattr(channel_state, 'mid_today') else None,  # 添加中轴价格
                    'upper_today': channel_state.upper_today if hasattr(channel_state, 'upper_today') else None
                    # 添加上沿价格
                })

        # 按距离下沿百分比排序（从小到大）
        normal_stocks.sort(key=lambda x: x['distance_to_lower'])

        self.logger.info(f"找到 {len(normal_stocks)} 只NORMAL状态股票")

        # 添加详细的通道信息日志
        if normal_stocks:
            self.logger.info("通道分析详情:")
            for i, stock_info in enumerate(normal_stocks[:5]):  # 只显示前5只股票
                self.logger.info(
                    f"  {i + 1}. {stock_info['stock_code']}: "
                    f"当前价格={stock_info['current_price']:.2f}, "
                    f"下沿={stock_info['lower_today']:.2f}, "
                    f"中轴={stock_info['mid_today']:.2f}, "
                    f"上沿={stock_info['upper_today']:.2f}, "
                    f"距离下沿={stock_info['distance_to_lower']:.2f}%, "
                    f"评分={stock_info['channel_score']:.1f}"
                )

        # 计算需要买入的股票数量
        current_position_count = len([p for p in self.current_positions.values() if p > 0])
        need_to_buy = min(self.params.max_positions - current_position_count, len(normal_stocks))

        # 买入前N只股票
        for i in range(need_to_buy):
            stock_info = normal_stocks[i]
            stock_code = stock_info['stock_code']

            # 检查是否已经持仓
            if stock_code in self.current_positions and self.current_positions[stock_code] > 0:
                continue

            # 买入股票
            self._buy_stock(stock_code, stock_info['current_price'], current_date)

    def _calculate_distance_to_lower(self, current_price: float, lower_price: float) -> float:
        """
        计算股价距离下沿的百分比距离
        
        Args:
            current_price: 当前价格
            lower_price: 下沿价格
            
        Returns:
            float: 距离下沿的百分比距离
        """
        if lower_price <= 0 or current_price <= 0:
            return float('inf')

        # 如果当前价格等于下沿价格，距离为0
        if current_price == lower_price:
            return 0.0

        # 计算距离下沿的百分比距离
        distance = (current_price - lower_price) / lower_price * 100

        # 确保距离不为负数（如果当前价格低于下沿，给予一个小的正值）
        if distance < 0:
            distance_config = RisingChannelConfig.get_distance_config()
            distance = distance_config['min_distance_below_lower']

        return distance

    def _get_stock_price(self, stock_code: str, current_date) -> float:
        """获取指定股票在指定日期的价格"""
        if stock_code not in self.all_stock_data:
            return 0.0

        stock_df = self.all_stock_data[stock_code]

        # 确保current_date是datetime类型
        if isinstance(current_date, datetime):
            current_datetime = current_date
        elif isinstance(current_date, pd.Timestamp):
            current_datetime = current_date.to_pydatetime()
        else:
            current_datetime = pd.to_datetime(current_date).to_pydatetime()

        # 使用trade_date列进行匹配
        current_data = stock_df[stock_df['trade_date'] == current_datetime.date()]
        if not current_data.empty:
            return current_data.iloc[0]['close']

        # 如果精确匹配失败，尝试获取最近的日期
        if len(stock_df) > 0:
            return stock_df.iloc[-1]['close']

        return 0.0

    def _log_trade(self, action: str, stock_code: str, size: int, price: float, current_date=None,
                   returns: float = None):
        """记录交易"""
        # 安全获取通道状态
        channel_status = None
        if (stock_code in self.channel_states and
                self.channel_states[stock_code] and
                hasattr(self.channel_states[stock_code], 'channel_status') and
                self.channel_states[stock_code].channel_status):
            channel_status = self.channel_states[stock_code].channel_status.value

        trade = {
            'date': current_date or datetime.now(),
            'action': action,
            'stock_code': stock_code,
            'quantity': size,
            'price': price,
            'returns': returns,
            'channel_status': channel_status,
            'channel_score': self.channel_scores.get(stock_code, 0.0)
        }
        self.trades.append(trade)
        self.trade_count += 1

    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        # 安全获取通道状态
        channel_states_info = {}
        for k, v in self.channel_states.items():
            if v and hasattr(v, 'channel_status') and v.channel_status:
                channel_states_info[k] = v.channel_status.value
            else:
                channel_states_info[k] = None

        return {
            'strategy_name': 'RisingChannelBacktestStrategy',
            'parameters': self._get_parameters(),
            'current_status': {
                'position_count': len([p for p in self.current_positions.values() if p > 0]),
                'total_positions': self.current_positions,
                'channel_states': channel_states_info,
                'channel_scores': self.channel_scores
            },
            'performance': self.get_performance_summary()
        }

    def _get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            'max_positions': self.params.max_positions,
            'min_channel_score': self.params.min_channel_score,
            'k': self.params.k,
            'L_max': self.params.L_max,
            'delta_cut': self.params.delta_cut,
            'pivot_m': self.params.pivot_m,
            'gain_trigger': self.params.gain_trigger,
            'beta_delta': self.params.beta_delta,
            'break_days': self.params.break_days,
            'reanchor_fail_max': self.params.reanchor_fail_max,
            'min_data_points': self.params.min_data_points,
            'R2_min': self.params.R2_min,
            'width_pct_min': self.params.width_pct_min,
            'width_pct_max': self.params.width_pct_max
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'total_trades': self.trade_count,
            'trades': self.trades,
            'channel_analysis_records': self.channel_analysis_records
        }


def create_rising_channel_backtest_strategy(
        max_positions: int = 50,
        min_channel_score: float = 60.0,
        **params
) -> RisingChannelBacktestStrategy:
    """
    创建上升通道回测策略实例
    
    Args:
        max_positions: 最大持仓数量
        min_channel_score: 最小通道评分
        **params: 其他策略参数
        
    Returns:
        策略实例
    """
    # 构建参数字典
    strategy_params = {
        'max_positions': max_positions,
        'min_channel_score': min_channel_score,
        **params
    }

    # 创建策略实例
    strategy = RisingChannelBacktestStrategy()

    # 设置参数
    for param_name, param_value in strategy_params.items():
        if hasattr(strategy.params, param_name):
            setattr(strategy.params, param_name, param_value)

    return strategy
