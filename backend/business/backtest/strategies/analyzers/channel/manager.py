#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通道分析器管理器
使用工厂模式管理通道分析器的创建和使用
提供统一的通道分析接口
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from backend.business.factor.core.engine.library.channel_analysis.channel_state import ChannelStatus
from backend.utils.logger import setup_logger


class ChannelAnalyzerInterface(ABC):
    """
    通道分析器接口
    定义通道分析器的统一接口
    """

    @abstractmethod
    def fit_channel(self, data: pd.DataFrame) -> Any:
        """
        拟合通道
        
        Args:
            data: 股票数据
            
        Returns:
            通道状态对象
        """
        pass


class MockChannelAnalyzer(ChannelAnalyzerInterface):
    """
    模拟通道分析器
    用于测试和开发环境
    """

    def __init__(self, **params):
        """
        初始化模拟分析器
        
        Args:
            **params: 分析器参数
        """
        self.params = params
        self.logger = setup_logger("backtest")

    def fit_channel(self, data: pd.DataFrame) -> 'MockChannelState':
        """
        模拟通道拟合
        
        Args:
            data: 股票数据
            
        Returns:
            模拟通道状态
        """
        # 模拟分析逻辑
        return MockChannelState(data)


class MockChannelState:
    """模拟通道状态"""

    def __init__(self, data: pd.DataFrame):
        """
        初始化模拟通道状态
        
        Args:
            data: 股票数据
        """
        self.data = data

        # 模拟通道状态（60%概率为NORMAL）
        import random
        rand_val = random.random()
        if rand_val < 0.6:
            self.channel_status = ChannelStatus.NORMAL
        elif rand_val < 0.8:
            self.channel_status = ChannelStatus.ACCEL_BREAKOUT
        else:
            self.channel_status = ChannelStatus.BREAKDOWN

        # 模拟通道参数
        self.r2 = random.uniform(0.2, 0.9)
        self.slope = random.uniform(0.01, 0.05)
        self.width_pct = random.uniform(0.05, 0.15)

        # 模拟今日通道价格
        if not data.empty:
            current_price = data.iloc[-1]['close']
            self.lower_today = current_price * random.uniform(0.92, 0.98)
            self.mid_today = current_price * random.uniform(0.98, 1.02)
            self.upper_today = current_price * random.uniform(1.02, 1.08)
        else:
            self.lower_today = 10.0
            self.mid_today = 11.0
            self.upper_today = 12.0


class RealChannelAnalyzer(ChannelAnalyzerInterface):
    """
    真实通道分析器
    封装实际的上升通道分析算法
    """

    def __init__(self, **params):
        """
        初始化真实分析器
        
        Args:
            **params: 分析器参数
        """
        self.params = params
        self.logger = setup_logger("backtest")
        self._analyzer = None
        self._init_analyzer()

    def _init_analyzer(self):
        """初始化实际的分析器"""
        try:
            from backend.business.factor.core.engine.library.channel_analysis.rising_channel import \
                AscendingChannelRegression

            self._analyzer = AscendingChannelRegression(**self.params)
            self.logger.info("成功初始化真实通道分析器")

        except ImportError as e:
            self.logger.warning(f"无法导入真实通道分析器: {e}，将使用模拟分析器")
            self._analyzer = MockChannelAnalyzer(**self.params)
        except Exception as e:
            self.logger.error(f"初始化通道分析器失败: {e}，使用模拟分析器")
            self._analyzer = MockChannelAnalyzer(**self.params)

    def fit_channel(self, data: pd.DataFrame) -> Any:
        """
        真实通道拟合
        
        Args:
            data: 股票数据
            
        Returns:
            通道状态对象
        """
        try:
            return self._analyzer.fit_channel(data)
        except Exception as e:
            self.logger.error(f"通道分析失败: {e}")
            # 降级为模拟分析器
            mock_analyzer = MockChannelAnalyzer(**self.params)
            return mock_analyzer.fit_channel(data)


class ChannelAnalyzerFactory:
    """
    通道分析器工厂
    使用工厂模式创建不同类型的通道分析器
    """

    _analyzers = {
        'mock': MockChannelAnalyzer,
        'real': RealChannelAnalyzer
    }

    @classmethod
    def create_analyzer(cls, analyzer_type: str = 'real', **params) -> ChannelAnalyzerInterface:
        """
        创建通道分析器
        
        Args:
            analyzer_type: 分析器类型 ('mock', 'real')
            **params: 分析器参数
            
        Returns:
            通道分析器实例
        """
        if analyzer_type not in cls._analyzers:
            raise ValueError(f"未知的分析器类型: {analyzer_type}")

        analyzer_class = cls._analyzers[analyzer_type]
        return analyzer_class(**params)

    @classmethod
    def register_analyzer(cls, analyzer_type: str, analyzer_class):
        """
        注册新的分析器类型
        
        Args:
            analyzer_type: 分析器类型名称
            analyzer_class: 分析器类
        """
        cls._analyzers[analyzer_type] = analyzer_class


class ChannelAnalyzerManager:
    """
    通道分析器管理器
    
    功能：
    - 管理通道分析器实例
    - 缓存分析结果
    - 提供统一的分析接口
    - 计算通道评分
    """

    def __init__(self, analyzer_type: str = 'real', **analyzer_params):
        """
        初始化通道分析器管理器
        
        Args:
            analyzer_type: 分析器类型
            **analyzer_params: 分析器参数
        """
        self.logger = setup_logger("backtest")

        # 创建分析器
        self.analyzer = ChannelAnalyzerFactory.create_analyzer(analyzer_type, **analyzer_params)
        self.analyzer_params = analyzer_params

        # 分析结果缓存 {股票代码: {日期: 分析结果}}
        self.results_cache = {}

        # 通道评分缓存 {股票代码: {日期: 评分}}
        self.scores_cache = {}

        # 分析统计
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        self.logger.info(f"通道分析器管理器初始化完成，类型: {analyzer_type}")

    def get_analyzer(self):
        """
        获取分析器实例
        
        Returns:
            分析器实例
        """
        return self.analyzer

    def analyze_channel(self, stock_code: str, data: pd.DataFrame,
                        date: datetime = None, use_cache: bool = True) -> Optional[Any]:
        """
        分析股票通道
        
        Args:
            stock_code: 股票代码
            data: 股票数据
            date: 分析日期（用于缓存）
            use_cache: 是否使用缓存
            
        Returns:
            通道状态对象，失败返回None
        """
        # 检查缓存
        if use_cache and date and self._check_cache(stock_code, date):
            self.analysis_stats['cache_hits'] += 1
            return self.results_cache[stock_code][date]

        self.analysis_stats['cache_misses'] += 1

        try:
            # 执行分析
            channel_state = self.analyzer.fit_channel(data)

            # 缓存结果
            if date:
                self._cache_result(stock_code, date, channel_state)

            self.analysis_stats['total_analyses'] += 1
            self.analysis_stats['successful_analyses'] += 1

            return channel_state

        except Exception as e:
            self.logger.error(f"分析股票 {stock_code} 通道失败: {e}")
            self.analysis_stats['total_analyses'] += 1
            self.analysis_stats['failed_analyses'] += 1
            return None

    def calculate_channel_score(self, stock_code: str, channel_state: Any,
                                data: pd.DataFrame, date: datetime = None) -> float:
        """
        计算通道评分
        
        Args:
            stock_code: 股票代码
            channel_state: 通道状态对象
            data: 股票数据
            date: 评分日期（用于缓存）
            
        Returns:
            通道评分 (0-100)
        """
        # 检查评分缓存
        if date and self._check_score_cache(stock_code, date):
            return self.scores_cache[stock_code][date]

        try:
            score = self._calculate_score(channel_state, data)

            # 缓存评分
            if date:
                self._cache_score(stock_code, date, score)

            return score

        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 通道评分失败: {e}")
            return 0.0

    def _calculate_score(self, channel_state: Any, data: pd.DataFrame) -> float:
        """
        内部评分计算逻辑
        
        Args:
            channel_state: 通道状态
            data: 股票数据
            
        Returns:
            评分
        """
        if not channel_state or not hasattr(channel_state, 'channel_status'):
            return 0.0

        # 基础评分：通道状态
        base_score = 0.0
        status = channel_state.channel_status

        if status == ChannelStatus.NORMAL:
            base_score = 60.0  # 提高NORMAL状态的基础评分
        elif status == ChannelStatus.ACCEL_BREAKOUT:
            base_score = 50.0
        elif status == ChannelStatus.BREAKDOWN:
            base_score = 20.0
        else:
            base_score = 0.0

        # TODO：先不要做几分，之后我会在做，AI别修改评分标准
        r2_score = 0
        slope_score = 0

        # # R²评分（0-20分）
        # r2_score = 0.0
        # if hasattr(channel_state, 'r2') and channel_state.r2:
        #     r2_score = min(channel_state.r2 * 20, 20.0)
        #
        # # 斜率评分（0-10分）
        # slope_score = 0.0
        # if hasattr(channel_state, 'slope') and channel_state.slope:
        #     # 斜率在0.01-0.05之间得分较高
        #     if 0.01 <= channel_state.slope <= 0.05:
        #         slope_score = 10.0
        #     elif channel_state.slope > 0:
        #         slope_score = 5.0

        total_score = base_score + r2_score + slope_score
        return min(total_score, 100.0)

    def batch_analyze(self, stock_data_dict: Dict[str, pd.DataFrame],
                      date: datetime) -> Dict[str, Dict[str, Any]]:
        """
        批量分析多只股票
        
        Args:
            stock_data_dict: 股票数据字典
            date: 分析日期
            
        Returns:
            批量分析结果 {股票代码: {'channel_state': 状态, 'score': 评分}}
        """
        results = {}

        for stock_code, data in stock_data_dict.items():
            try:
                # 分析通道
                channel_state = self.analyze_channel(stock_code, data, date)

                # 计算评分
                score = 0.0
                if channel_state:
                    score = self.calculate_channel_score(stock_code, channel_state, data, date)

                results[stock_code] = {
                    'channel_state': channel_state,
                    'score': score
                }

            except Exception as e:
                self.logger.error(f"批量分析股票 {stock_code} 失败: {e}")
                results[stock_code] = {
                    'channel_state': None,
                    'score': 0.0
                }

        return results

    def filter_normal_channels(self, analysis_results: Dict[str, Dict[str, Any]],
                               min_score: float = 60.0,
                               r2_min: float | None = None,
                               r2_max: float | None = None) -> List[Dict[str, Any]]:
        """
        筛选NORMAL状态的通道
        
        Args:
            analysis_results: 批量分析结果
            min_score: 最小评分阈值
            r2_min: 选股阶段的 R² 下限（None 表示不限制）
            r2_max: 选股阶段的 R² 上限（None 表示不限制）
            
        Returns:
            符合条件的股票列表
        """
        normal_stocks = []

        for stock_code, result in analysis_results.items():
            channel_state = result['channel_state']
            score = result['score']

            if (channel_state and
                    hasattr(channel_state, 'channel_status') and
                    channel_state.channel_status == ChannelStatus.NORMAL and
                    score >= min_score):
                # R² 区间过滤（仅用于选股阶段）
                if (r2_min is not None) or (r2_max is not None):
                    # 缺失 r2 时不通过过滤
                    r2_value = getattr(channel_state, 'r2', None)
                    if r2_value is None:
                        continue
                    if (r2_min is not None) and (r2_value < r2_min):
                        continue
                    if (r2_max is not None) and (r2_value > r2_max):
                        continue
                normal_stocks.append({
                    'stock_code': stock_code,
                    'channel_state': channel_state,
                    'score': score
                })

        return normal_stocks

    def _check_cache(self, stock_code: str, date: datetime) -> bool:
        """检查分析结果缓存"""
        return (stock_code in self.results_cache and
                date in self.results_cache[stock_code])

    def _cache_result(self, stock_code: str, date: datetime, result: Any):
        """缓存分析结果"""
        if stock_code not in self.results_cache:
            self.results_cache[stock_code] = {}

        self.results_cache[stock_code][date] = result

    def _check_score_cache(self, stock_code: str, date: datetime) -> bool:
        """检查评分缓存"""
        return (stock_code in self.scores_cache and
                date in self.scores_cache[stock_code])

    def _cache_score(self, stock_code: str, date: datetime, score: float):
        """缓存评分"""
        if stock_code not in self.scores_cache:
            self.scores_cache[stock_code] = {}

        self.scores_cache[stock_code][date] = score

    def clear_cache(self, stock_code: str = None):
        """
        清理缓存
        
        Args:
            stock_code: 股票代码，None表示清理所有
        """
        if stock_code:
            if stock_code in self.results_cache:
                del self.results_cache[stock_code]
            if stock_code in self.scores_cache:
                del self.scores_cache[stock_code]
        else:
            self.results_cache.clear()
            self.scores_cache.clear()

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        获取分析统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.analysis_stats.copy()

        # 计算成功率
        if stats['total_analyses'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['total_analyses']
        else:
            stats['success_rate'] = 0.0

        # 计算缓存命中率
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0

        # 缓存状态
        stats['cache_status'] = {
            'cached_stocks': len(self.results_cache),
            'total_cached_results': sum(len(dates) for dates in self.results_cache.values()),
            'total_cached_scores': sum(len(dates) for dates in self.scores_cache.values())
        }

        return stats
