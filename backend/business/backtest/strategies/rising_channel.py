#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道回测策略
基于上升通道回归分析的月度调仓策略

策略逻辑：
1. 每月最后一天扫描所有股票，找出通道状态为NORMAL的股票
2. 在下一个月的第一个交易日买入，平均分配资金
3. 持有到月底，然后重新调仓
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import backtrader as bt

from ..core.base_strategy import BaseStrategy
from ..core.trading_rules import ASHARE_RULES
from backend.utils.indicators import CalIndicators


class RisingChannelBacktestStrategy(BaseStrategy):
    """
    上升通道回测策略
    
    策略特点：
    - 月度调仓：每月最后一天扫描，下月第一个交易日买入
    - 平均分配：符合条件的股票平均分配资金
    - 通道筛选：只买入通道状态为NORMAL的股票
    - A股规则：严格遵守A股交易规则
    """
    
    def __init__(self, 
                 max_positions: int = 10,
                 min_channel_score: float = 60.0,
                 **params):
        """
        初始化策略参数
        
        Args:
            max_positions: 最大持仓数量，默认10
            min_channel_score: 最小通道评分，默认60.0
            **params: 上升通道计算参数字典，支持以下参数：
                - k: 通道斜率参数，默认2.0
                - L_max: 最大通道长度，默认120
                - delta_cut: 通道宽度参数，默认5
                - pivot_m: 枢轴点参数，默认3
                - gain_trigger: 收益触发阈值，默认0.30
                - beta_delta: Beta变化阈值，默认0.15
                - break_days: 突破确认天数，默认3
                - reanchor_fail_max: 重新锚定最大失败次数，默认2
                - min_data_points: 最小数据点数，默认60
                - R2_min: 最小R²值，默认0.20
                - width_pct_min: 最小通道宽度百分比，默认0.04
                - width_pct_max: 最大通道宽度百分比，默认0.15
        """
        super().__init__()
        
        # 策略基本参数
        self.max_positions = max_positions
        self.min_channel_score = min_channel_score
        
        # 上升通道参数（使用**params方式，支持部分参数传入）
        self.channel_params = {
            "k": params.get('k', 2.0),
            "L_max": params.get('L_max', 120),
            "delta_cut": params.get('delta_cut', 5),
            "pivot_m": params.get('pivot_m', 3),
            "gain_trigger": params.get('gain_trigger', 0.30),
            "beta_delta": params.get('beta_delta', 0.15),
            "break_days": params.get('break_days', 3),
            "reanchor_fail_max": params.get('reanchor_fail_max', 2),
            "min_data_points": params.get('min_data_points', 60),
            "R2_min": params.get('R2_min', 0.20),
            "width_pct_min": params.get('width_pct_min', 0.04),
            "width_pct_max": params.get('width_pct_max', 0.15)
        }
        
        # 策略状态
        self.last_rebalance_date = None
        self.current_positions = {}  # {股票代码: 持仓信息}
        self.candidate_stocks = []  # 候选股票列表
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 交易记录
        self.rebalance_records = []
        self.channel_analysis_records = []
    
    def next(self):
        """
        策略核心逻辑
        每个交易日都会调用此方法
        """
        current_date = self.data.datetime.date()
        
        # 检查是否需要调仓（每月最后一天）
        if self._should_rebalance(current_date):
            self._perform_rebalance(current_date)
        
        # 检查是否需要卖出（持仓股票通道状态变化）
        self._check_exit_signals(current_date)
    
    def _should_rebalance(self, current_date: datetime.date) -> bool:
        """
        检查是否应该调仓
        
        Args:
            current_date: 当前日期
            
        Returns:
            是否应该调仓
        """
        # 如果是第一次运行，需要调仓
        if self.last_rebalance_date is None:
            return True
        
        # 检查是否是新的月份
        last_month = self.last_rebalance_date.month
        current_month = current_date.month
        last_year = self.last_rebalance_date.year
        current_year = current_date.year
        
        # 跨月或跨年都需要调仓
        return (current_month != last_month) or (current_year != last_year)
    
    def _perform_rebalance(self, current_date: datetime.date):
        """
        执行调仓操作
        
        Args:
            current_date: 当前日期
        """
        self.logger.info(f"开始调仓: {current_date}")
        
        # 1. 卖出所有现有持仓
        self._sell_all_positions(current_date)
        
        # 2. 扫描候选股票（这里需要外部数据，实际使用时需要修改）
        # 在实际回测中，我们需要获取所有股票的数据
        # 这里先使用模拟数据演示逻辑
        self._scan_candidate_stocks(current_date)
        
        # 3. 买入候选股票
        self._buy_candidate_stocks(current_date)
        
        # 4. 更新调仓日期
        self.last_rebalance_date = current_date
        
        # 5. 记录调仓信息
        self._record_rebalance(current_date)
    
    def _sell_all_positions(self, current_date: datetime.date):
        """
        卖出所有持仓
        
        Args:
            current_date: 当前日期
        """
        for stock_code, position_info in self.current_positions.items():
            try:
                # 计算卖出数量
                sell_quantity = position_info['quantity']
                
                # 执行卖出
                if sell_quantity > 0:
                    # 这里需要根据实际的backtrader数据源来执行卖出
                    # 由于这是多股票策略，需要特殊处理
                    self.logger.info(f"卖出 {stock_code}: {sell_quantity} 股")
                    
                    # 记录卖出交易
                    self._log_trade(
                        "SELL", 
                        sell_quantity, 
                        self.dataclose[0],
                        stock_code=stock_code
                    )
                    
            except Exception as e:
                self.logger.error(f"卖出 {stock_code} 失败: {e}")
        
        # 清空持仓记录
        self.current_positions.clear()
    
    def _scan_candidate_stocks(self, current_date: datetime.date):
        """
        扫描候选股票
        在实际使用中，这里需要获取所有股票的数据并计算上升通道
        
        Args:
            current_date: 当前日期
        """
        self.logger.info("开始扫描候选股票...")
        
        # 这里应该获取所有股票的数据
        # 由于backtrader的限制，这里先使用模拟数据
        # 实际使用时需要修改为获取真实股票数据
        
        # 模拟候选股票（实际使用时需要替换为真实数据）
        self.candidate_stocks = [
            {
                'code': '000001.SZ',
                'name': '平安银行',
                'channel_score': 85.5,
                'channel_status': 'NORMAL',
                'beta': 0.025,
                'r2': 0.65
            },
            {
                'code': '000002.SZ', 
                'name': '万科A',
                'channel_score': 72.3,
                'channel_status': 'NORMAL',
                'beta': 0.018,
                'r2': 0.58
            }
        ]
        
        # 筛选符合条件的股票
        qualified_stocks = []
        for stock in self.candidate_stocks:
            if (stock['channel_status'] == 'NORMAL' and 
                stock['channel_score'] >= self.min_channel_score):
                qualified_stocks.append(stock)
        
        # 按评分排序，取前N只
        qualified_stocks.sort(key=lambda x: x['channel_score'], reverse=True)
        self.candidate_stocks = qualified_stocks[:self.max_positions]
        
        self.logger.info(f"找到 {len(self.candidate_stocks)} 只候选股票")
        
        # 记录分析结果
        self._record_channel_analysis(current_date, self.candidate_stocks)
    
    def _buy_candidate_stocks(self, current_date: datetime.date):
        """
        买入候选股票
        
        Args:
            current_date: 当前日期
        """
        if not self.candidate_stocks:
            self.logger.info("没有候选股票，跳过买入")
            return
        
        # 计算每只股票的资金分配
        available_cash = self.broker.getcash()
        cash_per_stock = available_cash / len(self.candidate_stocks)
        
        self.logger.info(f"可用资金: {available_cash:.2f}, 每只股票分配: {cash_per_stock:.2f}")
        
        for stock in self.candidate_stocks:
            try:
                # 计算买入数量
                current_price = self.dataclose[0]  # 这里需要获取对应股票的价格
                buy_quantity = int(cash_per_stock / current_price)
                
                # 调整数量符合A股规则
                buy_quantity = ASHARE_RULES.adjust_trade_quantity(buy_quantity)
                
                if buy_quantity > 0:
                    # 执行买入
                    self.logger.info(f"买入 {stock['code']}: {buy_quantity} 股")
                    
                    # 记录买入交易
                    self._log_trade(
                        "BUY",
                        buy_quantity,
                        current_price,
                        stock_code=stock['code']
                    )
                    
                    # 记录持仓信息
                    self.current_positions[stock['code']] = {
                        'quantity': buy_quantity,
                        'buy_price': current_price,
                        'buy_date': current_date,
                        'channel_score': stock['channel_score']
                    }
                    
            except Exception as e:
                self.logger.error(f"买入 {stock['code']} 失败: {e}")
    
    def _check_exit_signals(self, current_date: datetime.date):
        """
        检查退出信号
        
        Args:
            current_date: 当前日期
        """
        # 这里可以添加额外的退出条件
        # 比如通道状态变为非NORMAL，或者达到止损条件等
        pass
    
    def _record_rebalance(self, current_date: datetime.date):
        """
        记录调仓信息
        
        Args:
            current_date: 当前日期
        """
        record = {
            'date': current_date,
            'candidate_count': len(self.candidate_stocks),
            'position_count': len(self.current_positions),
            'total_value': self.broker.getvalue(),
            'cash': self.broker.getcash(),
            'candidates': self.candidate_stocks.copy()
        }
        
        self.rebalance_records.append(record)
    
    def _record_channel_analysis(self, current_date: datetime.date, stocks: List[Dict]):
        """
        记录通道分析结果
        
        Args:
            current_date: 当前日期
            stocks: 股票列表
        """
        record = {
            'date': current_date,
            'analysis_count': len(stocks),
            'stocks': stocks.copy()
        }
        
        self.channel_analysis_records.append(record)
    
    def _log_trade(self, action: str, size: int, price: float, stock_code: str = None):
        """
        记录交易信息（重写父类方法以支持多股票）
        
        Args:
            action: 交易动作 (BUY/SELL)
            size: 交易数量
            price: 交易价格
            stock_code: 股票代码
        """
        trade = {
            "id": self.trade_count,
            "date": self.data.datetime.date(),
            "action": action,
            "stock_code": stock_code,
            "price": price,
            "size": size,
            "value": price * size,
            "cash": self.broker.getcash(),
            "portfolio_value": self.broker.getvalue()
        }
        
        self.trades.append(trade)
        self.trade_count += 1
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            策略信息字典
        """
        return {
            "strategy_name": "上升通道回测策略",
            "parameters": self._get_parameters(),
            "trade_count": self.trade_count,
            "current_positions": len(self.current_positions),
            "current_cash": self.broker.getcash(),
            "portfolio_value": self.broker.getvalue(),
            "rebalance_count": len(self.rebalance_records),
            "last_rebalance_date": self.last_rebalance_date
        }
    
    def _get_parameters(self) -> Dict[str, Any]:
        """
        获取策略参数
        
        Returns:
            策略参数字典
        """
        return {
            "max_positions": self.max_positions,
            "min_channel_score": self.min_channel_score,
            "channel_params": self.channel_params
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取策略表现摘要
        
        Returns:
            表现摘要字典
        """
        return {
            "total_rebalances": len(self.rebalance_records),
            "total_trades": self.trade_count,
            "current_positions": len(self.current_positions),
            "portfolio_value": self.broker.getvalue(),
            "cash": self.broker.getcash(),
            "rebalance_records": self.rebalance_records,
            "channel_analysis_records": self.channel_analysis_records
        }


# 便捷函数，用于创建策略实例
def create_rising_channel_backtest_strategy(
    max_positions: int = 10,
    min_channel_score: float = 60.0,
    **params
) -> RisingChannelBacktestStrategy:
    """
    创建上升通道回测策略实例
    
    Args:
        max_positions: 最大持仓数量，默认10
        min_channel_score: 最小通道评分，默认60.0
        **params: 上升通道计算参数字典，支持以下参数：
            - k: 通道斜率参数，默认2.0
            - L_max: 最大通道长度，默认120
            - delta_cut: 通道宽度参数，默认5
            - pivot_m: 枢轴点参数，默认3
            - gain_trigger: 收益触发阈值，默认0.30
            - beta_delta: Beta变化阈值，默认0.15
            - break_days: 突破确认天数，默认3
            - reanchor_fail_max: 重新锚定最大失败次数，默认2
            - min_data_points: 最小数据点数，默认60
            - R2_min: 最小R²值，默认0.20
            - width_pct_min: 最小通道宽度百分比，默认0.04
            - width_pct_max: 最大通道宽度百分比，默认0.15
            
    Returns:
        策略实例
    """
    return RisingChannelBacktestStrategy(
        max_positions=max_positions,
        min_channel_score=min_channel_score,
        **params
    ) 