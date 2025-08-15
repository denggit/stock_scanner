#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易记录器
负责统一管理所有交易记录和日志
提供详细的交易数据记录和查询功能
"""

from datetime import datetime
from typing import Dict, List, Any, Optional

from backend.utils.logger import setup_logger


class TradeLogger:
    """
    交易记录器
    
    功能：
    - 记录所有交易详情
    - 提供交易查询和统计
    - 支持交易过滤和分析
    - 管理交易历史记录
    """

    def __init__(self):
        """初始化交易记录器"""
        # 所有交易记录
        self.trades = []

        # 交易计数器
        self.trade_count = 0

        # 日志记录器
        self.logger = setup_logger("backtest")

        # 交易统计
        self.statistics = {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }

    def log_trade(self, action: str, stock_code: str, size: int, price: float,
                  date: datetime, reason: str = '', confidence: float = 0.0,
                  returns: float = None, profit_amount: float = None,
                  buy_price: float = None, holding_days: int = None,
                  trade_cost: float = None, **additional_data) -> int:
        """
        记录交易
        
        Args:
            action: 交易动作 ('BUY', 'SELL')
            stock_code: 股票代码
            size: 交易数量
            price: 交易价格
            date: 交易日期
            reason: 交易原因
            confidence: 信心度 (0-1)
            returns: 收益率百分比（卖出时）
            profit_amount: 绝对收益金额（卖出时）
            buy_price: 买入价格（卖出时）
            holding_days: 持仓天数（卖出时）
            trade_cost: 交易成本（佣金+印花税+过户费等）
            **additional_data: 其他附加数据
            
        Returns:
            交易ID
        """
        # 生成交易ID
        trade_id = self.trade_count + 1

        # 计算交易金额
        trade_value = size * price

        # 构建基础交易记录
        trade_record = {
            # 基础信息
            'trade_id': trade_id,
            '交易日期': date,
            '交易动作': action,
            '股票代码': stock_code,
            '交易数量': size,
            '交易价格': price,
            '交易金额': trade_value,
            '交易原因': reason,
            '信心度': confidence,

            # 兼容性字段
            'date': date,
            'action': action,
            'stock_code': stock_code,
            'quantity': size,
            'size': size,
            'price': price,
            'value': trade_value,
            'reason': reason,
            'confidence': confidence,

            # 时间戳
            'logged_at': datetime.now()
        }

        # 添加收益相关信息（卖出时）
        if action == 'SELL':
            trade_record.update({
                '收益率': returns,
                '绝对收益': profit_amount,
                '买入价格': buy_price,
                '持仓天数': holding_days,
                '交易成本': trade_cost,

                # 兼容性字段
                'returns': returns,
                'profit_amount': profit_amount,
                'buy_price': buy_price,
                'holding_days': holding_days,
                'trade_cost': trade_cost
            })

            # 如果有收益率信息，添加额外字段
            if returns is not None:
                trade_record['收益率百分比'] = returns
                if profit_amount is not None:
                    if profit_amount >= 0:
                        trade_record['交易结果'] = '盈利'
                    else:
                        trade_record['交易结果'] = '亏损'

        # 添加其他附加数据
        trade_record.update(additional_data)

        # 添加到交易记录
        self.trades.append(trade_record)
        self.trade_count += 1

        # 更新统计信息
        self._update_statistics(trade_record)

        # 记录日志
        self._log_trade_details(trade_record)

        return trade_id

    def _update_statistics(self, trade_record: Dict[str, Any]):
        """
        更新交易统计信息
        
        Args:
            trade_record: 交易记录
        """
        self.statistics['total_trades'] += 1

        action = trade_record['交易动作']

        if action == 'BUY':
            self.statistics['buy_trades'] += 1
        elif action == 'SELL':
            self.statistics['sell_trades'] += 1

            # 更新盈亏统计
            profit_amount = trade_record.get('绝对收益')
            if profit_amount is not None:
                if profit_amount > 0:
                    self.statistics['profitable_trades'] += 1
                    self.statistics['total_profit'] += profit_amount
                elif profit_amount < 0:
                    self.statistics['losing_trades'] += 1
                    self.statistics['total_loss'] += abs(profit_amount)

    def _log_trade_details(self, trade_record: Dict[str, Any]):
        """
        记录交易详情到日志
        
        Args:
            trade_record: 交易记录
        """
        action = trade_record['交易动作']
        stock_code = trade_record['股票代码']
        size = trade_record['交易数量']
        price = trade_record['交易价格']

        if action == 'BUY':
            self.logger.info(f"交易记录: 买入 {stock_code} {size}股 @ {price:.2f}元")
        elif action == 'SELL':
            returns = trade_record.get('收益率')
            profit_amount = trade_record.get('绝对收益')
            holding_days = trade_record.get('持仓天数', 0)

            if returns is not None and profit_amount is not None:
                profit_sign = "+" if profit_amount >= 0 else ""
                returns_sign = "+" if returns >= 0 else ""
                self.logger.info(
                    f"交易记录: 卖出 {stock_code} {size}股 @ {price:.2f}元, "
                    f"收益: {profit_sign}{profit_amount:.2f}元 ({returns_sign}{returns:.2f}%), "
                    f"持仓{holding_days}天"
                )
            else:
                self.logger.info(f"交易记录: 卖出 {stock_code} {size}股 @ {price:.2f}元")

    def get_trades(self) -> List[Dict[str, Any]]:
        """
        获取所有交易记录（主要接口）
        
        Returns:
            所有交易记录列表
        """
        return self.trades.copy()

    def get_all_trades(self) -> List[Dict[str, Any]]:
        """
        获取所有交易记录（别名方法）
        
        Returns:
            所有交易记录列表
        """
        return self.get_trades()

    def get_trades_by_action(self, action: str) -> List[Dict[str, Any]]:
        """
        按交易动作获取交易记录
        
        Args:
            action: 交易动作 ('BUY', 'SELL')
            
        Returns:
            符合条件的交易记录列表
        """
        return [trade for trade in self.trades if trade['交易动作'] == action]

    def get_trades_by_stock(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        按股票代码获取交易记录
        
        Args:
            stock_code: 股票代码
            
        Returns:
            该股票的所有交易记录
        """
        return [trade for trade in self.trades if trade['股票代码'] == stock_code]

    def get_trades_by_date_range(self, start_date: datetime,
                                 end_date: datetime) -> List[Dict[str, Any]]:
        """
        按日期范围获取交易记录
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            日期范围内的交易记录
        """
        return [
            trade for trade in self.trades
            if start_date <= trade['交易日期'] <= end_date
        ]

    def get_profitable_trades(self) -> List[Dict[str, Any]]:
        """
        获取盈利的交易记录
        
        Returns:
            盈利交易记录列表
        """
        return [
            trade for trade in self.trades
            if (trade['交易动作'] == 'SELL' and
                trade.get('绝对收益', 0) > 0)
        ]

    def get_losing_trades(self) -> List[Dict[str, Any]]:
        """
        获取亏损的交易记录
        
        Returns:
            亏损交易记录列表
        """
        return [
            trade for trade in self.trades
            if (trade['交易动作'] == 'SELL' and
                trade.get('绝对收益', 0) < 0)
        ]

    def get_trade_count(self) -> int:
        """
        获取交易总数
        
        Returns:
            交易总数
        """
        return self.trade_count

    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        获取交易统计信息
        
        Returns:
            交易统计字典
        """
        stats = self.statistics.copy()

        # 计算衍生统计
        if stats['sell_trades'] > 0:
            stats['win_rate'] = (stats['profitable_trades'] / stats['sell_trades']) * 100
            stats['loss_rate'] = (stats['losing_trades'] / stats['sell_trades']) * 100
        else:
            stats['win_rate'] = 0.0
            stats['loss_rate'] = 0.0

        # 计算平均盈亏
        if stats['profitable_trades'] > 0:
            stats['avg_profit'] = stats['total_profit'] / stats['profitable_trades']
        else:
            stats['avg_profit'] = 0.0

        if stats['losing_trades'] > 0:
            stats['avg_loss'] = stats['total_loss'] / stats['losing_trades']
        else:
            stats['avg_loss'] = 0.0

        # 净盈亏
        stats['net_profit'] = stats['total_profit'] - stats['total_loss']

        # 盈亏比
        if stats['avg_loss'] > 0:
            stats['profit_loss_ratio'] = stats['avg_profit'] / stats['avg_loss']
        else:
            stats['profit_loss_ratio'] = float('inf') if stats['avg_profit'] > 0 else 0.0

        return stats

    def get_trade_summary_by_stock(self) -> Dict[str, Dict[str, Any]]:
        """
        按股票获取交易汇总
        
        Returns:
            按股票分组的交易汇总 {股票代码: 汇总信息}
        """
        stock_summary = {}

        for trade in self.trades:
            stock_code = trade['股票代码']

            if stock_code not in stock_summary:
                stock_summary[stock_code] = {
                    'total_trades': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'total_volume': 0,
                    'total_value': 0.0,
                    'total_profit': 0.0,
                    'profitable_trades': 0,
                    'losing_trades': 0
                }

            summary = stock_summary[stock_code]
            summary['total_trades'] += 1
            summary['total_volume'] += trade['交易数量']
            summary['total_value'] += trade['交易金额']

            if trade['交易动作'] == 'BUY':
                summary['buy_trades'] += 1
            elif trade['交易动作'] == 'SELL':
                summary['sell_trades'] += 1

                profit = trade.get('绝对收益', 0)
                if profit > 0:
                    summary['profitable_trades'] += 1
                elif profit < 0:
                    summary['losing_trades'] += 1

                summary['total_profit'] += profit or 0

        return stock_summary

    def export_trades_to_dataframe(self):
        """
        导出交易记录为DataFrame
        
        Returns:
            包含所有交易记录的DataFrame
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.trades)
        except ImportError:
            self.logger.warning("pandas未安装，无法导出DataFrame")
            return None

    def clear_trades(self):
        """清空所有交易记录"""
        self.trades.clear()
        self.trade_count = 0

        # 重置统计
        self.statistics = {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }

        self.logger.info("已清空所有交易记录")

    def get_last_trade(self) -> Optional[Dict[str, Any]]:
        """
        获取最后一笔交易记录
        
        Returns:
            最后一笔交易记录，无记录返回None
        """
        return self.trades[-1] if self.trades else None

    def get_trade_by_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        根据交易ID获取交易记录
        
        Args:
            trade_id: 交易ID
            
        Returns:
            交易记录，未找到返回None
        """
        for trade in self.trades:
            if trade.get('trade_id') == trade_id:
                return trade
        return None

    def filter_trades(self, **filters) -> List[Dict[str, Any]]:
        """
        根据条件过滤交易记录
        
        Args:
            **filters: 过滤条件，如 action='BUY', stock_code='000001'
            
        Returns:
            符合条件的交易记录列表
        """
        filtered_trades = []

        for trade in self.trades:
            match = True

            for key, value in filters.items():
                # 支持中英文字段名
                trade_value = trade.get(key) or trade.get({
                                                              'action': '交易动作',
                                                              'stock_code': '股票代码',
                                                              'date': '交易日期',
                                                              'size': '交易数量',
                                                              'price': '交易价格',
                                                              'value': '交易金额'
                                                          }.get(key, key))

                if trade_value != value:
                    match = False
                    break

            if match:
                filtered_trades.append(trade)

        return filtered_trades
