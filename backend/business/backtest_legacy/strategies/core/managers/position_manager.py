#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仓位管理器
负责管理策略的所有持仓信息
使用单一职责原则，专注于持仓管理
"""

from datetime import datetime
from typing import Dict, Any, Optional, List


class PositionManager:
    """
    仓位管理器
    
    功能：
    - 记录和管理所有股票持仓
    - 提供持仓查询功能
    - 计算持仓统计信息
    """

    def __init__(self):
        """初始化仓位管理器"""
        # 当前持仓信息 {股票代码: 持仓信息}
        self._positions = {}

        # 持仓历史记录
        self._position_history = []

    def add_position(self, stock_code: str, shares: int, buy_price: float, buy_date: datetime):
        """
        添加持仓
        
        Args:
            stock_code: 股票代码
            shares: 持仓数量
            buy_price: 买入价格
            buy_date: 买入日期
        """
        position_info = {
            'stock_code': stock_code,
            'shares': shares,
            'buy_price': buy_price,
            'buy_date': buy_date,
            'buy_value': shares * buy_price,
            'created_at': datetime.now()
        }

        self._positions[stock_code] = position_info

        # 记录历史
        self._position_history.append({
            'action': 'ADD',
            'stock_code': stock_code,
            'position_info': position_info.copy(),
            'timestamp': datetime.now()
        })

    def remove_position(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        移除持仓
        
        Args:
            stock_code: 股票代码
            
        Returns:
            被移除的持仓信息，如果不存在返回None
        """
        if stock_code not in self._positions:
            return None

        position_info = self._positions[stock_code].copy()
        del self._positions[stock_code]

        # 记录历史
        self._position_history.append({
            'action': 'REMOVE',
            'stock_code': stock_code,
            'position_info': position_info,
            'timestamp': datetime.now()
        })

        return position_info

    def update_position(self, stock_code: str, shares: int = None, buy_price: float = None):
        """
        更新持仓信息
        
        Args:
            stock_code: 股票代码
            shares: 新的持仓数量（可选）
            buy_price: 新的买入价格（可选）
        """
        if stock_code not in self._positions:
            return

        position = self._positions[stock_code]
        old_info = position.copy()

        if shares is not None:
            position['shares'] = shares
            position['buy_value'] = shares * position['buy_price']

        if buy_price is not None:
            position['buy_price'] = buy_price
            position['buy_value'] = position['shares'] * buy_price

        # 记录历史
        self._position_history.append({
            'action': 'UPDATE',
            'stock_code': stock_code,
            'old_info': old_info,
            'new_info': position.copy(),
            'timestamp': datetime.now()
        })

    def get_position(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取指定股票的持仓信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            持仓信息字典，如果不存在返回None
        """
        return self._positions.get(stock_code)

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有持仓信息
        
        Returns:
            所有持仓信息字典
        """
        return self._positions.copy()

    def get_position_count(self) -> int:
        """
        获取当前持仓数量
        
        Returns:
            持仓股票数量
        """
        return len([p for p in self._positions.values() if p['shares'] > 0])

    def get_total_position_value(self) -> float:
        """
        获取总持仓价值（按买入价计算）
        
        Returns:
            总持仓价值
        """
        total_value = 0.0
        for position in self._positions.values():
            total_value += position['buy_value']
        return total_value

    def has_position(self, stock_code: str) -> bool:
        """
        检查是否持有指定股票
        
        Args:
            stock_code: 股票代码
            
        Returns:
            是否持有该股票
        """
        position = self._positions.get(stock_code)
        return position is not None and position['shares'] > 0

    def get_position_codes(self) -> List[str]:
        """
        获取所有持仓股票代码列表
        
        Returns:
            持仓股票代码列表
        """
        return [code for code, position in self._positions.items() if position['shares'] > 0]

    def get_position_statistics(self) -> Dict[str, Any]:
        """
        获取持仓统计信息
        
        Returns:
            持仓统计字典
        """
        positions = [p for p in self._positions.values() if p['shares'] > 0]

        if not positions:
            return {
                'total_positions': 0,
                'total_value': 0.0,
                'average_position_value': 0.0,
                'min_position_value': 0.0,
                'max_position_value': 0.0
            }

        values = [p['buy_value'] for p in positions]

        return {
            'total_positions': len(positions),
            'total_value': sum(values),
            'average_position_value': sum(values) / len(values),
            'min_position_value': min(values),
            'max_position_value': max(values)
        }

    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        获取持仓变动历史
        
        Returns:
            持仓历史记录列表
        """
        return self._position_history.copy()

    def clear_all_positions(self):
        """清空所有持仓"""
        # 记录清空操作
        self._position_history.append({
            'action': 'CLEAR_ALL',
            'cleared_positions': self._positions.copy(),
            'timestamp': datetime.now()
        })

        self._positions.clear()

    def calculate_position_returns(self, stock_code: str, current_price: float) -> Optional[Dict[str, float]]:
        """
        计算指定持仓的收益情况
        
        Args:
            stock_code: 股票代码
            current_price: 当前价格
            
        Returns:
            收益信息字典，包含收益率和绝对收益
        """
        position = self.get_position(stock_code)
        if not position:
            return None

        buy_price = position['buy_price']
        shares = position['shares']

        if buy_price <= 0:
            return None

        # 计算收益
        returns_pct = (current_price - buy_price) / buy_price * 100
        profit_amount = (current_price - buy_price) * shares
        current_value = current_price * shares

        return {
            'returns_pct': returns_pct,
            'profit_amount': profit_amount,
            'buy_value': position['buy_value'],
            'current_value': current_value,
            'buy_price': buy_price,
            'current_price': current_price,
            'shares': shares
        }
