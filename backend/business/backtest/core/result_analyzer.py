#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果分析器
使用观察者模式分析回测结果
"""

import backtrader as bt
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ResultAnalyzer:
    """
    结果分析器
    负责分析回测结果并生成报告
    """
    
    def __init__(self):
        """初始化结果分析器"""
        self.analyzers = {}
        self.results = {}
    
    def add_analyzer(self, name: str, analyzer_class, **kwargs):
        """
        添加分析器
        
        Args:
            name: 分析器名称
            analyzer_class: 分析器类
            **kwargs: 分析器参数
        """
        self.analyzers[name] = (analyzer_class, kwargs)
    
    def analyze(self, cerebro: bt.Cerebro, strategy_results: List) -> Dict[str, Any]:
        """
        分析回测结果
        
        Args:
            cerebro: backtrader引擎
            strategy_results: 策略结果列表
            
        Returns:
            分析结果字典
        """
        if not strategy_results:
            return {}
        
        strat = strategy_results[0]
        
        # 基础指标
        basic_metrics = self._calculate_basic_metrics(cerebro, strat)
        
        # 风险指标
        risk_metrics = self._calculate_risk_metrics(strat)
        
        # 交易指标
        trade_metrics = self._calculate_trade_metrics(strat)
        
        # 性能指标
        performance_metrics = self._calculate_performance_metrics(strat)
        
        # 合并所有指标
        all_metrics = {
            **basic_metrics,
            **risk_metrics,
            **trade_metrics,
            **performance_metrics
        }
        
        return {
            "metrics": all_metrics,
            "trades": getattr(strat, 'trades', []),
            "portfolio_value": cerebro.broker.getvalue(),
            "total_return": all_metrics.get('总收益率', 0)
        }
    
    def _calculate_basic_metrics(self, cerebro: bt.Cerebro, strat) -> Dict[str, Any]:
        """计算基础指标"""
        initial_cash = cerebro.broker.startingcash
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        return {
            "初始资金": initial_cash,
            "最终资金": final_value,
            "总收益率": total_return,
            "绝对收益": final_value - initial_cash
        }
    
    def _calculate_risk_metrics(self, strat) -> Dict[str, Any]:
        """计算风险指标"""
        try:
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            
            return {
                "夏普比率": sharpe.get('sharperatio', 0),
                "最大回撤": drawdown.get('max', {}).get('drawdown', 0) * 100,
                "最大回撤期间": drawdown.get('max', {}).get('len', 0),
                "当前回撤": drawdown.get('current', {}).get('drawdown', 0) * 100
            }
        except:
            return {
                "夏普比率": 0,
                "最大回撤": 0,
                "最大回撤期间": 0,
                "当前回撤": 0
            }
    
    def _calculate_trade_metrics(self, strat) -> Dict[str, Any]:
        """计算交易指标"""
        try:
            trades = strat.analyzers.trades.get_analysis()
            
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                "交易次数": total_trades,
                "盈利交易": won_trades,
                "亏损交易": lost_trades,
                "胜率": win_rate,
                "平均收益": trades.get('pnl', {}).get('average', 0),
                "最大单笔收益": trades.get('pnl', {}).get('max', 0),
                "最大单笔亏损": trades.get('pnl', {}).get('min', 0),
                "平均持仓时间": trades.get('len', {}).get('average', 0)
            }
        except:
            return {
                "交易次数": 0,
                "盈利交易": 0,
                "亏损交易": 0,
                "胜率": 0,
                "平均收益": 0,
                "最大单笔收益": 0,
                "最大单笔亏损": 0,
                "平均持仓时间": 0
            }
    
    def _calculate_performance_metrics(self, strat) -> Dict[str, Any]:
        """计算性能指标"""
        try:
            returns = strat.analyzers.returns.get_analysis()
            
            return {
                "年化收益率": returns.get('rnorm100', 0),
                "年化波动率": returns.get('std', 0) * 100,
                "信息比率": returns.get('ir', 0),
                "索提诺比率": returns.get('sortino', 0)
            }
        except:
            return {
                "年化收益率": 0,
                "年化波动率": 0,
                "信息比率": 0,
                "索提诺比率": 0
            }
    
    def generate_report(self, results: Dict[str, Any], strategy_name: str = "策略") -> str:
        """
        生成回测报告
        
        Args:
            results: 回测结果
            strategy_name: 策略名称
            
        Returns:
            报告文本
        """
        metrics = results.get('metrics', {})
        trades = results.get('trades', [])
        
        # 安全获取指标值，处理None值
        def safe_get(key, default=0):
            value = metrics.get(key, default)
            return value if value is not None else default
        
        report = f"""
=== {strategy_name} 回测报告 ===

【基础指标】
初始资金: {safe_get('初始资金', 0):,.2f}
最终资金: {safe_get('最终资金', 0):,.2f}
总收益率: {safe_get('总收益率', 0):.2f}%
绝对收益: {safe_get('绝对收益', 0):,.2f}

【风险指标】
夏普比率: {safe_get('夏普比率', 0):.4f}
最大回撤: {safe_get('最大回撤', 0):.2f}%
最大回撤期间: {safe_get('最大回撤期间', 0)} 天
当前回撤: {safe_get('当前回撤', 0):.2f}%

【交易指标】
交易次数: {safe_get('交易次数', 0)}
盈利交易: {safe_get('盈利交易', 0)}
亏损交易: {safe_get('亏损交易', 0)}
胜率: {safe_get('胜率', 0):.2f}%
平均收益: {safe_get('平均收益', 0):.2f}
最大单笔收益: {safe_get('最大单笔收益', 0):.2f}
最大单笔亏损: {safe_get('最大单笔亏损', 0):.2f}
平均持仓时间: {safe_get('平均持仓时间', 0):.1f} 天

【性能指标】
年化收益率: {safe_get('年化收益率', 0):.2f}%
年化波动率: {safe_get('年化波动率', 0):.2f}%
信息比率: {safe_get('信息比率', 0):.4f}
索提诺比率: {safe_get('索提诺比率', 0):.4f}

【交易记录】
总交易数: {len(trades)}
"""
        
        if trades:
            report += "\n前5笔交易:\n"
            for i, trade in enumerate(trades[:5]):
                report += f"  {i+1}. {trade['date']} {trade['action']} "
                report += f"价格:{trade['price']:.2f} 数量:{trade['size']} "
                if 'returns' in trade and trade['returns'] is not None:
                    report += f"收益率:{trade['returns']:.2f}%\n"
                else:
                    report += "\n"
        
        return report
    
    def compare_strategies(self, strategy_results: Dict[str, Dict]) -> str:
        """
        比较多个策略的结果
        
        Args:
            strategy_results: 策略结果字典 {策略名: 结果}
            
        Returns:
            比较报告
        """
        if not strategy_results:
            return "没有策略结果可比较"
        
        report = "=== 策略比较报告 ===\n\n"
        report += f"{'策略名称':<15} {'总收益率':<10} {'夏普比率':<10} {'最大回撤':<10} {'交易次数':<10} {'胜率':<10}\n"
        report += "-" * 80 + "\n"
        
        for name, results in strategy_results.items():
            metrics = results.get('metrics', {})
            
            # 安全获取指标值，处理None值
            def safe_get(key, default=0):
                value = metrics.get(key, default)
                return value if value is not None else default
            
            total_return = safe_get('总收益率', 0)
            sharpe = safe_get('夏普比率', 0)
            drawdown = safe_get('最大回撤', 0)
            trades = safe_get('交易次数', 0)
            win_rate = safe_get('胜率', 0)
            
            report += f"{name:<15} {total_return:<10.2f} {sharpe:<10.4f} "
            report += f"{drawdown:<10.2f} {trades:<10.0f} {win_rate:<10.2f}\n"
        
        return report 