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
import logging


class ResultAnalyzer:
    """
    结果分析器
    负责分析回测结果并生成报告
    """
    
    def __init__(self):
        """初始化结果分析器"""
        self.analyzers = {}
        self.results = {}
        # 添加日志记录器
        self.logger = logging.getLogger("backtest")
    
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
            
            # 调试信息：打印原始回撤数据
            self.logger.debug(f"原始回撤数据: {drawdown}")
            
            # 获取最大回撤数据
            max_drawdown_data = drawdown.get('max', {})
            current_drawdown_data = drawdown.get('current', {})
            
            # 提取回撤值
            max_drawdown = max_drawdown_data.get('drawdown', 0)
            current_drawdown = current_drawdown_data.get('drawdown', 0)
            
            # 调试信息
            self.logger.debug(f"最大回撤原始值: {max_drawdown}")
            self.logger.debug(f"当前回撤原始值: {current_drawdown}")
            
            # backtrader的drawdown返回的是0-1之间的小数，需要转换为百分比
            # 但需要检查是否为None或异常值
            if max_drawdown is not None and isinstance(max_drawdown, (int, float)):
                # 检查值的范围来判断是否需要转换
                if max_drawdown > 1.0:
                    # 如果值大于1，说明已经是百分比形式，直接使用
                    max_drawdown_pct = max_drawdown
                else:
                    # 如果值小于等于1，说明是小数形式，需要转换为百分比
                    max_drawdown_pct = max_drawdown * 100
                
                # 确保值在合理范围内
                max_drawdown_pct = max(0.0, min(max_drawdown_pct, 100.0))
            else:
                max_drawdown_pct = 0.0
                
            if current_drawdown is not None and isinstance(current_drawdown, (int, float)):
                # 同样的逻辑处理当前回撤
                if current_drawdown > 1.0:
                    current_drawdown_pct = current_drawdown
                else:
                    current_drawdown_pct = current_drawdown * 100
                
                current_drawdown_pct = max(0.0, min(current_drawdown_pct, 100.0))
            else:
                current_drawdown_pct = 0.0
            
            # 调试信息
            self.logger.debug(f"转换后最大回撤: {max_drawdown_pct:.2f}%")
            self.logger.debug(f"转换后当前回撤: {current_drawdown_pct:.2f}%")
            
            # 验证合理性：如果最大回撤接近100%，但总收益为正，说明计算有问题
            if max_drawdown_pct > 95.0:
                self.logger.warning(f"最大回撤异常高: {max_drawdown_pct:.2f}%，可能存在计算错误")
                # 尝试手动计算回撤
                max_drawdown_pct = self._manual_calculate_drawdown(strat)
            
            return {
                "夏普比率": sharpe.get('sharperatio', 0),
                "最大回撤": max_drawdown_pct,
                "最大回撤期间": max_drawdown_data.get('len', 0),
                "当前回撤": current_drawdown_pct
            }
        except Exception as e:
            self.logger.error(f"计算风险指标时出错: {e}")
            # 尝试手动计算
            try:
                manual_drawdown = self._manual_calculate_drawdown(strat)
                return {
                    "夏普比率": 0,
                    "最大回撤": manual_drawdown,
                    "最大回撤期间": 0,
                    "当前回撤": 0
                }
            except:
                return {
                    "夏普比率": 0,
                    "最大回撤": 0,
                    "最大回撤期间": 0,
                    "当前回撤": 0
                }
    
    def _manual_calculate_drawdown(self, strat) -> float:
        """
        手动计算最大回撤率
        
        Args:
            strat: 策略对象
            
        Returns:
            最大回撤率（百分比）
        """
        try:
            # 获取策略的权益曲线
            if hasattr(strat, 'value') and len(strat.value) > 0:
                # 使用策略的value数组
                values = strat.value.array
            elif hasattr(strat, '_value') and len(strat._value) > 0:
                # 使用策略的_value数组
                values = strat._value.array
            else:
                # 尝试从broker获取
                if hasattr(strat, 'broker') and hasattr(strat.broker, 'value'):
                    values = [strat.broker.value]
                else:
                    self.logger.warning("无法获取权益数据，返回默认回撤值")
                    return 0.0
            
            if len(values) < 2:
                return 0.0
            
            # 计算最大回撤
            peak = values[0]  # 初始值作为第一个峰值
            max_drawdown = 0.0
            
            for value in values:
                if value > peak:
                    peak = value  # 更新峰值
                else:
                    # 计算当前回撤
                    if peak > 0:
                        drawdown = (peak - value) / peak
                        max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown * 100
            
        except Exception as e:
            self.logger.error(f"手动计算回撤时出错: {e}")
            return 0.0
    
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
                # 格式化日期显示
                trade_date = trade.get('date', 'Unknown')
                if hasattr(trade_date, 'strftime'):
                    date_str = trade_date.strftime('%Y-%m-%d')
                else:
                    date_str = str(trade_date)
                
                report += f"  {i+1}. {date_str} {trade['action']} "
                report += f"价格:{trade['price']:.2f} "
                
                # 兼容不同的数量字段名
                quantity = trade.get('size', trade.get('quantity', 0))
                report += f"数量:{quantity} "
                
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