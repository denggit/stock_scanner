#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果分析器
使用观察者模式分析回测结果
"""

import logging
from typing import Dict, Any, List

import backtrader as bt


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
        # 添加详细的调试输出
        self.logger.info("开始计算交易指标...")
        
        try:
            # 首先尝试使用backtrader的TradeAnalyzer
            trades_analysis = None
            try:
                trades_analysis = strat.analyzers.trades.get_analysis()
                self.logger.debug(f"Backtrader交易分析结果: {trades_analysis}")
            except Exception as e:
                self.logger.debug(f"无法获取backtrader交易分析: {e}")

            total_trades = 0
            won_trades = 0
            lost_trades = 0
            
            if trades_analysis:
                total_trades = trades_analysis.get('total', {}).get('total', 0)
                won_trades = trades_analysis.get('won', {}).get('total', 0)
                lost_trades = trades_analysis.get('lost', {}).get('total', 0)
                self.logger.info(f"Backtrader分析器数据 - 总交易:{total_trades}, 盈利:{won_trades}, 亏损:{lost_trades}")

            # 如果backtrader分析器没有数据，使用策略自定义的交易记录
            if total_trades == 0:
                self.logger.info("Backtrader分析器无数据，尝试使用策略自定义交易记录...")
                
                strategy_trades = getattr(strat, 'trades', [])
                self.logger.info(f"策略交易记录总数: {len(strategy_trades)}")
                
                if not strategy_trades:
                    self.logger.warning("策略交易记录为空！")
                    return self._get_empty_trade_metrics()
                
                # 打印前几条交易记录的结构用于调试
                if len(strategy_trades) > 0:
                    self.logger.debug(f"第一条交易记录结构: {strategy_trades[0].keys()}")
                    self.logger.debug(f"第一条交易记录内容: {strategy_trades[0]}")
                
                # 分析策略自定义的交易记录
                all_trades = strategy_trades
                sell_trades = []
                buy_trades = []
                
                for trade in all_trades:
                    action = trade.get('action') or trade.get('交易动作', '')
                    if action == 'SELL':
                        sell_trades.append(trade)
                    elif action == 'BUY':
                        buy_trades.append(trade)
                
                self.logger.info(f"筛选结果 - 卖出交易:{len(sell_trades)}, 买入交易:{len(buy_trades)}")
                
                if not sell_trades:
                    self.logger.warning("没有找到卖出交易记录！")
                    return self._get_empty_trade_metrics()

                # 以卖出交易作为完整交易进行分析
                total_trades = len(sell_trades)
                won_trades = 0
                lost_trades = 0
                total_profit = 0
                max_profit = 0
                max_loss = 0
                total_holding_days = 0
                valid_holding_periods = 0

                self.logger.info(f"开始分析 {total_trades} 笔卖出交易...")

                for i, trade in enumerate(sell_trades):
                    # 调试输出每笔交易
                    self.logger.debug(f"分析第{i+1}笔交易: {trade}")
                    
                    # 尝试多种可能的收益率字段名
                    returns = None
                    for field in ['returns', '收益率', '收益率(%)', 'return_rate']:
                        if field in trade and trade[field] is not None:
                            returns = trade[field]
                            self.logger.debug(f"在字段'{field}'中找到收益率: {returns}")
                            break
                    
                    if returns is None:
                        self.logger.warning(f"第{i+1}笔交易缺少收益率数据: {trade}")
                        continue

                    # 处理numpy.float64类型和其他数值类型
                    try:
                        if hasattr(returns, 'item'):
                            returns = returns.item()
                        returns = float(returns)
                        self.logger.debug(f"转换后的收益率: {returns}")
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"收益率转换失败: {returns}, 错误: {e}")
                        continue

                    # 分类盈利和亏损交易
                    if returns > 0:
                        won_trades += 1
                        self.logger.debug(f"盈利交易: {returns:.2f}%")
                    elif returns < 0:
                        lost_trades += 1
                        self.logger.debug(f"亏损交易: {returns:.2f}%")
                    else:
                        # 收益率为0的交易计入亏损（或可以单独分类）
                        lost_trades += 1

                    # 计算收益金额（如果有交易金额）
                    trade_value = 0
                    for field in ['value', '交易金额', 'amount']:
                        if field in trade and trade[field] is not None:
                            try:
                                trade_value = float(trade[field])
                                if hasattr(trade[field], 'item'):
                                    trade_value = trade[field].item()
                                trade_value = float(trade_value)
                                break
                            except (ValueError, TypeError):
                                continue

                    if trade_value > 0:
                        profit_amount = trade_value * returns / 100
                        total_profit += profit_amount
                        max_profit = max(max_profit, profit_amount)
                        max_loss = min(max_loss, profit_amount)
                        self.logger.debug(f"交易金额: {trade_value:.2f}, 收益金额: {profit_amount:.2f}")
                    
                    # 计算持仓时间
                    buy_date = self._find_buy_date_for_sell(trade, buy_trades)
                    if buy_date:
                        sell_date = trade.get('date') or trade.get('交易日期')
                        if sell_date and buy_date:
                            try:
                                holding_days = self._calculate_holding_days(buy_date, sell_date)
                                if holding_days > 0:
                                    total_holding_days += holding_days
                                    valid_holding_periods += 1
                                    self.logger.debug(f"持仓天数: {holding_days}")
                            except Exception as e:
                                self.logger.debug(f"计算持仓天数失败: {e}")

                # 计算统计指标
                avg_profit = total_profit / total_trades if total_trades > 0 else 0
                win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                avg_holding_days = total_holding_days / valid_holding_periods if valid_holding_periods > 0 else 0

                self.logger.info(f"交易指标计算完成:")
                self.logger.info(f"  总交易数: {total_trades}")
                self.logger.info(f"  盈利交易: {won_trades}")
                self.logger.info(f"  亏损交易: {lost_trades}")
                self.logger.info(f"  胜率: {win_rate:.2f}%")
                self.logger.info(f"  平均收益: {avg_profit:.2f}")
                self.logger.info(f"  最大单笔收益: {max_profit:.2f}")
                self.logger.info(f"  最大单笔亏损: {max_loss:.2f}")
                self.logger.info(f"  平均持仓天数: {avg_holding_days:.1f}")

                return {
                    "交易次数": total_trades,
                    "盈利交易": won_trades,
                    "亏损交易": lost_trades,
                    "胜率": win_rate,
                    "平均收益": avg_profit,
                    "最大单笔收益": max_profit,
                    "最大单笔亏损": max_loss,
                    "平均持仓时间": avg_holding_days
                }

            # 如果backtrader分析器有数据，使用原有逻辑
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            self.logger.info(f"使用Backtrader分析器数据 - 胜率: {win_rate:.2f}%")

            return {
                "交易次数": total_trades,
                "盈利交易": won_trades,
                "亏损交易": lost_trades,
                "胜率": win_rate,
                "平均收益": trades_analysis.get('pnl', {}).get('average', 0),
                "最大单笔收益": trades_analysis.get('pnl', {}).get('max', 0),
                "最大单笔亏损": trades_analysis.get('pnl', {}).get('min', 0),
                "平均持仓时间": trades_analysis.get('len', {}).get('average', 0)
            }
            
        except Exception as e:
            self.logger.error(f"计算交易指标时发生异常: {e}")
            # 如果出错，尝试使用策略自定义交易记录作为备用
            try:
                strategy_trades = getattr(strat, 'trades', [])
                self.logger.info(f"异常处理 - 使用策略交易记录，总数: {len(strategy_trades)}")
                
                if not strategy_trades:
                    return self._get_empty_trade_metrics()
                
                sell_trades = [t for t in strategy_trades if 
                              t.get('action') == 'SELL' or t.get('交易动作') == 'SELL']

                total_trades = len(sell_trades)
                won_trades = 0
                lost_trades = 0

                for trade in sell_trades:
                    returns = None
                    for field in ['returns', '收益率', '收益率(%)', 'return_rate']:
                        if field in trade and trade[field] is not None:
                            returns = trade[field]
                            break

                    if returns is not None:
                        try:
                            if hasattr(returns, 'item'):
                                returns = returns.item()
                            returns = float(returns)

                            if returns > 0:
                                won_trades += 1
                            elif returns < 0:
                                lost_trades += 1
                            else:
                                lost_trades += 1
                        except (ValueError, TypeError):
                            lost_trades += 1

                win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                
                self.logger.info(f"异常处理结果 - 交易数:{total_trades}, 盈利:{won_trades}, 亏损:{lost_trades}, 胜率:{win_rate:.2f}%")

                return {
                    "交易次数": total_trades,
                    "盈利交易": won_trades,
                    "亏损交易": lost_trades,
                    "胜率": win_rate,
                    "平均收益": 0,
                    "最大单笔收益": 0,
                    "最大单笔亏损": 0,
                    "平均持仓时间": 0
                }
            except Exception as inner_e:
                self.logger.error(f"异常处理也失败: {inner_e}")
                return self._get_empty_trade_metrics()
    
    def _get_empty_trade_metrics(self) -> Dict[str, Any]:
        """返回空的交易指标"""
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
    
    def _find_buy_date_for_sell(self, sell_trade, buy_trades):
        """为卖出交易找到对应的买入日期"""
        try:
            stock_code = sell_trade.get('stock_code') or sell_trade.get('股票代码')
            sell_date = sell_trade.get('date') or sell_trade.get('交易日期')
            
            if not stock_code or not sell_date:
                return None
            
            # 查找同一股票最近的买入交易
            matching_buys = [
                t for t in buy_trades 
                if (t.get('stock_code') or t.get('股票代码')) == stock_code
            ]
            
            if not matching_buys:
                return None
            
            # 找到最近的买入日期（在卖出日期之前）
            valid_buys = []
            for buy_trade in matching_buys:
                buy_date = buy_trade.get('date') or buy_trade.get('交易日期')
                if buy_date and buy_date <= sell_date:
                    valid_buys.append((buy_date, buy_trade))
            
            if valid_buys:
                # 选择最近的买入
                valid_buys.sort(key=lambda x: x[0], reverse=True)
                return valid_buys[0][0]
            
            return None
            
        except Exception as e:
            self.logger.debug(f"查找买入日期失败: {e}")
            return None
    
    def _calculate_holding_days(self, buy_date, sell_date):
        """计算持仓天数"""
        try:
            from datetime import datetime
            
            # 处理不同的日期格式
            if hasattr(buy_date, 'strftime'):
                buy_dt = buy_date
            elif isinstance(buy_date, str):
                buy_dt = datetime.strptime(buy_date, '%Y-%m-%d')
            else:
                buy_dt = buy_date
                
            if hasattr(sell_date, 'strftime'):
                sell_dt = sell_date
            elif isinstance(sell_date, str):
                sell_dt = datetime.strptime(sell_date, '%Y-%m-%d')
            else:
                sell_dt = sell_date
            
            delta = sell_dt - buy_dt
            return delta.days
            
        except Exception as e:
            self.logger.debug(f"计算持仓天数失败: {e}")
            return 0

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

                report += f"  {i + 1}. {date_str} {trade['action']} "
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
