#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果分析器
使用观察者模式分析回测结果
"""

from datetime import datetime
from typing import Dict, Any, List

import backtrader as bt
import numpy as np

from backend.utils.logger import setup_logger


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
        self.logger = setup_logger("backtest")

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

        # 风险指标（调整为依赖cerebro，保持与年化指标一致）
        risk_metrics = self._calculate_risk_metrics(strat, cerebro)

        # 交易指标（优先使用策略自定义交易记录）
        trade_metrics = self._calculate_trade_metrics(strat)

        # 性能指标
        performance_metrics = self._calculate_performance_metrics(strat, cerebro)

        # 提取日收益序列（来自TimeReturn分析器），返回为 {date: daily_return_decimal}
        daily_returns = self._get_daily_returns_from_analyzer(strat)

        # 推导策略开始生效日期（剔除min_data_points之前的日期）
        active_start_date = self._derive_active_start_date(strat, daily_returns)

        # 合并所有指标
        all_metrics = {
            **basic_metrics,
            **risk_metrics,
            **trade_metrics,
            **performance_metrics
        }

        # 优先从trade_logger获取交易记录，保证与交易指标一致
        trade_records = []
        try:
            if hasattr(strat, 'trade_logger'):
                trade_records = strat.trade_logger.get_all_trades() or []
            elif hasattr(strat, 'trades'):
                trade_records = getattr(strat, 'trades', []) or []
        except Exception:
            trade_records = getattr(strat, 'trades', []) or []

        return {
            "metrics": all_metrics,
            "trades": trade_records,
            "portfolio_value": cerebro.broker.getvalue(),
            "total_return": all_metrics.get('总收益率', 0),
            "daily_returns": daily_returns,
            "active_start_date": active_start_date,
        }

    def _calculate_basic_metrics(self, cerebro: bt.Cerebro, strat) -> Dict[str, Any]:
        """计算基础指标"""
        try:
            initial_cash = cerebro.broker.startingcash
            final_cash = cerebro.broker.getcash()  # 现金
            final_value = cerebro.broker.getvalue()  # 总资产
            
            # 计算未平仓持仓价值
            position_value = final_value - final_cash
            
            # 检测异常情况并修复
            if final_value < final_cash:
                self.logger.error(f"异常：总资产({final_value:.2f})小于现金({final_cash:.2f})，这是不可能的！")
                self.logger.error("这可能是backtrader的bug或持仓计算错误")
                
                # 修复方案1：使用现金作为总资产（假设所有持仓都已平仓）
                corrected_final_value = final_cash
                self.logger.warning(f"修复：将总资产修正为现金值 {corrected_final_value:.2f}")
                
                # 修复方案2：如果策略有持仓记录，重新计算总资产
                if hasattr(strat, 'position_manager'):
                    try:
                        position_manager = strat.position_manager
                        manual_position_value = position_manager.get_total_position_value()
                        corrected_final_value = final_cash + manual_position_value
                        self.logger.warning(f"使用策略持仓管理器重新计算总资产: {corrected_final_value:.2f}")
                    except Exception as e:
                        self.logger.warning(f"无法使用策略持仓管理器重新计算: {e}")
                
                final_value = corrected_final_value
                position_value = final_value - final_cash
            
            # 统一基于总资产(final_value)计算核心指标
            net_profit_loss = final_value - initial_cash  # 净盈亏
            total_return_on_value = (final_value / initial_cash - 1) * 100 if initial_cash > 0 else 0
            
            # 现实策略（无杠杆）下总收益率不会小于 -100%，若资产异常为负，做展示层兜底
            if total_return_on_value < -100:
                total_return_on_value = -100.0

            self.logger.info(
                f"基础指标 - 初始资金: {initial_cash}, 最终现金: {final_cash}, 最终总资产: {final_value}")
            self.logger.info(
                f"未平仓持仓价值: {position_value:.2f}, 净盈亏: {net_profit_loss:.2f}, 总收益率: {total_return_on_value:.2f}%")
            
            # 检查其他异常情况
            if final_cash == 0 and position_value > 0:
                self.logger.warning("警告：最终现金为0，但有未平仓持仓，可能存在计算错误")
            if final_value < 0:
                self.logger.warning("警告：最终总资产为负值，可能存在计算错误")
            if position_value < 0:
                self.logger.warning("警告：未平仓持仓价值为负值，可能存在计算错误")

            return {
                "初始资金": initial_cash,
                "最终现金": final_cash,
                "最终总资产": final_value,
                "未平仓持仓价值": position_value,
                "净盈亏": net_profit_loss,
                "总收益率": total_return_on_value
            }
        except Exception as e:
            self.logger.error(f"计算基础指标时发生异常: {e}")
            return {
                "初始资金": 0,
                "最终总资产": 0,
                "净盈亏": 0,
                "总收益率": 0,
                "最终现金": 0,
                "未平仓持仓价值": 0
            }

    def _calculate_risk_metrics(self, strat, cerebro) -> Dict[str, Any]:
        """计算风险指标（与年化指标对齐）"""
        try:
            # 先尝试backtrader分析器（用于回撤）
            max_drawdown = 0
            max_drawdown_duration = 0
            current_drawdown = 0

            try:
                drawdown_analysis = strat.analyzers.drawdown.get_analysis()
                max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0) or 0
                max_drawdown_duration = drawdown_analysis.get('max', {}).get('len', 0) or 0
            except Exception as e:
                self.logger.debug(f"获取回撤数据失败: {e}")

            # 与性能指标一致的夏普比率：Sharpe = 年化收益率 / 年化波动率（rf≈0）
            perf = self._manual_calculate_performance_metrics(strat, cerebro)
            annual_return = perf.get('年化收益率', 0)
            annual_vol = perf.get('年化波动率', 0)
            # 避免复数/无穷：只在波动率为正且数值有效时计算
            try:
                annual_return = float(annual_return)
                annual_vol = float(annual_vol)
            except Exception:
                annual_return, annual_vol = 0.0, 0.0
            sharpe_ratio = (annual_return / annual_vol) if (annual_vol and annual_vol > 0) else 0.0

            # 最大回撤同样做合理范围约束（0~100%）
            max_drawdown = min(abs(max_drawdown), 100.0)
            self.logger.info(f"风险指标 - 夏普比率(统一口径): {sharpe_ratio:.4f}, 最大回撤: {max_drawdown:.2f}%")

            return {
                "夏普比率": sharpe_ratio,
                "最大回撤": max_drawdown,  # 确保回撤为正值且不过 100%
                "最大回撤期间": max_drawdown_duration,
                "当前回撤": abs(current_drawdown)
            }

        except Exception as e:
            self.logger.error(f"计算风险指标时发生异常: {e}")
            return {
                "夏普比率": 0,
                "最大回撤": 0,
                "最大回撤期间": 0,
                "当前回撤": 0
            }

    def _manual_calculate_risk_metrics(self, strat) -> Dict[str, Any]:
        """手动计算风险指标"""
        try:
            # 获取组合价值历史数据
            portfolio_values = []

            # 尝试从策略中获取价值记录
            if hasattr(strat, 'portfolio_values'):
                portfolio_values = strat.portfolio_values
            else:
                # 如果没有记录，使用初始和最终值估算
                initial_value = getattr(strat.broker, 'startingcash', 100000)
                final_value = strat.broker.getvalue()
                portfolio_values = [initial_value, final_value]

            if len(portfolio_values) < 2:
                return {'sharpe_ratio': 0, 'max_drawdown': 0, 'max_drawdown_duration': 0}

            # 计算收益率序列
            returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i - 1] > 0:
                    ret = (portfolio_values[i] / portfolio_values[i - 1] - 1)
                    returns.append(ret)

            if not returns:
                return {'sharpe_ratio': 0, 'max_drawdown': 0, 'max_drawdown_duration': 0}

            returns = np.array(returns)

            # 计算夏普比率
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

            # 计算最大回撤
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative / running_max - 1) * 100
            max_drawdown = abs(np.min(drawdown))

            self.logger.debug(f"手动计算 - 夏普比率: {sharpe_ratio:.4f}, 最大回撤: {max_drawdown:.2f}%")

            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': 0
            }

        except Exception as e:
            self.logger.debug(f"手动计算风险指标失败: {e}")
            return {'sharpe_ratio': 0, 'max_drawdown': 0, 'max_drawdown_duration': 0}

    def _calculate_trade_metrics(self, strat) -> Dict[str, Any]:
        """计算交易指标 - 改进版本"""
        self.logger.info("开始计算交易指标...")

        try:
            # 优先使用策略自定义的交易记录（trade_logger）
            strategy_trades = []
            try:
                if hasattr(strat, 'trade_logger'):
                    strategy_trades = strat.trade_logger.get_all_trades() or []
                elif hasattr(strat, 'trades'):
                    strategy_trades = getattr(strat, 'trades', []) or []
            except Exception:
                strategy_trades = getattr(strat, 'trades', []) or []

            self.logger.info(f"策略交易记录总数: {len(strategy_trades)}")

            if len(strategy_trades) > 0:
                self.logger.info("检测到策略自定义交易记录，直接使用策略数据进行分析")
                return self._calculate_manual_trade_metrics(strat)

            # 如果没有策略交易记录，尝试使用backtrader分析器
            trades_analysis = None
            total_trades = 0
            won_trades = 0
            lost_trades = 0

            try:
                trades_analysis = strat.analyzers.trades.get_analysis()
                self.logger.debug(f"Backtrader交易分析结果: {trades_analysis}")

                if trades_analysis:
                    total_trades = trades_analysis.get('total', {}).get('total', 0)
                    won_trades = trades_analysis.get('won', {}).get('total', 0)
                    lost_trades = trades_analysis.get('lost', {}).get('total', 0)
                    self.logger.info(
                        f"Backtrader分析器数据 - 总交易:{total_trades}, 盈利:{won_trades}, 亏损:{lost_trades}")
            except Exception as e:
                self.logger.debug(f"无法获取backtrader交易分析: {e}")

            if total_trades > 0:
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
            else:
                return self._get_empty_trade_metrics()

        except Exception as e:
            self.logger.error(f"计算交易指标时发生异常: {e}")
            return self._calculate_manual_trade_metrics(strat)

    def _calculate_manual_trade_metrics(self, strat) -> Dict[str, Any]:
        """
        手动计算交易指标 - 简化版本
        
        使用标准字段名，遵循"约定优于配置"原则：
        - action: 交易动作 (BUY/SELL)
        - returns: 收益率（百分比）
        - holding_days: 持仓天数
        - buy_price: 买入价格
        """
        self.logger.info("使用策略自定义交易记录进行手动计算...")

        # 获取交易记录
        strategy_trades = []
        try:
            if hasattr(strat, 'trade_logger'):
                strategy_trades = strat.trade_logger.get_all_trades() or []
            elif hasattr(strat, 'trades'):
                strategy_trades = getattr(strat, 'trades', []) or []
        except Exception:
            strategy_trades = getattr(strat, 'trades', []) or []
        
        self.logger.info(f"策略交易记录总数: {len(strategy_trades)}")

        if not strategy_trades:
            self.logger.warning("策略交易记录为空！")
            return self._get_empty_trade_metrics()

        # 筛选卖出交易（只有卖出交易才有收益率）
        sell_trades = [trade for trade in strategy_trades if trade.get('action') == 'SELL']
        
        self.logger.info(f"卖出交易数量: {len(sell_trades)}")

        if not sell_trades:
            self.logger.warning("没有找到卖出交易记录！")
            return self._get_empty_trade_metrics()

        # 直接分析卖出交易
        return self._analyze_sell_trades_simplified(sell_trades)

    def _analyze_sell_trades_simplified(self, sell_trades: List[Dict]) -> Dict[str, Any]:
        """
        简化版卖出交易分析 - 只使用标准字段名
        
        标准字段约定：
        - returns: 收益率（百分比）
        - holding_days: 持仓天数
        - buy_price: 买入价格
        """
        total_trades = len(sell_trades)
        won_trades = 0
        lost_trades = 0
        total_returns = 0
        max_profit = float('-inf')
        max_loss = float('inf')
        total_holding_days = 0
        valid_holding_periods = 0
        valid_returns = []

        self.logger.info(f"开始分析 {total_trades} 笔卖出交易...")

        for i, trade in enumerate(sell_trades):
            # 获取收益率 - 只使用标准字段
            returns = trade.get('returns')
            
            if returns is None:
                self.logger.warning(f"第{i + 1}笔交易缺少收益率数据，跳过")
                continue

            # 处理收益率数据类型
            try:
                if hasattr(returns, 'item'):
                    returns = returns.item()
                returns = float(returns)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"收益率转换失败: {returns}, 错误: {e}")
                continue

            # 记录有效的收益率
            valid_returns.append(returns)

            # 分类盈利和亏损交易
            if returns > 0:
                won_trades += 1
            elif returns < 0:
                lost_trades += 1
            else:
                # 收益率为0的交易计入亏损
                lost_trades += 1

            total_returns += returns
            max_profit = max(max_profit, returns)
            max_loss = min(max_loss, returns)

            # 获取持仓时间 - 只使用标准字段
            holding_days = trade.get('holding_days')
            if holding_days is not None and holding_days > 0:
                total_holding_days += holding_days
                valid_holding_periods += 1

        # 如果没有有效的收益率数据
        if not valid_returns:
            self.logger.warning("没有找到有效的收益率数据！")
            return self._get_empty_trade_metrics()

        # 计算平均指标
        effective_trades = len(valid_returns)
        avg_returns = total_returns / effective_trades if effective_trades > 0 else 0
        win_rate = (won_trades / effective_trades * 100) if effective_trades > 0 else 0
        avg_holding_days = total_holding_days / valid_holding_periods if valid_holding_periods > 0 else 0

        # 确保最大值有效
        if max_profit == float('-inf'):
            max_profit = 0
        if max_loss == float('inf'):
            max_loss = 0

        self.logger.info(f"交易分析完成:")
        self.logger.info(f"  有效交易数: {effective_trades}")
        self.logger.info(f"  盈利交易: {won_trades}")
        self.logger.info(f"  亏损交易: {lost_trades}")
        self.logger.info(f"  胜率: {win_rate:.2f}%")
        self.logger.info(f"  平均收益率: {avg_returns:.2f}%")
        self.logger.info(f"  最大单笔收益: {max_profit:.2f}%")
        self.logger.info(f"  最大单笔亏损: {max_loss:.2f}%")
        self.logger.info(f"  平均持仓天数: {avg_holding_days:.1f}")

        return {
            "交易次数": effective_trades,
            "盈利交易": won_trades,
            "亏损交易": lost_trades,
            "胜率": win_rate,
            "平均收益": avg_returns,
            "最大单笔收益": max_profit,
            "最大单笔亏损": abs(max_loss),  # 转换为正值显示
            "平均持仓时间": avg_holding_days,
            "收益率序列": valid_returns  # 添加收益率序列用于波动率计算
        }

    def _analyze_sell_trades_directly(self, sell_trades: List[Dict]) -> Dict[str, Any]:
        """
        直接分析卖出交易的收益率 - 简化版本
        
        注意：此方法已被 _analyze_sell_trades_simplified 替代，
        保留此方法是为了向后兼容，但建议使用简化版本。
        """
        self.logger.warning("使用已弃用的 _analyze_sell_trades_directly 方法，建议使用 _analyze_sell_trades_simplified")
        return self._analyze_sell_trades_simplified(sell_trades)

    def _create_trade_pairs(self, buy_trades: List[Dict], sell_trades: List[Dict]) -> List[Dict]:
        """
        创建买卖交易配对 - 简化版本
        
        使用标准字段名：
        - stock_code: 股票代码
        - date: 交易日期
        """
        trade_pairs = []

        # 按股票代码对买入交易进行分组和排序
        buy_trades_by_stock = {}
        for trade in buy_trades:
            stock_code = trade.get('stock_code')
            if stock_code and stock_code not in buy_trades_by_stock:
                buy_trades_by_stock[stock_code] = []
            if stock_code:
                buy_trades_by_stock[stock_code].append(trade)

        # 对每只股票的买入交易按日期排序
        for stock_code in buy_trades_by_stock:
            buy_trades_by_stock[stock_code].sort(key=lambda x: x.get('date', datetime.min))

        # 为每个卖出交易寻找对应的买入交易
        for sell_trade in sell_trades:
            stock_code = sell_trade.get('stock_code')
            sell_date = sell_trade.get('date')

            if not stock_code or not sell_date:
                continue

            # 查找该股票的买入交易
            if stock_code in buy_trades_by_stock:
                buy_trades_for_stock = buy_trades_by_stock[stock_code]

                # 寻找最近的买入交易（FIFO原则）
                matching_buy = None
                for i, buy_trade in enumerate(buy_trades_for_stock):
                    buy_date = buy_trade.get('date')
                    if buy_date and buy_date <= sell_date:
                        matching_buy = buy_trade
                        # 从可用买入交易中移除（避免重复配对）
                        buy_trades_for_stock.pop(i)
                        break

                if matching_buy:
                    # 计算收益率
                    returns = self._calculate_trade_returns(matching_buy, sell_trade)

                    trade_pair = {
                        'buy_trade': matching_buy,
                        'sell_trade': sell_trade,
                        'returns': returns,
                        'stock_code': stock_code,
                        'buy_date': matching_buy.get('date'),
                        'sell_date': sell_date
                    }
                    trade_pairs.append(trade_pair)

        return trade_pairs

    def _calculate_trade_returns(self, buy_trade: Dict, sell_trade: Dict) -> float:
        """
        计算交易收益率 - 简化版本
        
        使用标准字段名：
        - returns: 收益率（百分比）
        - price: 交易价格
        """
        try:
            # 首先尝试从卖出交易中直接获取收益率
            returns = sell_trade.get('returns')
            if returns is not None:
                if hasattr(returns, 'item'):
                    returns = returns.item()
                return float(returns)

            # 如果没有直接的收益率，通过价格计算
            buy_price = buy_trade.get('price', 0)
            sell_price = sell_trade.get('price', 0)

            if buy_price > 0 and sell_price > 0:
                returns = (sell_price / buy_price - 1) * 100
                return returns

            return 0.0

        except (ValueError, TypeError) as e:
            self.logger.debug(f"计算收益率失败: {e}")
            return 0.0

    def _analyze_trade_pairs(self, trade_pairs: List[Dict]) -> Dict[str, Any]:
        """分析交易配对数据"""
        total_trades = len(trade_pairs)
        won_trades = 0
        lost_trades = 0
        total_returns = 0
        max_profit = 0
        max_loss = 0
        total_holding_days = 0
        valid_holding_periods = 0

        for pair in trade_pairs:
            returns = pair['returns']

            # 分类盈利和亏损交易
            if returns > 0:
                won_trades += 1
            elif returns < 0:
                lost_trades += 1
            else:
                lost_trades += 1  # 无收益视为亏损

            total_returns += returns
            max_profit = max(max_profit, returns)
            max_loss = min(max_loss, returns)

            # 统计持仓天数
            holding_days = pair.get('holding_days', 0)
            if holding_days > 0:
                total_holding_days += holding_days
                valid_holding_periods += 1

        # 计算平均指标
        avg_returns = total_returns / total_trades if total_trades > 0 else 0
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        avg_holding_days = total_holding_days / valid_holding_periods if valid_holding_periods > 0 else 0

        self.logger.info(f"交易配对分析完成:")
        self.logger.info(f"  总交易数: {total_trades}")
        self.logger.info(f"  盈利交易: {won_trades}")
        self.logger.info(f"  亏损交易: {lost_trades}")
        self.logger.info(f"  胜率: {win_rate:.2f}%")
        self.logger.info(f"  平均收益率: {avg_returns:.2f}%")
        self.logger.info(f"  最大单笔收益: {max_profit:.2f}%")
        self.logger.info(f"  最大单笔亏损: {max_loss:.2f}%")
        self.logger.info(f"  平均持仓天数: {avg_holding_days:.1f}")

        return {
            "交易次数": total_trades,
            "盈利交易": won_trades,
            "亏损交易": lost_trades,
            "胜率": win_rate,
            "平均收益": avg_returns,
            "最大单笔收益": max_profit,
            "最大单笔亏损": abs(max_loss),  # 转换为正值显示
            "平均持仓时间": avg_holding_days
        }

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

    def _calculate_holding_days(self, buy_date, sell_date):
        """计算持仓天数"""
        try:
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
            return max(delta.days, 1)  # 至少1天

        except Exception as e:
            self.logger.debug(f"计算持仓天数失败: {e}")
            return 1

    def _calculate_performance_metrics(self, strat, cerebro) -> Dict[str, Any]:
        """计算性能指标 - 重构版本
        
        改进的波动率计算逻辑：
        1. 优先使用日收益率序列计算年化波动率（最准确）
        2. 如果无法获取日收益率，尝试从交易记录计算
        3. 如果以上方法都失败，将波动率和夏普比率设为0，避免提供误导性数据
        """
        try:
            # 首先尝试使用backtrader分析器
            backtrader_annual_return = 0
            backtrader_annual_volatility = 0

            try:
                returns_analysis = strat.analyzers.returns.get_analysis()
                backtrader_annual_return = returns_analysis.get('rnorm100', 0)
                backtrader_annual_volatility = returns_analysis.get('std', 0) * 100

                self.logger.debug(
                    f"Backtrader性能指标 - 年化收益率: {backtrader_annual_return}%, 年化波动率: {backtrader_annual_volatility}%")

            except Exception as e:
                self.logger.debug(f"获取backtrader性能指标失败: {e}")

            # 总是执行手动计算，然后与backtrader结果比较
            manual_metrics = self._manual_calculate_performance_metrics(strat, cerebro)
            self.logger.info(f"手动计算性能指标 - 年化收益率: {manual_metrics['年化收益率']:.2f}%, "
                             f"年化波动率: {manual_metrics['年化波动率']:.2f}%")

            # 选择更好的结果：优先使用有年化波动率的结果
            final_annual_return = manual_metrics['年化收益率']
            final_annual_volatility = manual_metrics['年化波动率']
            volatility_calculation_method = manual_metrics.get('波动率计算方法', '未知')

            # 如果手动计算的波动率为0，但backtrader有波动率，则使用backtrader的波动率
            if final_annual_volatility == 0 and backtrader_annual_volatility > 0:
                final_annual_volatility = backtrader_annual_volatility
                volatility_calculation_method = "Backtrader分析器"
                self.logger.info(f"使用backtrader的年化波动率: {final_annual_volatility:.2f}%")

            # 如果手动计算的收益率为0，但backtrader有收益率，则使用backtrader的收益率
            if final_annual_return == 0 and backtrader_annual_return > 0:
                final_annual_return = backtrader_annual_return
                self.logger.info(f"使用backtrader的年化收益率: {final_annual_return:.2f}%")

            # 计算信息比率和索提诺比率
            info_ratio = 0
            sortino_ratio = 0

            if final_annual_volatility > 0:
                info_ratio = final_annual_return / final_annual_volatility
                # 简化的Sortino比率计算（假设负收益的标准差为总波动率的70%）
                downside_volatility = final_annual_volatility * 0.7
                sortino_ratio = final_annual_return / downside_volatility if downside_volatility > 0 else 0
            else:
                # 如果波动率为0，记录警告
                self.logger.warning("年化波动率为0，信息比率和索提诺比率将设为0")

            # 记录最终使用的计算方法
            self.logger.info(f"最终波动率计算方法: {volatility_calculation_method}")

            return {
                "年化收益率": final_annual_return,
                "年化波动率": final_annual_volatility,
                "信息比率": info_ratio,
                "索提诺比率": sortino_ratio,
                "波动率计算方法": volatility_calculation_method  # 新增字段，用于调试
            }

        except Exception as e:
            self.logger.error(f"计算性能指标失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "年化收益率": 0,
                "年化波动率": 0,
                "信息比率": 0,
                "索提诺比率": 0,
                "波动率计算方法": "计算失败"
            }

    def _manual_calculate_performance_metrics(self, strat, cerebro) -> Dict[str, Any]:
        """手动计算性能指标 - 重构版本
        
        改进的波动率计算逻辑：
        1. 优先使用日收益率序列计算年化波动率（最准确）
        2. 如果无法获取日收益率，尝试从交易记录计算
        3. 如果以上方法都失败，将波动率和夏普比率设为0，避免提供误导性数据
        """
        try:
            # 计算总收益率 - 统一口径：基于最终总资产(final_value)
            initial_cash = cerebro.broker.startingcash
            final_cash = cerebro.broker.getcash()
            final_value = cerebro.broker.getvalue()
            total_return_ratio = final_value / initial_cash - 1  # 基于总资产的收益率

            # 获取实际回测天数 - 改进计算方法
            backtest_days = 252  # 默认一年

            # 尝试从策略中获取更准确的回测时间
            if hasattr(strat, 'datas') and len(strat.datas) > 0:
                # 从主数据源获取回测期间
                data = strat.datas[0]
                if hasattr(data, 'datetime') and len(data.datetime) > 0:
                    # 计算数据的实际天数
                    total_bars = len(data.datetime)
                    if total_bars > 0:
                        # 假设是日线数据，直接使用bar数量
                        backtest_days = total_bars
                        self.logger.debug(f"从数据源获取回测天数: {backtest_days} 天")

            # 如果上述方法失败，尝试从交易记录获取
            if backtest_days == 252:  # 仍然是默认值
                strategy_trades = getattr(strat, 'trades', [])
                if strategy_trades:
                    # 获取第一个和最后一个交易的日期
                    first_date = None
                    last_date = None

                    for trade in strategy_trades:
                        trade_date = trade.get('date') or trade.get('交易日期')
                        if trade_date:
                            if first_date is None or trade_date < first_date:
                                first_date = trade_date
                            if last_date is None or trade_date > last_date:
                                last_date = trade_date

                    if first_date and last_date:
                        delta = last_date - first_date
                        backtest_days = max(delta.days, 1)
                        self.logger.debug(f"从交易记录获取回测期间: {first_date} 到 {last_date}, 共 {backtest_days} 天")

            # 计算年化收益率（避免复数）：统一使用总资产口径
            years = backtest_days / 365.25
            if years <= 0:
                annual_return = 0.0
            elif final_value <= 0:
                annual_return = -100.0
            else:
                annual_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100

            self.logger.debug(
                f"回测基础数据 - 天数: {backtest_days}, 年数: {years:.2f}, 总收益率: {total_return_ratio * 100:.2f}%")

            # ==================== 年化波动率计算 - 重构版本 ====================
            annual_volatility = 0
            volatility_calculation_method = "未计算"

            # 方法1：优先使用日收益率序列计算（最准确）
            try:
                daily_returns = self._get_daily_returns_from_analyzer(strat)
                if daily_returns and len(daily_returns) > 1:
                    # 获取有效的日收益率序列
                    returns_values = list(daily_returns.values())
                    returns_array = np.array(returns_values)
                    
                    # 计算日收益率的标准差
                    daily_std = np.std(returns_array, ddof=1)
                    
                    # 年化波动率 = 日收益率标准差 * sqrt(252) * 100
                    annual_volatility = daily_std * np.sqrt(252) * 100
                    
                    volatility_calculation_method = f"日收益率序列({len(returns_values)}个数据点)"
                    self.logger.info(f"基于日收益率序列计算年化波动率: {annual_volatility:.2f}% "
                                   f"(日标准差: {daily_std:.4f})")
                    
            except Exception as e:
                self.logger.debug(f"日收益率序列计算波动率失败: {e}")

            # 方法2：如果方法1失败，尝试从交易记录计算
            if annual_volatility == 0:
                try:
                    strategy_trades = getattr(strat, 'trades', [])
                    sell_trades = [t for t in strategy_trades if
                                   t.get('action') == 'SELL' or t.get('交易动作') == 'SELL']

                    if sell_trades:
                        returns = []
                        self.logger.debug(f"正在分析{len(sell_trades)}笔卖出交易的收益率...")

                        for i, trade in enumerate(sell_trades):
                            # 检查多个可能的收益率字段
                            return_value = None
                            for field in ['returns', '收益率', '收益率(%)', 'return_rate']:
                                if field in trade and trade[field] is not None:
                                    return_value = trade[field]
                                    break

                            if return_value is not None:
                                try:
                                    # 处理不同的收益率格式
                                    if isinstance(return_value, str):
                                        if return_value.strip().upper() in ['N/A', 'NA', 'NULL', '']:
                                            self.logger.debug(f"交易{i + 1}收益率为空或N/A，跳过")
                                            continue
                                        # 移除百分号
                                        return_value = return_value.strip().replace('%', '')

                                    # 转换为数值
                                    if hasattr(return_value, 'item'):
                                        return_value = return_value.item()

                                    ret = float(return_value)

                                    # 如果收益率是百分比格式（>1），转换为小数
                                    if abs(ret) > 1:
                                        ret = ret / 100

                                    returns.append(ret)
                                    self.logger.debug(f"交易{i + 1}收益率: {ret:.4f}")

                                except (ValueError, TypeError) as e:
                                    self.logger.debug(f"交易{i + 1}收益率解析失败: {return_value}, 错误: {e}")
                                    continue
                            else:
                                self.logger.debug(f"交易{i + 1}没有找到收益率字段")

                        self.logger.info(f"成功解析{len(returns)}笔交易的收益率，总卖出交易{len(sell_trades)}笔")

                        if len(returns) > 1:
                            returns_array = np.array(returns)
                            # 计算交易收益率的标准差
                            returns_std = np.std(returns_array, ddof=1)
                            returns_mean = np.mean(returns_array)

                            self.logger.debug(f"收益率统计: 均值={returns_mean:.4f}, 标准差={returns_std:.4f}")

                            # 使用时间频率来年化波动率
                            # 假设交易是均匀分布在回测期间的
                            average_holding_days = backtest_days / len(returns) if len(returns) > 0 else 30

                            # 年化系数：基于平均持仓天数，使用交易日252天作为基准
                            if average_holding_days > 0:
                                annualization_factor = np.sqrt(252 / average_holding_days)
                            else:
                                annualization_factor = np.sqrt(252)  # 默认日频

                            annual_volatility = returns_std * annualization_factor * 100

                            volatility_calculation_method = f"交易记录({len(returns)}笔交易)"
                            self.logger.info(f"基于{len(returns)}笔交易计算年化波动率: {annual_volatility:.2f}%")
                            self.logger.debug(f"平均持仓天数: {average_holding_days:.1f}, 年化系数: {annualization_factor:.2f}")

                        elif len(returns) == 1:
                            # 单笔交易，使用保守的估算
                            single_return = abs(returns[0])
                            # 假设月频交易
                            annual_volatility = single_return * np.sqrt(12) * 100
                            volatility_calculation_method = "单笔交易估算"
                            self.logger.info(f"基于单笔交易估算年化波动率: {annual_volatility:.2f}%")
                        else:
                            self.logger.warning("没有有效的收益率数据用于计算波动率")
                            
                except Exception as e:
                    self.logger.debug(f"从交易记录计算波动率失败: {e}")

            # 方法3：如果以上方法都失败，将波动率设为0，避免提供误导性数据
            if annual_volatility == 0:
                volatility_calculation_method = "无法计算"
                self.logger.warning("无法计算年化波动率，将设为0以避免提供误导性数据")
                self.logger.warning("建议检查策略是否有足够的交易记录或日收益率数据")

            # 计算其他指标
            info_ratio = 0
            sortino_ratio = 0

            if annual_volatility > 0:
                info_ratio = annual_return / annual_volatility
                # 简化的Sortino比率计算（假设负收益的标准差为总波动率的70%）
                downside_volatility = annual_volatility * 0.7
                sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0

            # 记录计算方法的详细信息
            self.logger.info(f"波动率计算方法: {volatility_calculation_method}")

            return {
                "年化收益率": annual_return,
                "年化波动率": annual_volatility,
                "信息比率": info_ratio,
                "索提诺比率": sortino_ratio,
                "波动率计算方法": volatility_calculation_method  # 新增字段，用于调试
            }

        except Exception as e:
            self.logger.error(f"手动计算性能指标失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "年化收益率": 0,
                "年化波动率": 0,
                "信息比率": 0,
                "索提诺比率": 0,
                "波动率计算方法": "计算失败"
            }

    def _get_daily_returns_from_analyzer(self, strat) -> Dict:
        """从TimeReturn分析器获取日收益序列。
        返回 {datetime.date: float_return_decimal}，若不可用返回空dict。
        """
        try:
            if hasattr(strat, 'analyzers') and hasattr(strat.analyzers, 'timereturn'):
                analysis = strat.analyzers.timereturn.get_analysis()
                if not analysis:
                    return {}
                normalized = {}
                for k, v in analysis.items():
                    # k 可能是 datetime/date/num，将其转为date
                    try:
                        if hasattr(k, 'date'):
                            d = k.date()
                        else:
                            d = k
                    except Exception:
                        d = k
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    normalized[d] = val
                return normalized
        except Exception:
            return {}
        return {}

    def _derive_active_start_date(self, strat, daily_returns: Dict) -> Any:
        """根据策略的min_data_points与daily_returns推导开始生效日期。
        
        新规则：
        - 如果策略有缓存适配器且预加载了数据，从第一天开始生效
        - 否则，当 len(self.data) < min_data_points 时跳过，第一天生效为第(min_data_points-1)个日期
        
        若无法获取min_data_points或daily_returns为空，则返回None。
        """
        try:
            if not daily_returns:
                return None

            sorted_dates = sorted(daily_returns.keys())
            if not sorted_dates:
                return None

            # 检查策略是否使用了缓存数据
            if self._strategy_uses_cache_data(strat):
                # 使用缓存数据的策略从第一天开始生效
                return sorted_dates[0]

            # 传统逻辑：获取min_data_points
            min_pts = None
            if hasattr(strat, 'params') and hasattr(strat.params, 'min_data_points'):
                min_pts = getattr(strat.params, 'min_data_points', None)
            elif hasattr(strat, 'min_data_points'):
                min_pts = getattr(strat, 'min_data_points', None)

            if not min_pts or min_pts <= 1:
                # 没有min_data_points限制，从第一天开始
                return sorted_dates[0]

            if len(sorted_dates) < min_pts:
                return None

            # 第一生效日为第(min_pts-1)个
            return sorted_dates[min_pts - 1]
        except Exception:
            return None

    def _strategy_uses_cache_data(self, strat) -> bool:
        """
        检查策略是否使用了缓存数据
        
        Args:
            strat: 策略实例
            
        Returns:
            bool: True表示使用了缓存数据
        """
        try:
            # 检查是否有缓存适配器
            if not (hasattr(strat, 'cache_adapter') and strat.cache_adapter is not None):
                return False

            # 检查是否有预加载的通道数据
            if hasattr(strat, 'preloaded_channel_data') and strat.preloaded_channel_data:
                return True

            return False
        except Exception:
            return False

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
最终总资产: {safe_get('最终总资产', 0):,.2f}
净盈亏: {safe_get('净盈亏', 0):,.2f}
总收益率: {safe_get('总收益率', 0):.2f}%
最终现金: {safe_get('最终现金', 0):,.2f}
现金盈亏: {safe_get('现金盈亏', 0):,.2f}

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
平均收益: {safe_get('平均收益', 0):.2f}%
最大单笔收益: {safe_get('最大单笔收益', 0):.2f}%
最大单笔亏损: {safe_get('最大单笔亏损', 0):.2f}%
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
