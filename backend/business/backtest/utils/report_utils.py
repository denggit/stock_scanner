#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
报告工具类
"""

from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd


class ReportUtils:
    """
    报告工具类
    提供报告生成和可视化功能
    """

    @staticmethod
    def _build_equity_series(results: Dict[str, Any]) -> pd.DataFrame:
        """
        基于 daily_returns 重建权益曲线
        返回包含两列：'return'(日收益小数) 与 'equity'(权益数值)
        优先读取 metrics['初始资金'] 作为初始权益，否则回退为 1.0
        """
        daily_ret_map = results.get('daily_returns') or {}
        if not daily_ret_map:
            return pd.DataFrame(columns=['return', 'equity'])

        # 解析初始资金
        init_equity = 1.0
        metrics = results.get('metrics') or {}
        if metrics:
            try:
                init_equity = float(metrics.get('初始资金', init_equity))
            except Exception:
                pass

        # 构造日收益序列
        try:
            ser = pd.Series(daily_ret_map)
            # 索引转为 datetime
            ser.index = pd.to_datetime(ser.index)
            ser = ser.sort_index()
            ser = ser.astype(float)
        except Exception:
            # 退化处理：尝试把 key 当作日期字符串
            keys = list(daily_ret_map.keys())
            vals = [float(daily_ret_map[k]) for k in keys]
            ser = pd.Series(vals, index=pd.to_datetime(keys)).sort_index()

        equity = (1.0 + ser).cumprod() * init_equity
        df = pd.DataFrame({'return': ser, 'equity': equity})
        return df

    @staticmethod
    def _aggregate_period_returns(results: Dict[str, Any], trades: List[Dict],
                                  period_col_name: str, period_code: str) -> pd.DataFrame:
        """
        基于权益曲线聚合周期收益（与整体业绩一致），必要时回退到卖出交易聚合。
        period_code: 'W' 周, 'M' 月, 'Y' 年
        """
        eq = ReportUtils._build_equity_series(results)
        if not eq.empty:
            # 构建周期标签
            eq = eq.copy()
            eq[period_col_name] = eq.index.to_period(period_code)

            rows = []
            for period, g in eq.groupby(period_col_name):
                start_equity = g['equity'].iloc[0]
                end_equity = g['equity'].iloc[-1]
                period_return = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0 else 0.0
                avg_daily = g['return'].mean() * 100.0 if len(g) > 0 else 0.0

                # 交易统计（可选）
                trade_cnt = 0
                total_value = 0.0
                invested_capital = 0.0
                returned_capital = 0.0
                if trades:
                    df_t = pd.DataFrame(trades)
                    # 日期列
                    date_col = None
                    for c in ['date', '交易日期']:
                        if c in df_t.columns:
                            date_col = c
                            break
                    if date_col is not None:
                        df_t = df_t.copy()
                        df_t['__date__'] = pd.to_datetime(df_t[date_col], errors='coerce')
                        df_t.dropna(subset=['__date__'], inplace=True)
                        df_t['__period__'] = df_t['__date__'].dt.to_period(period_code)
                        gtr = df_t[df_t['__period__'] == period]
                        # 仅卖出交易用于金额和投资估算
                        if 'action' in gtr.columns:
                            g_sell = gtr[gtr['action'] == 'SELL']
                        else:
                            g_sell = gtr
                        trade_cnt = int(len(g_sell))
                        if trade_cnt > 0:
                            # 交易金额
                            for vc in ['value', '交易金额']:
                                if vc in g_sell.columns:
                                    total_value = float(pd.to_numeric(g_sell[vc], errors='coerce').fillna(0).sum())
                                    break
                            # 投入/收回资本估算
                            returns_col = None
                            for rc in ['returns', '收益率', '收益率(%)']:
                                if rc in g_sell.columns:
                                    returns_col = rc
                                    break
                            if returns_col is not None:
                                for _, r in g_sell.iterrows():
                                    try:
                                        v = float(r.get('value', r.get('交易金额', 0)) or 0)
                                        rr = float(r.get(returns_col, 0) or 0)
                                    except Exception:
                                        v, rr = 0.0, 0.0
                                    invested = v / (1 + rr / 100.0) if rr != -100 else v
                                    invested_capital += invested
                                    returned_capital += v

                rows.append({
                    period_col_name: str(period),
                    '交易次数': trade_cnt,
                    '总收益率': round(period_return, 2),
                    '平均收益率': round(avg_daily, 2),
                    '交易金额': round(total_value, 2),
                    '投入资本': round(invested_capital, 2),
                    '收回资本': round(returned_capital, 2),
                })

            return pd.DataFrame(rows)

        # 回退：无 daily_returns 时，仍按卖出交易粗略聚合（与原实现一致）
        if period_code == 'M':
            return ReportUtils.create_monthly_returns(trades)
        if period_code == 'W':
            return ReportUtils.create_weekly_returns(trades)
        if period_code == 'Y':
            # 退化汇总（基于交易），与月/周类似
            return ReportUtils.create_yearly_returns(trades)
        return pd.DataFrame()

    @staticmethod
    def create_performance_summary(results: Dict[str, Any]) -> pd.DataFrame:
        """
        创建性能汇总表
        
        Args:
            results: 回测结果
            
        Returns:
            性能汇总表
        """
        metrics = results.get('metrics', {})

        # 安全获取指标值，处理None值
        def safe_get(key, default=0):
            value = metrics.get(key, default)
            return value if value is not None else default

        # 确保所有值都是字符串格式
        summary_data = {
            '指标': [
                '初始资金', '最终资金', '总收益率', '绝对收益',
                '夏普比率', '最大回撤', '交易次数', '胜率',
                '年化收益率', '年化波动率'
            ],
            '数值': [
                f"{safe_get('初始资金', 0):,.2f}",
                f"{safe_get('最终资金', 0):,.2f}",
                f"{safe_get('总收益率', 0):.2f}%",
                f"{safe_get('绝对收益', 0):,.2f}",
                f"{safe_get('夏普比率', 0):.4f}",
                f"{safe_get('最大回撤', 0):.2f}%",
                f"{safe_get('交易次数', 0)}",
                f"{safe_get('胜率', 0):.2f}%",
                f"{safe_get('年化收益率', 0):.2f}%",
                f"{safe_get('年化波动率', 0):.2f}%"
            ]
        }

        # 确保数据长度一致
        if len(summary_data['指标']) != len(summary_data['数值']):
            # 如果长度不一致，截取较短的长度
            min_length = min(len(summary_data['指标']), len(summary_data['数值']))
            summary_data['指标'] = summary_data['指标'][:min_length]
            summary_data['数值'] = summary_data['数值'][:min_length]

        return pd.DataFrame(summary_data)

    @staticmethod
    def create_trade_summary(trades: List[Dict]) -> pd.DataFrame:
        """
        创建交易汇总表
        
        Args:
            trades: 交易记录列表
            
        Returns:
            交易汇总表
        """
        if not trades:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(trades)

            # 计算交易统计
            buy_trades = df[df['action'] == 'BUY'] if 'action' in df.columns else pd.DataFrame()
            sell_trades = df[df['action'] == 'SELL'] if 'action' in df.columns else pd.DataFrame()

            # 安全计算统计值
            def safe_calc(series, calc_type='mean'):
                if series.empty:
                    return "N/A"
                try:
                    if calc_type == 'mean':
                        return f"{series.mean():.2f}"
                    elif calc_type == 'max':
                        return f"{series.max():,.2f}"
                    elif calc_type == 'min':
                        return f"{series.min():,.2f}"
                except:
                    return "N/A"

            summary_data = {
                '统计项': [
                    '总交易数', '买入交易数', '卖出交易数',
                    '平均买入价格', '平均卖出价格',
                    '最大单笔交易金额', '最小单笔交易金额'
                ],
                '数值': [
                    str(len(df)),
                    str(len(buy_trades)),
                    str(len(sell_trades)),
                    safe_calc(buy_trades['price'], 'mean') if 'price' in buy_trades.columns else "N/A",
                    safe_calc(sell_trades['price'], 'mean') if 'price' in sell_trades.columns else "N/A",
                    safe_calc(df['value'], 'max') if 'value' in df.columns else "N/A",
                    safe_calc(df['value'], 'min') if 'value' in df.columns else "N/A"
                ]
            }

            # 确保数据长度一致
            if len(summary_data['统计项']) != len(summary_data['数值']):
                min_length = min(len(summary_data['统计项']), len(summary_data['数值']))
                summary_data['统计项'] = summary_data['统计项'][:min_length]
                summary_data['数值'] = summary_data['数值'][:min_length]

            return pd.DataFrame(summary_data)

        except Exception as e:
            # 如果处理失败，返回基本的汇总信息
            return pd.DataFrame({
                '统计项': ['总交易数', '处理状态'],
                '数值': [str(len(trades)), f'处理失败: {str(e)}']
            })

    @staticmethod
    def create_monthly_returns(trades: List[Dict]) -> pd.DataFrame:
        """
        创建月度收益表
        
        Args:
            trades: 交易记录列表
            
        Returns:
            月度收益表
        """
        # 为保持向后兼容，这里仍支持仅传 trades 的老签名
        results_like = {'daily_returns': {}, 'metrics': {}, 'trades': trades}
        return ReportUtils._aggregate_period_returns(results_like, trades, '月份', 'M')

    @staticmethod
    def create_weekly_returns(trades: List[Dict]) -> pd.DataFrame:
        """
        创建周度收益表
        
        说明：
        - 基于 SELL 交易，按周聚合；
        - 周定义采用 pandas Period('W')（自然周），输出字符串化的周标识；
        - 同月度收益一致，包含总收益率（基于总资本变化）、平均收益率、交易金额、投入资本、收回资本。
        
        Args:
            trades: 交易记录列表
        
        Returns:
            周度收益表 DataFrame
        """
        results_like = {'daily_returns': {}, 'metrics': {}, 'trades': trades}
        return ReportUtils._aggregate_period_returns(results_like, trades, '周', 'W')

    @staticmethod
    def create_yearly_returns(trades: List[Dict]) -> pd.DataFrame:
        """
        创建年度收益表
        
        说明：
        - 基于 SELL 交易，按年聚合（自然年，Period('Y')）；
        - 指标与月度/周度一致：总收益率（基于总资本变化）、平均收益率、交易金额、投入资本、收回资本。
        
        Args:
            trades: 交易记录列表
        
        Returns:
            年度收益表 DataFrame
        """
        results_like = {'daily_returns': {}, 'metrics': {}, 'trades': trades}
        return ReportUtils._aggregate_period_returns(results_like, trades, '年度', 'Y')

    @staticmethod
    def plot_performance_comparison(strategy_results: Dict[str, Dict],
                                    metrics: List[str] = None) -> None:
        """
        绘制策略性能比较图
        
        Args:
            strategy_results: 策略结果字典
            metrics: 要比较的指标列表
        """
        if not strategy_results:
            print("没有策略结果可比较")
            return

        if metrics is None:
            metrics = ['总收益率', '夏普比率', '最大回撤', '胜率']

        # 准备数据
        comparison_data = []
        for name, results in strategy_results.items():
            result_metrics = results.get('metrics', {})
            row = {'策略名称': name}
            for metric in metrics:
                row[metric] = result_metrics.get(metric, 0)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('策略性能比较', fontsize=16)

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            if metric in ['总收益率', '胜率']:
                # 柱状图
                bars = ax.bar(df['策略名称'], df[metric])
                ax.set_title(f'{metric}比较')
                ax.set_ylabel(metric)

                # 添加数值标签
                for bar, value in zip(bars, df[metric]):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{value:.2f}', ha='center', va='bottom')
            else:
                # 条形图（水平）
                bars = ax.barh(df['策略名称'], df[metric])
                ax.set_title(f'{metric}比较')
                ax.set_xlabel(metric)

                # 添加数值标签
                for bar, value in zip(bars, df[metric]):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                            f'{value:.4f}', ha='left', va='center')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_trade_analysis(trades: List[Dict]) -> None:
        """
        绘制交易分析图
        
        Args:
            trades: 交易记录列表
        """
        if not trades:
            print("没有交易记录可分析")
            return

        df = pd.DataFrame(trades)
        df['date'] = pd.to_datetime(df['date'])

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('交易分析', fontsize=16)

        # 1. 交易时间分布
        ax1 = axes[0, 0]
        df['date'].value_counts().sort_index().plot(kind='line', ax=ax1)
        ax1.set_title('交易时间分布')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('交易次数')

        # 2. 交易金额分布
        ax2 = axes[0, 1]
        df['value'].hist(bins=20, ax=ax2, alpha=0.7)
        ax2.set_title('交易金额分布')
        ax2.set_xlabel('交易金额')
        ax2.set_ylabel('频次')

        # 3. 买卖交易对比
        ax3 = axes[1, 0]
        trade_counts = df['action'].value_counts()
        ax3.pie(trade_counts.values, labels=trade_counts.index, autopct='%1.1f%%')
        ax3.set_title('买卖交易比例')

        # 4. 收益率分布（仅卖出交易）
        ax4 = axes[1, 1]
        sell_trades = df[df['action'] == 'SELL']
        if len(sell_trades) > 0 and 'returns' in sell_trades.columns:
            sell_trades['returns'].hist(bins=20, ax=ax4, alpha=0.7)
            ax4.set_title('收益率分布')
            ax4.set_xlabel('收益率 (%)')
            ax4.set_ylabel('频次')
        else:
            ax4.text(0.5, 0.5, '无收益率数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('收益率分布')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_report_to_excel(results: Dict[str, Any], filename: str) -> None:
        """
        保存报告到Excel文件
        
        Args:
            results: 回测结果
            filename: 文件名
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 性能汇总表（始终创建，确保至少有一个工作表）
                performance_summary = ReportUtils.create_performance_summary(results)
                performance_summary.to_excel(writer, sheet_name='性能汇总', index=False)

                # 处理交易记录
                trades = results.get('trades', [])
                if not trades and 'performance' in results:
                    # 从performance中获取交易记录
                    performance = results.get('performance', {})
                    trades = performance.get('trades', [])

                if trades:
                    # 交易汇总表
                    trade_summary = ReportUtils.create_trade_summary(trades)
                    if not trade_summary.empty:
                        trade_summary.to_excel(writer, sheet_name='交易汇总', index=False)

                    # 月度收益表
                    # 优先基于权益曲线聚合，保持与整体业绩一致
                    monthly_returns = ReportUtils._aggregate_period_returns(results, trades, '月份', 'M')
                    if not monthly_returns.empty:
                        monthly_returns.to_excel(writer, sheet_name='月度收益', index=False)

                    # 周度收益表
                    weekly_returns = ReportUtils._aggregate_period_returns(results, trades, '周', 'W')
                    if not weekly_returns.empty:
                        weekly_returns.to_excel(writer, sheet_name='周度收益', index=False)

                    # 年度收益表（新增）
                    yearly_returns = ReportUtils._aggregate_period_returns(results, trades, '年度', 'Y')
                    if not yearly_returns.empty:
                        yearly_returns.to_excel(writer, sheet_name='年度收益', index=False)

                    # 详细交易记录 - 保存到"交易记录"sheet
                    trades_df = ReportUtils.create_detailed_trade_records(trades)
                    if not trades_df.empty:
                        trades_df.to_excel(writer, sheet_name='交易记录', index=False)
                    
                    # 交易分析表 - 新增
                    trade_analysis_df = ReportUtils.create_trade_analysis(trades)
                    if not trade_analysis_df.empty:
                        trade_analysis_df.to_excel(writer, sheet_name='交易分析', index=False)

                # 每日收益表（使用日收益和交易记录构建）
                daily_table = ReportUtils.create_daily_returns_table(results)
                if not daily_table.empty:
                    daily_table.to_excel(writer, sheet_name='每日收益', index=False)

                # 策略报告
                if 'report' in results and results['report']:
                    report_df = pd.DataFrame({'报告': [results['report']]})
                    report_df.to_excel(writer, sheet_name='策略报告', index=False)

                # 策略信息
                if 'strategy_info' in results:
                    strategy_info = results['strategy_info']
                    if strategy_info:
                        # 创建策略信息表
                        strategy_info_df = ReportUtils.create_strategy_info_table(strategy_info)
                        if not strategy_info_df.empty:
                            strategy_info_df.to_excel(writer, sheet_name='策略信息', index=False)

            print(f"报告已保存到: {filename}")

        except Exception as e:
            print(f"保存报告失败: {e}")
            # 如果保存失败，尝试创建一个简单的报告
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # 创建一个基本的性能汇总表
                    basic_summary = pd.DataFrame({
                        '指标': ['状态'],
                        '数值': ['回测完成，但报告生成失败']
                    })
                    basic_summary.to_excel(writer, sheet_name='性能汇总', index=False)
                print(f"已创建基本报告: {filename}")
            except Exception as e2:
                print(f"创建基本报告也失败: {e2}")

    @staticmethod
    def create_detailed_trade_records(trades: List[Dict]) -> pd.DataFrame:
        """
        创建详细的交易记录表
        
        Args:
            trades: 交易记录列表
            
        Returns:
            详细的交易记录DataFrame
        """
        if not trades:
            return pd.DataFrame()

        try:
            # 创建DataFrame
            df = pd.DataFrame(trades)

            # 定义需要的字段映射（只保留用户需要的7个字段）
            required_fields = {
                '交易日期': ['交易日期', 'date'],
                '交易动作': ['交易动作', 'action'],
                '股票代码': ['股票代码', 'stock_code'],
                '交易数量': ['交易数量', 'quantity', 'size'],
                '交易价格': ['交易价格', 'price'],
                '交易金额': ['交易金额', 'value'],
                '收益率(%)': ['收益率', 'returns', '收益率(%)']
            }

            # 创建新的DataFrame，只包含需要的字段
            result_data = {}

            for final_name, possible_sources in required_fields.items():
                # 寻找可用的源字段
                found_field = None
                for source_field in possible_sources:
                    if source_field in df.columns:
                        found_field = source_field
                        break

                if found_field:
                    result_data[final_name] = df[found_field]
                else:
                    # 如果没有找到对应字段，创建空列
                    result_data[final_name] = [None] * len(df)

            # 创建新的DataFrame，只包含指定的字段
            df_result = pd.DataFrame(result_data)

            # 追加通道字段（放在收益率(%)之后）
            channel_fields_in_order = [
                '通道状态',
                '通道评分',
                '斜率β',
                'R²',
                '今日中轴',
                '今日上沿',
                '今日下沿',
                '通道宽度',
                '距下沿(%)',
            ]

            # 收集存在于原df中的通道字段
            available_channel_fields = [f for f in channel_fields_in_order if f in df.columns]

            # 将可用通道字段添加到结果df，并构建新的列顺序
            for f in available_channel_fields:
                df_result[f] = df[f]

            # 格式化数值列
            if '交易价格' in df_result.columns:
                df_result['交易价格'] = pd.to_numeric(df_result['交易价格'], errors='coerce').round(2)

            if '交易金额' in df_result.columns:
                df_result['交易金额'] = pd.to_numeric(df_result['交易金额'], errors='coerce').round(2)

            if '收益率(%)' in df_result.columns:
                df_result['收益率(%)'] = pd.to_numeric(df_result['收益率(%)'], errors='coerce').round(2)

            # 对通道数值列进行格式化
            numeric_channel_fields = [
                '通道评分', '斜率β', 'R²', '今日中轴', '今日上沿', '今日下沿', '通道宽度', '距下沿(%)'
            ]
            for f in available_channel_fields:
                if f in numeric_channel_fields:
                    df_result[f] = pd.to_numeric(df_result[f], errors='coerce').round(2)

            # 格式化日期列
            if '交易日期' in df_result.columns:
                df_result['交易日期'] = pd.to_datetime(df_result['交易日期'], errors='coerce')
                df_result['交易日期'] = df_result['交易日期'].dt.strftime('%Y-%m-%d')

            # 格式化整数列
            if '交易数量' in df_result.columns:
                df_result['交易数量'] = pd.to_numeric(df_result['交易数量'], errors='coerce').fillna(0).astype(int)

            # 重新排序列：确保通道字段位于“收益率(%)”之后
            if available_channel_fields:
                base_cols = list(df_result.columns)
                if '收益率(%)' in base_cols:
                    # 先移除已添加的通道列
                    base_cols_no_channels = [c for c in base_cols if c not in available_channel_fields]
                    insert_pos = base_cols_no_channels.index('收益率(%)') + 1
                    new_cols = (
                            base_cols_no_channels[:insert_pos]
                            + available_channel_fields
                            + base_cols_no_channels[insert_pos:]
                    )
                    df_result = df_result[new_cols]

            return df_result

        except Exception as e:
            print(f"创建详细交易记录时出错: {e}")
            return pd.DataFrame()

    @staticmethod
    def create_strategy_info_table(strategy_info: Dict[str, Any]) -> pd.DataFrame:
        """
        创建策略信息表
        
        Args:
            strategy_info: 策略信息字典
            
        Returns:
            策略信息DataFrame
        """
        if not strategy_info:
            return pd.DataFrame()

        try:
            info_data = []

            # 基本信息
            info_data.append({
                '信息类型': '策略名称',
                '数值': strategy_info.get('strategy_name', '未知策略')
            })

            # 参数信息
            parameters = strategy_info.get('parameters', {})
            for param_name, param_value in parameters.items():
                info_data.append({
                    '信息类型': f'参数_{param_name}',
                    '数值': str(param_value)
                })

            # 当前状态
            current_status = strategy_info.get('current_status', {})
            for status_name, status_value in current_status.items():
                if isinstance(status_value, dict):
                    info_data.append({
                        '信息类型': f'状态_{status_name}',
                        '数值': str(status_value)
                    })
                else:
                    info_data.append({
                        '信息类型': f'状态_{status_name}',
                        '数值': str(status_value)
                    })

            if info_data:
                return pd.DataFrame(info_data)
            else:
                return pd.DataFrame()

        except Exception as e:
            # 如果处理失败，返回基本的策略信息
            return pd.DataFrame({
                '信息类型': ['策略名称', '处理状态'],
                '数值': [strategy_info.get('strategy_name', '未知策略'), f'处理失败: {str(e)}']
            })

    @staticmethod
    def create_daily_returns_table(results: Dict[str, Any]) -> pd.DataFrame:
        """
        创建每日收益表。
        字段：日期、买入次数、卖出次数、今日收益率、整体收益率。
        今日收益率来源于TimeReturn分析器（results['daily_returns']，日频小数），
        买入/卖出次数来自交易记录（当日汇总）。
        """
        try:
            trades = results.get('trades', []) or []
            daily_ret = results.get('daily_returns', {}) or {}
            active_start_date = results.get('active_start_date')

            # 解析交易记录生成每日买卖次数
            buy_counts = {}
            sell_counts = {}
            if trades:
                df_t = pd.DataFrame(trades)
                # 兼容列名
                date_col = None
                for c in ['交易日期', 'date']:
                    if c in df_t.columns:
                        date_col = c
                        break
                action_col = 'action' if 'action' in df_t.columns else (
                    '交易动作' if '交易动作' in df_t.columns else None)
                if date_col is not None and action_col is not None:
                    # 规范日期
                    dates = pd.to_datetime(df_t[date_col], errors='coerce').dt.date
                    actions = df_t[action_col].fillna('')
                    for d, a in zip(dates, actions):
                        if pd.isna(d):
                            continue
                        if active_start_date and d < active_start_date:
                            continue
                        if a == 'BUY':
                            buy_counts[d] = buy_counts.get(d, 0) + 1
                        elif a == 'SELL':
                            sell_counts[d] = sell_counts.get(d, 0) + 1

            # 统一日期集合
            date_set = set()
            date_set.update(buy_counts.keys())
            date_set.update(sell_counts.keys())
            # 过滤daily_ret到active_start_date及之后
            if active_start_date:
                daily_ret = {d: v for d, v in daily_ret.items() if d >= active_start_date}
            date_set.update(daily_ret.keys())
            if not date_set:
                return pd.DataFrame()

            dates_sorted = sorted(date_set)

            rows = []
            cumulative = 1.0
            for d in dates_sorted:
                day_ret = daily_ret.get(d, 0.0)
                try:
                    day_ret = float(day_ret)
                except Exception:
                    day_ret = 0.0
                cumulative *= (1.0 + day_ret)
                rows.append({
                    '日期': pd.to_datetime(d).strftime('%Y-%m-%d'),
                    '买入次数': int(buy_counts.get(d, 0)),
                    '卖出次数': int(sell_counts.get(d, 0)),
                    '今日收益率': round(day_ret * 100.0, 2),
                    '整体收益率': round((cumulative - 1.0) * 100.0, 2),
                })

            return pd.DataFrame(rows)
        except Exception as e:
            print(f"创建每日收益表时出错: {e}")
            return pd.DataFrame()

    @staticmethod
    def create_trade_analysis(trades: List[Dict]) -> pd.DataFrame:
        """
        创建交易分析表，包含买入卖出配对和通道数据
        
        Args:
            trades: 交易记录列表
            
        Returns:
            交易分析DataFrame
        """
        if not trades:
            return pd.DataFrame()
        
        try:
            # 分离买入和卖出交易
            buy_trades = {}  # stock_code -> list of buy trades
            sell_trades = []
            
            for trade in trades:
                action = trade.get('action', trade.get('交易动作', ''))
                stock_code = trade.get('stock_code', trade.get('股票代码', ''))
                
                if action == 'BUY':
                    if stock_code not in buy_trades:
                        buy_trades[stock_code] = []
                    buy_trades[stock_code].append(trade)
                elif action == 'SELL':
                    sell_trades.append(trade)
            
            # 按股票代码和日期排序买入交易
            for stock_code in buy_trades:
                buy_trades[stock_code].sort(key=lambda x: x.get('date', x.get('交易日期', '')))
            
            # 创建交易分析记录
            analysis_records = []
            
            for sell_trade in sell_trades:
                stock_code = sell_trade.get('stock_code', sell_trade.get('股票代码', ''))
                
                # 找到对应的买入交易（FIFO原则）
                if stock_code in buy_trades and buy_trades[stock_code]:
                    buy_trade = buy_trades[stock_code].pop(0)  # 取出最早的买入交易
                    
                    # 提取数据
                    buy_date = buy_trade.get('date', buy_trade.get('交易日期', ''))
                    sell_date = sell_trade.get('date', sell_trade.get('交易日期', ''))
                    buy_price = buy_trade.get('price', buy_trade.get('交易价格', 0))
                    sell_price = sell_trade.get('price', sell_trade.get('交易价格', 0))
                    
                    # 计算收益率
                    returns = 0
                    if buy_price > 0:
                        returns = (sell_price - buy_price) / buy_price * 100
                    
                    # 如果卖出交易中已有收益率，使用它
                    if 'returns' in sell_trade and sell_trade['returns'] is not None:
                        returns = sell_trade['returns']
                    elif '收益率' in sell_trade and sell_trade['收益率'] is not None:
                        returns = sell_trade['收益率']
                    
                    # 提取通道数据（从买入交易）
                    record = {
                        '买入日期': buy_date,
                        '卖出日期': sell_date,
                        '股票代码': stock_code,
                        '买入价格': round(buy_price, 2) if buy_price else 0,
                        '卖出价格': round(sell_price, 2) if sell_price else 0,
                        '收益率(%)': round(returns, 2) if returns else 0,
                        '通道评分': buy_trade.get('通道评分', ''),
                        '斜率β': buy_trade.get('斜率β', ''),
                        'R²': buy_trade.get('R²', ''),
                        '买入中轴': round(buy_trade.get('今日中轴', 0), 2) if buy_trade.get('今日中轴') else '',
                        '买入下沿': round(buy_trade.get('今日下沿', 0), 2) if buy_trade.get('今日下沿') else '',
                        '买入上沿': round(buy_trade.get('今日上沿', 0), 2) if buy_trade.get('今日上沿') else '',
                        '通道宽度': round(buy_trade.get('通道宽度', 0), 2) if buy_trade.get('通道宽度') else '',
                        '距下沿(%)': buy_trade.get('距下沿(%)', ''),
                    }
                    
                    analysis_records.append(record)
            
            # 创建DataFrame
            df = pd.DataFrame(analysis_records)
            
            # 格式化日期
            if not df.empty:
                for date_col in ['买入日期', '卖出日期']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            print(f"创建交易分析表时出错: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
