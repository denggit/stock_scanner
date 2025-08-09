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
        if not trades:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(trades)

            # 兼容不同的列名
            returns_col = None
            for col in ['returns', '收益率', '收益率(%)']:
                if col in df.columns:
                    returns_col = col
                    break

            if returns_col is None:
                return pd.DataFrame()

            sell_trades = df[df['action'] == 'SELL'].copy()

            if len(sell_trades) == 0:
                return pd.DataFrame()

            # 转换日期格式
            date_col = None
            for col in ['date', '交易日期']:
                if col in sell_trades.columns:
                    date_col = col
                    break

            if date_col is None:
                return pd.DataFrame()

            sell_trades['date'] = pd.to_datetime(sell_trades[date_col])
            sell_trades['month'] = sell_trades['date'].dt.to_period('M')

            # 按月度汇总 - 基于总资本变化计算总收益率
            monthly_data = []

            for month, group in sell_trades.groupby('month'):
                # 获取交易金额列
                value_col = None
                for col in ['value', '交易金额']:
                    if col in group.columns:
                        value_col = col
                        break

                if not value_col:
                    continue

                # 计算总资本变化
                total_capital_invested = 0  # 总投入资本
                total_capital_returned = 0  # 总收回资本

                for _, trade in group.iterrows():
                    trade_value = trade.get(value_col, 0)
                    trade_return = trade.get(returns_col, 0)

                    if trade_value > 0 and trade_return is not None:
                        # 根据收益率计算投入资本
                        # 收益率 = (卖出价格 - 买入价格) / 买入价格
                        # 投入资本 = 交易金额 / (1 + 收益率/100)
                        if trade_return != -100:  # 避免除零错误
                            invested_capital = trade_value / (1 + trade_return / 100)
                        else:
                            # 如果收益率是-100%，说明全部亏损
                            invested_capital = trade_value

                        total_capital_invested += invested_capital
                        total_capital_returned += trade_value

                # 计算基于总资本的总收益率
                if total_capital_invested > 0:
                    total_return_rate = ((
                                                 total_capital_returned - total_capital_invested) / total_capital_invested) * 100
                else:
                    total_return_rate = 0

                # 计算平均收益率（基于单笔交易收益率）
                avg_return_rate = group[returns_col].mean() if returns_col in group.columns else 0

                month_data = {
                    '月份': str(month),
                    '交易次数': len(group),
                    '总收益率': total_return_rate,  # 基于总资本变化
                    '平均收益率': avg_return_rate,  # 基于单笔交易收益率
                    '交易金额': group[value_col].sum() if value_col else 0,
                    '投入资本': total_capital_invested,  # 新增：显示投入资本
                    '收回资本': total_capital_returned  # 新增：显示收回资本
                }

                monthly_data.append(month_data)

            if monthly_data:
                result_df = pd.DataFrame(monthly_data)
                # 格式化数值
                numeric_cols = ['总收益率', '平均收益率', '交易金额', '投入资本', '收回资本']
                for col in numeric_cols:
                    if col in result_df.columns:
                        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').round(2)

                return result_df
            else:
                return pd.DataFrame()

        except Exception as e:
            # 如果处理失败，返回空的DataFrame
            print(f"创建月度收益表时出错: {e}")
            return pd.DataFrame()

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
                    monthly_returns = ReportUtils.create_monthly_returns(trades)
                    if not monthly_returns.empty:
                        monthly_returns.to_excel(writer, sheet_name='月度收益', index=False)

                    # 详细交易记录 - 保存到"交易记录"sheet
                    trades_df = ReportUtils.create_detailed_trade_records(trades)
                    if not trades_df.empty:
                        trades_df.to_excel(writer, sheet_name='交易记录', index=False)

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
