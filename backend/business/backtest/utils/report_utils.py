#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
报告工具类
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


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
        
        df = pd.DataFrame(trades)
        
        # 计算交易统计
        buy_trades = df[df['action'] == 'BUY']
        sell_trades = df[df['action'] == 'SELL']
        
        summary = {
            '统计项': [
                '总交易数', '买入交易数', '卖出交易数',
                '平均买入价格', '平均卖出价格',
                '最大单笔交易金额', '最小单笔交易金额'
            ],
            '数值': [
                len(df),
                len(buy_trades),
                len(sell_trades),
                f"{buy_trades['price'].mean():.2f}" if len(buy_trades) > 0 else "N/A",
                f"{sell_trades['price'].mean():.2f}" if len(sell_trades) > 0 else "N/A",
                f"{df['value'].max():,.2f}",
                f"{df['value'].min():,.2f}"
            ]
        }
        
        return pd.DataFrame(summary)
    
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
        
        df = pd.DataFrame(trades)
        sell_trades = df[df['action'] == 'SELL'].copy()
        
        if len(sell_trades) == 0:
            return pd.DataFrame()
        
        # 转换日期格式
        sell_trades['date'] = pd.to_datetime(sell_trades['date'])
        sell_trades['month'] = sell_trades['date'].dt.to_period('M')
        
        # 按月度汇总
        monthly_returns = sell_trades.groupby('month').agg({
            'returns': ['count', 'sum', 'mean'],
            'value': 'sum'
        }).round(2)
        
        monthly_returns.columns = ['交易次数', '总收益率', '平均收益率', '交易金额']
        monthly_returns = monthly_returns.reset_index()
        monthly_returns['月份'] = monthly_returns['month'].astype(str)
        
        return monthly_returns[['月份', '交易次数', '总收益率', '平均收益率', '交易金额']]
    
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
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.2f}', ha='center', va='bottom')
            else:
                # 条形图（水平）
                bars = ax.barh(df['策略名称'], df[metric])
                ax.set_title(f'{metric}比较')
                ax.set_xlabel(metric)
                
                # 添加数值标签
                for bar, value in zip(bars, df[metric]):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
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
                
                # 交易汇总表
                trades = results.get('trades', [])
                if trades:
                    trade_summary = ReportUtils.create_trade_summary(trades)
                    if not trade_summary.empty:
                        trade_summary.to_excel(writer, sheet_name='交易汇总', index=False)
                    
                    # 月度收益表
                    monthly_returns = ReportUtils.create_monthly_returns(trades)
                    if not monthly_returns.empty:
                        monthly_returns.to_excel(writer, sheet_name='月度收益', index=False)
                    
                    # 详细交易记录
                    trades_df = pd.DataFrame(trades)
                    if not trades_df.empty:
                        trades_df.to_excel(writer, sheet_name='交易详情', index=False)
                
                # 策略报告
                if 'report' in results and results['report']:
                    report_df = pd.DataFrame({'报告': [results['report']]})
                    report_df.to_excel(writer, sheet_name='策略报告', index=False)
            
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