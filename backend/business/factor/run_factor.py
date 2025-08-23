#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : run_factor.py
@Description: 因子运行入口 - 简化后的线性流程
@Author     : Zijun Deng
@Date       : 2025-08-20
"""
import os
import sys
from datetime import date
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# 导入配置
from backend.business.factor.core.config import (
    DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_STOCK_POOL,
    DEFAULT_TOP_N, DEFAULT_N_GROUPS, DEFAULT_BATCH_SIZE,
    DEFAULT_OPTIMIZE_DATA_FETCH_FOR_WORLDQUANT, DEFAULT_REPORT_OUTPUT_DIR
)

from backend.business.factor import main
from backend.utils.logger import setup_logger

logger = setup_logger("backtest_factor")


class FactorRunner:
    """
    因子运行器 - 使用简化的线性流程
    """
    
    def __init__(self, report_dir: str = DEFAULT_REPORT_OUTPUT_DIR):
        """
        初始化因子运行器
        
        Args:
            report_dir: 报告输出目录
        """
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # 因子配置
        self.factor_configs = {
            'momentum': {
                'factors': ['momentum_5d', 'momentum_20d', 'momentum_60d'],
                'description': '动量类因子'
            },
            'volatility': {
                'factors': ['volatility_20d', 'volatility_60d', 'intraday_volatility'],
                'description': '波动率类因子'
            },
            'technical': {
                'factors': ['rsi_14', 'rsi_21', 'bollinger_position', 'macd_histogram'],
                'description': '技术指标因子'
            },
            'volume': {
                'factors': ['volume_ratio_5d', 'volume_ratio_20d', 'volume_price_momentum'],
                'description': '成交量类因子'
            },
            'channel': {
                'factors': ['channel_distance', 'channel_breakout', 'channel_width', 'channel_trend'],
                'description': '通道分析因子'
            },
            'fundamental': {
                'factors': ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio'],
                'description': '基本面因子'
            }
        }
        
        logger.info("因子运行器初始化完成")

    def run_single_factor_type(self, factor_type: str, stock_pool: str, start_date: str, end_date: str,
                              top_n: int = DEFAULT_TOP_N, n_groups: int = DEFAULT_N_GROUPS):
        """
        Runs the analysis for a single type of factor using the simplified flow.
        """
        logger.info(f"=== 运行 {factor_type} 类因子 ===")

        factor_config = self.factor_configs.get(factor_type, {})
        factor_names = factor_config.get('factors', [])
        if not factor_names:
            logger.warning(f"因子类型 '{factor_type}' 没有配置任何因子，跳过。")
            return

        # 1. 创建唯一的 FactorFramework 实例
        framework = main.FactorResearchFramework(output_dir=self.report_dir)

        # 2. 调用新的、统一的执行和报告方法
        report_path = framework.run_and_report(
            factor_names=factor_names,
            batch_name=f"{factor_type} 因子分析报告",
            output_dir=self.report_dir,
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            top_n=top_n,
            n_groups=n_groups
        )

        logger.info(f"{factor_type} 因子分析完成，报告路径: {report_path}")
        return report_path

    def run_worldquant_factors(self, start_date: str, end_date: str, stock_pool: str,
                              top_n: int = DEFAULT_TOP_N, n_groups: int = DEFAULT_N_GROUPS,
                              batch_size: int = DEFAULT_BATCH_SIZE,
                              optimize_data_fetch: bool = DEFAULT_OPTIMIZE_DATA_FETCH_FOR_WORLDQUANT):
        """
        运行WorldQuant Alpha因子
        """
        logger.info("=== 运行WorldQuant Alpha因子 ===")

        # 获取所有WorldQuant Alpha因子
        from backend.business.factor.core.factor.factor_registry import factor_registry

        worldquant_factors = [f for f in factor_registry._factors.keys() if f.startswith('alpha_')]
        worldquant_factors.sort(key=lambda x: int(x.split('_')[1]))  # 按数字排序

        logger.info(f"找到 {len(worldquant_factors)} 个WorldQuant Alpha因子")

        # 分批运行
        all_results = {}
        for i in range(0, len(worldquant_factors), batch_size):
            batch_factors = worldquant_factors[i:i + batch_size]
            logger.info(f"运行第 {i // batch_size + 1} 批WorldQuant因子: {batch_factors}")

            try:
                # 创建唯一的 FactorFramework 实例
                framework = main.FactorResearchFramework(output_dir=self.report_dir)
                
                # 调用新的、统一的执行和报告方法
                report_path = framework.run_and_report(
                    factor_names=batch_factors,
                    batch_name=f"WorldQuant因子批次{i // batch_size + 1}",
                    output_dir=self.report_dir,
                    start_date=start_date,
                    end_date=end_date,
                    stock_pool=stock_pool,
                    top_n=top_n,
                    n_groups=n_groups,
                    optimize_data_fetch=optimize_data_fetch
                )
                
                all_results[f'worldquant_batch_{i // batch_size + 1}'] = {
                    'report_path': report_path,
                    'factors': batch_factors
                }
                logger.info(f"第 {i // batch_size + 1} 批WorldQuant因子运行成功，报告路径: {report_path}")
            except Exception as e:
                logger.error(f"第 {i // batch_size + 1} 批WorldQuant因子运行失败: {e}")

        logger.info("所有WorldQuant Alpha因子分析完成")
        return all_results

    def run_single_factor(self, factor_name: str, start_date: str, end_date: str, stock_pool: str,
                         top_n: int = DEFAULT_TOP_N, n_groups: int = DEFAULT_N_GROUPS):
        """
        运行单个因子
        """
        logger.info(f"=== 运行单个因子: {factor_name} ===")

        # 创建唯一的 FactorFramework 实例
        framework = main.FactorResearchFramework(output_dir=self.report_dir)
        
        # 调用新的、统一的执行和报告方法
        report_path = framework.run_and_report(
            factor_names=[factor_name],
            batch_name=f"单个因子分析 - {factor_name}",
            output_dir=self.report_dir,
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            top_n=top_n,
            n_groups=n_groups
        )

        logger.info(f"因子 {factor_name} 分析完成，报告路径: {report_path}")
        return report_path

    def run_all_factors(self, start_date: str, end_date: str, stock_pool: str,
                       top_n: int = DEFAULT_TOP_N, n_groups: int = DEFAULT_N_GROUPS,
                       batch_size: int = DEFAULT_BATCH_SIZE):
        """
        运行所有因子
        """
        logger.info("=== 运行所有因子 ===")

        # 获取所有可用因子
        framework = main.FactorResearchFramework(output_dir=self.report_dir)
        all_factors = framework.get_available_factors()
        factor_names = all_factors['name'].tolist()

        logger.info(f"找到 {len(factor_names)} 个因子")

        # 分批运行
        all_results = {}
        for i in range(0, len(factor_names), batch_size):
            batch_factors = factor_names[i:i + batch_size]
            logger.info(f"运行第 {i // batch_size + 1} 批因子: {batch_factors}")

            try:
                # 创建唯一的 FactorFramework 实例
                batch_framework = main.FactorResearchFramework(output_dir=self.report_dir)
                
                # 调用新的、统一的执行和报告方法
                report_path = batch_framework.run_and_report(
                    factor_names=batch_factors,
                    batch_name=f"全因子分析批次{i // batch_size + 1}",
                    output_dir=self.report_dir,
                    start_date=start_date,
                    end_date=end_date,
                    stock_pool=stock_pool,
                    top_n=top_n,
                    n_groups=n_groups
                )
                
                all_results[f'batch_{i // batch_size + 1}'] = {
                    'report_path': report_path,
                    'factors': batch_factors
                }
            except Exception as e:
                logger.error(f"第 {i // batch_size + 1} 批因子运行失败: {e}")

        logger.info("所有因子分析完成")
        return all_results


# 便捷函数 - 保持向后兼容性
def run_momentum_factors(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                         stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行动量类因子"""
    runner = FactorRunner()
    return runner.run_single_factor_type('momentum', stock_pool, start_date, end_date, top_n, n_groups)


def run_volatility_factors(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                           stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行波动率类因子"""
    runner = FactorRunner()
    return runner.run_single_factor_type('volatility', stock_pool, start_date, end_date, top_n, n_groups)


def run_technical_indicators(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                             stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行技术指标因子"""
    runner = FactorRunner()
    return runner.run_single_factor_type('technical', stock_pool, start_date, end_date, top_n, n_groups)


def run_volume_factors(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                       stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行成交量类因子"""
    runner = FactorRunner()
    return runner.run_single_factor_type('volume', stock_pool, start_date, end_date, top_n, n_groups)


def run_channel_factors(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                        stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行通道分析因子"""
    runner = FactorRunner()
    return runner.run_single_factor_type('channel', stock_pool, start_date, end_date, top_n, n_groups)


def run_worldquant_factors(start_date: str = DEFAULT_START_DATE, end_date: str = None,
                           stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS,
                           optimize_data_fetch=DEFAULT_OPTIMIZE_DATA_FETCH_FOR_WORLDQUANT):
    """运行WorldQuant Alpha因子"""
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    
    runner = FactorRunner()
    return runner.run_worldquant_factors(start_date, end_date, stock_pool, top_n, n_groups, 
                                       DEFAULT_BATCH_SIZE, optimize_data_fetch)


def run_fundamental_factors(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                            stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行基本面因子"""
    runner = FactorRunner()
    return runner.run_single_factor_type('fundamental', stock_pool, start_date, end_date, top_n, n_groups)


def run_single_factor(factor_name: str, start_date: str = DEFAULT_START_DATE, end_date: str = None,
                      stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行单个因子"""
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    
    runner = FactorRunner()
    return runner.run_single_factor(factor_name, start_date, end_date, stock_pool, top_n, n_groups)


def run_all_factors(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
                    stock_pool=DEFAULT_STOCK_POOL, top_n=DEFAULT_TOP_N, n_groups=DEFAULT_N_GROUPS):
    """运行所有因子"""
    runner = FactorRunner()
    return runner.run_all_factors(start_date, end_date, stock_pool, top_n, n_groups, DEFAULT_BATCH_SIZE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='因子运行工具')
    parser.add_argument('--factor_type', type=str, default='momentum',
                        choices=['momentum', 'volatility', 'technical', 'volume',
                                 'channel', 'worldquant', 'fundamental', 'single', 'all'],
                        help='要运行的因子类型')
    parser.add_argument('--factor_name', type=str, default=None,
                        help='单个因子名称（当factor_type为single时使用）')
    parser.add_argument('--start_date', type=str, default=DEFAULT_START_DATE,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=DEFAULT_END_DATE,
                        help='结束日期 (YYYY-MM-DD)，默认为今天')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='批处理大小')
    parser.add_argument('--stock_pool', type=str, default=DEFAULT_STOCK_POOL,
                        help='股票池')
    parser.add_argument('--top_n', type=int, default=DEFAULT_TOP_N,
                        help='买入因子排名前n只股票')
    parser.add_argument('--n_groups', type=int, default=DEFAULT_N_GROUPS,
                        help='分组数量')

    args = parser.parse_args()

    try:
        runner = FactorRunner()
        
        if args.factor_type == 'momentum':
            runner.run_single_factor_type('momentum', args.stock_pool, args.start_date, args.end_date, args.top_n, args.n_groups)
        elif args.factor_type == 'volatility':
            runner.run_single_factor_type('volatility', args.stock_pool, args.start_date, args.end_date, args.top_n, args.n_groups)
        elif args.factor_type == 'technical':
            runner.run_single_factor_type('technical', args.stock_pool, args.start_date, args.end_date, args.top_n, args.n_groups)
        elif args.factor_type == 'volume':
            runner.run_single_factor_type('volume', args.stock_pool, args.start_date, args.end_date, args.top_n, args.n_groups)
        elif args.factor_type == 'channel':
            runner.run_single_factor_type('channel', args.stock_pool, args.start_date, args.end_date, args.top_n, args.n_groups)
        elif args.factor_type == 'worldquant':
            runner.run_worldquant_factors(args.start_date, args.end_date, args.stock_pool, args.top_n, args.n_groups, args.batch_size)
        elif args.factor_type == 'fundamental':
            runner.run_single_factor_type('fundamental', args.stock_pool, args.start_date, args.end_date, args.top_n, args.n_groups)
        elif args.factor_type == 'single':
            if args.factor_name is None:
                logger.error("运行单个因子时必须指定factor_name参数")
                sys.exit(1)
            runner.run_single_factor(args.factor_name, args.start_date, args.end_date, args.stock_pool, args.top_n, args.n_groups)
        elif args.factor_type == 'all':
            runner.run_all_factors(args.start_date, args.end_date, args.stock_pool, args.top_n, args.n_groups, args.batch_size)
        else:
            logger.error(f"不支持的因子类型: {args.factor_type}")
            sys.exit(1)

        logger.info("因子运行完成！")

    except Exception as e:
        logger.error(f"因子运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
