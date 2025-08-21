#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : run_factor.py
@Description: 因子运行入口 - 简单粗暴的运行方式
@Author     : Zijun Deng
@Date       : 2025-08-20
"""
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# 导入配置，设置环境
from backend.business.factor.core.config import setup_environment

setup_environment()

from backend.business.factor import create_factor_research_framework
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_momentum_factors():
    """运行动量类因子"""
    logger.info("=== 运行动量类因子 ===")

    framework = create_factor_research_framework()

    factor_names = ['momentum_5d', 'momentum_20d', 'momentum_60d']

    results = framework.run_factor_comparison(
        factor_names=factor_names,
        start_date='2025-01-01',
        end_date=datetime.date.today().strftime("%Y-%m-%d"),
        stock_pool='no_st',
        top_n=10,
        n_groups=5
    )

    logger.info(f"动量因子分析完成，报告路径: {results['report_path']}")
    return results


def run_volatility_factors():
    """运行波动率类因子"""
    logger.info("=== 运行波动率类因子 ===")

    framework = create_factor_research_framework()

    factor_names = ['volatility_20d', 'volatility_60d', 'intraday_volatility']

    results = framework.run_factor_comparison(
        factor_names=factor_names,
        start_date='2025-01-01',
        end_date=datetime.date.today().strftime("%Y-%m-%d"),
        stock_pool='no_st',
        top_n=10,
        n_groups=5
    )

    logger.info(f"波动率因子分析完成，报告路径: {results['report_path']}")
    return results


def run_technical_indicators():
    """运行技术指标因子"""
    logger.info("=== 运行技术指标因子 ===")

    framework = create_factor_research_framework()

    factor_names = ['rsi_14', 'rsi_21', 'bollinger_position', 'macd_histogram']

    results = framework.run_factor_comparison(
        factor_names=factor_names,
        start_date='2025-01-01',
        end_date=datetime.date.today().strftime("%Y-%m-%d"),
        stock_pool='no_st',
        top_n=10,
        n_groups=5
    )

    logger.info(f"技术指标因子分析完成，报告路径: {results['report_path']}")
    return results


def run_volume_factors():
    """运行成交量类因子"""
    logger.info("=== 运行成交量类因子 ===")

    framework = create_factor_research_framework()

    factor_names = ['volume_ratio_5d', 'volume_ratio_20d', 'volume_price_momentum']

    results = framework.run_factor_comparison(
        factor_names=factor_names,
        start_date='2025-01-01',
        end_date=datetime.date.today().strftime("%Y-%m-%d"),
        stock_pool='no_st',
        top_n=10,
        n_groups=5
    )

    logger.info(f"成交量因子分析完成，报告路径: {results['report_path']}")
    return results


def run_channel_factors():
    """运行通道分析因子"""
    logger.info("=== 运行通道分析因子 ===")

    framework = create_factor_research_framework()

    factor_names = ['channel_distance', 'channel_breakout', 'channel_width', 'channel_trend']

    results = framework.run_factor_comparison(
        factor_names=factor_names,
        start_date='2025-01-01',
        end_date=datetime.date.today().strftime("%Y-%m-%d"),
        stock_pool='no_st',
        top_n=10,
        n_groups=5
    )

    logger.info(f"通道分析因子分析完成，报告路径: {results['report_path']}")
    return results


def run_worldquant_factors(start_date: str = '2025-01-01', end_date: str = None):
    """运行WorldQuant Alpha因子"""
    logger.info("=== 运行WorldQuant Alpha因子 ===")

    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    framework = create_factor_research_framework()

    # 获取所有WorldQuant Alpha因子
    from backend.business.factor.core.factor.factor_registry import factor_registry
    
    worldquant_factors = [f for f in factor_registry._factors.keys() if f.startswith('alpha_')]
    worldquant_factors.sort(key=lambda x: int(x.split('_')[1]))  # 按数字排序
    
    logger.info(f"找到 {len(worldquant_factors)} 个WorldQuant Alpha因子")
    logger.info(f"因子列表: {worldquant_factors}")

    # 分批运行，避免内存不足
    batch_size = 10
    all_results = {}

    for i in range(0, len(worldquant_factors), batch_size):
        batch_factors = worldquant_factors[i:i + batch_size]
        logger.info(f"运行第 {i // batch_size + 1} 批WorldQuant因子: {batch_factors}")

        try:
            results = framework.run_factor_comparison(
                factor_names=batch_factors,
                start_date=start_date,
                end_date=end_date,
                stock_pool='no_st',
                top_n=10,
                n_groups=5
            )
            all_results[f'worldquant_batch_{i // batch_size + 1}'] = results
            logger.info(f"第 {i // batch_size + 1} 批WorldQuant因子运行成功，报告路径: {results['report_path']}")
        except Exception as e:
            logger.error(f"第 {i // batch_size + 1} 批WorldQuant因子运行失败: {e}")

    logger.info("所有WorldQuant Alpha因子分析完成")
    return all_results


def run_worldquant_factors_merged(start_date: str = '2025-01-01', end_date: str = None, batch_size: int = 10):
    """运行WorldQuant Alpha因子并合并为单一报告"""
    logger.info("=== 运行WorldQuant Alpha因子（合并报告模式） ===")

    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    framework = create_factor_research_framework()

    # 获取所有WorldQuant Alpha因子
    from backend.business.factor.core.factor.factor_registry import factor_registry
    
    worldquant_factors = [f for f in factor_registry._factors.keys() if f.startswith('alpha_')]
    worldquant_factors.sort(key=lambda x: int(x.split('_')[1]))  # 按数字排序
    
    logger.info(f"找到 {len(worldquant_factors)} 个WorldQuant Alpha因子")
    logger.info(f"因子列表: {worldquant_factors}")

    # 分批运行，收集所有结果
    all_backtest_results = {}
    all_effectiveness_results = {}
    all_comparison_results = {}
    successful_factors = []
    failed_factors = []

    for i in range(0, len(worldquant_factors), batch_size):
        batch_factors = worldquant_factors[i:i + batch_size]
        logger.info(f"运行第 {i // batch_size + 1} 批WorldQuant因子: {batch_factors}")

        try:
            results = framework.run_factor_comparison(
                factor_names=batch_factors,
                start_date=start_date,
                end_date=end_date,
                stock_pool='no_st',
                top_n=10,
                n_groups=5
            )
            
            # 合并结果
            if 'backtest_results' in results:
                all_backtest_results.update(results['backtest_results'])
            if 'effectiveness_results' in results:
                all_effectiveness_results.update(results['effectiveness_results'])
            if 'comparison_results' in results:
                all_comparison_results.update(results['comparison_results'])
            
            successful_factors.extend(batch_factors)
            logger.info(f"第 {i // batch_size + 1} 批WorldQuant因子运行成功")
            
        except Exception as e:
            logger.error(f"第 {i // batch_size + 1} 批WorldQuant因子运行失败: {e}")
            failed_factors.extend(batch_factors)

    # 生成合并的综合报告
    logger.info("开始生成合并的综合报告...")
    
    # 创建合并的结果字典
    merged_results = {
        'backtest_results': all_backtest_results,
        'effectiveness_results': all_effectiveness_results,
        'comparison_results': all_comparison_results,
        'successful_factors': successful_factors,
        'failed_factors': failed_factors,
        'total_factors': len(worldquant_factors),
        'success_rate': len(successful_factors) / len(worldquant_factors) if worldquant_factors else 0
    }
    
    # 生成合并报告
    try:
        report_path = framework.generate_merged_comprehensive_report(
            factor_names=successful_factors,
            merged_results=merged_results,
            start_date=start_date,
            end_date=end_date,
            stock_pool='no_st',
            top_n=10,
            n_groups=5
        )
        
        logger.info(f"合并综合报告生成成功: {report_path}")
        logger.info(f"成功运行因子: {len(successful_factors)}/{len(worldquant_factors)}")
        if failed_factors:
            logger.warning(f"失败的因子: {failed_factors}")
        
        return {
            'report_path': report_path,
            'successful_factors': successful_factors,
            'failed_factors': failed_factors,
            'success_rate': merged_results['success_rate'],
            'merged_results': merged_results
        }
        
    except Exception as e:
        logger.error(f"生成合并报告失败: {e}")
        raise


def run_fundamental_factors():
    """运行基本面因子"""
    logger.info("=== 运行基本面因子 ===")

    framework = create_factor_research_framework()

    factor_names = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio']

    results = framework.run_factor_comparison(
        factor_names=factor_names,
        start_date='2025-01-01',
        end_date=datetime.date.today().strftime("%Y-%m-%d"),
        stock_pool='no_st',
        top_n=10,
        n_groups=5
    )

    logger.info(f"基本面因子分析完成，报告路径: {results['report_path']}")
    return results


def run_single_factor(factor_name: str, start_date: str = '2025-01-01', end_date: str = None):
    """运行单个因子"""
    logger.info(f"=== 运行单个因子: {factor_name} ===")

    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    framework = create_factor_research_framework()

    results = framework.run_single_factor_analysis(
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        stock_pool='no_st',
        top_n=10,
        n_groups=5
    )

    logger.info(f"因子 {factor_name} 分析完成，报告路径: {results['report_path']}")
    return results


def run_all_factors():
    """运行所有因子"""
    logger.info("=== 运行所有因子 ===")

    framework = create_factor_research_framework()

    # 获取所有可用因子
    all_factors = framework.get_available_factors()
    factor_names = all_factors['name'].tolist()

    logger.info(f"找到 {len(factor_names)} 个因子")

    # 分批运行，避免内存不足
    batch_size = 10
    all_results = {}

    for i in range(0, len(factor_names), batch_size):
        batch_factors = factor_names[i:i + batch_size]
        logger.info(f"运行第 {i // batch_size + 1} 批因子: {batch_factors}")

        try:
            results = framework.run_factor_comparison(
                factor_names=batch_factors,
                start_date='2025-01-01',
                end_date=datetime.date.today().strftime("%Y-%m-%d"),
                stock_pool='no_st',
                top_n=10,
                n_groups=5
            )
            all_results[f'batch_{i // batch_size + 1}'] = results
        except Exception as e:
            logger.error(f"第 {i // batch_size + 1} 批因子运行失败: {e}")

    logger.info("所有因子分析完成")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='因子运行工具')
    parser.add_argument('--factor_type', type=str, default='momentum',
                        choices=['momentum', 'volatility', 'technical', 'volume',
                                 'channel', 'worldquant', 'fundamental', 'single', 'all'],
                        help='要运行的因子类型')
    parser.add_argument('--factor_name', type=str, default=None,
                        help='单个因子名称（当factor_type为single时使用）')
    parser.add_argument('--start_date', type=str, default='2025-01-01',
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.date.today().strftime("%Y-%m-%d"),
                        help='结束日期 (YYYY-MM-DD)，默认为今天')
    parser.add_argument('--merged', action='store_true',
                        help='使用合并报告模式（分批运行但输出单一报告）')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='批处理大小（仅在合并模式下使用）')

    args = parser.parse_args()

    try:
        if args.factor_type == 'momentum':
            run_momentum_factors()
        elif args.factor_type == 'volatility':
            run_volatility_factors()
        elif args.factor_type == 'technical':
            run_technical_indicators()
        elif args.factor_type == 'volume':
            run_volume_factors()
        elif args.factor_type == 'channel':
            run_channel_factors()
        elif args.factor_type == 'worldquant':
            if args.merged:
                run_worldquant_factors_merged(args.start_date, args.end_date, args.batch_size)
            else:
                run_worldquant_factors(args.start_date, args.end_date)
        elif args.factor_type == 'fundamental':
            run_fundamental_factors()
        elif args.factor_type == 'single':
            if args.factor_name is None:
                logger.error("运行单个因子时必须指定factor_name参数")
                sys.exit(1)
            run_single_factor(args.factor_name, args.start_date, args.end_date)
        elif args.factor_type == 'all':
            run_all_factors()
        else:
            logger.error(f"不支持的因子类型: {args.factor_type}")
            sys.exit(1)

        logger.info("因子运行完成！")

    except Exception as e:
        logger.error(f"因子运行失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
