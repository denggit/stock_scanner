#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : example_with_real_data.py
@Description: 使用真实数据的因子回测框架示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.business.backtest_factor import FactorFramework
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

def example_alpha_8_factor(open: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """
    Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    
    -1乘以(过去5日开盘价之和乘以过去5日收益率之和)与其10日前的值的差的排名。
    
    Args:
        open: 开盘价序列
        pct_chg: 收益率序列（百分比形式）
    Returns:
        Alpha#8因子值
    """
    # 转换百分比收益率为小数形式
    returns = pct_chg / 100.0
    
    sum_open = open.rolling(5).sum()
    sum_returns = returns.rolling(5).sum()
    product = sum_open * sum_returns
    delay_product = product.shift(10)
    
    factor_value = -1 * (product - delay_product)
    return factor_value.rank(pct=True)

def example_momentum_factor(close: pd.Series, **kwargs) -> pd.Series:
    """
    简单动量因子：过去20日收益率
    
    Args:
        close: 收盘价序列
    Returns:
        动量因子值
    """
    return close.pct_change(20)

def example_volatility_factor(close: pd.Series, **kwargs) -> pd.Series:
    """
    波动率因子：过去20日收益率的标准差
    
    Args:
        close: 收盘价序列
    Returns:
        波动率因子值
    """
    returns = close.pct_change()
    return returns.rolling(20).std()

def example_pe_factor(pe_ttm: pd.Series, **kwargs) -> pd.Series:
    """
    PE倒数因子：1/PE，PE越低因子值越大
    
    Args:
        pe_ttm: 市盈率TTM序列
    Returns:
        PE倒数因子值
    """
    # 处理PE异常值
    pe_clean = pe_ttm.replace([np.inf, -np.inf], np.nan)
    pe_clean = pe_clean[(pe_clean > 0) & (pe_clean < 100)]  # 过滤极端PE值
    
    return 1.0 / pe_clean

def run_single_factor_example():
    """运行单因子分析示例"""
    logger.info("=== 单因子分析示例 ===")
    
    try:
        # 创建框架实例
        framework = FactorFramework()
        
        # 注册自定义因子
        framework.register_custom_factor('alpha_8', 'Alpha#8因子', example_alpha_8_factor)
        framework.register_custom_factor('momentum_20d', '20日动量因子', example_momentum_factor)
        framework.register_custom_factor('volatility_20d', '20日波动率因子', example_volatility_factor)
        framework.register_custom_factor('pe_inverse', 'PE倒数因子', example_pe_factor)
        
        # 设置分析参数
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        logger.info(f"分析期间: {start_date} 到 {end_date}")
        
        # 运行单因子分析 - 使用沪深300股票池
        results = framework.run_single_factor_analysis(
            factor_name='alpha_8',
            start_date=start_date,
            end_date=end_date,
            stock_pool='hs300',  # 使用沪深300股票池
            top_n=10,
            n_groups=5
        )
        
        # 查看结果
        logger.info("✓ 单因子分析完成")
        
        # 显示数据信息
        data_info = results['data_info']
        logger.info(f"数据信息: {data_info}")
        
        # 显示因子摘要
        factor_summary = results['factor_summary']
        if factor_summary is not None and not factor_summary.empty:
            logger.info(f"因子摘要:\n{factor_summary}")
        
        # 显示回测结果
        backtest_results = results['backtest_results']
        for result_key, result in backtest_results.items():
            logger.info(f"回测结果 {result_key}: {result['stats']}")
        
        # 显示报告路径
        report_paths = results['report_paths']
        logger.info(f"报告文件路径: {report_paths}")
        
        return results
        
    except Exception as e:
        logger.error(f"单因子分析失败: {e}")
        raise

def run_multi_factor_example():
    """运行多因子对比分析示例"""
    logger.info("=== 多因子对比分析示例 ===")
    
    try:
        # 创建框架实例
        framework = FactorFramework()
        
        # 注册多个因子
        framework.register_custom_factor('alpha_8', 'Alpha#8因子', example_alpha_8_factor)
        framework.register_custom_factor('momentum_20d', '20日动量因子', example_momentum_factor)
        framework.register_custom_factor('volatility_20d', '20日波动率因子', example_volatility_factor)
        framework.register_custom_factor('pe_inverse', 'PE倒数因子', example_pe_factor)
        
        # 设置分析参数
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")  # 短一点的时间窗口
        
        logger.info(f"分析期间: {start_date} 到 {end_date}")
        
        # 运行多因子对比分析 - 使用中证500股票池
        results = framework.run_factor_comparison(
            factor_names=['alpha_8', 'momentum_20d', 'pe_inverse'],
            start_date=start_date,
            end_date=end_date,
            stock_pool='zz500',  # 使用中证500股票池
            top_n=15,
            n_groups=5,
            weights=[0.5, 0.3, 0.2]  # 多因子权重
        )
        
        # 查看对比结果
        logger.info("✓ 多因子对比分析完成")
        
        if 'comparison_results' in results:
            comparison_results = results['comparison_results']
            
            if 'ic_comparison' in comparison_results:
                ic_comparison = comparison_results['ic_comparison']
                logger.info(f"IC对比结果:\n{ic_comparison}")
            
            if 'backtest_comparison' in comparison_results:
                backtest_comparison = comparison_results['backtest_comparison']
                logger.info(f"回测对比结果:\n{backtest_comparison}")
        
        return results
        
    except Exception as e:
        logger.error(f"多因子对比分析失败: {e}")
        raise

def run_stock_pool_example():
    """运行不同股票池对比示例"""
    logger.info("=== 不同股票池对比示例 ===")
    
    try:
        # 创建框架实例
        framework = FactorFramework()
        
        # 注册因子
        framework.register_custom_factor('momentum_20d', '20日动量因子', example_momentum_factor)
        
        # 设置分析参数
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")  # 更短的时间窗口
        
        logger.info(f"分析期间: {start_date} 到 {end_date}")
        
        # 不同股票池的对比
        stock_pools = ['sz50', 'hs300', 'zz500']
        pool_results = {}
        
        for pool in stock_pools:
            logger.info(f"分析股票池: {pool}")
            
            try:
                results = framework.run_single_factor_analysis(
                    factor_name='momentum_20d',
                    start_date=start_date,
                    end_date=end_date,
                    stock_pool=pool,
                    top_n=10,
                    n_groups=3
                )
                
                pool_results[pool] = results
                
                # 显示关键指标
                data_info = results['data_info']
                logger.info(f"股票池 {pool} 股票数量: {data_info.get('stock_count', 'N/A')}")
                
                backtest_results = results['backtest_results']
                if f'topn_momentum_20d' in backtest_results:
                    stats = backtest_results[f'topn_momentum_20d']['stats']
                    logger.info(f"股票池 {pool} 回测结果: {stats}")
                
            except Exception as e:
                logger.warning(f"股票池 {pool} 分析失败: {e}")
                continue
        
        logger.info("✓ 股票池对比分析完成")
        return pool_results
        
    except Exception as e:
        logger.error(f"股票池对比分析失败: {e}")
        raise

def main():
    """主函数"""
    logger.info("=== 因子回测框架真实数据示例 ===")
    
    try:
        # 运行单因子分析
        single_results = run_single_factor_example()
        
        # 运行多因子对比分析
        multi_results = run_multi_factor_example()
        
        # 运行股票池对比分析
        pool_results = run_stock_pool_example()
        
        logger.info("=== 所有示例运行完成 ===")
        
        return {
            'single_factor': single_results,
            'multi_factor': multi_results,
            'stock_pools': pool_results
        }
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        raise

if __name__ == "__main__":
    main()
