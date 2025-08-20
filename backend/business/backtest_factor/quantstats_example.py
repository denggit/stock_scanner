#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/19/25 11:24 PM
@File       : quantstats_example.py
@Description: QuantStats报告生成示例
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

def momentum_factor(close: pd.Series, **kwargs) -> pd.Series:
    """动量因子：过去10日收益率"""
    return close.pct_change(10)

def volatility_factor(close: pd.Series, **kwargs) -> pd.Series:
    """波动率因子：过去20日收益率标准差"""
    returns = close.pct_change()
    return returns.rolling(20).std()

def main():
    """QuantStats报告生成示例"""
    logger.info("=== QuantStats报告生成示例 ===")
    
    try:
        # 创建框架实例
        framework = FactorFramework()
        
        # 注册因子
        framework.register_custom_factor('momentum_10d', '10日动量因子', momentum_factor)
        framework.register_custom_factor('volatility_20d', '20日波动率因子', volatility_factor)
        
        # 设置分析参数
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        logger.info(f"分析期间: {start_date} 到 {end_date}")
        
        # 运行单因子分析
        results = framework.run_single_factor_analysis(
            factor_name='momentum_10d',
            start_date=start_date,
            end_date=end_date,
            stock_pool='sz50',  # 使用上证50
            top_n=10,
            n_groups=5
        )
        
        logger.info("✓ 单因子分析完成")
        
        # 查看汇总报告路径
        summary_path = results['report_paths']['summary']
        logger.info("生成的汇总报告:")
        logger.info(f"  汇总报告: {summary_path}")
        
        # 运行多因子对比分析
        multi_results = framework.run_factor_comparison(
            factor_names=['momentum_10d', 'volatility_20d'],
            start_date=start_date,
            end_date=end_date,
            stock_pool='sz50',
            top_n=10,
            n_groups=5
        )
        
        logger.info("✓ 多因子对比分析完成")
        
        # 查看多因子汇总报告路径
        multi_summary_path = multi_results['report_paths']['summary']
        logger.info("生成的多因子汇总报告:")
        logger.info(f"  汇总报告: {multi_summary_path}")
        
        logger.info("=== 示例完成 ===")
        logger.info("请打开HTML汇总报告文件查看完整的因子分析结果")
        logger.info("汇总报告包含：")
        logger.info("  - 📊 分析总览")
        logger.info("  - 📈 TopN回测结果")
        logger.info("  - 📊 分组回测结果")
        logger.info("  - 🔗 多因子回测结果")
        logger.info("  - 📊 IC分析结果")
        logger.info("  - 📈 有效性分析结果")
        
        return results, multi_results
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
