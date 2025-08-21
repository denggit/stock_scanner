#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 8/20/25
@File       : update_financial_data.py
@Description: 增量式财务数据更新脚本，支持智能更新策略
"""
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
os.chdir(root_dir)

import dotenv

dotenv.load_dotenv()

from backend.business.data.data_update import DataUpdateManager
from backend.utils.logger import setup_logger


def update_financial_data_incremental(args, logger):
    """增量式更新财务数据"""
    
    try:
        logger.info("开始增量式财务数据更新...")
        
        data_manager = DataUpdateManager()
        
        # 获取包含财务数据更新时间的股票列表
        stock_list = data_manager.get_stock_list_with_financial_updates()
        logger.info(f"获取到 {len(stock_list)} 只股票")
        
        # 确定更新范围
        current_year = datetime.date.today().year
        
        if args.full:
            # 全量更新：从指定开始年份到当前年份
            start_year = args.start_year
            end_year = current_year
            logger.info(f"执行全量更新: {start_year} - {end_year}")
        else:
            # 增量更新：智能确定需要更新的年份
            start_year, end_year = _determine_incremental_range(stock_list, current_year, logger)
            logger.info(f"执行增量更新: {start_year} - {end_year}")
        
        if start_year > end_year:
            logger.info("所有财务数据已是最新，无需更新")
            return
        
        # 执行更新
        result = data_manager.update_all_financial_data(
            start_year=start_year,
            end_year=end_year,
        )
        
        logger.info("财务数据更新完成")
        
        # 显示更新统计
        if result:
            total_updated = sum(stats.get('updated', 0) for stats in result.values())
            total_failed = sum(stats.get('failed', 0) for stats in result.values())
            logger.info(f"更新统计: 成功 {total_updated}, 失败 {total_failed}")
            
            if total_failed > 0:
                failed_details = {data_type: stats.get('failed_codes', []) 
                                for data_type, stats in result.items() 
                                if stats.get('failed', 0) > 0}
                logger.warning(f"更新失败详情: {failed_details}")

    except Exception as e:
        logger.exception(f"财务数据更新失败: {e}")
        if not args.silent:
            raise
        sys.exit(1)


def _determine_incremental_range(stock_list, current_year, logger):
    """智能确定增量更新的年份范围"""
    
    # 财务数据类型
    financial_data_types = ['profit', 'balance', 'cashflow', 'growth', 'operation', 'dupont', 'dividend']
    
    # 分析每种财务数据的更新情况
    min_update_year = current_year
    
    for data_type in financial_data_types:
        update_time_col = f'update_time_{data_type}'
        
        if update_time_col in stock_list.columns:
            # 获取该数据类型的最小更新年份（排除NULL值）
            valid_updates = stock_list[update_time_col].dropna()
            if not valid_updates.empty:
                # 解析更新时间格式 (YYYY-Q1/Q2/Q3/Q4)
                years = []
                for update_time in valid_updates:
                    if isinstance(update_time, str) and '-' in update_time:
                        year = int(update_time.split('-')[0])
                        years.append(year)
                    elif isinstance(update_time, (int, float)):
                        years.append(int(update_time))
                
                if years:
                    type_min_year = min(years)
                    min_update_year = min(min_update_year, type_min_year)
                    logger.debug(f"{data_type} 数据最早需要从 {type_min_year} 年开始更新")
    
    # 设定合理的起始年份
    if min_update_year == current_year:
        # 如果所有数据都没有更新记录，从去年开始
        start_year = current_year - 1
        logger.info("检测到初次更新，从去年开始更新财务数据")
    else:
        # 从最小更新年份开始，确保数据完整性
        start_year = min_update_year
        logger.info(f"检测到部分数据需要更新，从 {start_year} 年开始")
    
    # 财务数据通常有发布延迟，当前年份的数据可能不完整
    # 但仍要尝试更新以获取最新可用数据
    end_year = current_year
    
    return start_year, end_year


def _check_latest_financial_data(data_manager, logger):
    """检查最新财务数据情况"""
    
    try:
        stock_list = data_manager.get_stock_list_with_financial_updates()
        financial_data_types = ['profit', 'balance', 'cashflow', 'growth', 'operation', 'dupont', 'dividend']
        
        logger.info("=== 财务数据更新状态检查 ===")
        
        for data_type in financial_data_types:
            update_time_col = f'update_time_{data_type}'
            
            if update_time_col in stock_list.columns:
                valid_data = stock_list[update_time_col].dropna()
                
                if not valid_data.empty:
                    latest_year = valid_data.max()
                    earliest_year = valid_data.min()
                    updated_count = len(valid_data)
                    total_count = len(stock_list)
                    
                    logger.info(f"{data_type:10} | 覆盖率: {updated_count:4}/{total_count:4} ({updated_count/total_count*100:5.1f}%) | 年份范围: {earliest_year}-{latest_year}")
                else:
                    logger.info(f"{data_type:10} | 覆盖率:    0/{len(stock_list):4} (  0.0%) | 年份范围: 无数据")
            else:
                logger.warning(f"{data_type:10} | 缺少更新时间字段")
        
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"检查财务数据状态失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="增量式财务数据更新工具")
    parser.add_argument("--full", action="store_true", help="强制全量更新（默认增量更新）")
    parser.add_argument("--silent", action="store_true", help="静默模式，减少输出信息")
    parser.add_argument("--start-year", type=int, default=2015, help="全量更新的起始年份（默认2015）")
    parser.add_argument("--check", action="store_true", help="仅检查当前财务数据状态，不执行更新")
    args = parser.parse_args()

    # 设置日志
    if args.silent:
        logger = setup_logger("update_financial_data", log_level=logging.WARNING, set_root_logger=True)
    else:
        logger = setup_logger("update_financial_data", log_level=logging.INFO, set_root_logger=True)

    try:
        if args.check:
            # 仅检查数据状态
            data_manager = DataUpdateManager()
            _check_latest_financial_data(data_manager, logger)
        else:
            # 执行更新
            update_financial_data_incremental(args, logger)
            
    except KeyboardInterrupt:
        print("\n用户中断更新过程")
        sys.exit(1)


if __name__ == "__main__":
    """
    使用示例:
    python bin/update_financial_data.py                    # 增量更新
    python bin/update_financial_data.py --full             # 全量更新
    python bin/update_financial_data.py --check            # 检查状态
    python bin/update_financial_data.py --start-year 2020  # 从2020年开始全量更新
    python bin/update_financial_data.py --silent           # 静默模式
    """
    main()
