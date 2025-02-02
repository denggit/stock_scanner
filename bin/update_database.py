#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/2/2025 12:16 PM
@File       : update_database.py
@Description: 
"""
import argparse
import logging
import sys
import time
import traceback

from tqdm import tqdm

from backend.data.data_manager import DataUpdateManager
from backend.utils import format_time
from backend.utils.logger import setup_logger


def update_database(args, logger):
    """更新数据库"""

    try:
        logger.info("更新数据库")
        start_time = time.time()

        data_manager = DataUpdateManager()
        stock_list = data_manager.get_stock_list()
        data_manager.update_stock_list(stock_list)
        total_stocks = len(stock_list)

        # 创建进度条
        with tqdm(total=total_stocks, desc="更新数据库", unit="只股票") as pbar:
            result = data_manager.update_all_stocks(force_full_update=args.full,
                                                    progress_callback=lambda: pbar.update(1))

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 显示最终结果
        logger.info("更新数据库完成")
        logger.info(f"更新数据库完成，耗时 {format_time(elapsed_time)}")
        logger.info(f"处理速度：{total_stocks / elapsed_time:.2f} 只/秒")
        logger.info(f"更新结果: {result}")

        if result['failed'] > 0:
            logger.warning(f"更新失败的股票数量: {result['failed']}")
            logger.warning(f"更新失败的股票代码: {result['failed_codes']}")

    except Exception as e:
        logger.error(f"更新数据库失败: {e}")
        if not args.silent:
            traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="更新股票数据库")
    parser.add_argument("--full", action="store_true", help="强制全量更新(默认增量更新)")
    parser.add_argument("--silent", action="store_true", help="静默模式，减少输出信息")
    args = parser.parse_args()

    # 设置日志
    if args.silent:
        logger = setup_logger("update_database", log_level=logging.WARNING)
    else:
        logger = setup_logger("update_database", log_level=logging.INFO)

    try:
        update_database(args, logger=logger)
    except KeyboardInterrupt:
        print("\n用户终端更新过程")
        sys.exit(1)


if __name__ == "__main__":
    main()
