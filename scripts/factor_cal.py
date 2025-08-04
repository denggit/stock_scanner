#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 3/14/2025 12:38 AM
@File       : factor_cal.py
@Description: 单因子计算与导出工具
"""
import datetime
import inspect
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# 设置项目根目录
root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
os.chdir(root_dir)

# 导入项目模块
from backend.data.stock_data_fetcher import StockDataFetcher
from backend.quant.core.factor_engine.factor_generator import get_registered_factors
from backend.utils.logger import setup_logger

# 设置日志
logger = setup_logger("factor_cal")


def calculate_single_factor(
        factor_name: str,
        start_date: Optional[str] = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
        end_date: Optional[str] = datetime.date.today().strftime("%Y-%m-%d"),
        pool_name: str = "hs300",
        ipo_days: int = 365,
        min_amount: float = 5000000,
        output_dir: str = None
) -> pd.DataFrame:
    """
    计算单个因子在指定日期的值，并导出为Excel文件
    
    Args:
        factor_name: 因子名称
        start_date: 开始日期，获取股票数据的长度，格式为YYYY-MM-DD，默认为一年前
        end_date: 计算日期，格式为YYYY-MM-DD，默认为最近交易日
        pool_name: 股票池名称，如'hs300', 'zz500', 'zz1000', 'all'等
        ipo_days: 上市天数限制，默认365天
        min_amount: 最小成交额限制，默认500万
        output_dir: 输出目录，默认为results/{当前日期}
        
    Returns:
        包含因子值的DataFrame
    """
    logger.info(f"开始计算因子: {factor_name}")

    # 初始化数据获取器
    fetcher = StockDataFetcher()

    # 获取日期
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        logger.info(f"未指定日期，使用最近交易日: {end_date}")

    # 获取股票列表
    logger.info(f"获取股票池: {pool_name}，上市天数>={ipo_days}天，最小成交额>={min_amount / 10000}万")
    ipo_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=ipo_days)).strftime(
        "%Y-%m-%d")

    stock_list = fetcher.get_stock_list_with_cond(
        pool_name=pool_name,
        ipo_date=ipo_date,
        min_amount=min_amount,
    )
    logger.info(f"符合条件的股票数量: {len(stock_list)}")

    # 获取因子函数
    try:
        # 获取注册的因子函数
        all_factors = get_registered_factors()

        if factor_name not in all_factors:
            logger.error(f"因子 '{factor_name}' 未找到，请确认因子名称正确")
            return pd.DataFrame()

        factor_func = all_factors[factor_name]
        logger.info(f"成功加载因子函数: {factor_name}")

    except Exception as e:
        logger.exception(f"加载因子函数失败: {e}")
        return pd.DataFrame()

    # 计算因子值
    try:
        logger.info(f"正在计算因子值...")

        # 获取因子函数所需参数
        params = inspect.signature(factor_func).parameters
        param_names = list(params.keys())
        logger.info(f"因子函数需要的参数: {param_names}")

        # 为每只股票计算因子值
        factor_values = {}

        for stock_code in stock_list['code']:
            try:
                # 获取股票历史数据
                df = fetcher.fetch_stock_data(
                    code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="1"  # 后复权
                )

                if df is None or df.empty:
                    logger.info(f"无法获取股票 {stock_code} 的历史数据")
                    continue

                # 准备参数
                args = {}
                for param in param_names:
                    if param in df.columns:
                        args[param] = df[param]
                    elif param == "open_price" and "open" in df.columns:
                        # 因为open是一个built_in参数，所以尽量不直接使用它
                        args["open_price"] = df["open"]

                # 计算因子
                if args:
                    factor_value = factor_func(**args)

                    # 获取最后一个日期的因子值（即目标日期）
                    if isinstance(factor_value, pd.Series):
                        if not factor_value.empty:
                            factor_values[stock_code] = factor_value.iloc[-1]
                    else:
                        factor_values[stock_code] = factor_value

            except Exception as e:
                logger.info(f"计算股票 {stock_code} 的因子值时出错: {e}")

        # 检查计算结果
        if not factor_values:
            logger.error(f"因子 {factor_name} 计算结果为空")
            return pd.DataFrame()

        # 转换为DataFrame
        factor_df = pd.DataFrame({
            "股票代码": list(factor_values.keys()),
            "因子值": list(factor_values.values())
        })

        # 去除无效数据
        factor_df = factor_df.dropna().copy()

        # 获取股票名称
        stocks_basic = stock_list.set_index('code')
        factor_df["股票名称"] = factor_df["股票代码"].map(
            lambda x: stocks_basic.loc[x, 'name'] if x in stocks_basic.index else "")

        # 按因子值排序
        factor_df = factor_df.sort_values("因子值", ascending=False).reset_index(drop=True)

        # 导出到Excel
        if output_dir is None:
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_dir = os.path.join("results", current_date)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{factor_name}.xlsx")

        # 创建Excel写入器
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 写入因子值表
            factor_df.to_excel(writer, sheet_name='因子值', index=False)

            # 写入计算参数表
            params_df = pd.DataFrame({
                "参数": ["因子名称", "计算日期", "股票池", "上市天数限制", "最小成交额限制", "股票数量"],
                "值": [
                    factor_name,
                    end_date,
                    pool_name,
                    f">={ipo_date}天",
                    f">={min_amount / 10000}万",
                    len(factor_df)
                ]
            })
            params_df.to_excel(writer, sheet_name='计算参数', index=False)

        logger.info(f"✅ 因子计算完成，结果已保存至: {output_file}")
        return factor_df

    except Exception as e:
        logger.exception(f"计算因子 {factor_name} 时发生错误: {e}")
        return pd.DataFrame()


def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="单因子计算工具")
    parser.add_argument("factor_name", help="因子名称")
    parser.add_argument("--date", help="计算日期，格式为YYYY-MM-DD，默认为最近交易日")
    parser.add_argument("--pool", default="hs300", help="股票池名称，如'hs300', 'zz500', 'zz1000', 'all'等")
    parser.add_argument("--ipo_days", type=int, default=365, help="上市天数限制")
    parser.add_argument("--min_amount", type=float, default=5000000, help="最小成交额限制")

    args = parser.parse_args()

    calculate_single_factor(
        factor_name=args.factor_name,
        start_date=(datetime.datetime.strptime(args.date, "%Y-%m-%d") - datetime.timedelta(days=365)).strftime(
            "%Y-%m-%d"),
        end_date=args.date,
        pool_name=args.pool,
        ipo_days=args.ipo_days,
        min_amount=args.min_amount
    )


if __name__ == "__main__":
    # main()

    factor_name = "alpha_37"
    start_date = "2024-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    pool_name = "no_st"
    ipo_days = 365
    min_amount = 5000000

    calculate_single_factor(
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        pool_name=pool_name,
        ipo_days=ipo_days,
        min_amount=min_amount,
    )
