#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:59 PM
@File       : backtest_service.py
@Description: 回测服务
"""

from typing import Dict, Any, List

import pandas as pd

from backend.business.data.data_fetcher import StockDataFetcher
from backend.utils.logger import setup_logger

logger = setup_logger("backtest_service")


class BacktestService:
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        # 注册所有可用的策略
        self.backtest = {
            # 可以添加更多策略
        }

    async def run_backtest(self,
                           strategy: str,
                           start_date: str,
                           end_date: str,
                           backtest_init_params: Dict[str, Any],
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """运行回测
        
        Args:
            strategy: 策略名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            backtest_init_params: 回测初始化参数
            params: 策略参数
        """
        try:
            if strategy not in self.backtest:
                raise ValueError(f"Strategy '{strategy}' not found")

            # 获取回测方法
            backtest_func = self.backtest.get(strategy)

            # 获取股票池
            stock_pool = backtest_init_params.get("stock_pool")
            stock_list = await self._get_stock_pool(stock_pool)
            if not stock_list:
                raise ValueError(f"No stocks found in pool: {stock_pool}")

            # 获取所有股票的历史数据
            extended_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=100)
            all_stock_data = {}
            for stock_code in stock_list:
                try:
                    data = await self._get_backtest_data(
                        stock_code,
                        extended_start_date.strftime("%Y-%m-%d"),
                        end_date
                    )
                    if not data.empty:
                        all_stock_data[stock_code] = data
                except Exception as e:
                    logger.warning(f"获取股票 {stock_code} 数据失败: {e}")
                    continue

            if not all_stock_data:
                raise ValueError("No valid stock data available for backtest_event")

            # 运行回测
            results = backtest_func(
                data=all_stock_data,
                backtest_init_params=backtest_init_params,
                **params
            )

            # 生成回测ID并保存结果
            backtest_id = self._generate_backtest_id(strategy, stock_pool, start_date, end_date)
            await self._save_results(backtest_id, results)

            # 只返回实际回测期间的结果
            results = self._trim_results_to_period(results, start_date, end_date)

            return {
                "backtest_id": backtest_id,
                "summary": results["summary"],
                "stock_pool": stock_pool
            }

        except Exception as e:
            logger.exception(f"回测执行失败: {e}")
            raise

    async def _get_stock_pool(self, pool_name: str) -> List[str]:
        """获取股票池列表"""
        try:
            if pool_name == "全量股票":
                return self.data_fetcher.get_stock_list("full")['code'].tolist()

            elif pool_name == "非ST股票":
                return self.data_fetcher.get_stock_list("no_st")['code'].tolist()

            elif pool_name == "上证50":
                return self.data_fetcher.get_stock_list("sz50")['code'].tolist()

            elif pool_name == "沪深300":
                return self.data_fetcher.get_stock_list("hs300")['code'].tolist()

            elif pool_name == "中证500":
                return self.data_fetcher.get_stock_list("zz500")['code'].tolist()

            # elif pool_name == "创业板指":
            #     index_code = "399006.SZ"
            #     return self.data_fetcher.get_index_stocks(index_code)

            else:
                raise ValueError(f"Unknown stock pool: {pool_name}")

        except Exception as e:
            logger.exception(f"获取股票池失败: {e}")
            raise

    async def _get_backtest_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取单个股票的回测数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            pd.DataFrame: 包含股票历史数据的DataFrame
        """
        try:
            # 使用data_fetcher获取股票数据
            stock_data = await self.data_fetcher.fetch_stock_data(
                code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            return stock_data
        except Exception as e:
            logger.warning(f"获取股票 {stock_code} 的历史数据失败: {e}")
            return pd.DataFrame()

    def get_backtest_results(self, backtest_id: str):
        """获取回测结果"""
        pass
