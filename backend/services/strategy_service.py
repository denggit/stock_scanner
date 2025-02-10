#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 5:59 PM
@File       : strategy_service.py
@Description: 
"""
import datetime
from typing import Any, Dict, List

from backend.data.stock_data_fetcher import StockDataFetcher
from backend.strategies.breakout import BreakoutStrategy
from backend.strategies.double_up import DoubleUpStrategy
from backend.strategies.explosive_stock import ExplosiveStockStrategy
from backend.strategies.hs_bottom import HSBottom
from backend.strategies.long_term_uptrend import LongTermUpTrendStrategy
from backend.strategies.ma_pullback import MAPullbackStrategy
from backend.strategies.swing_trading import SwingTradingStrategy
from backend.utils.api_response import convert_to_python_types
from backend.utils.logger import setup_logger

logger = setup_logger("strategy_service", set_root_logger=True)


class StrategyService:
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.strategies = {
            "均线回踩策略": MAPullbackStrategy,
            "突破策略": BreakoutStrategy,
            "波段交易策略": SwingTradingStrategy,
            "扫描翻倍股": DoubleUpStrategy,
            "长期上涨策略": LongTermUpTrendStrategy,
            "头肩底形态策略": HSBottom,
            "爆发式选股策略": ExplosiveStockStrategy
        }
        self.pool_trans = {
            "全量股票": "full",
            "非ST股票": "no_st",
            "上证50": "sz50",
            "沪深300": "hs300",
            "中证500": "zz500"
        }

    async def scan_stocks(self, strategy: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用策略扫描股票"""
        try:
            logger.info(f"Scanning stocks with strategy: {strategy}, params: {params}")

            if strategy not in self.strategies:
                raise ValueError(f"Strategy {strategy} not found")

            # 转化stock_pool
            stock_pool = params.get('stock_pool')
            if stock_pool in self.pool_trans.keys():
                stock_pool = self.pool_trans.get(stock_pool)
                params["stock_pool"] = stock_pool
            # 获取股票列表
            if stock_pool:
                stocks = self.data_fetcher.get_stock_list(pool_name=stock_pool)
                results = []

            # 获取足量数据
            ma_period = params.get("ma_period", 20)  # 假设设置了均线 
            max_ma_period = max(params.get("ma_periods", [20]))
            long_ma_period = params.get("long_ma_period", 20)  # 假设设置了长期均线
            lookback_period = params.get("lookback_period", 20)
            period = params.get("period", 20)  # 假设直接设置了获取数据的周期
            period = max(ma_period, long_ma_period, period, max_ma_period, lookback_period)

            years = period // 250
            months = period % 250 // 20
            days = period % 250 % 20
            take_period = 0
            take_period += (years + 1) * 365 if years > 0 else 0
            take_period += (months + 1) * 31 if months > 0 else 0
            take_period += days * 2 if days > 0 else 0
            take_period += 30       # 预留一个月

            end_date = params.get("end_date", datetime.date.today().strftime("%Y-%m-%d"))
            start_date = params.get(
                "start_date",
                (datetime.datetime.strptime(end_date, "%Y-%m-%d") -
                 datetime.timedelta(days=take_period)).strftime("%Y-%m-%d")
            )

            # 对每只股票进行策略扫描
            for stock in stocks.itertuples():
                if "st" in stock.name.lower():
                    # 跳过ST股票
                    continue

                # 获取股票数据
                stock_data = self.data_fetcher.fetch_stock_data(code=stock.code, start_date=start_date,
                                                                end_date=end_date)
                strategy_instance = self.strategies[strategy]()
                strategy_instance.set_parameters(params)
                # 生成信号
                signals = strategy_instance.generate_signal(stock_data)

                if not signals.empty:
                    # 信号为0，跳过
                    if "signal" in signals and signals.signal == 0:
                        continue

                    result = {
                        'code': stock.code,
                        'name': stock.name,
                    }
                    for column, value in signals.items():
                        result[column] = value

                    results.append(result)

            logger.info(f"Found {len(results)} stocks with strategy: {strategy}")
            results = convert_to_python_types(results)
            return results
        except Exception as e:
            logger.exception(f"Error scanning stocks: {e}")
            raise Exception(f"Error scanning stocks: {e}")

    async def list_strategies(self) -> List[Dict[str, Any]]:
        """列出所有策略"""
        try:
            return [{"name": strategy_name, "description": strategy_instance().get_description()} for
                    strategy_name, strategy_instance in self.strategies.items()]
        except Exception as e:
            logger.error(f"Error listing strategies: {e}", exc_info=True)
            raise Exception(f"Error listing strategies: {e}")
