#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/3/2025 8:33 PM
@File       : double_up.py
@Description: 
"""
import logging

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy


class DoubleUpStrategy(BaseStrategy):
    """扫描翻倍股票"""

    def __init__(self):
        super().__init__(name="扫描翻倍股", description="扫描过去周期内曾经翻倍过的股票")
        self.params = {
            "double_period": 20,  # 观察期（交易日）
            "max_drawdown": 0.05,  # 最大回撤
            "times": 2.00,  # 翻倍倍数
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号，返回所有不重叠的翻倍记录"""
        if not self.validate_data(data):
            return pd.DataFrame([{'signal': 0}])

        # 1. 数据预处理
        df = data.copy()
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)

        # 2. 计算在观察期内的最大涨幅
        max_drawndown = self._params['max_drawdown']
        times = float(self._params['times'])
        double_period = int(self._params['double_period'])

        # 3. 找出所有翻倍记录
        double_records = []

        start_idx = 0
        while start_idx < len(df) - 1:
            start_price = df['close'].iloc[start_idx]
            max_return = 1.0
            max_return_idx = start_idx
            found_peak = False

            # 确定搜索范围
            end_search_idx = len(df)
            if double_period > 0:
                end_search_idx = min(start_idx + double_period, len(df))

            # 从起始点向后查找，直到找到最大收益点
            for end_idx in range(start_idx + 1, end_search_idx):
                end_price = df['close'].iloc[end_idx]
                return_rate = end_price / start_price

                # 更新最大收益
                if return_rate > max_return:
                    max_return = return_rate
                    max_return_idx = end_idx
                # 如果收益开始下降，且已经达到翻倍要求，则记录该区间
                elif return_rate < max_return * (1 - max_drawndown) and max_return >= times:  # 回撤容忍度
                    found_peak = True
                    break

            # 如果找到了符合条件的翻倍记录
            if max_return >= times:
                start_date = df.index[start_idx]
                end_date = df.index[max_return_idx]

                # 检查是否与已有记录重叠
                overlap = False
                update_existing = False
                for record in double_records:
                    # 判断日期区间是否有交集
                    if not (end_date < record['start_date'] or start_date > record['end_date']):
                        # 如果新记录收益率更高，则更新原记录
                        if max_return > record['max_return']:
                            record.update({
                                'start_date': start_date,
                                'end_date': end_date,
                                'max_return': max_return,
                                'start_price': start_price,
                                'end_price': df['close'].iloc[max_return_idx]
                            })
                            update_existing = True
                        overlap = True
                        break

                # 如果没有重叠或者更新了现有记录，添加新记录
                if not overlap:
                    double_records.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'max_return': max_return,
                        'start_price': start_price,
                        'end_price': df['close'].iloc[max_return_idx]
                    })

            # 更新起始索引
            if found_peak:
                # 如果找到了回撤点，从最高点后开始下一轮搜索
                start_idx = max_return_idx + 1
            else:
                # 如果没有找到回撤点，继续往后搜索
                start_idx += 1

        # 4. 生成结果DataFrame
        if double_records:
            results = []
            for record in double_records:
                results.append({
                    'signal': 1,
                    'start_date': data[data.index == record['start_date']].trade_date.iloc[0].strftime("%Y-%m-%d"),
                    'end_date': data[data.index == record['end_date']].trade_date.iloc[0].strftime("%Y-%m-%d"),
                    'max_return': record['max_return'],
                    'start_price': record['start_price'],
                    'end_price': record['end_price']
                })
            return pd.DataFrame(results)
        else:
            return pd.DataFrame([{
                'signal': 0,
                'start_date': None,
                'end_date': None,
                'max_return': np.nan,
                'start_price': np.nan,
                'end_price': np.nan
            }])

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据是否符合策略要求"""
        try:
            if len(data) < 250:  # 至少需要250个交易日的数据
                logging.warning(f"股票{data['code'].iloc[-1]}数据不足250天，跳过该股票")
                return False

            # 检查必要的列是否存在
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_date']
            if not all(col in data.columns for col in required_columns):
                logging.warning(f"股票{data['code'].iloc[-1]}缺少必要的列，跳过该股票")
                return False

            return True
        except Exception as e:
            logging.exception(f"验证数据时发生错误: {e}")
            return False
