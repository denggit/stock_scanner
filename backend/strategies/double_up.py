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
            "allowed_drawdown": 0.05,  # 最大回撤
            "target_return": 100.00,  # 目标收益率(%)
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        if not self.validate_data(data):
            return pd.DataFrame([{'signal': 0}])

        # 1. 数据预处理 - 确保所有数值列都是float类型
        df = data.copy()

        # 2. 获取参数
        allowed_drawdown = self._params['allowed_drawdown']
        target_return = float(self._params['target_return']) / 100  # 参数单位为百分比(%)
        double_period = int(self._params['double_period'])

        # 3. 找出所有翻倍记录
        match_records = []
        i = 0
        while i < len(df) - 1:
            # 找到价格开始上涨的点作为起始点
            start_idx = i
            while start_idx < len(df) - 1:
                if df['close'].iloc[start_idx + 1] > df['close'].iloc[start_idx]:
                    break
                start_idx += 1

            start_price = df['close'].iloc[start_idx]
            start_date = df['trade_date'].iloc[start_idx]
            max_price = start_price
            max_idx = start_idx
            max_date = start_date
            found_valid_period = False

            # 在double_period区间内寻找符合条件的区域
            for j in range(start_idx + 1, min(start_idx + double_period, len(df))):
                current_price = df['close'].iloc[j]

                # 更新最高价和对应索引及日期
                if current_price > max_price:
                    max_price = current_price
                    max_idx = j
                    max_date = df['trade_date'].iloc[j]

                # 计算当前回撤
                if max_price > 0:  # 防止除零错误
                    current_drawdown = abs(max_price - current_price) / max_price

                    # 如果出现超过最大回撤的情况
                    if current_drawdown > allowed_drawdown:
                        # 记录之前的有效区间（如果存在）
                        current_return = (max_price - start_price) / start_price
                        if current_return >= target_return:
                            match_records.append({
                                'start_date': start_date,
                                'end_date': max_date,
                                'max_return': current_return,
                                'start_price': start_price,
                                'end_price': max_price
                            })
                        # 从回撤点后一天开始新的扫描
                        i = j
                        found_valid_period = True
                        break

            # 如果完整扫描了double_period且没有触发最大回撤
            if not found_valid_period:
                current_return = (max_price - start_price) / start_price
                if current_return >= target_return:
                    match_records.append({
                        'start_date': start_date,
                        'end_date': max_date,
                        'max_return': current_return,
                        'start_price': start_price,
                        'end_price': max_price
                    })
                    i = max_idx + 1  # 从最高点后一天开始新的扫描
                else:
                    # 如果当前起始点不是区间内的最低点，找到最低点作为新的起始点
                    min_idx = df['close'].iloc[start_idx:min(start_idx + double_period, len(df))].idxmin()
                    if min_idx > start_idx:
                        i = min_idx  # 使用最低点作为新的起始点
                    else:
                        # 如果当前点就是最低点，向后移动一天
                        i += 1

        # 4. 生成结果DataFrame
        if match_records:
            # 按开始日期排序
            match_records.sort(key=lambda x: df.index[df['trade_date'] == x['start_date']].values[0])

            # 合并相邻的记录
            merged_records = []
            current_record = match_records[0]

            for i in range(1, len(match_records)):
                # 获取当前记录和下一条记录在原始数据中的索引
                current_end_idx = df.index[df['trade_date'] == current_record['end_date']].values[0]
                next_start_idx = df.index[df['trade_date'] == match_records[i]['start_date']].values[0]

                # 如果两条记录的索引相差1，则合并
                if next_start_idx - current_end_idx == 1:
                    # 更新结束日期和最大收益率
                    current_record['end_date'] = match_records[i]['end_date']
                    current_record['end_price'] = match_records[i]['end_price']
                    current_record['max_return'] = (current_record['end_price'] - current_record['start_price']) / \
                                                   current_record['start_price']
                else:
                    merged_records.append(current_record)
                    current_record = match_records[i]

            # 添加最后一条记录
            merged_records.append(current_record)

            # 转换为DataFrame格式
            results = []
            for record in merged_records:
                # 获取区间数据
                start_idx = df.index[df['trade_date'] == record['start_date']].values[0]
                end_idx = df.index[df['trade_date'] == record['end_date']].values[0]
                period_data = df.iloc[start_idx:end_idx + 1]

                # 找到区间内最低价对应的日期
                min_price_idx = period_data['close'].idxmin()
                real_start_date = df['trade_date'].loc[min_price_idx]
                real_start_price = df['close'].loc[min_price_idx]

                # 从最低点开始计算最大回撤
                period_data = df.iloc[min_price_idx:end_idx + 1]
                rolling_max = period_data['close'].expanding().max()
                drawdowns = (rolling_max - period_data['close']) / rolling_max
                max_drawdown = drawdowns.max().round(4)

                # 重新计算最大收益率
                max_return = (record['end_price'] - real_start_price) / real_start_price

                results.append({
                    'signal': 1,
                    'start_date': real_start_date.strftime("%Y-%m-%d"),
                    'end_date': record['end_date'].strftime("%Y-%m-%d"),
                    'max_return': max_return,
                    'start_price': real_start_price,
                    'end_price': record['end_price'],
                    'max_drawdown': max_drawdown
                })
            return pd.DataFrame(results)
        else:
            return pd.DataFrame([{
                'signal': 0,
                'start_date': None,
                'end_date': None,
                'max_return': np.nan,
                'start_price': np.nan,
                'end_price': np.nan,
                'max_drawdown': np.nan
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
