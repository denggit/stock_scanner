#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道数据库适配器

提供与原缓存适配器相同的外部接口，但底层改为使用数据库：
- 从数据库读取通道数据
- 缺失则调用 AscendingChannelRegression 计算并写回数据库
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.business.backtest.database.channel_db.channel_db_manager import ChannelDBManager
from backend.business.factor.core.engine.library.channel_analysis.rising_channel import (
    AscendingChannelRegression,
)
from backend.utils.logger import setup_logger


class ChannelDBAdapter:
    """数据库通道适配器

    对外方法保持与原 ChannelCacheAdapter 一致，便于无缝替换。
    """

    def __init__(self, min_window_size: int = 60):
        self.logger = setup_logger(__name__)
        self.db = ChannelDBManager()
        self.min_window_size = min_window_size
        # 分析器实例缓存：按参数键缓存
        self._analyzer_cache: Dict[str, AscendingChannelRegression] = {}

    # ---------- 对外接口 ----------
    def get_channel_history_data(
        self,
        stock_data_dict: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        start_date: str | None = None,
        end_date: str | None = None,
        min_window_size: int | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取通道历史数据：优先从数据库读取；缺失则计算并写回。
        返回结构: {stock_code: DataFrame}
        """
        window = min_window_size or self.min_window_size
        stock_codes = list(stock_data_dict.keys())

        # 1) 读取数据库已有数据
        existing: Dict[str, List[Dict[str, Any]]] = self.db.fetch_records(
            params, stock_codes=stock_codes, start_date=start_date, end_date=end_date
        )

        # 2) 识别每只股票缺失的日期范围
        missing_by_stock: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        req_start = pd.to_datetime(start_date) if start_date else None
        req_end = pd.to_datetime(end_date) if end_date else None

        for code in stock_codes:
            stock_df = stock_data_dict.get(code)
            if stock_df is None or stock_df.empty:
                continue
            stock_df = stock_df.copy()
            stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])

            # 请求范围
            s_date = req_start or stock_df['trade_date'].min()
            e_date = req_end or stock_df['trade_date'].max()

            # 已有日期集合
            have = set()
            if code in existing:
                for r in existing[code]:
                    try:
                        have.add(pd.to_datetime(r.get('trade_date')))
                    except Exception:
                        pass

            # 找到请求区间内缺失的连续段
            date_series = stock_df[(stock_df['trade_date'] >= s_date) & (stock_df['trade_date'] <= e_date)]['trade_date']
            sorted_dates = list(pd.to_datetime(date_series).sort_values())
            segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            seg_start: Optional[pd.Timestamp] = None
            prev_date: Optional[pd.Timestamp] = None
            for d in sorted_dates:
                if d not in have:
                    if seg_start is None:
                        seg_start = d
                    # 如果和前一天不连续，结束上一段
                    if prev_date is not None and (d - prev_date).days > 1:
                        segments.append((seg_start, prev_date))
                        seg_start = d
                else:
                    if seg_start is not None:
                        segments.append((seg_start, prev_date or seg_start))
                        seg_start = None
                prev_date = d
            if seg_start is not None:
                segments.append((seg_start, prev_date or seg_start))

            if segments:
                missing_by_stock[code] = segments

        # 3) 计算缺失并写回DB
        if missing_by_stock:
            analyzer = self._get_analyzer(params)
            to_write: Dict[str, List[Dict[str, Any]]] = {}
            for code, ranges in missing_by_stock.items():
                try:
                    sdf = stock_data_dict[code].copy()
                    sdf['trade_date'] = pd.to_datetime(sdf['trade_date'])

                    # 核心修正：不再截取 sub_df，直接将全部可用历史数据传递给计算函数
                    # fit_channel_history_optimized 内部会处理窗口问题
                    ch_df = analyzer.fit_channel_history_optimized(sdf, window)

                    if ch_df.empty:
                        continue

                    ch_df['trade_date'] = pd.to_datetime(ch_df['trade_date'])
                    
                    # 在计算完成后，再裁剪到我们需要的日期范围
                    computed_records: List[Dict[str, Any]] = []
                    for (seg_start, seg_end) in ranges:
                        mask = (ch_df['trade_date'] >= seg_start) & (ch_df['trade_date'] <= seg_end)
                        seg_df = ch_df.loc[mask]
                        
                        # 确保beta值非空
                        seg_df = seg_df.dropna(subset=['beta'])
                        if not seg_df.empty:
                            computed_records.extend(seg_df.to_dict('records'))

                    if computed_records:
                        # 直接调用 ChannelDBManager 的静态方法来清洗数据
                        to_write[code] = [ChannelDBManager._normalize_record(r) for r in computed_records]
                except Exception as e:
                    self.logger.debug(f"跳过股票 {code} 的通道计算，原因: {e}")
                    continue
            
            if to_write:
                self.db.upsert_records(params, to_write)
                # 合并写回结果到 existing
                for code, recs in to_write.items():
                    existing.setdefault(code, []).extend(recs)

        # 4) 构造成 DataFrame 返回
        result: Dict[str, pd.DataFrame] = {}
        for code, recs in existing.items():
            if not recs:
                continue
            df = pd.DataFrame(recs)
            if not df.empty and 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
            result[code] = df
        return result

    def get_single_channel_data(
        self,
        stock_code: str,
        stock_df: pd.DataFrame,
        params: Dict[str, Any],
        target_date: str | None = None,
    ) -> Optional[pd.DataFrame]:
        """单只股票查询，缺失则计算并写回数据库。"""
        # 先读库
        existing = self.db.fetch_records(params, stock_codes=[stock_code], start_date=target_date, end_date=target_date)
        if existing and stock_code in existing and existing[stock_code]:
            df = pd.DataFrame(existing[stock_code])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df

        # 计算
        stock_df = stock_df.copy()
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
        analyzer = self._get_analyzer(params)
        ch_df = analyzer.fit_channel_history_optimized(stock_df, self.min_window_size)
        if ch_df.empty:
            return None
        ch_df['trade_date'] = pd.to_datetime(ch_df['trade_date'])
        # 写回
        recs = ch_df.to_dict('records')
        # 直接调用 ChannelDBManager 的静态方法来清洗数据
        self.db.upsert_records(params, {stock_code: [ChannelDBManager._normalize_record(r) for r in recs]})
        return ch_df

    def preload_cache_for_backtest(
        self,
        stock_data_dict: Dict[str, pd.DataFrame],
        params_list: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> bool:
        """为回测预加载（确保DB中有数据）。"""
        try:
            for i, p in enumerate(params_list, 1):
                self.logger.info(f"预加载参数组合 {i}/{len(params_list)}")
                _ = self.get_channel_history_data(stock_data_dict, p, start_date, end_date, self.min_window_size)
            return True
        except Exception as e:
            self.logger.error(f"预加载失败: {e}")
            return False

    # 兼容接口：返回元信息
    def get_cache_statistics(self) -> Dict[str, Any]:
        meta = self.db.meta_manager.get_meta_info()
        total = len(meta.get('params_map', {}))
        return {"param_sets": total}

    # ---------- 内部 ----------
    def _get_analyzer(self, params: Dict[str, Any]) -> AscendingChannelRegression:
        key = str(sorted(params.items()))
        inst = self._analyzer_cache.get(key)
        if inst is None:
            inst = AscendingChannelRegression(**params)
            self._analyzer_cache[key] = inst
        return inst


