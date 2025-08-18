#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道数据库适配器

提供与原缓存适配器相同的外部接口，但底层改为使用数据库：
- 从数据库读取通道数据
- 缺失则调用 AscendingChannelRegression 计算并写回数据库
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.business.backtest.database.channel_db.channel_db_manager import ChannelDBManager, ParameterHasher
from backend.business.factor.core.engine.library.channel_analysis.rising_channel import (
    AscendingChannelRegression,
)
from backend.utils.logger import setup_logger


class ChannelDBAdapter:
    """数据库通道适配器

    对外方法保持与原 ChannelCacheAdapter 一致，便于无缝替换。
    """

    def __init__(self, min_window_size: int = 60, db_manager: ChannelDBManager | None = None):
        self.logger = setup_logger("backtest")
        # 允许注入自定义DB管理器，便于测试或替换存储实现
        self.db = db_manager if db_manager is not None else ChannelDBManager()
        self.min_window_size = min_window_size
        # 分析器实例缓存：按参数键缓存
        self._analyzer_cache: Dict[str, AscendingChannelRegression] = {}

    # ---------- 对外接口（逐日按需） ----------
    def get_channels_for_date(
            self,
            stock_data_dict: Dict[str, pd.DataFrame],
            params: Dict[str, Any],
            target_date: str,
            min_window_size: int | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        逐日按需获取通道数据：
        - 仅查询 target_date 当天的记录
        - 缺失则使用 fit_channel 计算当日状态，并仅写入该日一条记录

        Returns: {stock_code: 单行DataFrame}
        """
        t0 = time.perf_counter()
        window = int(min_window_size or self.min_window_size)
        stock_codes = list(stock_data_dict.keys())
        params_hash = ParameterHasher.generate_hash(params)
        try:
            table_name = self.db.meta_manager.get_or_create_table_meta(params).table_name
        except Exception:
            table_name = f"channel_rcn_{params_hash}"

        self.logger.info(
            f"[CHD-Daily] 获取当日通道 | date={target_date} | stocks={len(stock_codes)} | window={window} | table={table_name}"
        )

        # 1) 查询当日已有记录
        existing: Dict[str, List[Dict[str, Any]]] = self.db.fetch_records(
            params, stock_codes=stock_codes, start_date=target_date, end_date=target_date
        )

        have_codes = {code for code, recs in (existing or {}).items() if recs}
        missing_codes = [c for c in stock_codes if c not in have_codes]

        # 2) 对缺失股票进行当日计算并写回
        if missing_codes:
            analyzer = self._get_analyzer(params)
            to_write: Dict[str, List[Dict[str, Any]]] = {}
            t_calc = time.perf_counter()
            td = pd.to_datetime(target_date)
            for code in missing_codes:
                try:
                    sdf = stock_data_dict.get(code)
                    if sdf is None or sdf.empty:
                        continue

                    # 避免整体复制，按需视图切片并仅使用必要列，减少内存占用
                    # 假设上游已确保 trade_date 为 datetime 类型
                    sdt = sdf
                    try:
                        from pandas.api import types as ptypes
                        if not ptypes.is_datetime64_any_dtype(sdt['trade_date']):
                            # 仅当确实不是datetime时才转换，避免重复分配
                            sdt = sdt.assign(trade_date=pd.to_datetime(sdt['trade_date']))
                    except Exception:
                        sdt = sdt.assign(trade_date=pd.to_datetime(sdt['trade_date']))

                    mask = sdt['trade_date'] <= td
                    if not mask.any():
                        continue
                    # 仅保留必需列视图
                    cols = [c for c in ['trade_date', 'open', 'high', 'low', 'close', 'volume'] if c in sdt.columns]
                    sdt = sdt.loc[mask, cols]

                    if len(sdt) < window:
                        continue
                    # 若最后一条并非目标日，跳过（该股当日无数据）
                    last_row = sdt.iloc[-1]
                    if pd.to_datetime(last_row['trade_date']).date() != td.date():
                        continue

                    state = analyzer.fit_channel(sdt)
                    if state is None:
                        continue
                    # 构造成记录并写回
                    close_price = float(last_row['close'])
                    rec = self._state_to_record(state, td, close_price)
                    to_write[code] = [ChannelDBManager._normalize_record(rec)]
                except Exception as e:
                    self.logger.debug(f"跳过股票 {code} 的当日通道计算，原因: {e}")
                    continue

            if to_write:
                ok = self.db.upsert_records(params, to_write)
                self.logger.info(
                    f"[CHD-Daily] 写回数据库 {'成功' if ok else '失败'} | 表={table_name} | 写回记录数={sum(len(v) for v in to_write.values())} | 计算耗时={(time.perf_counter() - t_calc) * 1000:.0f}ms"
                )
                # 合并写回结果
                for code, recs in to_write.items():
                    existing.setdefault(code, []).extend(recs)

        # 3) 返回 {code: 单行DataFrame}
        result: Dict[str, pd.DataFrame] = {}
        for code, recs in (existing or {}).items():
            if not recs:
                continue
            df = pd.DataFrame(recs)
            if not df.empty and 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                # 仅保留目标日
                df = df[df['trade_date'] == pd.to_datetime(target_date)]
                df = df.sort_values('trade_date').reset_index(drop=True)
            if not df.empty:
                result[code] = df

        self.logger.info(
            f"[CHD-Daily] 完成 | 返回股票={len(result)} | 总耗时={(time.perf_counter() - t0) * 1000:.0f}ms"
        )
        return result

    def get_single_channel_data(
            self,
            stock_code: str,
            stock_df: pd.DataFrame,
            params: Dict[str, Any],
            target_date: str | None = None,
    ) -> Optional[pd.DataFrame]:
        """单只股票当日查询：先读库；缺失则当日计算并仅写入该日记录。"""
        # 先读库
        if target_date:
            existing = self.db.fetch_records(
                params, stock_codes=[stock_code], start_date=target_date, end_date=target_date
            )
            if existing and stock_code in existing and existing[stock_code]:
                df = pd.DataFrame(existing[stock_code])
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df

        # 计算（仅当日）
        td = pd.to_datetime(target_date) if target_date else pd.to_datetime(stock_df['trade_date'].max())
        sdf = stock_df.copy()
        sdf['trade_date'] = pd.to_datetime(sdf['trade_date'])
        sdf = sdf[sdf['trade_date'] <= td]
        if len(sdf) < self.min_window_size or sdf.empty or sdf.iloc[-1]['trade_date'].date() != td.date():
            return None

        analyzer = self._get_analyzer(params)
        state = analyzer.fit_channel(sdf)
        if state is None:
            return None

        rec = self._state_to_record(state, td, float(sdf.iloc[-1]['close']))
        ok = self.db.upsert_records(params, {stock_code: [ChannelDBManager._normalize_record(rec)]})
        if not ok:
            return None
        df = pd.DataFrame([rec])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df

    # 已移除预加载与整段历史计算接口，遵循逐日按需策略

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

    def _state_to_record(self, state: "AscendingChannelRegression".state.__class__ | Any,
                         current_date: pd.Timestamp, current_close: float) -> Dict[str, Any]:
        """
        将通道状态转换为数据库记录格式（单日）。

        Args:
            state: ChannelState 对象
            current_date: 当日日期
            current_close: 当日收盘价

        Returns:
            记录字典，包含窄表所需字段。
        """
        # state.to_dict() 已包含大部分字段
        base = {}
        try:
            base = state.to_dict() if hasattr(state, 'to_dict') else {}
        except Exception:
            base = {}

        record: Dict[str, Any] = {
            'trade_date': pd.to_datetime(current_date).strftime('%Y-%m-%d'),
            'close': current_close,
            'beta': base.get('beta'),
            'sigma': base.get('sigma'),
            'r2': base.get('r2'),
            'mid_today': base.get('mid_today'),
            'upper_today': base.get('upper_today'),
            'lower_today': base.get('lower_today'),
            'mid_tomorrow': base.get('mid_tomorrow'),
            'upper_tomorrow': base.get('upper_tomorrow'),
            'lower_tomorrow': base.get('lower_tomorrow'),
            'channel_status': base.get('channel_status'),
            'anchor_date': base.get('anchor_date'),
            'anchor_price': base.get('anchor_price'),
            'break_cnt_up': base.get('break_cnt_up'),
            'break_cnt_down': base.get('break_cnt_down'),
            'reanchor_fail_up': base.get('reanchor_fail_up'),
            'reanchor_fail_down': base.get('reanchor_fail_down'),
            'cumulative_gain': base.get('cumulative_gain'),
            'window_size': base.get('window_size'),
            'days_since_anchor': base.get('days_since_anchor'),
            'break_reason': base.get('break_reason'),
            'width_pct': base.get('width_pct'),
            'slope_deg': base.get('slope_deg'),
            'volatility': base.get('volatility'),
        }
        return record
