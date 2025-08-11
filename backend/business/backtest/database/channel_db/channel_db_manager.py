#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道数据库管理器

职责分离：
- 参数→表名映射的meta管理（JSON文件）
- 按参数组合创建并维护MySQL表（宽表：每个股票代码一列JSON）
- 读取/写入指定参数、股票、日期范围的通道数据

设计说明：
- 表结构：channel_rc_{hash}
  - trade_date DATE PRIMARY KEY
  - <sanitized_code> JSON
- 列命名：将股票代码中的非法字符转换为下划线（如 sz.000001 → sz_000001）
- 元数据：backend/business/backtest/database/channel_db/channel_meta.json

注意：
- 使用 MySQL JSON 类型存储单元格内的通道记录(dict)
- 对于不存在的股票列，会自动ALTER TABLE新增JSON列
- 写入采用 UPSERT：INSERT ... ON DUPLICATE KEY UPDATE
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pymysql as mysql

from backend.business.data.configs.db_conn import config as global_db_config
from backend.utils.logger import setup_logger


class ParameterHasher:
    """参数哈希生成器

    - 生成稳定的短哈希（前16位MD5）
    - 生成可读参数键（用于诊断与meta展示）
    """

    @staticmethod
    def generate_hash(params: Dict[str, Any]) -> str:
        import hashlib
        if params is None:
            return ""
        sorted_params = dict(sorted(params.items()))
        param_str = json.dumps(sorted_params, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(param_str.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def get_readable_params_key(params: Dict[str, Any]) -> str:
        parts: List[str] = []
        for k, v in sorted(params.items()):
            if isinstance(v, float):
                parts.append(f"{k}_{v:.3f}")
            else:
                parts.append(f"{k}_{v}")
        return "_".join(parts)


@dataclass
class ChannelTableMeta:
    params_hash: str
    table_name: str
    params: Dict[str, Any]
    readable_key: str
    created_at: str


class ChannelDBMetaManager:
    """参数到表名的映射管理，持久化到JSON meta文件"""

    def __init__(self, meta_file: Optional[str] = None):
        self.logger = setup_logger(__name__)
        default_path = Path(__file__).parent / "channel_meta.json"
        self.meta_path = Path(meta_file) if meta_file else default_path
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {"params_map": {}}
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "params_map" not in data:
                return {"params_map": {}}
            return data
        except Exception as e:
            self.logger.warning(f"加载meta文件失败，将重建: {e}")
            return {"params_map": {}}

    def _save(self) -> None:
        tmp_path = self.meta_path.with_suffix(".tmp.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, self.meta_path)

    def get_or_create_table_meta(self, params: Dict[str, Any]) -> ChannelTableMeta:
        ph = ParameterHasher.generate_hash(params)
        params_map: Dict[str, Any] = self._meta.setdefault("params_map", {})
        if ph in params_map:
            m = params_map[ph]
            return ChannelTableMeta(
                params_hash=ph,
                table_name=m["table"],
                params=m.get("params", {}),
                readable_key=m.get("readable_key", ""),
                created_at=m.get("created_at", datetime.now().isoformat()),
            )

        table_name = f"channel_rc_{ph}"
        meta = ChannelTableMeta(
            params_hash=ph,
            table_name=table_name,
            params=params,
            readable_key=ParameterHasher.get_readable_params_key(params),
            created_at=datetime.now().isoformat(),
        )
        params_map[ph] = {
            "table": table_name,
            "params": params,
            "readable_key": meta.readable_key,
            "created_at": meta.created_at,
        }
        self._save()
        return meta

    def get_meta_info(self) -> Dict[str, Any]:
        return self._meta


class ChannelDBManager:
    """上升通道数据库操作封装

    - 创建参数表（若不存在）
    - 确保列存在（股票代码列）
    - 批量读取/写入 JSON 单元格
    """

    def __init__(self, meta_file: Optional[str] = None, schema_name: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.meta_manager = ChannelDBMetaManager(meta_file)
        self.schema_name = schema_name or os.getenv('MYSQL_CHANNEL_DB', 'stock_channel')
        self.conn = self._create_channel_connection()
        self._ensure_database()
        try:
            self.conn.select_db(self.schema_name)
        except Exception:
            self.conn.close()
            self.conn = self._create_channel_connection(select_db=True)

    # ---------- 表与列管理 ----------
    def _sanitize_code_to_column(self, code: str) -> str:
        col = code.replace(".", "_").replace("-", "_")
        # 防止列名以数字开头
        if col and col[0].isdigit():
            col = f"c_{col}"
        return col

    def _ensure_table(self, table_name: str) -> None:
        # 创建仅含trade_date的空表
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name}(
            trade_date DATE PRIMARY KEY
        )
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
        self.conn.commit()

    def _get_existing_columns(self, table_name: str) -> List[str]:
        query = (
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s"
        )
        with self.conn.cursor() as cursor:
            cursor.execute(query, (self.schema_name, table_name))
            rows = cursor.fetchall()
        return [r[0] for r in rows] if rows else []

    def _ensure_columns(self, table_name: str, codes: List[str]) -> List[str]:
        if not codes:
            return []
        existing = set(self._get_existing_columns(table_name))
        new_columns: List[str] = []
        alters: List[str] = []
        for code in codes:
            col = self._sanitize_code_to_column(code)
            if col not in existing:
                alters.append(f"ADD COLUMN `{col}` JSON NULL")
                new_columns.append(col)
        if alters:
            sql = f"ALTER TABLE {table_name} " + ", ".join(alters)
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            self.conn.commit()
            self.logger.info(f"为表 {table_name} 新增列: {new_columns}")
        return [self._sanitize_code_to_column(c) for c in codes]

    # ---------- 写入 ----------
    def upsert_records(self, params: Dict[str, Any], code_to_records: Dict[str, List[Dict[str, Any]]]) -> bool:
        if not code_to_records:
            return True
        meta = self.meta_manager.get_or_create_table_meta(params)
        self._ensure_table(meta.table_name)

        # 确保列
        codes = list(code_to_records.keys())
        _ = self._ensure_columns(meta.table_name, codes)

        # 组装 per-date UPSERT 数据
        # 我们一次处理一个日期，合并多只股票的JSON到同一行
        # 收集所有涉及到的日期
        all_dates: Dict[str, Dict[str, Any]] = {}
        for code, records in code_to_records.items():
            col = self._sanitize_code_to_column(code)
            for rec in records or []:
                td = rec.get("trade_date")
                if td is None:
                    continue
                # 标准化日期
                td_norm = self._normalize_date_str(td)
                all_dates.setdefault(td_norm, {})[col] = json.dumps(self._normalize_record(rec), ensure_ascii=False, default=str)

        if not all_dates:
            return True

        # 构建批量 UPSERT
        # INSERT INTO table (trade_date, colA, colB) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE colA=VALUES(colA), colB=VALUES(colB)
        sample_date, sample_payload = next(iter(all_dates.items()))
        cols = ["trade_date"] + list(sample_payload.keys())
        col_list = ",".join(f"`{c}`" for c in cols)
        placeholders = ",".join(["%s"] * len(cols))
        update_part = ",".join([f"`{c}`=VALUES(`{c}`)" for c in cols if c != "trade_date"])
        sql = f"INSERT INTO {meta.table_name} ({col_list}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_part}"

        values: List[Tuple[Any, ...]] = []
        for td, payload in all_dates.items():
            row = [td] + [payload.get(c) for c in cols if c != "trade_date"]
            values.append(tuple(row))

        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(sql, values)
            self.conn.commit()
            self.logger.info(f"写入 {meta.table_name}: {len(values)} 天的数据")
            return True
        except Exception as e:
            self.logger.error(f"写入 {meta.table_name} 失败: {e}")
            return False

    # ---------- 读取 ----------
    def fetch_records(self,
                      params: Dict[str, Any],
                      stock_codes: Optional[List[str]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        meta = self.meta_manager.get_or_create_table_meta(params)
        self._ensure_table(meta.table_name)

        # 若未指定codes，则列出表中的所有股票列
        existing_cols = self._get_existing_columns(meta.table_name)
        if stock_codes:
            cols_to_read = [self._sanitize_code_to_column(c) for c in stock_codes if self._sanitize_code_to_column(c) in existing_cols]
        else:
            # 排除 trade_date
            cols_to_read = [c for c in existing_cols if c != "trade_date"]

        if not cols_to_read:
            return {}

        # 构建查询
        col_list = ",".join(f"`{c}`" for c in ["trade_date"] + cols_to_read)
        sql = f"SELECT {col_list} FROM {meta.table_name}"
        params_list: List[Any] = []
        where_parts: List[str] = []
        if start_date:
            where_parts.append("trade_date >= %s")
            params_list.append(start_date)
        if end_date:
            where_parts.append("trade_date <= %s")
            params_list.append(end_date)
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY trade_date ASC"

        with self.conn.cursor() as cursor:
            cursor.execute(sql, tuple(params_list))
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]

        # 解析为 {code: [record_dict, ...]}
        result: Dict[str, List[Dict[str, Any]]] = {}
        # 建立反向映射：sanitized_col -> raw_code(尽量恢复)
        reverse_code_map: Dict[str, str] = {}
        if stock_codes:
            for raw in stock_codes:
                reverse_code_map[self._sanitize_code_to_column(raw)] = raw
        else:
            # 无原始codes，保持sanitized作为code
            for c in cols_to_read:
                reverse_code_map[c] = c

        if not rows:
            return {}

        td_idx = col_names.index("trade_date")
        for row in rows:
            trade_date = self._normalize_date_str(row[td_idx])
            for idx, c in enumerate(col_names):
                if c == "trade_date":
                    continue
                cell = row[idx]
                if cell is None:
                    continue
                try:
                    # MySQL JSON → Python
                    rec = cell if isinstance(cell, dict) else json.loads(cell)
                except Exception:
                    # 有些驱动会直接返回str
                    try:
                        rec = json.loads(str(cell))
                    except Exception:
                        continue
                rec["trade_date"] = trade_date
                code = reverse_code_map.get(c, c)
                result.setdefault(code, []).append(rec)

        # 按日期排序
        for code in result.keys():
            result[code].sort(key=lambda r: r.get("trade_date", ""))
        return result

    # ---------- 工具 ----------
    def _create_channel_connection(self, select_db: bool = False):
        """创建到通道库的连接。select_db=True 时直接连到目标库。"""
        try:
            conn = mysql.connect(
                host=global_db_config.MYSQL_HOST,
                port=global_db_config.MYSQL_PORT,
                user=global_db_config.MYSQL_USER,
                password=global_db_config.MYSQL_PASSWORD,
                database=self.schema_name if select_db else None,
                charset='utf8mb4',
                autocommit=True,
                connect_timeout=10,
            )
            return conn
        except Exception as e:
            self.logger.error(f"连接通道数据库失败: {e}")
            raise

    def _ensure_database(self) -> None:
        """确保 stock_channel（或指定）库存在"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.schema_name} DEFAULT CHARACTER SET utf8mb4")
            self.conn.commit()
            self.logger.info(f"已确保通道数据库存在: {self.schema_name}")
        except Exception as e:
            self.logger.error(f"创建/确保数据库失败: {e}")
            raise

    @staticmethod
    def _normalize_date_str(d: Any) -> str:
        if d is None:
            return ""
        try:
            return pd.to_datetime(d).strftime("%Y-%m-%d")
        except Exception:
            s = str(d)
            if len(s) == 8 and s.isdigit():
                return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
            return s

    @staticmethod
    def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化记录数据，确保可以安全地序列化为JSON
        
        Args:
            rec: 原始记录字典
            
        Returns:
            标准化后的记录字典
        """
        import math
        
        new_r = dict(rec)
        
        # 处理日期字段
        if "trade_date" in new_r:
            new_r["trade_date"] = ChannelDBManager._normalize_date_str(new_r["trade_date"])
        if "anchor_date" in new_r and new_r["anchor_date"] is not None:
            try:
                new_r["anchor_date"] = ChannelDBManager._normalize_date_str(new_r["anchor_date"])
            except Exception:
                new_r["anchor_date"] = None
        
        # 处理所有数值字段，将NaN、Infinity等转换为None
        for k, v in list(new_r.items()):
            # 处理pandas.Timestamp对象
            if hasattr(v, "isoformat"):
                new_r[k] = v.isoformat() if k not in ("trade_date", "anchor_date") else ChannelDBManager._normalize_date_str(v)
            # 处理数值类型
            elif isinstance(v, (int, float)):
                # 检查是否为NaN或Infinity
                if math.isnan(v) or math.isinf(v):
                    new_r[k] = None
                # 处理pandas的NaN
                elif pd.isna(v):
                    new_r[k] = None
            # 处理pandas.Series或DataFrame中的NaN
            elif pd.isna(v):
                new_r[k] = None
                
        return new_r


