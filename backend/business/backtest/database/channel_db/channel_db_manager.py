#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道数据库管理器

职责分离：
- 参数→表名映射的meta管理（JSON文件）
- 按参数组合创建并维护MySQL表（窄表：一行=一个(code, trade_date)）
- 读取/写入指定参数、股票、日期范围的通道数据

设计说明（新：窄表 v1）：
- 表结构：channel_rcn_{hash}
  - code VARCHAR NOT NULL
  - trade_date DATE NOT NULL
  - 其余为通道字段列（DOUBLE/INT/VARCHAR），允许为 NULL
  - PRIMARY KEY (code, trade_date)
- 元数据：backend/business/backtest/database/channel_db/channel_meta.json
- 为兼容旧meta（宽表：channel_rc_{hash}），首次访问时将自动迁移映射到新表名(channel_rcn_{hash})并标注 schema_version='narrow_v1'

注意：
- 不再使用按股票列的宽表/JSON列，彻底解决“Too many columns”问题
- 写入采用 UPSERT：INSERT ... ON DUPLICATE KEY UPDATE 全量字段
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
            elif isinstance(v, bool):
                parts.append(f"{k}_{str(v).lower()}")
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
        self.logger = setup_logger("backtest")
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
            # 若为旧映射（无 schema_version 或旧表前缀），迁移到窄表命名
            table_name = m.get("table", "")
            schema_version = m.get("schema_version")
            if not schema_version or not str(table_name).startswith("channel_rcn_"):
                table_name = f"channel_rcn_{ph}"
                m.update({
                    "table": table_name,
                    "schema_version": "narrow_v1",
                })
                self._save()
            return ChannelTableMeta(
                params_hash=ph,
                table_name=table_name,
                params=m.get("params", {}),
                readable_key=m.get("readable_key", ""),
                created_at=m.get("created_at", datetime.now().isoformat()),
            )

        # 新建窄表映射
        table_name = f"channel_rcn_{ph}"
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
            "schema_version": "narrow_v1",
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
        self.logger = setup_logger("backtest")
        self.meta_manager = ChannelDBMetaManager(meta_file)
        self.schema_name = schema_name or os.getenv('MYSQL_CHANNEL_DB', 'stock_channel')
        self.conn = self._create_channel_connection()
        self._ensure_database()
        try:
            self.conn.select_db(self.schema_name)
        except Exception:
            self.conn.close()
            self.conn = self._create_channel_connection(select_db=True)

        # 定义窄表字段与SQL类型
        self._narrow_columns: List[Tuple[str, str]] = [
            ("code", "VARCHAR(32) NOT NULL"),
            ("trade_date", "DATE NOT NULL"),
            ("close", "DOUBLE NULL"),
            ("beta", "DOUBLE NULL"),
            ("sigma", "DOUBLE NULL"),
            ("r2", "DOUBLE NULL"),
            ("mid_today", "DOUBLE NULL"),
            ("upper_today", "DOUBLE NULL"),
            ("lower_today", "DOUBLE NULL"),
            ("mid_tomorrow", "DOUBLE NULL"),
            ("upper_tomorrow", "DOUBLE NULL"),
            ("lower_tomorrow", "DOUBLE NULL"),
            ("channel_status", "VARCHAR(32) NULL"),
            ("anchor_date", "DATE NULL"),
            ("anchor_price", "DOUBLE NULL"),
            ("break_cnt_up", "INT NULL"),
            ("break_cnt_down", "INT NULL"),
            ("reanchor_fail_up", "INT NULL"),
            ("reanchor_fail_down", "INT NULL"),
            ("cumulative_gain", "DOUBLE NULL"),
            ("window_size", "INT NULL"),
            ("days_since_anchor", "INT NULL"),
            ("break_reason", "VARCHAR(32) NULL"),
            ("width_pct", "DOUBLE NULL"),
            ("slope_deg", "DOUBLE NULL"),
            ("volatility", "DOUBLE NULL"),
        ]

    # ---------- 表与列管理 ----------
    def _sanitize_code_to_column(self, code: str) -> str:
        # 兼容旧接口，无实际用途（窄表不再需要按股票动态列）
        return code

    def _ensure_table(self, table_name: str) -> None:
        # 创建窄表：包含(code, trade_date)主键及所有字段
        columns_sql = ",\n            ".join([f"`{name}` {sql_type}" for name, sql_type in self._narrow_columns])
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {columns_sql},
            PRIMARY KEY (`code`, `trade_date`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_code` (`code`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
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

    def _ensure_columns(self, table_name: str) -> None:
        """确保窄表的所有字段齐全（新增字段时可自动补齐）"""
        existing = set(self._get_existing_columns(table_name))
        alters: List[str] = []
        for name, sql_type in self._narrow_columns:
            if name not in existing:
                alters.append(f"ADD COLUMN `{name}` {sql_type}")
        # 主键/索引若缺失，此处不强制追加，避免复杂迁移；新建表会具备
        if alters:
            sql = f"ALTER TABLE {table_name} " + ", ".join(alters)
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            self.conn.commit()
            self.logger.info(f"为表 {table_name} 新增字段: {[a.split(' ')[2].strip('`') for a in alters]}")

    # ---------- 写入 ----------
    def upsert_records(self, params: Dict[str, Any], code_to_records: Dict[str, List[Dict[str, Any]]]) -> bool:
        if not code_to_records:
            return True
        meta = self.meta_manager.get_or_create_table_meta(params)
        self._ensure_table(meta.table_name)
        self._ensure_columns(meta.table_name)

        # 准备列与SQL
        all_columns = [name for name, _ in self._narrow_columns]
        non_pk_columns = [c for c in all_columns if c not in ("code", "trade_date")]
        col_list = ",".join(f"`{c}`" for c in all_columns)
        placeholders = ",".join(["%s"] * len(all_columns))
        update_part = ",".join([f"`{c}`=VALUES(`{c}`)" for c in non_pk_columns])
        sql = f"INSERT INTO {meta.table_name} ({col_list}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_part}"

        # 组装批量值
        values: List[Tuple[Any, ...]] = []
        for code, records in code_to_records.items():
            for rec in records or []:
                nr = self._normalize_record(rec)
                row = [
                    code,
                    self._normalize_date_str(nr.get("trade_date")),
                    nr.get("close"),
                    nr.get("beta"),
                    nr.get("sigma"),
                    nr.get("r2"),
                    nr.get("mid_today"),
                    nr.get("upper_today"),
                    nr.get("lower_today"),
                    nr.get("mid_tomorrow"),
                    nr.get("upper_tomorrow"),
                    nr.get("lower_tomorrow"),
                    nr.get("channel_status"),
                    self._normalize_date_str(nr.get("anchor_date")) if nr.get("anchor_date") else None,
                    nr.get("anchor_price"),
                    nr.get("break_cnt_up"),
                    nr.get("break_cnt_down"),
                    nr.get("reanchor_fail_up"),
                    nr.get("reanchor_fail_down"),
                    nr.get("cumulative_gain"),
                    nr.get("window_size"),
                    nr.get("days_since_anchor"),
                    nr.get("break_reason"),
                    nr.get("width_pct"),
                    nr.get("slope_deg"),
                    nr.get("volatility"),
                ]
                # 跳过无日期
                if row[1] in (None, ""):
                    continue
                values.append(tuple(row))

        if not values:
            return True

        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(sql, values)
            self.conn.commit()
            self.logger.info(f"写入 {meta.table_name}: {len(values)} 条记录")
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

        # 构建查询（窄表）
        all_columns = [name for name, _ in self._narrow_columns]
        col_list = ",".join(f"`{c}`" for c in all_columns)
        sql = f"SELECT {col_list} FROM {meta.table_name}"
        params_list: List[Any] = []
        where_parts: List[str] = []
        if stock_codes:
            placeholders = ",".join(["%s"] * len(stock_codes))
            where_parts.append(f"code IN ({placeholders})")
            params_list.extend(stock_codes)
        if start_date:
            where_parts.append("trade_date >= %s")
            params_list.append(start_date)
        if end_date:
            where_parts.append("trade_date <= %s")
            params_list.append(end_date)
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY code ASC, trade_date ASC"

        with self.conn.cursor() as cursor:
            cursor.execute(sql, tuple(params_list))
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]

        if not rows:
            return {}

        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            rec = {col_names[i]: row[i] for i in range(len(col_names))}
            # 规范化日期字段
            rec["trade_date"] = self._normalize_date_str(rec.get("trade_date"))
            if rec.get("anchor_date"):
                rec["anchor_date"] = self._normalize_date_str(rec.get("anchor_date"))
            code = rec.get("code")
            if code is None:
                continue
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
    def _normalize_date_str(d: Any):
        """
        规范化日期到 YYYY-MM-DD；无法解析或为 NaT/NaN/空值则返回 None。
        """
        try:
            if d is None:
                return None
            # pandas 友好解析，失败返回 NaT
            ts = pd.to_datetime(d, errors='coerce')
            if pd.isna(ts):
                return None
            return pd.Timestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化记录数据，确保可以安全地写入DB
        
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
                if k in ("trade_date", "anchor_date"):
                    new_r[k] = ChannelDBManager._normalize_date_str(v)
                else:
                    new_r[k] = v.isoformat()
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
            # 处理字符串形式的日期字段
            if k in ("trade_date", "anchor_date"):
                new_r[k] = ChannelDBManager._normalize_date_str(new_r.get(k))

        return new_r
