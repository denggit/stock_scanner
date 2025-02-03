#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 8:23 PM
@File       : database.py
@Description: 
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pymysql as mysql
from backend.config.database import Config


class DatabaseManager:

    def __init__(self):
        self.config = Config()
        self.conn = self._create_connection()
        self._init_database()

    def _create_connection(self):
        """创建数据库连接"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                conn = mysql.connect(
                    host=self.config.MYSQL_HOST,
                    port=self.config.MYSQL_PORT,
                    user=self.config.MYSQL_USER,
                    password=self.config.MYSQL_PASSWORD,
                    database=self.config.MYSQL_DATABASE,
                    charset='utf8mb4',
                    connect_timeout=10,
                    autocommit=True,
                )
                return conn
            except Exception as e:
                if attempt == self.config.MAX_RETRIES - 1:
                    raise Exception("Failed to connect to database after multiple attempts")
                logging.warning(f"Failed to connect to database: {e}")
                time.sleep(self.config.RETRY_DELAY)

    def _ensure_connection(self):
        """确保数据库连接可用"""
        try:
            self.conn.ping(reconnect=True)
        except:
            self.conn = self._create_connection()

    def _init_database(self):
        """初始化数据库"""
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_list(
            code VARCHAR(10) PRIMARY KEY,
            name VARCHAR(100),
            ipo_date DATE,
            out_date DATE,
            type CHAR(1),
            status CHAR(1),
            update_time TIMESTAMP,
            INDEX idx_status(status),
            INDEX idx_type(type))
        """)
        self.conn.commit()
        cursor.close()

    def _table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables
            WHERE table_name = %s 
            AND table_schema = %s
        """, (self.config.MYSQL_DATABASE, table_name))
        cursor.close()
        return cursor.fetchone()[0] > 0

    def _ensure_daily_table(self, year: int):
        """确保年份表存在"""
        table_name = f"stock_daily_{year}"
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name}(
            code VARCHAR(10),
            trade_date DATE,
            open DECIMAL(10, 2),
            high DECIMAL(10, 2),
            low DECIMAL(10, 2),
            close DECIMAL(10, 2),
            preclose DECIMAL(10, 2),
            volume BIGINT,
            amount DECIMAL(16, 2),
            turn DECIMAL(10, 2),
            tradestatus TINYINT(1),
            pct_chg DECIMAL(10, 2),
            pe_ttm DECIMAL(20, 4),
            pb_mrq DECIMAL(20, 4),
            ps_ttm DECIMAL(20, 4),
            pcf_ncf_ttm DECIMAL(20, 4),
            is_st TINYINT(1),
            PRIMARY KEY (code, trade_date)
        ) PARTITION BY RANGE (MONTH(trade_date)) (
            PARTITION p1 VALUES LESS THAN (4),
            PARTITION p2 VALUES LESS THAN (7),
            PARTITION p3 VALUES LESS THAN (10),
            PARTITION p4 VALUES LESS THAN (13)
        )
        """
        cursor = self.conn.cursor()
        cursor.execute(create_table_sql)
        self.conn.commit()
        cursor.close()

    def save_stock_update_time(self, stock_df: pd.DataFrame):
        """保存股票更新时间"""
        cursor = self.conn.cursor()
        for _, row in stock_df.iterrows():
            cursor.execute("""
            INSERT INTO stock_list (code, update_time)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE update_time = VALUES(update_time)
            """, (row['code'], row['update_time']))
        self.conn.commit()
        cursor.close()

    def save_stock_basic(self, stock_basic: pd.DataFrame):
        """保存股票基本信息"""
        cursor = self.conn.cursor()
        for _, row in stock_basic.iterrows():
            # 处理日期字段的 NaT 值
            ipo_date = row['ipo_date'] if not pd.isna(row['ipo_date']) else None
            out_date = row['out_date'] if not pd.isna(row['out_date']) else None

            cursor.execute("""
            INSERT INTO stock_list (code, name, ipo_date, out_date, type, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE name = VALUES(name), ipo_date = VALUES(ipo_date), out_date = VALUES(out_date), type = VALUES(type), status = VALUES(status)
            """, (row['code'], row['name'], ipo_date, out_date, row['type'], row['status']))
            logging.debug(f"Saved stock basic info for {row['code']}")
        self.conn.commit()
        cursor.close()

    def update_stock_daily(self, code: str, stock_df: pd.DataFrame):
        """更新股票日线数据"""
        if stock_df.empty:
            return

        # 确保所有必须的列存在
        required_columns = ['code', 'trade_date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                            'turn', 'tradestatus', 'pct_chg', 'pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm', 'is_st']
        missing_columns = [col for col in required_columns if col not in stock_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in stock_df for code: {code}. Columns: {missing_columns}")

        # 检查NaN值
        for col in required_columns:
            nan_count = stock_df[col].isna().sum()
            if nan_count > 0:
                logging.debug(f"NaN values found in {col} for code: {code}")
                # 对不同类型的列使用不同的填充策略
                if col in ["volume", "amount", "turn", "tradestatus", "is_st"]:
                    stock_df[col] = stock_df[col].fillna(0)
                elif col in ['pe_ttm', 'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm']:
                    stock_df[col] = stock_df[col].fillna(0)
                elif col == 'pct_chg':
                    # 使用向量化操作计算涨跌幅
                    mask = stock_df[col].isna() & stock_df['close'].notna() & stock_df['preclose'].notna()
                    stock_df.loc[mask, col] = (stock_df.loc[mask, 'close'] - stock_df.loc[mask, 'preclose']) / stock_df.loc[mask, 'preclose'] * 100
                    # 如果还有NaN值，填充0
                    stock_df[col] = stock_df[col].fillna(0)
                elif col in ['open', 'high', 'low', 'close', 'preclose']:
                    # 使用前一个有效值填充
                    stock_df[col] = stock_df[col].fillna(method='ffill')
                    # 如果还有NaN（比如第一行就是NaN），使用后一个有效值填充
                    stock_df[col] = stock_df[col].fillna(method='bfill')
                    # 如果仍然有NaN，填充0
                    stock_df[col] = stock_df[col].fillna(0)

        total_records = 0
        # 按年份分组处理数据
        stock_df['year'] = pd.to_datetime(stock_df['trade_date']).dt.year
        for year, group in stock_df.groupby('year'):
            try:
                self._ensure_daily_table(year)
                table_name = f"stock_daily_{year}"

                # 准备SQL语句
                columns = ','.join(required_columns)
                placeholders = ','.join(['%s'] * len(required_columns))

                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {', '.join([f'{col} = VALUES({col})' for col in required_columns if col not in ('code', 'trade_date')])}"

                # 准备数据
                values = []
                for _, row in group.iterrows():
                    value = tuple(row[col] for col in required_columns)
                    values.append(value)

                # 执行更新
                with self.conn.cursor() as cursor:
                    cursor.executemany(insert_sql, values)

                self.conn.commit()
                total_records += len(values)
                logging.debug(f"Updated {len(values)} records for {code} in {year}")
            except Exception as e:
                logging.error(f"Failed to update {code} in {year}: {e}")
                logging.error(f"问题数据实例：{group.head().to_dict('records')}")
                raise

        logging.info(f"Total records updated for {code}: {total_records}")

        # 更新股票列表中的更新时间
        self.save_stock_update_time(pd.DataFrame({'code': [code], 'update_time': [datetime.now()]}))

    def get_stock_update_time(self, stock_code: str) -> Optional[datetime]:
        """获取股票更新时间"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT update_time FROM stock_list WHERE code = %s", (stock_code,))
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None

    def get_all_update_time(self) -> dict:
        """获取所有股票更新时间"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT code, update_time FROM stock_list")
            result = cursor.fetchall()
            cursor.close()
            return {row[0]: row[1].strftime('%Y-%m-%d') for row in result}
        except Exception as e:
            logging.error(f"Failed to get all update time: {e}")
            return {}

    def get_stock_list(self, fields='*') -> pd.DataFrame:
        """获取股票列表"""
        try:
            if fields != "*":
                fields = ",".join(fields) if isinstance(fields, (list, tuple)) else fields
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT {fields} FROM stock_list")
            result = cursor.fetchall()
            cursor.close()
            if result:  # 检查data是否为空
                return pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
            else:
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Failed to get stock list: {e}")
            return pd.DataFrame()

    def get_stock_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票日线数据"""
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year

        # 获取所有年份的数据
        all_data = []
        for year in range(start_year, end_year + 1):
            table_name = f"stock_daily_{year}"

            # 确保年份表存在
            if not self._table_exists(table_name):
                logging.warning(f"Table {table_name} does not exist")
                continue

            # 获取年份表中的数据
            query = f"SELECT * FROM {table_name} WHERE code = %s AND trade_date BETWEEN %s AND %s ORDER BY trade_date ASC"
            with self.conn.cursor() as cursor:
                cursor.execute(query, (code, start_date, end_date))
                result = cursor.fetchall()
                if result:
                    df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
                    all_data.append(df)

        # 合并所有年份的数据
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def need_update(self, stock_code: str) -> bool:
        """判断是否需要更新"""
        update_time = self.get_stock_update_time(stock_code)
        if update_time is None:
            return True
        return datetime.now() - update_time > timedelta(hours=self.config.DATA_UPDATE_INTERVAL)
