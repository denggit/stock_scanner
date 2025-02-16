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
from datetime import datetime

import pandas as pd
import pymysql as mysql
import numpy as np

from backend.config.database import Config


class DatabaseManager:

    def __init__(self):
        self.config = Config()
        self.conn = self._create_connection()
        self._init_database()

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, "conn"):
            self.conn.close()

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
            update_time_back TIMESTAMP,  # 后复权数据更新时间
            update_time_profit INT,  # 利润表更新时间
            update_time_balance INT,  # 资产负债表更新时间
            update_time_cashflow INT,  # 现金流量表更新时间
            update_time_growth INT,  # 成长能力更新时间
            update_time_operation INT,  # 营运能力更新时间
            update_time_dupont INT,  # 杜邦分析更新时间
            update_time_dividend INT,  # 分红数据更新时间
            INDEX idx_status(status),
            INDEX idx_type(type))
        """)
        # 创建财务报表相关表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_profit(
            code VARCHAR(10),
            pubDate DATE,
            statDate DATE,
            roeAvg DECIMAL(15,4),
            npMargin DECIMAL(15,4),
            gpMargin DECIMAL(15,4),
            netProfit DECIMAL(20,4),
            epsTTM DECIMAL(15,4),
            MBRevenue DECIMAL(20,4),
            totalShare DECIMAL(20,4),
            liqaShare DECIMAL(20,4),
            PRIMARY KEY (code, statDate)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_balance(
            code VARCHAR(10),
            pubDate DATE,
            statDate DATE,
            currentRatio DECIMAL(15,4),
            quickRatio DECIMAL(15,4),
            cashRatio DECIMAL(15,4),
            YOYLiability DECIMAL(15,4),
            liabilityToAsset DECIMAL(15,4),
            assetToEquity DECIMAL(15,4),
            PRIMARY KEY (code, statDate)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_cashflow(
            code VARCHAR(10),
            pubDate DATE,
            statDate DATE,
            CAToAsset DECIMAL(15,4),
            NCAToAsset DECIMAL(15,4),
            tangibleAssetToAsset DECIMAL(15,4),
            ebitToInterest DECIMAL(15,4),
            CFOToOR DECIMAL(15,4),
            CFOToNP DECIMAL(15,4),
            CFOToGr DECIMAL(15,4),
            PRIMARY KEY (code, statDate)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_growth(
            code VARCHAR(10),
            pubDate DATE,
            statDate DATE,
            YOYEquity DECIMAL(15,4),
            YOYAsset DECIMAL(15,4),
            YOYNI DECIMAL(15,4),
            YOYEPSBasic DECIMAL(15,4),
            YOYPNI DECIMAL(15,4),
            PRIMARY KEY (code, statDate)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_operation(
            code VARCHAR(10),
            pubDate DATE,
            statDate DATE,
            NRTurnRatio DECIMAL(15,4),
            NRTurnDays DECIMAL(15,4),
            INVTurnRatio DECIMAL(15,4),
            INVTurnDays DECIMAL(15,4),
            CATurnRatio DECIMAL(15,4),
            AssetTurnRatio DECIMAL(15,4),
            PRIMARY KEY (code, statDate)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_dupont(
            code VARCHAR(10),
            pubDate DATE,
            statDate DATE,
            dupontROE DECIMAL(15,4),
            dupontAssetStoEquity DECIMAL(15,4),
            dupontAssetTurn DECIMAL(15,4),
            dupontPnitoni DECIMAL(15,4),
            dupontNitogr DECIMAL(15,4),
            dupontTaxBurden DECIMAL(15,4),
            dupontIntburden DECIMAL(15,4),
            dupontEbittogr DECIMAL(15,4),
            PRIMARY KEY (code, statDate)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_dividend(
            code VARCHAR(10),
            dividPreNoticeDate DATE,
            dividAgmPumDate DATE,
            dividPlanAnnounceDate DATE,
            dividPlanDate DATE,
            dividRegistDate DATE,
            dividOperateDate DATE,
            dividPayDate DATE,
            dividStockMarketDate DATE,
            dividCashPsBeforeTax DECIMAL(15,4),
            dividCashPsAfterTax DECIMAL(15,4),
            dividStocksPs DECIMAL(15,4),
            dividCashStock DECIMAL(15,4),
            dividReserveToStockPs DECIMAL(15,4),
            PRIMARY KEY (code, dividOperateDate)
        )""")

        self.conn.commit()
        cursor.close()

    def _table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables
            WHERE table_schema = %s 
            AND table_name = %s
        """, (self.config.MYSQL_DATABASE, table_name))
        cursor.close()
        return cursor.fetchone()[0] > 0

    def _ensure_daily_table(self, year: int, adjust: str = '3'):
        """确保特定年份的日线数据表存在
        
        Args:
            year: 年份
            adjust: 复权类型，1:后复权，2:前复权，3:不复权
        """
        table_suffix = f"back_{year}" if adjust == '1' else str(year)
        table_name = f"stock_daily_{table_suffix}"

        if not self._table_exists(table_name):
            cursor = self.conn.cursor()
            cursor.execute(f"""
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
            """)
            self.conn.commit()
            cursor.close()

    def update_stock_update_time(self, df: pd.DataFrame, adjust: str = '3'):
        """保存股票行情数据更新时间
        
        Args:
            df: 包含code和update_time的DataFrame
            adjust: 复权类型，'1'表示后复权，'3'表示不复权
        """
        cursor = self.conn.cursor()
        time_field = 'update_time_back' if adjust == '1' else 'update_time'

        for _, row in df.iterrows():
            cursor.execute(f"""
            INSERT INTO stock_list (code, {time_field})
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE {time_field} = VALUES({time_field})
            """, (row['code'], row['update_time']))
        self.conn.commit()
        cursor.close()

    def update_financial_update_time(self, df: pd.DataFrame, data_type: str):
        """保存股票行情数据更新时间

        Args:
            df: 包含code和update_time的DataFrame
            data_type: 财务数据类型 -> [dividend, dupont, growth, operation, profit, balance, cashflow]
        """
        cursor = self.conn.cursor()
        column = f"update_time_{data_type}"

        for _, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO stock_list (code, {column})
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE {column} = VALUES({column})
                """, (row['code'], row['year']))
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

    def update_stock_daily(self, code: str, stock_df: pd.DataFrame, adjust: str = '3'):
        """更新股票日线数据
        
        Args:
            code: 股票代码
            stock_df: 股票数据
            adjust: 复权类型，'1'表示后复权，'3'表示不复权
        """
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
                    stock_df.loc[mask, col] = (stock_df.loc[mask, 'close'] - stock_df.loc[mask, 'preclose']) / \
                                              stock_df.loc[mask, 'preclose'] * 100
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
                # 根据adjust参数决定使用哪个表
                if adjust == '1':
                    self._ensure_daily_table(year, adjust='1')
                    table_name = f"stock_daily_back_{year}"
                else:
                    self._ensure_daily_table(year, adjust='3')
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
                logging.error(f"Failed to update {code}_{adjust} in {year}: {e}")
                logging.error(f"问题数据实例：{group.head().to_dict('records')}")
                raise

        logging.info(f"Total records updated for {code}_{adjust}: {total_records}")

        # 更新股票列表中的更新时间
        self.update_stock_update_time(pd.DataFrame({'code': [code], 'update_time': [datetime.now()]}), adjust)

    def get_all_update_time(self, adjust: str = '3') -> dict:
        """获取所有股票更新时间
        
        Args:
            adjust: 复权类型，'1'表示后复权，'3'表示不复权
            
        Returns:
            股票代码到更新时间的映射字典
        """
        try:
            cursor = self.conn.cursor()
            time_field = 'update_time_back' if adjust == '1' else 'update_time'

            cursor.execute(f"SELECT code, {time_field} FROM stock_list")
            data = cursor.fetchall()
            cursor.close()
            result = {}
            for row in data:
                if pd.isnull(row[1]):
                    result[row[0]] = None
                else:
                    result[row[0]] = row[1].strftime('%Y-%m-%d %H:%M:%S')

            return result
        except Exception as e:
            logging.exception(f"Failed to get all update time: {e}")
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

    def get_stock_daily(self, code: str, start_date: str, end_date: str, adjust: str = '3') -> pd.DataFrame:
        """获取股票日线数据"""
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year

        # 获取所有年份的数据
        all_data = []
        for year in range(start_year, end_year + 1):
            if adjust == '1':
                table_name = f"stock_daily_back_{year}"
            elif adjust == '2':
                table_name = f"stock_daily_forward_{year}"
            else:
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

    def _save_financial_data(self, df: pd.DataFrame, table_name: str, columns: list):
        """通用的财务数据保存方法"""
        if df.empty:
            return
        
        # 定义各字段合理范围
        range_limits = {
            'NRTurnRatio': (-1e10, 1e10),
            'NRTurnDays': (-1e10, 1e10),
            'INVTurnRatio': (-1e10, 1e10),
            'INVTurnDays': (-1e10, 1e10),
            'CATurnRatio': (-1e10, 1e10),
            'CATurnDays': (-1e10, 1e10),
            'AssetTurnRatio': (-1e10, 1e10),
            'AssetTurnDays': (-1e10, 1e10),
            'roeAvg': (-100, 100),
            'npMargin': (-100, 100),
            'gpMargin': (-100, 100),
            'netProfit': (-1e15, 1e15),
            'epsTTM': (-1e10, 1e10),
            'MBRevenue': (-1e15, 1e15),
            'totalShare': (-1e15, 1e15),
            'liqaShare': (-1e15, 1e15),
            'currentRatio': (-1e10, 1e10),
            'quickRatio': (-1e10, 1e10), 
            'cashRatio': (-1e10, 1e10),
            'YOYLiability': (-1e10, 1e10),
            'liabilityToAsset': (-1e10, 1e10),
            'assetToEquity': (-1e10, 1e10),
            'dupontROE': (-1e10, 1e10),
            'dupontAssetStoEquity': (-1e10, 1e10),
            'dupontAssetTurn': (-1e10, 1e10),
            'dupontPnitoni': (-1e10, 1e10),
            'dupontNitogr': (-1e10, 1e10),
            'dupontTaxBurden': (-1e10, 1e10),
            'dupontIntburden': (-1e10, 1e10),
            'dupontEbittogr': (-1e10, 1e10),
            'dividCashPsBeforeTax': (-1e10, 1e10),
            'dividCashPsAfterTax': (-1e10, 1e10),
            'dividStocksPs': (-1e10, 1e10),
            'dividCashStock': (-1e10, 1e10),
            'dividReserveToStockPs': (-1e10, 1e10)
        }
        
        # 清洗数据
        for col in df.columns:
            if col in range_limits:
                min_val, max_val = range_limits[col]
                # 确保列是浮点数类型
                df[col] = df[col].astype('float64', errors='ignore')
                
                # 先处理无穷大值
                mask = np.isinf(df[col])
                df.loc[mask, col] = np.nan
                
                # 检查异常值并记录
                outliers = df[(df[col] < min_val) | (df[col] > max_val)]
                if not outliers.empty:
                    logging.warning(f"在 {table_name} 表中发现 {len(outliers)} 条 {col} 异常数据：")
                    for _, row in outliers.iterrows():
                        logging.warning(f"代码：{row['code']}, 日期：{row.get('statDate', 'N/A')}, {col}：{row[col]}")
                
                # 限制数值范围
                df[col] = df[col].clip(lower=min_val, upper=max_val)
            
        # 记录 NaN 值的数量
        nan_count = df.isna().sum()
        if nan_count.any():
            logging.debug(f"在 {table_name} 表中发现 NaN 值:\n{nan_count[nan_count > 0]}")
        
        # 将所有数值列的 NaN 转换为 None
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = df[col].astype('float64', errors='ignore')  # 确保类型一致
            df[col] = df[col].where(pd.notnull(df[col]), None)
        
        # 过滤无效数据
        if 'statDate' in df.columns:
            df = df.dropna(subset=['code', 'statDate'], how='any')
        else:
            df = df.dropna(subset=['code'], how='any')
        
        # 准备 SQL 语句
        placeholders = ','.join(['%s'] * len(columns))
        columns_str = ','.join(columns)
        update_str = ','.join([f"{col}=VALUES({col})" for col in columns if col not in ['code', 'statDate']])

        sql = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_str}
        """

        # 准备数据并确保没有 NaN
        values = []
        for _, row in df[columns].iterrows():
            value = []
            for v in row:
                if isinstance(v, float) and np.isnan(v):
                    value.append(None)
                else:
                    value.append(v)
            values.append(tuple(value))
        
        # 执行更新
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(sql, values)
            self.conn.commit()
        except Exception as e:
            logging.error(f"保存到 {table_name} 失败: {str(e)}")
            # 打印出问题数据的样本
            if values:
                logging.error(f"问题数据样本: {values[0]}")
            raise

    def save_profit_data(self, df: pd.DataFrame):
        """保存利润表数据"""
        columns = ['code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin',
                   'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare']
        self._save_financial_data(df, 'stock_profit', columns)

    def save_balance_data(self, df: pd.DataFrame):
        """保存资产负债表数据"""
        columns = ['code', 'pubDate', 'statDate', 'currentRatio', 'quickRatio',
                   'cashRatio', 'YOYLiability', 'liabilityToAsset', 'assetToEquity']
        self._save_financial_data(df, 'stock_balance', columns)

    def save_cashflow_data(self, df: pd.DataFrame):
        """保存现金流量表数据"""
        columns = ['code', 'pubDate', 'statDate', 'CAToAsset', 'NCAToAsset',
                   'tangibleAssetToAsset', 'ebitToInterest', 'CFOToOR',
                   'CFOToNP', 'CFOToGr']
        self._save_financial_data(df, 'stock_cashflow', columns)

    def save_growth_data(self, df: pd.DataFrame):
        """保存成长能力数据"""
        columns = ['code', 'pubDate', 'statDate', 'YOYEquity', 'YOYAsset',
                   'YOYNI', 'YOYEPSBasic', 'YOYPNI']
        self._save_financial_data(df, 'stock_growth', columns)

    def save_operation_data(self, df: pd.DataFrame):
        """保存营运能力数据"""
        columns = ['code', 'pubDate', 'statDate', 'NRTurnRatio', 'NRTurnDays',
                   'INVTurnRatio', 'INVTurnDays', 'CATurnRatio', 'AssetTurnRatio']
        self._save_financial_data(df, 'stock_operation', columns)

    def save_dupont_data(self, df: pd.DataFrame):
        """保存杜邦分析数据"""
        columns = ['code', 'pubDate', 'statDate', 'dupontROE', 'dupontAssetStoEquity',
                   'dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden',
                   'dupontIntburden', 'dupontEbittogr']
        self._save_financial_data(df, 'stock_dupont', columns)

    def save_dividend_data(self, df: pd.DataFrame):
        """保存分红数据"""
        columns = ['code', 'dividPreNoticeDate', 'dividAgmPumDate', 'dividPlanAnnounceDate',
                   'dividPlanDate', 'dividRegistDate', 'dividOperateDate', 'dividPayDate',
                   'dividStockMarketDate', 'dividCashPsBeforeTax', 'dividCashPsAfterTax',
                   'dividStocksPs', 'dividCashStock', 'dividReserveToStockPs']
        self._save_financial_data(df, 'stock_dividend', columns)

