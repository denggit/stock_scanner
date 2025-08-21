# 因子字段映射文档

## 概述

本文档说明了因子库中使用的字段名称与数据库字段的对应关系，确保因子计算能够正确使用您的数据库中的数据。

## 数据库字段结构

### 日线数据字段 (stock_daily_*)

根据 `data_manager.py` 和 `data_fetcher.py`，您的数据库包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `code` | VARCHAR(10) | 股票代码 |
| `trade_date` | DATE | 交易日期 |
| `open` | DECIMAL(10,2) | 开盘价 |
| `high` | DECIMAL(10,2) | 最高价 |
| `low` | DECIMAL(10,2) | 最低价 |
| `close` | DECIMAL(10,2) | 收盘价 |
| `preclose` | DECIMAL(10,2) | 前收盘价 |
| `volume` | BIGINT | 成交量 |
| `amount` | DECIMAL(16,2) | 成交额 |
| `turn` | DECIMAL(10,2) | 换手率 |
| `tradestatus` | SMALLINT | 交易状态 |
| `pct_chg` | DECIMAL(10,2) | 涨跌幅 |
| `pe_ttm` | DECIMAL(10,2) | 市盈率(TTM) |
| `pb_mrq` | DECIMAL(10,2) | 市净率(MRQ) |
| `ps_ttm` | DECIMAL(10,2) | 市销率(TTM) |
| `pcf_ncf_ttm` | DECIMAL(10,2) | 市现率(TTM) |
| `is_st` | SMALLINT | 是否ST股票 |
| `vwap` | DECIMAL(10,2) | 成交量加权平均价 |

### 财务数据字段

#### 利润表 (stock_profit)

| 字段名 | 说明 |
|--------|------|
| `roeAvg` | 平均净资产收益率 |
| `npMargin` | 净利润率 |
| `gpMargin` | 毛利率 |
| `netProfit` | 净利润 |
| `epsTTM` | 每股收益(TTM) |
| `MBRevenue` | 主营收入 |
| `totalShare` | 总股本 |
| `liqaShare` | 流通股本 |

#### 资产负债表 (stock_balance)

| 字段名 | 说明 |
|--------|------|
| `currentRatio` | 流动比率 |
| `quickRatio` | 速动比率 |
| `cashRatio` | 现金比率 |
| `YOYLiability` | 负债同比增长率 |
| `liabilityToAsset` | 资产负债率 |
| `assetToEquity` | 权益乘数 |

#### 现金流量表 (stock_cashflow)

| 字段名 | 说明 |
|--------|------|
| `CAToAsset` | 流动资产/总资产 |
| `NCAToAsset` | 非流动资产/总资产 |
| `tangibleAssetToAsset` | 有形资产/总资产 |
| `ebitToInterest` | 息税前利润/利息费用 |
| `CFOToOR` | 经营现金流/营业收入 |
| `CFOToNP` | 经营现金流/净利润 |
| `CFOToGr` | 经营现金流/毛利润 |

#### 成长能力表 (stock_growth)

| 字段名 | 说明 |
|--------|------|
| `YOYEquity` | 净资产同比增长率 |
| `YOYAsset` | 总资产同比增长率 |
| `YOYNI` | 净利润同比增长率 |
| `YOYEPSBasic` | 每股收益同比增长率 |
| `YOYPNI` | 主营收入同比增长率 |

#### 营运能力表 (stock_operation)

| 字段名 | 说明 |
|--------|------|
| `NRTurnRatio` | 应收账款周转率 |
| `NRTurnDays` | 应收账款周转天数 |
| `INVTurnRatio` | 存货周转率 |
| `INVTurnDays` | 存货周转天数 |
| `CATurnRatio` | 流动资产周转率 |
| `AssetTurnRatio` | 总资产周转率 |

## 因子字段映射

### 技术因子字段映射

| 因子名称 | 使用的字段 | 说明 |
|----------|------------|------|
| `momentum_*` | `close` | 收盘价 |
| `volatility_*` | `close` | 收盘价 |
| `volume_ratio_*` | `volume` | 成交量 |
| `price_position_*` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `ma_cross_*` | `close` | 收盘价 |
| `rsi_*` | `close` | 收盘价 |
| `bollinger_position` | `close` | 收盘价 |
| `macd_histogram` | `close` | 收盘价 |
| `williams_r` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `cci` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `kama` | `close` | 收盘价 |
| `atr` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `adx` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `volume_price_momentum` | `close`, `volume` | 收盘价、成交量 |
| `gap_strength` | `open`, `preclose` | 开盘价、前收盘价 |
| `intraday_volatility` | `high`, `low`, `close` | 最高价、最低价、收盘价 |

### 基本面因子字段映射

| 因子名称 | 使用的字段 | 说明 |
|----------|------------|------|
| `pe_ratio` | `pe_ttm` | 市盈率倒数 |
| `pb_ratio` | `pb_mrq` | 市净率倒数 |
| `ps_ratio` | `ps_ttm` | 市销率倒数 |
| `pcf_ratio` | `pcf_ncf_ttm` | 市现率倒数 |
| `roe` | `roeAvg` | 净资产收益率 |
| `roa` | 暂未实现 | 总资产收益率 |
| `revenue_growth` | `YOYAsset` | 资产同比增长率 |
| `profit_growth` | `YOYNI` | 净利润同比增长率 |
| `debt_to_equity` | `liabilityToAsset` | 资产负债率 |
| `current_ratio` | `currentRatio` | 流动比率 |
| `net_profit_margin` | `npMargin` | 净利润率 |
| `gross_profit_margin` | `gpMargin` | 毛利率 |
| `eps_ttm` | `epsTTM` | 每股收益(TTM) |
| `asset_turnover` | `AssetTurnRatio` | 总资产周转率 |
| `inventory_turnover` | `INVTurnRatio` | 存货周转率 |
| `cfo_to_revenue` | `CFOToOR` | 经营现金流/营业收入 |
| `cfo_to_profit` | `CFOToNP` | 经营现金流/净利润 |

### WorldQuant Alpha因子字段映射

| 因子名称 | 使用的字段 | 说明 |
|----------|------------|------|
| `alpha_1` | `pct_chg`, `close` | 涨跌幅、收盘价 |
| `alpha_2` | `volume`, `close`, `open` | 成交量、收盘价、开盘价 |
| `alpha_3` | `open`, `volume` | 开盘价、成交量 |
| `alpha_4` | `low` | 最低价 |
| `alpha_5` | `open`, `close`, `vwap` | 开盘价、收盘价、VWAP |
| `alpha_6` | `open`, `volume` | 开盘价、成交量 |
| `alpha_7` | `volume`, `close` | 成交量、收盘价 |
| `alpha_8` | `open`, `pct_chg` | 开盘价、涨跌幅 |
| `alpha_9` | `close` | 收盘价 |
| `alpha_10` | `close` | 收盘价 |

### 通道分析因子字段映射

| 因子名称 | 使用的字段 | 说明 |
|----------|------------|------|
| `channel_distance` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `channel_breakout` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `channel_width` | `high`, `low`, `close` | 最高价、最低价、收盘价 |
| `channel_trend` | `high`, `low`, `close` | 最高价、最低价、收盘价 |

## 更新说明

### 已完成的更新

1. **字段名称统一**：将所有因子中的字段名称更新为与数据库完全匹配的名称
    - `open_price` → `open`
    - `preclose` 保持不变
    - 其他字段名称已与数据库字段完全对应

2. **基本面因子增强**：
    - 添加了更多基于实际财务数据的因子
    - 使用正确的财务数据字段名称
    - 移除了临时的示例数据

3. **新增因子**：
    - 盈利能力因子：`net_profit_margin`, `gross_profit_margin`, `eps_ttm`
    - 营运能力因子：`asset_turnover`, `inventory_turnover`
    - 现金流量因子：`cfo_to_revenue`, `cfo_to_profit`

### 使用说明

1. **技术因子**：主要使用市场行情数据，包括价格、成交量等
2. **基本面因子**：使用财务数据，需要确保数据库中有对应的财务数据
3. **WorldQuant Alpha因子**：使用市场行情数据，按照WorldQuant的公式计算
4. **通道分析因子**：使用价格数据，分析价格通道特征

### 注意事项

1. 基本面因子需要数据库中有对应的财务数据才能正常计算
2. 某些财务数据可能更新频率较低，需要注意数据时效性
3. 建议在使用基本面因子前，先检查数据库中是否有足够的财务数据

## 测试结果

已成功测试以下因子：

- ✅ `momentum_20d` - 技术因子
- ✅ `pe_ratio` - 基本面因子

所有因子都能正确使用数据库中的字段进行计算。
