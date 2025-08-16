# 🚀 股票扫描器 - API接口文档

## 📋 概述

- **基础URL**: `http://localhost:8000`
- **API文档**: `http://localhost:8000/docs`
- **认证**: 目前无需认证
- **数据格式**: JSON
- **字符编码**: UTF-8

---

## 🏠 基础接口

### 1. 根路径

```http
GET /
```

**响应示例**:

```json
{
  "message": "Welcome to the Stock Screener API"
}
```

### 2. 测试接口

```http
GET /test
```

**响应示例**:

```json
{
  "message": "Test successful"
}
```

---

## 📊 股票数据接口

### 1. 获取股票数据

```http
GET /api/stock/{code}
```

**路径参数**:

- `code` (string, required): 股票代码，如 `000001` 或 `000001.SZ`

**查询参数**:

- `period` (string, optional): 数据周期，默认 `daily`
  - `daily`: 日线数据
  - `weekly`: 周线数据
  - `monthly`: 月线数据
- `start_date` (string, optional): 开始日期，格式 `YYYY-MM-DD`
- `end_date` (string, optional): 结束日期，格式 `YYYY-MM-DD`
- `ma_periods` (array, optional): 移动平均线周期列表，如 `[5, 10, 20]`

**请求示例**:

```bash
# 基本请求
curl "http://localhost:8000/api/stock/000001"

# 带参数的请求
curl "http://localhost:8000/api/stock/000001?period=daily&start_date=2024-01-01&end_date=2024-12-31&ma_periods=5&ma_periods=20"
```

**成功响应** (200):

```json
[
  {
    "trade_date": "2024-01-02",
    "open": 10.50,
    "high": 10.80,
    "low": 10.30,
    "close": 10.75,
    "volume": 1000000,
    "amount": 10750000,
    "pct_chg": 2.38,
    "turn": 0.85,
    "ma5": 10.60,
    "ma10": 10.45,
    "ma20": 10.30
  }
]
```

**错误响应**:

```json
// 404 - 股票不存在
{
  "detail": "股票代码 999999 不存在或无效"
}

// 400 - 参数错误
{
  "detail": "请求参数错误，请检查日期格式"
}

// 500 - 服务器错误
{
  "detail": "无法连接到后端服务，请检查服务是否启动"
}
```

### 2. 获取股票指标数据

```http
POST /api/stock/{code}/indicators
```

**路径参数**:

- `code` (string, required): 股票代码

**查询参数**:

- `indicators` (array, required): 指标列表，如 `["MA", "MACD", "RSI"]`
- `date` (string, optional): 指定日期，格式 `YYYY-MM-DD`

**请求示例**:

```bash
curl -X POST "http://localhost:8000/api/stock/000001/indicators?indicators=MA&indicators=MACD&indicators=RSI&date=2024-12-31"
```

**响应示例**:

```json
{
  "code": "000001",
  "date": "2024-12-31",
  "indicators": {
    "MA": {
      "ma5": 10.60,
      "ma10": 10.45,
      "ma20": 10.30
    },
    "MACD": {
      "macd": 0.15,
      "signal": 0.12,
      "histogram": 0.03
    },
    "RSI": {
      "rsi": 65.5
    }
  }
}
```

---

## 🎯 策略接口

### 1. 策略扫描

```http
POST /api/strategy/scan
```

**请求体**:

```json
{
  "strategy": "爆发式选股策略",
  "params": {
    "stock_pool": "非ST股票",
    "signal": 70.0,
    "volume_ratio": 1.5,
    "rsi_range": [45, 65],
    "explosion_probability": 0.5,
    "weights": {
      "volume": 0.35,
      "momentum": 0.30,
      "pattern": 0.20,
      "volatility": 0.15
    }
  }
}
```

**策略参数说明**:

- `stock_pool`: 股票池选择
  - `"全量股票"`: 所有股票
  - `"非ST股票"`: 排除ST股票
  - `"上证50"`: 上证50成分股
  - `"沪深300"`: 沪深300成分股
  - `"中证500"`: 中证500成分股
- `signal`: 信号强度阈值 (0-100)
- `volume_ratio`: 成交量比率阈值
- `rsi_range`: RSI范围 [最小值, 最大值]
- `explosion_probability`: 爆发概率阈值 (0-1)

**响应示例**:

```json
{
  "results": [
    {
      "code": "000001.SZ",
      "name": "平安银行",
      "signal": 85.6,
      "volume_ratio": 2.1,
      "rsi": 52.3,
      "explosion_probability": 0.68,
      "buy_signal": "建议买入",
      "risk_level": "中等风险",
      "price": 10.75,
      "volume_score": 78.5,
      "momentum_score": 82.3,
      "pattern_score": 75.2,
      "volatility_score": 68.9,
      "ml_prediction": 72.1
    }
  ],
  "total_count": 1,
  "scan_time": "2.5s"
}
```

### 2. 获取策略列表

```http
GET /api/strategy/list
```

**响应示例**:

```json
{
  "strategies": [
    {
      "name": "爆发式选股策略",
      "description": "寻找20个交易日内可能暴涨30%的股票"
    },
    {
      "name": "均线回踩策略",
      "description": "寻找价格回踩均线的买入机会"
    },
    {
      "name": "波段交易策略",
      "description": "基于技术指标的波段操作"
    },
    {
      "name": "扫描翻倍股",
      "description": "寻找具有翻倍潜力的股票"
    },
    {
      "name": "长期上涨策略",
      "description": "识别长期上涨趋势的股票"
    },
    {
      "name": "头肩底形态策略",
      "description": "识别头肩底形态的股票"
    },
    {
      "name": "上升通道策略",
      "description": "基于上升通道回归分析的策略"
    },
    {
      "name": "放量上涨策略",
      "description": "识别放量上涨的股票"
    }
  ]
}
```

---

## 📈 回测接口

### 1. 运行回测

```http
GET /api/backtest/run
```

**查询参数**:

- `strategy` (string, required): 策略名称
- `start_date` (string, required): 开始日期，格式 `YYYY-MM-DD`
- `end_date` (string, required): 结束日期，格式 `YYYY-MM-DD`
- `backtest_init_params` (object, required): 回测初始化参数
- `params` (object, optional): 策略参数

**请求示例**:

```bash
curl "http://localhost:8000/api/backtest/run?strategy=爆发式选股策略&start_date=2024-01-01&end_date=2024-12-31&backtest_init_params={\"stock_pool\":\"非ST股票\",\"initial_capital\":100000,\"max_positions\":4,\"allocation_strategy\":\"信号强度加权\"}&params={\"signal\":70.0,\"volume_ratio\":1.5}"
```

**响应示例**:

```json
{
  "backtest_id": "backtest_20241231_001",
  "summary": {
    "total_return": 0.156,
    "annual_return": 0.189,
    "max_drawdown": -0.085,
    "sharpe_ratio": 1.45,
    "win_rate": 0.68,
    "total_trades": 45,
    "profit_factor": 2.1
  },
  "stock_pool": "非ST股票",
  "period": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "days": 365
  },
  "performance": {
    "initial_capital": 100000,
    "final_capital": 115600,
    "total_profit": 15600,
    "max_capital": 118000,
    "min_capital": 91500
  }
}
```

### 2. 获取回测结果

```http
GET /api/backtest/backtest_results/{backtest_id}
```

**路径参数**:

- `backtest_id` (string, required): 回测ID

**响应示例**:

```json
{
  "backtest_id": "backtest_20241231_001",
  "trades": [
    {
      "date": "2024-01-15",
      "stock_code": "000001.SZ",
      "action": "BUY",
      "price": 10.50,
      "quantity": 1000,
      "commission": 3.15,
      "total": 10503.15
    }
  ],
  "daily_returns": [
    {
      "date": "2024-01-02",
      "return": 0.015,
      "cumulative_return": 0.015,
      "capital": 101500
    }
  ],
  "risk_metrics": {
    "volatility": 0.18,
    "var_95": -0.025,
    "cvar_95": -0.035,
    "calmar_ratio": 2.22
  }
}
```

---

## 🔧 错误处理

### 错误响应格式

所有错误响应都遵循以下格式：

```json
{
  "detail": "错误描述信息"
}
```

### 常见错误码

- **400 Bad Request**: 请求参数错误
- **404 Not Found**: 资源不存在
- **500 Internal Server Error**: 服务器内部错误

### 错误示例

```json
// 股票代码不存在
{
  "detail": "股票代码 999999 不存在或无效"
}

// 日期格式错误
{
  "detail": "请求参数错误，请检查日期格式"
}

// 策略不存在
{
  "detail": "Strategy '不存在的策略' not found"
}

// 数据为空
{
  "detail": "股票 000001 在指定时间范围内没有数据"
}

// 服务连接错误
{
  "detail": "无法连接到后端服务，请检查服务是否启动"
}
```

---

## 📊 数据格式说明

### 股票数据字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| trade_date | string | 交易日期 (YYYY-MM-DD) |
| open | number | 开盘价 |
| high | number | 最高价 |
| low | number | 最低价 |
| close | number | 收盘价 |
| volume | number | 成交量 |
| amount | number | 成交额 |
| pct_chg | number | 涨跌幅 (%) |
| turn | number | 换手率 (%) |
| ma5, ma10, ma20 | number | 移动平均线 |

### 策略结果字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| code | string | 股票代码 |
| name | string | 股票名称 |
| signal | number | 信号强度 (0-100) |
| volume_ratio | number | 成交量比率 |
| rsi | number | RSI值 |
| explosion_probability | number | 爆发概率 |
| buy_signal | string | 买入信号 |
| risk_level | string | 风险等级 |

### 回测结果字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| total_return | number | 总收益率 |
| annual_return | number | 年化收益率 |
| max_drawdown | number | 最大回撤 |
| sharpe_ratio | number | 夏普比率 |
| win_rate | number | 胜率 |
| total_trades | number | 总交易次数 |

---

## 🚀 使用示例

### Python 示例

```python
import requests
import json

# 基础配置
BASE_URL = "http://localhost:8000"

# 获取股票数据
def get_stock_data(code, start_date, end_date):
    url = f"{BASE_URL}/api/stock/{code}"
    params = {
        "period": "daily",
        "start_date": start_date,
        "end_date": end_date,
        "ma_periods": [5, 10, 20]
    }
    response = requests.get(url, params=params)
    return response.json()

# 策略扫描
def scan_stocks(strategy, params):
    url = f"{BASE_URL}/api/strategy/scan"
    data = {
        "strategy": strategy,
        "params": params
    }
    response = requests.post(url, json=data)
    return response.json()

# 运行回测
def run_backtest(strategy, start_date, end_date, init_params, params=None):
    url = f"{BASE_URL}/api/backtest/run"
    request_params = {
        "strategy": strategy,
        "start_date": start_date,
        "end_date": end_date,
        "backtest_init_params": json.dumps(init_params)
    }
    if params:
        request_params["params"] = json.dumps(params)
    
    response = requests.get(url, params=request_params)
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 获取股票数据
    stock_data = get_stock_data("000001", "2024-01-01", "2024-12-31")
    print(f"获取到 {len(stock_data)} 条数据")
    
    # 策略扫描
    scan_params = {
        "stock_pool": "非ST股票",
        "signal": 70.0,
        "volume_ratio": 1.5
    }
    scan_results = scan_stocks("爆发式选股策略", scan_params)
    print(f"扫描到 {len(scan_results['results'])} 只股票")
    
    # 运行回测
    backtest_params = {
        "stock_pool": "非ST股票",
        "initial_capital": 100000,
        "max_positions": 4
    }
    backtest_results = run_backtest(
        "爆发式选股策略", 
        "2024-01-01", 
        "2024-12-31", 
        backtest_params
    )
    print(f"回测总收益率: {backtest_results['summary']['total_return']:.2%}")
```

### JavaScript 示例

```javascript
// 获取股票数据
async function getStockData(code, startDate, endDate) {
    const url = `http://localhost:8000/api/stock/${code}`;
    const params = new URLSearchParams({
        period: 'daily',
        start_date: startDate,
        end_date: endDate,
        ma_periods: [5, 10, 20]
    });
    
    const response = await fetch(`${url}?${params}`);
    return await response.json();
}

// 策略扫描
async function scanStocks(strategy, params) {
    const url = 'http://localhost:8000/api/strategy/scan';
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            strategy: strategy,
            params: params
        })
    });
    return await response.json();
}

// 使用示例
async function main() {
    try {
        // 获取股票数据
        const stockData = await getStockData('000001', '2024-01-01', '2024-12-31');
        console.log(`获取到 ${stockData.length} 条数据`);
        
        // 策略扫描
        const scanResults = await scanStocks('爆发式选股策略', {
            stock_pool: '非ST股票',
            signal: 70.0,
            volume_ratio: 1.5
        });
        console.log(`扫描到 ${scanResults.results.length} 只股票`);
        
    } catch (error) {
        console.error('API调用失败:', error);
    }
}

main();
```

---

## 📚 更多资源

- **在线API文档**: http://localhost:8000/docs
- **项目文档**: `docs/` 目录
- **使用示例**: `docs/模块使用示例/`
- **开发文档**: `docs/模块开发文档/`

---

**最后更新时间**: 2025年8月5日
**文档版本**: v2.0 