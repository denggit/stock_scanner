# 股票筛选系统 - API接口文档

## 📋 概述

- **基础URL**: `http://localhost:8000`
- **API文档**: `http://localhost:8000/docs`
- **认证**: 目前无需认证

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

**参数**:
- `code` (path): 股票代码，如 `000001.SZ`
- `period` (query): 周期，可选值 `daily`, `weekly`, `monthly`
- `start_date` (query): 开始日期，格式 `YYYY-MM-DD`
- `end_date` (query): 结束日期，格式 `YYYY-MM-DD`
- `ma_periods` (query): 移动平均线周期列表

**请求示例**:
```bash
curl "http://localhost:8000/api/stock/000001.SZ?period=daily&start_date=2024-01-01&end_date=2024-12-31"
```

**响应示例**:
```json
{
  "code": "000001.SZ",
  "data": [
    {
      "trade_date": "2024-01-02",
      "open": 10.50,
      "high": 10.80,
      "low": 10.30,
      "close": 10.75,
      "volume": 1000000,
      "amount": 10750000,
      "pct_chg": 2.38,
      "ma5": 10.60,
      "ma10": 10.45,
      "ma20": 10.30
    }
  ]
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
    "signal": 70.0,
    "volume_ratio": 1.5,
    "rsi_range": [45, 65],
    "explosion_probability": 0.5,
    "stock_pool": "非ST股票",
    "ipo_date": "2023-01-01",
    "min_amount": 100000000
  }
}
```

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
    }
  ]
}
```

---

## 📈 回测接口

### 1. 运行回测
```http
POST /api/backtest/run
```

**请求体**:
```json
{
  "strategy": "爆发式选股策略",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "backtest_init_params": {
    "stock_pool": "非ST股票",
    "initial_capital": 100000,
    "max_positions": 4,
    "allocation_strategy": "信号强度加权"
  },
  "params": {
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

**响应示例**:
```json
{
  "backtest_id": "bt_20250204_001",
  "summary": {
    "total_return": 25.6,
    "annual_return": 28.3,
    "max_drawdown": -8.5,
    "sharpe_ratio": 1.85,
    "win_rate": 0.68,
    "total_trades": 156,
    "profit_trades": 106,
    "loss_trades": 50
  },
  "equity_curve": [
    {
      "date": "2024-01-02",
      "equity": 100000,
      "drawdown": 0
    }
  ],
  "trades": [
    {
      "date": "2024-01-02",
      "code": "000001.SZ",
      "action": "buy",
      "price": 10.75,
      "quantity": 1000,
      "value": 10750
    }
  ],
  "positions": [
    {
      "date": "2024-01-02",
      "holdings": [
        {
          "code": "000001.SZ",
          "quantity": 1000,
          "cost": 10.75,
          "current_price": 10.80,
          "unrealized_pnl": 50
        }
      ]
    }
  ],
  "performance_metrics": {
    "calmar_ratio": 3.33,
    "sortino_ratio": 2.15,
    "information_ratio": 1.45,
    "beta": 0.85,
    "alpha": 0.12
  }
}
```

### 2. 获取回测结果
```http
GET /api/backtest/backtest_results/{backtest_id}
```

**参数**:
- `backtest_id` (path): 回测ID

**响应示例**:
```json
{
  "backtest_id": "bt_20250204_001",
  "status": "completed",
  "results": {
    // 同上面的回测结果
  }
}
```

---

## 🔧 系统监控接口

### 1. 获取系统健康状态
```http
GET /api/system/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2025-02-04T10:30:00",
  "checks": {
    "cpu": {
      "status": "healthy",
      "value": 45.2
    },
    "memory": {
      "status": "healthy",
      "value": 62.8
    },
    "disk": {
      "status": "healthy",
      "value": 78.5
    },
    "database": {
      "status": "healthy",
      "value": 0.15
    },
    "cache": {
      "status": "healthy",
      "value": 0.85
    }
  },
  "message": "系统运行正常"
}
```

### 2. 获取性能指标
```http
GET /api/system/metrics
```

**响应示例**:
```json
{
  "timestamp": "2025-02-04T10:30:00",
  "cpu_percent": 45.2,
  "memory_percent": 62.8,
  "memory_used_mb": 8192,
  "disk_usage_percent": 78.5,
  "network_io": {
    "bytes_sent": 1024000,
    "bytes_recv": 2048000,
    "packets_sent": 1000,
    "packets_recv": 2000
  },
  "active_connections": 25,
  "cache_hit_rate": 0.85,
  "cache_memory_usage": {
    "memory_cache_size": 1500,
    "redis_keys": 500
  },
  "api_response_time": 0.25,
  "database_query_time": 0.15,
  "strategy_execution_time": 0.45
}
```

### 3. 获取统计信息
```http
GET /api/system/stats
```

**响应示例**:
```json
{
  "api_calls": {
    "get_stock_data": 1250,
    "strategy_scan": 89,
    "run_backtest": 23
  },
  "db_queries": {
    "get_stock_daily": 1500,
    "get_stock_list": 45,
    "save_financial_data": 12
  },
  "strategy_executions": {
    "爆发式选股策略": 89,
    "均线回踩策略": 45,
    "波段交易策略": 67
  },
  "metrics_count": 1000,
  "health_count": 100
}
```

---

## 📊 数据管理接口

### 1. 获取股票列表
```http
GET /api/data/stocks
```

**参数**:
- `pool` (query): 股票池类型，可选值 `full`, `no_st`, `sz50`, `hs300`, `zz500`

**响应示例**:
```json
{
  "stocks": [
    {
      "code": "000001.SZ",
      "name": "平安银行",
      "ipo_date": "1991-04-03",
      "type": "1",
      "status": "1"
    }
  ],
  "total_count": 5000
}
```

### 2. 更新数据状态
```http
GET /api/data/update_status
```

**响应示例**:
```json
{
  "stock_list": {
    "last_update": "2025-02-04T09:00:00",
    "total_stocks": 5000
  },
  "daily_data": {
    "last_update": "2025-02-04T15:30:00",
    "updated_stocks": 4500
  },
  "financial_data": {
    "last_update": "2025-02-04T16:00:00",
    "updated_stocks": 3000
  }
}
```

---

## 🛠️ 缓存管理接口

### 1. 获取缓存状态
```http
GET /api/cache/status
```

**响应示例**:
```json
{
  "memory_cache_size": 1500,
  "redis_keys": 500,
  "redis_available": true,
  "cache_hit_rate": 0.85
}
```

### 2. 清理缓存
```http
POST /api/cache/clear
```

**请求体**:
```json
{
  "pattern": "stock_data_*"  // 可选，清理特定模式的缓存
}
```

**响应示例**:
```json
{
  "cleared_keys": 150,
  "message": "缓存清理成功"
}
```

---

## 📝 错误处理

### 错误响应格式
```json
{
  "detail": "错误描述信息",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-02-04T10:30:00"
}
```

### 常见错误码
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误
- `503`: 服务不可用

### 错误示例
```json
{
  "detail": "股票代码不存在: 999999.SZ",
  "error_code": "STOCK_NOT_FOUND",
  "timestamp": "2025-02-04T10:30:00"
}
```

---

## 🔐 安全说明

### 当前状态
- 无需认证
- 无访问限制
- 建议在生产环境中添加认证机制

### 建议的安全措施
1. 添加API密钥认证
2. 实现请求频率限制
3. 添加IP白名单
4. 使用HTTPS

---

## 📞 技术支持

### 联系方式
- 查看API文档: `http://localhost:8000/docs`
- 查看日志文件: `logs/app.log`
- 系统监控: `http://localhost:8000/api/system/health`

### 常见问题
1. **连接超时**: 检查服务是否启动
2. **数据为空**: 检查数据源连接
3. **策略执行失败**: 检查参数配置
4. **回测结果异常**: 检查日期范围

---

*API接口文档 - 2025年2月4日* 