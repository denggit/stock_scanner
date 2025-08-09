# 回测HTML结果图形

本文档说明如何使用 `HtmlResultPlotter` 生成策略与基线对比的累计收益曲线（HTML自包含）。

## 模块位置
- `backend/business/backtest/utils/html_result_plotter.py`

## 设计原则
- 解耦：不依赖回测引擎内部结构，只需要日期与策略累计收益。
- 复用：基线累计收益计算复用 `DataUtils.calculate_returns`，避免重复代码。
- 单一职责：仅负责数据对齐、曲线绘制与HTML导出。

## 使用方式

### 1. 直接传入日期与收益率
```python
from backend.business.backtest.utils.html_result_plotter import HtmlResultPlotter

plotter = HtmlResultPlotter()
html_path = plotter.generate_performance_html(
    dates=dates,                       # List[pd.Timestamp|str]
    strategy_cum_returns=returns_pct,  # List[float]，单位：百分比
    start_date="2020-01-01",
    end_date="2025-08-08",
    baseline_code="sh.000001",        # 基准上证指数
    output_dir="results",             # 输出目录
    title_prefix="上升通道回测"
)
print("HTML输出:", html_path)
```

### 2. 从标准回测结果结构生成
要求 `results['summary']` 中包含：
- `dates`: 日期序列
- `returns`: 策略累计收益率(%)序列

```python
plotter = HtmlResultPlotter()
html_path = plotter.generate_from_results(
    results=results,
    start_date="2020-01-01",
    end_date="2025-08-08",
    baseline_code="sh.000001",
    output_dir="results",
    title_prefix="上升通道回测"
)
```

## 输出说明
- 文件命名：`results/backtest_<timestamp>.html`
- 自包含HTML（使用CDN加载plotly.js）。
- 曲线：
  - 蓝色：策略累计收益(%)
  - 红色：基准累计收益(%)（来自数据库的 `baseline_code` 收盘价计算）

## 注意事项
- 模块内部使用 `StockDataFetcher.fetch_stock_data` 读取基线数据，需确保数据库中存在该指数代码。
- 若基线数据在某些日期缺失，曲线会采用前向填充以对齐策略日期序列。 