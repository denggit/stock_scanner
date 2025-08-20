# VectorBT框架使用说明

## 概述

VectorBT框架是基于`vectorbt`库的高性能向量化回测框架，相比传统的`backtrader`框架，具有以下优势：

- **高性能**：基于向量化计算，回测速度显著提升
- **易用性**：简洁的API设计，易于理解和使用
- **功能丰富**：内置多种技术指标和性能分析工具
- **可视化**：强大的图表和报告生成功能

## 框架结构

```
backend/business/backtest_vectorbt/
├── __init__.py                 # 框架入口
├── core/                       # 核心模块
│   ├── __init__.py
│   ├── base_strategy.py        # 策略基类
│   ├── backtest_engine.py      # 回测引擎
│   ├── data_manager.py         # 数据管理器
│   └── result_analyzer.py      # 结果分析器
├── strategies/                 # 策略实现
│   ├── __init__.py
│   └── rising_channel_strategy.py  # 上升通道策略
├── configs/                    # 配置文件
│   ├── __init__.py
│   └── rising_channel_config.py    # 上升通道策略配置
├── execution/                  # 执行模块
│   ├── __init__.py
│   └── rising_channel_backtest.py  # 上升通道策略执行器
└── utils/                      # 工具模块
    ├── __init__.py
    ├── data_utils.py           # 数据工具
    └── report_utils.py         # 报告工具
```

## 快速开始

### 1. 基本使用

```python
from backend.business.backtest_vectorbt import (
    BacktestEngine, 
    RisingChannelStrategy, 
    RisingChannelConfig
)

# 创建回测引擎
engine = BacktestEngine(initial_cash=100000.0, commission=0.0003)

# 运行单策略回测
results = engine.run_single_strategy(
    data=stock_data,
    strategy_class=RisingChannelStrategy,
    strategy_params={'max_positions': 10},
    strategy_name="上升通道策略"
)

# 查看结果
print(f"总收益率: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

### 2. 多策略对比

```python
# 定义多个策略
strategies = [
    {
        'name': '保守策略',
        'class': RisingChannelStrategy,
        'params': {'max_positions': 5, 'min_channel_score': 70}
    },
    {
        'name': '激进策略', 
        'class': RisingChannelStrategy,
        'params': {'max_positions': 20, 'min_channel_score': 50}
    }
]

# 运行多策略对比
results = engine.run_multi_strategy(data=stock_data, strategies=strategies)

# 分析对比结果
for name, result in results.items():
    print(f"{name}: 收益率 {result['total_return']:.2%}")
```

### 3. 参数优化

```python
# 定义参数范围
parameter_ranges = {
    'max_positions': [5, 10, 15, 20],
    'min_channel_score': [50, 60, 70, 80],
    'breakout_pullback_threshold': [2.0, 3.0, 4.0, 5.0]
}

# 运行参数优化
optimization_results = engine.optimize_parameters(
    data=stock_data,
    strategy_class=RisingChannelStrategy,
    parameter_ranges=parameter_ranges,
    optimization_target="total_return"
)

# 查看最佳参数
best_params = optimization_results['best_params']
best_result = optimization_results['best_result']
print(f"最佳参数: {best_params}")
print(f"最佳收益率: {best_result['total_return']:.2%}")
```

## 策略开发

### 1. 创建自定义策略

```python
from backend.business.backtest_vectorbt.core.base_strategy import BaseStrategy
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    """自定义策略示例"""
    
    def __init__(self):
        super().__init__(
            name="自定义策略",
            description="这是一个自定义策略示例"
        )
        self._params = {
            'ma_short': 5,
            'ma_long': 20
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        signals = pd.DataFrame(0, index=data.index, columns=['signal'])
        
        # 计算移动平均线
        ma_short = data['close'].rolling(window=self._params['ma_short']).mean()
        ma_long = data['close'].rolling(window=self._params['ma_long']).mean()
        
        # 生成信号
        for i in range(self._params['ma_long'], len(data)):
            if ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]:
                signals.loc[signals.index[i], 'signal'] = 1  # 买入
            elif ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]:
                signals.loc[signals.index[i], 'signal'] = -1  # 卖出
        
        return signals
```

### 2. 运行自定义策略

```python
# 创建策略实例
strategy = MyCustomStrategy()

# 设置参数
strategy.set_parameters({'ma_short': 10, 'ma_long': 30})

# 运行回测
results = strategy.run_backtest(
    data=stock_data,
    initial_cash=100000.0,
    commission=0.0003
)
```

## 数据处理

### 1. 数据格式要求

VectorBT框架要求数据包含以下列：
- `open`: 开盘价
- `high`: 最高价  
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

数据索引应为日期时间格式。

### 2. 数据预处理

```python
from backend.business.backtest_vectorbt.core.data_manager import DataManager

# 创建数据管理器
data_manager = DataManager()

# 加载和预处理数据
processed_data = data_manager.load_data(raw_data, name="股票数据")

# 验证数据质量
validation_result = data_manager.validate_data_quality(processed_data)
```

### 3. 使用项目数据源

```python
from backend.business.data.data_fetcher import StockDataFetcher
import datetime

# 获取股票数据
fetcher = StockDataFetcher()
start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
stock_data = fetcher.fetch_stock_data('sz.301383', start_date=start_date)
```

## 结果分析

### 1. 基础指标

```python
from backend.business.backtest_vectorbt.core.result_analyzer import ResultAnalyzer

# 创建结果分析器
analyzer = ResultAnalyzer()

# 分析结果
analysis = analyzer.analyze_results(results)

# 查看各项指标
print(f"基础指标: {analysis['basic_metrics']}")
print(f"风险指标: {analysis['risk_metrics']}")
print(f"交易指标: {analysis['trade_metrics']}")
```

### 2. 生成报告

```python
from backend.business.backtest_vectorbt.utils.report_utils import ReportUtils

# 创建报告工具
report_utils = ReportUtils()

# 生成Excel报告
report_utils.save_results_to_excel(
    results=results,
    filename="回测报告.xlsx",
    strategy_name="上升通道策略"
)

# 生成综合报告
comprehensive_report = report_utils.create_comprehensive_report(
    results=results,
    strategy_name="上升通道策略"
)
```

### 3. 可视化分析

```python
# 绘制权益曲线
analyzer.plot_equity_curve(results)

# 绘制回撤曲线
analyzer.plot_drawdown(results)

# 绘制交易分布
analyzer.plot_trade_distribution(results)
```

## 配置管理

### 1. 策略配置

```python
from backend.business.backtest_vectorbt.configs.rising_channel_config import RisingChannelConfig

# 获取配置
config = RisingChannelConfig()

# 查看配置参数
print(config.get_base_config())
print(config.get_strategy_params())
```

### 2. 环境配置

```python
# 开发环境配置
dev_config = config.get_development_config()

# 生产环境配置
prod_config = config.get_production_config()

# 测试环境配置
test_config = config.get_test_config()
```

## 性能优化

### 1. 数据缓存

```python
# 启用数据缓存
data_manager.enable_cache(True)

# 设置缓存目录
data_manager.set_cache_directory("/path/to/cache")
```

### 2. 并行计算

```python
# 启用并行优化
engine.enable_parallel_optimization(True)

# 设置并行进程数
engine.set_parallel_workers(4)
```

## 常见问题

### 1. 数据格式错误

**问题**: 数据列名不匹配或数据类型错误
**解决**: 确保数据包含必需的OHLCV列，且为数值类型

### 2. 信号生成错误

**问题**: 策略生成的信号格式不正确
**解决**: 确保信号DataFrame的值为1（买入）、-1（卖出）、0（持有）

### 3. 内存不足

**问题**: 处理大量数据时内存不足
**解决**: 使用数据分块处理或减少数据量

### 4. 性能问题

**问题**: 回测速度慢
**解决**: 使用数据缓存、减少参数组合数量、启用并行计算

## 最佳实践

1. **数据质量**: 确保输入数据的质量和完整性
2. **参数验证**: 在策略中验证参数的有效性
3. **错误处理**: 添加适当的异常处理机制
4. **日志记录**: 使用日志记录关键操作和错误信息
5. **测试验证**: 对策略进行充分的测试和验证
6. **文档维护**: 保持代码和文档的同步更新

## 迁移指南

### 从Backtrader迁移

1. **策略逻辑**: 将`next()`方法转换为`generate_signals()`方法
2. **数据格式**: 确保数据格式符合VectorBT要求
3. **信号生成**: 使用向量化方式生成信号
4. **结果分析**: 使用VectorBT的结果分析工具

### 性能对比

| 指标 | Backtrader | VectorBT | 提升 |
|------|------------|----------|------|
| 回测速度 | 基准 | 10-100x | 显著提升 |
| 内存使用 | 基准 | 50-80% | 减少 |
| 代码复杂度 | 基准 | 30-50% | 简化 |
| 功能丰富度 | 基准 | 200%+ | 大幅增加 |

## 总结

VectorBT框架提供了高性能、易用的回测解决方案，特别适合需要快速迭代和大量参数优化的量化交易策略开发。通过合理使用框架提供的功能，可以显著提升策略开发和回测的效率。
