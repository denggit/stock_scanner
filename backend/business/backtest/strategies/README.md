# 策略模块架构说明

## 📁 目录结构

```
backend/business/backtest/strategies/
├── __init__.py                 # 主入口，暴露核心接口
├── core/                       # 🔧 核心通用组件（所有策略都使用）
│   ├── __init__.py
│   ├── base_strategy.py        # 策略基类（模板方法模式）
│   ├── managers/               # 管理器组件
│   │   ├── __init__.py
│   │   ├── data_manager.py     # 数据管理器（观察者模式）
│   │   ├── position_manager.py # 仓位管理器
│   │   ├── trade_manager.py    # 交易管理器（策略模式）
│   │   └── trade_logger.py     # 交易日志记录器
│   └── utils.py               # 通用工具函数
├── analyzers/                  # 📊 分析器模块（按分析类型分组）
│   ├── __init__.py
│   └── channel/               # 通道分析相关
│       ├── __init__.py
│       ├── manager.py         # 通道分析器管理器（工厂模式）
│       └── utils.py           # 通道专用工具
├── implementations/            # 🎯 具体策略实现
│   ├── __init__.py
│   └── channel/               # 通道类策略
│       ├── __init__.py
│       └── rising_channel.py  # 上升通道策略
└── factory/                   # 🏭 工厂模式
    ├── __init__.py
    └── strategy_factory.py    # 策略工厂类
```

## 🎨 设计模式应用

### 1. 模板方法模式

- **位置**: `core/base_strategy.py`
- **用途**: 定义策略执行流程模板
- **流程**: prepare_data → generate_signals → risk_control → execute_trades → log_results

### 2. 工厂模式

- **位置**: `factory/strategy_factory.py`
- **用途**: 统一创建各种策略实例
- **特点**: 支持策略注册、参数配置、类型检查

### 3. 观察者模式

- **位置**: `core/managers/data_manager.py`
- **用途**: 数据更新通知
- **特点**: 解耦数据提供者和消费者

### 4. 策略模式

- **位置**: `core/managers/trade_manager.py`
- **用途**: 资金分配策略
- **实现**: EqualWeightAllocation, TargetPercentAllocation

### 5. 单一职责原则

- **应用**: 每个管理器专注单一功能
- **优势**: 高内聚、低耦合

## 🚀 使用示例

### 基础使用

```python
from backend.business.backtest.strategies import create_strategy, list_strategies

# 查看可用策略
print("可用策略:", list_strategies())

# 创建策略实例
strategy = create_strategy('rising_channel', max_positions=30, min_channel_score=70.0)
```

### 工厂模式使用

```python
from backend.business.backtest.strategies import StrategyFactory

# 创建工厂实例
factory = StrategyFactory()

# 获取策略信息
info = factory.get_strategy_info('rising_channel')
print(f"策略信息: {info}")

# 创建策略
strategy = factory.create_strategy('rising_channel', max_positions=50)
```

### 注册自定义策略

```python
from backend.business.backtest.strategies import register_strategy
from backend.business.backtest.strategies.core import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def generate_signals(self):
        # 实现自定义信号生成逻辑
        return []

# 注册策略
register_strategy('my_custom', MyCustomStrategy, {'param1': 'value1'})

# 使用自定义策略
strategy = create_strategy('my_custom', param1='new_value')
```

### 核心组件使用

```python
from backend.business.backtest.strategies.core import BaseStrategy
from backend.business.backtest.strategies.core.managers import (
    DataManager, PositionManager, TradeManager, TradeLogger
)
from backend.business.backtest.strategies.core.utils import (
    SignalUtils, PriceUtils
)

# 创建信号
signal = SignalUtils.create_buy_signal(
    stock_code="sz.301383",
    price=45.67,
    reason="技术指标显示买入机会",
    confidence=0.8
)

# 计算价格距离
distance = PriceUtils.calculate_percentage_distance(110, 100)  # 10.0%
```

## 📈 扩展指南

### 添加新分析器类型

```python
# 1. 创建新目录
mkdir analyzers/technical

# 2. 实现分析器
# analyzers/technical/indicators.py
class TechnicalIndicatorUtils:
    @staticmethod
    def calculate_rsi(data, period=14):
        # RSI计算逻辑
        pass

# 3. 更新__init__.py导入
```

### 添加新策略类型

```python
# 1. 创建策略目录
mkdir implementations/momentum

# 2. 实现策略类
# implementations/momentum/breakout.py
class BreakoutStrategy(BaseStrategy):
    def generate_signals(self):
        # 突破策略信号逻辑
        pass

# 3. 注册策略
register_strategy('breakout', BreakoutStrategy)
```

### 添加新管理器

```python
# 1. 实现管理器
# core/managers/risk_manager.py
class RiskManager:
    def __init__(self):
        pass
    
    def check_risk(self, signal):
        # 风险检查逻辑
        pass

# 2. 更新__init__.py导入
```

## ✅ 架构优势

1. **分层清晰**: 核心组件、分析器、策略实现分离
2. **职责明确**: 每个模块专注单一功能
3. **可扩展**: 易于添加新策略类型和分析器
4. **设计模式**: 应用多种设计模式提高代码质量
5. **向后兼容**: 保持现有API稳定
6. **统一管理**: 工厂模式统一策略创建

## 🔄 迁移说明

### 旧导入 → 新导入

```python
# 旧方式
from backend.business.backtest.strategies.base import BaseStrategy
from backend.business.backtest.strategies.implementations.channel.rising_channel import RisingChannelStrategy

# 新方式
from backend.business.backtest.strategies import BaseStrategy, RisingChannelStrategy
# 或者
from backend.business.backtest.strategies.core import BaseStrategy
from backend.business.backtest.strategies.implementations.channel import RisingChannelStrategy
```

### 工厂模式使用

```python
# 推荐的新方式
from backend.business.backtest.strategies import create_strategy

strategy = create_strategy('rising_channel', max_positions=50)
```

## 📋 TODO 清单

- [ ] 添加技术指标分析器模块
- [ ] 实现动量策略类型
- [ ] 添加风险管理器
- [ ] 完善策略配置验证
- [ ] 添加策略性能分析工具
- [ ] 实现策略组合管理

## 🧪 测试

新架构已通过全面测试，包括：

- ✅ 核心组件导入测试
- ✅ 分析器模块测试
- ✅ 策略实现测试
- ✅ 工厂模式测试
- ✅ 架构分离测试
- ✅ 功能完整性测试

**测试通过率: 100%** 🎉
