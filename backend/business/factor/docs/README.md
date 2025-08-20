# 因子研究框架 (Factor Research Framework)

## 📖 概述

因子研究框架是一个完整的量化因子开发到回测的全流程解决方案，整合了因子定义、计算、分析、回测和报告生成等核心功能。该框架基于 `@backtest_factor/` 的优秀回测系统，提供了更加完善和易用的因子研究环境。

## 🏗️ 架构设计

```
factor/
├── core/                          # 核心模块
│   ├── factor/                    # 因子定义和计算
│   │   ├── base_factor.py         # 基础因子类
│   │   ├── factor_registry.py     # 因子注册管理
│   │   └── factor_engine.py       # 因子计算引擎
│   ├── data/                      # 数据管理
│   │   ├── data_manager.py        # 数据管理器
│   │   ├── data_validator.py      # 数据验证
│   │   └── data_cleaner.py        # 数据清洗
│   ├── backtest/                  # 回测系统 (整合自backtest_factor)
│   │   ├── backtest_engine.py     # 回测引擎
│   │   ├── portfolio_manager.py   # 组合管理
│   │   └── risk_manager.py        # 风险管理
│   ├── analysis/                  # 因子分析
│   │   ├── factor_analyzer.py     # 因子有效性分析
│   │   ├── ic_analyzer.py         # IC分析
│   │   └── performance_analyzer.py # 绩效分析
│   └── reporting/                 # 报告生成
│       ├── report_generator.py    # 报告生成器
│       └── visualization.py       # 可视化
├── library/                       # 因子库 (用户主要修改的地方)
│   ├── technical_factors.py       # 技术因子
│   ├── fundamental_factors.py     # 基本面因子
│   ├── worldquant_factors.py      # WorldQuant Alpha因子
│   └── channel_factors.py         # 通道分析因子
├── configs/                       # 配置文件
├── storage/                       # 数据存储
├── utils/                         # 工具函数
├── main.py                        # 主入口文件
└── run_factor.py                  # 简化的运行文件
```

## 🚀 主要特性

### 1. 因子管理系统
- **因子注册**: 支持装饰器方式注册因子
- **因子分类**: 技术因子、基本面因子、WorldQuant因子、通道分析因子等
- **因子库**: 丰富的预定义因子库
- **自定义因子**: 支持用户自定义因子开发

### 2. 数据处理
- **数据获取**: 集成现有数据获取系统
- **数据清洗**: 自动处理异常数据和涨跌停
- **数据预处理**: 标准化、去极值、中性化等

### 3. 因子分析
- **IC分析**: 信息系数分析
- **RankIC分析**: 排序信息系数分析
- **因子有效性**: 因子稳定性、预测能力分析
- **因子相关性**: 因子间相关性分析

### 4. 回测系统
- **TopN回测**: 选择因子值最高的N只股票
- **分组回测**: 按因子值分组回测
- **多因子回测**: 多因子组合回测
- **风险控制**: 集成风险管理功能

### 5. 报告生成
- **统一报告**: 生成汇总HTML报告
- **可视化**: 丰富的图表展示
- **交互式**: 支持交互式查看

## 📦 安装依赖

```bash
pip install pandas numpy vectorbt quantstats plotly
```

## 🎯 快速开始

### 1. 基本使用

```python
from backend.business.factor import create_factor_research_framework

# 创建框架实例
framework = create_factor_research_framework()

# 运行单因子分析
results = framework.run_single_factor_analysis(
    factor_name='momentum_20d',
    start_date='2025-01-01',
    end_date='2025-08-19',
    stock_pool='hs300'
)

print(f"报告路径: {results['report_path']}")
```

### 2. 多因子对比

```python
# 运行多因子对比分析
factor_names = ['momentum_20d', 'rsi_14', 'bollinger_position']

results = framework.run_factor_comparison(
    factor_names=factor_names,
    start_date='2025-01-01',
    end_date='2025-08-19',
    stock_pool='hs300'
)
```

### 3. 自定义因子开发

```python
from backend.business.factor.core.factor.base_factor import register_technical_factor

@register_technical_factor(name='my_factor', description='我的自定义因子')
def my_custom_factor(close: pd.Series, **kwargs) -> pd.Series:
    """自定义因子计算逻辑"""
    return close.pct_change(10) * 100

# 分析自定义因子
results = framework.run_single_factor_analysis(
    factor_name='my_factor',
    start_date='2025-01-01',
    end_date='2025-08-19',
    stock_pool='hs300'
)
```

## 📊 因子库

### 技术因子
- `momentum_5d`: 5日动量因子
- `momentum_20d`: 20日动量因子
- `volatility_20d`: 20日波动率因子
- `rsi_14`: 14日RSI因子
- `bollinger_position`: 布林带位置因子
- `macd_histogram`: MACD柱状图因子
- `williams_r`: 威廉指标因子
- `cci`: 商品通道指数因子

### 基本面因子
- `pe_ratio`: 市盈率因子
- `pb_ratio`: 市净率因子
- `ps_ratio`: 市销率因子

### WorldQuant Alpha因子
- `alpha_1`: Alpha#1因子
- `alpha_8`: Alpha#8因子
- 更多Alpha因子...

### 通道分析因子
- 上升通道回归分析
- 通道突破检测
- 通道质量评估

## 🔧 高级功能

### 1. 因子注册管理

```python
from backend.business.factor.core.factor.factor_registry import factor_registry

# 查看所有因子
all_factors = factor_registry.list_factors()

# 查看特定类别因子
technical_factors = factor_registry.list_factors('technical')

# 获取因子信息
factor_info = factor_registry.get_factor_info('momentum_20d')
```

### 2. 数据管理

```python
# 获取市场数据
market_data = framework.data_manager.get_market_data(
    start_date='2025-01-01',
    end_date='2025-08-19',
    stock_pool='hs300'
)

# 获取财务数据
financial_data = framework.data_manager.get_financial_data(
    start_date='2025-01-01',
    end_date='2025-08-19'
)
```

### 3. 因子计算引擎

```python
# 批量计算因子
factor_data = framework.factor_engine.calculate_factors(['momentum_20d', 'rsi_14'])

# 因子标准化
standardized_factors = framework.factor_engine.standardize_factors(['momentum_20d'])

# 因子去极值
winsorized_factors = framework.factor_engine.winsorize_factors(['momentum_20d'])
```

### 4. 回测引擎

```python
# TopN回测
topn_result = framework.backtest_engine.run_topn_backtest(
    factor_name='momentum_20d',
    n=10
)

# 分组回测
group_result = framework.backtest_engine.run_group_backtest(
    factor_name='momentum_20d',
    n_groups=5
)

# 多因子回测
multifactor_result = framework.backtest_engine.run_multifactor_backtest(
    factor_names=['momentum_20d', 'rsi_14'],
    weights=[0.6, 0.4]
)
```

## 📈 报告系统

框架生成统一的汇总HTML报告，包含：

- **📊 分析总览**: 分析概况和基本信息
- **📈 TopN回测结果**: 各因子TopN策略的详细表现
- **📊 分组回测结果**: 因子分组回测的详细分析
- **🔗 多因子回测结果**: 多因子组合策略的回测结果
- **📊 IC分析结果**: 因子IC值和有效性指标
- **📈 有效性分析结果**: 因子有效性详细分析

## 🎨 报告特点

- **统一界面**: 所有分析结果集中在一个HTML文件中
- **导航菜单**: 顶部固定导航，快速跳转到各个部分
- **响应式设计**: 支持桌面和移动设备查看
- **美观样式**: 现代化的UI设计，清晰的数据展示
- **颜色编码**: 正负值用不同颜色区分，便于快速识别

## 🔍 示例代码

完整的使用示例请参考 `example_usage.py` 文件：

```bash
python example_usage.py
```

## 📝 开发指南

### 1. 添加新因子

```python
from backend.business.factor.core.factor.base_factor import register_technical_factor

@register_technical_factor(name='new_factor', description='新因子描述')
def new_factor(close: pd.Series, **kwargs) -> pd.Series:
    """
    新因子计算逻辑
    
    Args:
        close: 收盘价序列
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    # 因子计算逻辑
    return close.pct_change(5)
```

### 2. 因子参数配置

```python
# 在因子函数中使用参数
def factor_with_params(close: pd.Series, window: int = 20, **kwargs) -> pd.Series:
    return close.rolling(window).mean()
```

### 3. 数据字段映射

框架会自动将数据字段映射到因子参数：
- `close` → 收盘价
- `open` → 开盘价
- `high` → 最高价
- `low` → 最低价
- `volume` → 成交量
- `pct_chg` → 涨跌幅
- `pe_ttm` → 市盈率
- `pb_mrq` → 市净率
- 等等...

## 🚨 注意事项

1. **数据质量**: 确保输入数据的质量和完整性
2. **因子稳定性**: 注意因子的稳定性和预测能力
3. **过拟合风险**: 避免过度优化历史数据
4. **交易成本**: 考虑实际交易中的成本和滑点
5. **风险控制**: 合理设置风险控制参数

## 🔄 更新日志

### v2.0.0 (2025-08-20)
- **重构**: 完全重构因子研究框架架构
- **整合**: 整合backtest_factor的优秀回测系统
- **新增**: 因子注册管理系统
- **新增**: 统一的汇总报告系统
- **优化**: 改进数据管理和因子计算流程
- **增强**: 支持更多因子类型和分析功能

### v1.0.0 (2025-08-19)
- **初始版本**: 基础因子研究功能

## 📞 支持

如有问题或建议，请联系开发团队。
