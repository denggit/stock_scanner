# 因子回测框架

基于 `vectorbt` 的完整因子研究到策略落地流程框架，提供从数据准备到报告生成的一站式解决方案。

## 功能特性

### 🚀 核心功能
- **数据管理**: 自动获取市场行情和财务数据
- **因子计算**: 支持自定义因子函数，批量计算所有股票因子值
- **因子预处理**: 标准化、去极值、中性化等预处理功能
- **因子有效性验证**: IC/RankIC、因子稳定性、相关性分析
- **横截面回测**: TopN策略、分组策略、多因子组合策略
- **报告生成**: 生成统一的汇总HTML报告，包含所有分析结果

### 📊 支持的因子类型
- **技术指标因子**: 动量、波动率、成交量、RSI、布林带等
- **基本面因子**: PE、PB、PS、ROE等财务指标
- **自定义因子**: 支持任意复杂的因子计算逻辑
- **多因子组合**: 支持多因子加权组合策略

### 🔧 技术特点
- **模块化设计**: 各组件解耦，易于扩展和维护
- **高性能**: 基于 `vectorbt` 和 `pandas` 的高效计算
- **易用性**: 只需编写因子函数，自动完成全流程
- **可扩展**: 支持自定义数据源、因子类型、回测策略

## 快速开始

### 1. 安装依赖

```bash
pip install vectorbt pandas numpy matplotlib seaborn quantstats plotly
```

### 2. 基本使用

```python
from backend.business.backtest_factor import FactorFramework

# 创建框架实例
framework = FactorFramework()

# 定义自定义因子
def my_momentum_factor(close: pd.Series, **kwargs) -> pd.Series:
    """动量因子：过去20日收益率"""
    return close.pct_change(20)

# 注册因子
framework.register_custom_factor('momentum', '动量因子', my_momentum_factor)

# 运行分析 - 使用沪深300股票池
results = framework.run_single_factor_analysis(
    factor_name='momentum',
    start_date='2024-01-01',
    end_date='2024-12-31',
    stock_pool='hs300',  # 指定股票池
    top_n=10,
    n_groups=5
)

# 查看结果
print(results['effectiveness_results'])
print(results['backtest_results'])
print(results['report_paths'])
```

### 3. 多因子分析

```python
# 定义多个因子
def volatility_factor(close: pd.Series, **kwargs) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(20).std()

def volume_factor(volume: pd.Series, **kwargs) -> pd.Series:
    return volume.rolling(5).mean() / volume.rolling(20).mean()

# 注册因子
framework.register_custom_factor('volatility', '波动率因子', volatility_factor)
framework.register_custom_factor('volume', '成交量因子', volume_factor)

# 运行多因子分析 - 使用中证500股票池
results = framework.run_factor_comparison(
    factor_names=['momentum', 'volatility', 'volume'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    stock_pool='zz500',  # 指定股票池
    weights=[0.4, 0.3, 0.3]  # 多因子权重
)
```

## 详细使用指南

### 因子函数定义

因子函数需要遵循以下规范：

```python
def factor_function(data_series: pd.Series, **kwargs) -> pd.Series:
    """
    因子函数模板
    
    Args:
        data_series: 输入数据序列（如收盘价、成交量等）
        **kwargs: 其他参数
        
    Returns:
        因子值序列
    """
    # 因子计算逻辑
    factor_values = your_calculation_logic(data_series)
    return factor_values
```

### 支持的输入数据

框架自动提供以下数据字段：

- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `preclose`: 前收盘价
- `volume`: 成交量
- `amount`: 成交额
- `turn`: 换手率
- `pct_chg`: 涨跌幅（百分比）
- `returns`: 收益率（小数形式）
- `log_returns`: 对数收益率
- `pe_ttm`: 市盈率（TTM）
- `pb_mrq`: 市净率
- `ps_ttm`: 市销率
- `pcf_ncf_ttm`: 市现率
- `is_st`: ST标记
- `vwap`: 成交量加权平均价格
- `tradestatus`: 交易状态

### 支持的股票池

框架支持以下预定义股票池：

- `"full"` 或 `"全量股票"`: 全部股票
- `"no_st"` 或 `"非ST股票"`: 非ST股票
- `"st"`: ST股票
- `"sz50"` 或 `"上证50"`: 上证50成分股
- `"hs300"` 或 `"沪深300"`: 沪深300成分股
- `"zz500"` 或 `"中证500"`: 中证500成分股

#### 股票池过滤条件

可以对股票池添加额外的过滤条件：

```python
# 获取满足条件的股票池
stock_pool = framework.data_manager.get_stock_pool(
    pool_name='hs300',
    ipo_date='2020-01-01',  # 2020年前上市的股票
    min_amount=100000000,   # 日均成交额不低于1亿
    end_date='2024-12-31'
)
```

### 因子注册

```python
# 方法1：使用装饰器
@BaseFactor.register_factor(name='my_factor', description='我的因子')
def my_factor(close: pd.Series) -> pd.Series:
    return close.pct_change(10)

# 方法2：使用框架注册
framework.register_custom_factor(my_factor, 'my_factor', '我的因子')
```

### 回测策略

#### TopN策略
选择因子值最高的N只股票等权重持有：

```python
results = framework.run_single_factor_analysis(
    factor_name='momentum',
    top_n=10,  # 选择前10只股票
    rebalance_freq='1d'  # 每日调仓
)
```

#### 分组策略
按因子值将股票分为N组，分析各组表现：

```python
results = framework.run_single_factor_analysis(
    factor_name='momentum',
    n_groups=5,  # 分为5组
    rebalance_freq='1d'
)
```

#### 多因子组合策略
将多个因子加权组合：

```python
results = framework.run_factor_comparison(
    factor_names=['momentum', 'volatility', 'volume'],
    weights=[0.4, 0.3, 0.3],  # 权重分配
    top_n=10
)
```

### 因子有效性分析

框架自动计算以下指标：

- **IC (Information Coefficient)**: 因子值与未来收益的相关系数
- **RankIC**: 因子排名与未来收益排名的相关系数
- **IC衰减**: 不同持有期的IC变化
- **因子稳定性**: 因子值的时序稳定性
- **因子相关性**: 多因子间的相关性分析

### 报告生成

框架生成**统一的汇总报告**，将所有分析结果整合到一个HTML文件中：

#### 📊 汇总报告内容

- **📊 分析总览**: 分析概况和基本信息
- **📈 TopN回测结果**: 各因子TopN策略的详细表现
- **📊 分组回测结果**: 因子分组回测的详细分析
- **🔗 多因子回测结果**: 多因子组合策略的回测结果
- **📊 IC分析结果**: 因子IC值和有效性指标
- **📈 有效性分析结果**: 因子有效性详细分析

#### 🎨 报告特点

- **统一界面**: 所有分析结果集中在一个HTML文件中
- **导航菜单**: 顶部固定导航，快速跳转到各个部分
- **响应式设计**: 支持桌面和移动设备查看
- **美观样式**: 现代化的UI设计，清晰的数据展示
- **颜色编码**: 正负值用不同颜色区分，便于快速识别
- **数据表格**: 详细的统计指标表格，包含收益率、夏普比率、最大回撤等

## 高级功能

### 因子预处理

```python
# 标准化
framework.factor_engine.standardize_factors(method='zscore')

# 去极值
framework.factor_engine.winsorize_factors(limits=(0.01, 0.99))

# 中性化
framework.factor_engine.neutralize_factors(style_factors=['size', 'value'])
```

### 自定义数据源

```python
# 扩展数据管理器
class CustomDataManager(FactorDataManager):
    def get_custom_data(self, start_date, end_date):
        # 实现自定义数据获取逻辑
        pass
```

### 自定义回测策略

```python
# 扩展回测引擎
class CustomBacktestEngine(FactorBacktestEngine):
    def run_custom_strategy(self, factor_name, **kwargs):
        # 实现自定义回测策略
        pass
```

## 示例代码

### 完整示例

```python
from backend.business.backtest_factor import FactorFramework
import pandas as pd

# 创建框架
framework = FactorFramework()

# 定义Alpha#8因子
def alpha_8_factor(open_price: pd.Series, pct_chg: pd.Series, **kwargs) -> pd.Series:
    """Alpha#8因子"""
    sum_open = open_price.rolling(5).sum()
    sum_returns = pct_chg.rolling(5).sum()
    product = sum_open * sum_returns
    delay_product = product.shift(10)
    return -1 * (product - delay_product).rank(pct=True)

# 注册因子
framework.register_custom_factor(alpha_8_factor, 'alpha_8', 'Alpha#8因子')

# 运行完整分析
results = framework.run_complete_analysis(
    factor_names=['alpha_8'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    top_n=10,
    n_groups=5
)

# 查看结果
print("因子有效性:", results['effectiveness_results'])
print("回测结果:", results['backtest_results'])
print("报告路径:", results['report_paths'])

# 获取性能摘要
summary = framework.get_performance_summary('alpha_8')
print("性能摘要:", summary)
```

## 输出结果说明

### 因子有效性结果

```python
{
    'ic_alpha_8_pearson': {
        'ic_series': pd.Series,  # IC时间序列
        'ic_stats': pd.Series,   # IC统计指标
        'ic_decay': pd.Series    # IC衰减
    },
    'effectiveness_alpha_8': {
        'stability': float,      # 因子稳定性
        'turnover': float,       # 因子换手率
        'correlation': pd.DataFrame  # 因子相关性矩阵
    }
}
```

### 回测结果

```python
{
    'topn_alpha_8': {
        'portfolio': vectorbt.Portfolio,  # 回测组合对象
        'stats': dict,                    # 统计指标
        'signals': dict                   # 交易信号
    },
    'group_alpha_8': {
        'portfolios': dict,               # 各组组合
        'stats': pd.DataFrame,            # 各组统计
        'signals': dict                   # 分组信号
    }
}
```

### 报告文件

- `comprehensive_report_*.html`: 汇总分析报告（统一HTML格式）

**汇总报告特点**：
- **一站式查看**: 所有分析结果集中在一个文件中
- **导航便捷**: 顶部固定导航菜单，快速跳转
- **数据完整**: 包含TopN回测、分组回测、多因子回测、IC分析等所有结果
- **界面美观**: 现代化设计，响应式布局
- **易于理解**: 清晰的数据展示和颜色编码
- **专业格式**: 符合量化投资分析标准

## 注意事项

1. **数据质量**: 确保输入数据的质量和完整性
2. **因子逻辑**: 因子函数应避免未来信息泄露
3. **计算效率**: 复杂因子可能影响计算速度
4. **内存使用**: 大量股票数据可能占用较多内存
5. **参数调优**: 根据实际需求调整回测参数

## 常见问题

### Q: 如何处理缺失数据？
A: 框架自动处理缺失值，使用前向填充和插值方法。

### Q: 如何添加新的因子类型？
A: 只需定义因子函数并注册即可，框架会自动处理。

### Q: 如何自定义回测策略？
A: 可以继承 `FactorBacktestEngine` 类并重写相应方法。

### Q: 如何优化计算性能？
A: 可以使用并行计算、数据缓存等方法优化性能。

## 更新日志

### v1.1.0 (2025-08-20)
- **新增**: 统一的汇总报告功能，将所有分析结果整合到一个HTML文件
- **优化**: 改进报告界面设计，添加导航菜单和响应式布局
- **增强**: 支持颜色编码和现代化UI设计
- **简化**: 减少报告文件数量，提高用户体验

### v1.0.0 (2025-08-19)
- 初始版本发布
- 支持基本因子计算和回测功能
- 提供完整的报告生成功能

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进框架。

## 许可证

MIT License
