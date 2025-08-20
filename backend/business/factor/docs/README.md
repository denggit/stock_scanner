# 因子研究框架使用说明

## 🎯 快速开始

### 1. 运行单个因子分析

```bash
cd backend/business/factor
python run_factor.py --factor_type single --factor_name momentum_20d
```

### 2. 运行因子类型分析

```bash
# 动量类因子
python run_factor.py --factor_type momentum

# 技术指标因子
python run_factor.py --factor_type technical

# 波动率因子
python run_factor.py --factor_type volatility

# 成交量因子
python run_factor.py --factor_type volume

# 通道分析因子
python run_factor.py --factor_type channel

# WorldQuant Alpha因子
python run_factor.py --factor_type worldquant

# 基本面因子
python run_factor.py --factor_type fundamental

# 所有因子
python run_factor.py --factor_type all
```

## 📁 目录结构

```
factor/
├── library/                       # 因子库 (用户主要修改的地方)
│   ├── technical_factors.py       # 技术因子
│   ├── fundamental_factors.py     # 基本面因子
│   ├── worldquant_factors.py      # WorldQuant Alpha因子
│   └── channel_factors.py         # 通道分析因子
├── core/                          # 核心模块
│   ├── factor/                    # 因子定义和计算
│   ├── data/                      # 数据管理
│   ├── backtest/                  # 回测系统
│   ├── analysis/                  # 因子分析
│   └── reporting/                 # 报告生成
├── main.py                        # 主入口文件
├── run_factor.py                  # 简化的运行文件
└── storage/reports/               # 生成的报告
```

## 🔧 添加新因子

### 1. 在对应的因子库文件中添加因子函数

例如，在 `library/technical_factors.py` 中添加新的技术因子：

```python
@register_technical_factor(name='my_new_factor', description='我的新因子')
def my_new_factor(close: pd.Series, **kwargs) -> pd.Series:
    """我的新因子：自定义计算逻辑"""
    # 你的因子计算逻辑
    return close.rolling(10).mean() / close
```

### 2. 运行因子分析

```bash
python run_factor.py --factor_type single --factor_name my_new_factor
```

## 📊 查看报告

所有生成的报告都保存在 `storage/reports/` 目录下，以HTML格式提供，包含：

- 因子有效性分析
- IC分析结果
- 回测结果
- 可视化图表

## 🎨 支持的因子类型

### 技术因子 (technical_factors.py)
- 动量类：momentum_5d, momentum_20d, momentum_60d
- 波动率类：volatility_20d, volatility_60d, intraday_volatility
- 成交量类：volume_ratio_5d, volume_ratio_20d, volume_price_momentum
- 价格位置类：price_position_20d, price_position_60d
- 均线类：ma_cross_5_20, ma_cross_10_60
- 技术指标类：rsi_14, rsi_21, bollinger_position, macd_histogram, williams_r, cci, kama, atr, adx
- 其他：gap_strength

### 基本面因子 (fundamental_factors.py)
- 估值类：pe_ratio, pb_ratio, ps_ratio, pcf_ratio
- 盈利能力：roe, roa
- 成长性：revenue_growth, profit_growth
- 财务健康：debt_to_equity, current_ratio

### WorldQuant Alpha因子 (worldquant_factors.py)
- alpha_1 到 alpha_10

### 通道分析因子 (channel_factors.py)
- channel_distance, channel_breakout, channel_width, channel_trend

## 💡 使用建议

1. **开发新因子时**：主要在 `library/` 目录下修改，其他核心模块尽量不碰
2. **运行分析时**：使用 `run_factor.py` 进行简单粗暴的运行
3. **查看结果时**：打开生成的HTML报告，一次性查看所有分析结果
4. **调试因子时**：可以先用单个因子测试，确认无误后再进行多因子对比

## 🔍 注意事项

- 确保数据源连接正常
- 因子函数必须包含 `**kwargs` 参数
- 因子函数返回的必须是 `pd.Series` 类型
- 报告生成可能需要一些时间，请耐心等待
