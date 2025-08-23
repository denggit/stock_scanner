# 因子报告生成系统

## 概述

这是一个基于Jinja2模板的交互式HTML因子分析报告生成系统，专为量化因子研究设计。系统支持批量报告和合并报告两种模式，提供专业的可视化界面和交互式分析功能。

## 功能特性

### 🎯 核心功能
- **批量报告生成**: 为每个因子批次生成独立的HTML报告
- **合并报告生成**: 将所有批次结果合并为综合报告
- **交互式界面**: 基于Plotly.js的交互式图表
- **响应式设计**: 支持桌面和移动设备
- **多维度分析**: 性能指标、IC分析、风险分析等

### 📊 报告内容
- **执行摘要**: 推荐因子和问题因子识别
- **表现对比**: 可排序和搜索的因子表现表格
- **图表分析**: 累计收益率、分布图、散点图等
- **详细分析**: 分组回测、IC分析、风险指标等

### 🛠 技术特性
- **模块化设计**: 基于Jinja2模板系统
- **数据驱动**: 支持pandas DataFrame和字典数据
- **可扩展性**: 易于添加新的分析模块
- **高性能**: 优化的数据处理和渲染

## 文件结构

```
reporting/
├── __init__.py                 # 模块初始化
├── report_generator.py         # 核心报告生成器
├── README.md                   # 本文档
└── templates/                  # Jinja2模板目录
    ├── base_template.html      # 基础HTML模板
    ├── __init__.py
    ├── base_template.py        # 基础模板类
    ├── html_templates.py       # HTML模板管理器
    └── sections/               # 模板片段
        ├── _summary_card.html  # 总览卡片
        ├── _factor_table.html  # 因子表格
        ├── _charts.html        # 图表部分
        └── _details.html       # 详细分析
```

## 快速开始

### 1. 基本使用

```python
from backend.business.factor.core.reporting import FactorReportGenerator

# 初始化报告生成器
report_generator = FactorReportGenerator()

# 准备数据
report_data = {
    'factor_names': ['alpha_1', 'alpha_2', 'momentum_5d'],
    'performance_metrics': {
        'alpha_1': {'total_return': 0.15, 'sharpe_ratio': 1.2, ...},
        'alpha_2': {'total_return': 0.12, 'sharpe_ratio': 1.0, ...},
        # ...
    },
    'ic_metrics': {
        'alpha_1': {'mean_ic': 0.05, 'ic_ir': 0.8, ...},
        'alpha_2': {'mean_ic': 0.03, 'ic_ir': 0.6, ...},
        # ...
    },
    'time_series_returns': {
        'alpha_1': pd.Series(...),
        'alpha_2': pd.Series(...),
        # ...
    },
    'detailed_analysis': {
        'alpha_1': {
            'metrics': {...},
            'group_results': {...},
            'ic_stats': {...},
            'risk_metrics': {...}
        },
        # ...
    }
}

# 生成批次报告
batch_report_path = report_generator.generate_batch_report(
    batch_name="测试批次",
    report_data=report_data,
    output_path="reports/batch_report.html",
    start_date="2020-01-01",
    end_date="2020-12-31",
    stock_pool="no_st",
    top_n=10,
    n_groups=5
)
```

### 2. 合并报告生成

```python
# 准备多个批次的数据
all_batches_data = [batch1_data, batch2_data, batch3_data]

# 生成合并报告
merged_report_path = report_generator.generate_merged_report(
    all_batches_data=all_batches_data,
    output_path="reports/comprehensive_report.html",
    start_date="2020-01-01",
    end_date="2020-12-31",
    stock_pool="no_st",
    top_n=10,
    n_groups=5
)
```

## 数据格式

### 报告数据结构

```python
report_data = {
    # 必需字段
    'factor_names': List[str],           # 因子名称列表
    
    # 性能指标 (可选)
    'performance_metrics': Dict[str, Dict],  # 因子性能指标
    # 格式: {'factor_name': {'total_return': float, 'sharpe_ratio': float, ...}}
    
    # IC指标 (可选)
    'ic_metrics': Dict[str, Dict],       # 因子IC指标
    # 格式: {'factor_name': {'mean_ic': float, 'ic_ir': float, ...}}
    
    # 时间序列数据 (可选)
    'time_series_returns': Dict[str, pd.Series],  # 因子收益率时间序列
    # 格式: {'factor_name': pd.Series}
    
    # 详细分析数据 (可选)
    'detailed_analysis': Dict[str, Dict], # 因子详细分析
    # 格式: {'factor_name': {'metrics': {...}, 'group_results': {...}, ...}}
}
```

### 性能指标字段

```python
performance_metrics = {
    'factor_name': {
        'total_return': float,      # 总收益率
        'annual_return': float,     # 年化收益率
        'volatility': float,        # 年化波动率
        'sharpe_ratio': float,      # 夏普比率
        'max_drawdown': float,      # 最大回撤
        'trading_days': int         # 交易天数
    }
}
```

### IC指标字段

```python
ic_metrics = {
    'factor_name': {
        'mean_ic': float,           # IC均值
        'ic_ir': float,             # IC IR
        'win_rate': float,          # IC胜率
        'ic_std': float,            # IC标准差
        'ic_skew': float,           # IC偏度
        'ic_kurtosis': float        # IC峰度
    }
}
```

### 详细分析字段

```python
detailed_analysis = {
    'factor_name': {
        'metrics': Dict,            # 基础指标
        'group_results': Dict,      # 分组回测结果
        'ic_stats': Dict,           # IC统计
        'risk_metrics': Dict,       # 风险指标
        'returns_series': pd.Series, # 收益率序列
        'drawdown_series': pd.Series, # 回撤序列
        'ic_series': pd.Series,     # IC序列
        'monthly_returns': List     # 月度收益
    }
}
```

## 在run_factor.py中的集成

### 1. 导入报告生成器

```python
from backend.business.factor.core.reporting import FactorReportGenerator
```

### 2. 修改现有函数

```python
def run_worldquant_factors_merged_with_new_reporting(
    start_date: str = DEFAULT_START_DATE, 
    end_date: str = None,
    batch_size: int = DEFAULT_BATCH_SIZE, 
    stock_pool=DEFAULT_STOCK_POOL,
    top_n=DEFAULT_TOP_N, 
    n_groups=DEFAULT_N_GROUPS,
    optimize_data_fetch=DEFAULT_OPTIMIZE_DATA_FETCH_FOR_WORLDQUANT
):
    """运行WorldQuant Alpha因子并使用新的报告系统"""
    
    # 初始化报告生成器
    report_generator = FactorReportGenerator()
    
    # 获取因子列表
    worldquant_factors = [f for f in factor_registry._factors.keys() if f.startswith('alpha_')]
    
    # 分批处理
    all_batches_data = []
    for i in range(0, len(worldquant_factors), batch_size):
        batch_factors = worldquant_factors[i:i + batch_size]
        
        # 运行因子分析
        results = framework.run_factor_comparison(
            factor_names=batch_factors,
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            top_n=top_n,
            n_groups=n_groups,
            optimize_data_fetch=optimize_data_fetch
        )
        
        # 准备报告数据
        batch_report_data = {
            "factor_names": batch_factors,
            "performance_metrics": results.get('backtest_results', {}),
            "ic_metrics": results.get('effectiveness_results', {}),
            "time_series_returns": results.get('time_series_data', {}),
            "detailed_analysis": results.get('detailed_analysis', {})
        }
        
        all_batches_data.append(batch_report_data)
        
        # 生成批次报告
        batch_report_path = f"reports/batch_{i//batch_size + 1}_report.html"
        report_generator.generate_batch_report(
            batch_name=f"WorldQuant Alpha 批次 {i//batch_size + 1}",
            report_data=batch_report_data,
            output_path=batch_report_path,
            start_date=start_date,
            end_date=end_date,
            stock_pool=stock_pool,
            top_n=top_n,
            n_groups=n_groups
        )
    
    # 生成合并报告
    merged_report_path = f"reports/comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_generator.generate_merged_report(
        all_batches_data=all_batches_data,
        output_path=merged_report_path,
        start_date=start_date,
        end_date=end_date,
        stock_pool=stock_pool,
        top_n=top_n,
        n_groups=n_groups
    )
    
    return {
        'merged_report_path': merged_report_path,
        'successful_factors': successful_factors,
        'failed_factors': failed_factors,
        'success_rate': success_rate
    }
```

## 报告功能详解

### 1. 执行摘要 (仅合并报告)
- **推荐因子**: 基于综合评分的前5名因子
- **问题因子**: 表现最差的后5名因子
- **总体统计**: 因子数量、成功率、平均指标等

### 2. 表现对比表格
- **可排序**: 点击列标题进行排序
- **可搜索**: 支持因子名称搜索
- **交互式**: 点击行可添加/移除图表
- **指标完整**: 收益率、夏普比率、IC等

### 3. 图表分析
- **累计收益率**: 交互式时间序列图
- **分布图**: 夏普比率、IC分布直方图
- **散点图**: 收益率vs夏普比率
- **对数坐标**: 支持线性/对数切换

### 4. 详细分析
- **概览**: 关键指标卡片
- **表现分析**: 收益率走势、回撤分析
- **分组回测**: 分组表现对比
- **IC分析**: IC时间序列、统计指标
- **风险分析**: VaR、CVaR、Beta等

## 自定义和扩展

### 1. 添加新的分析模块

```python
# 在templates/sections/中添加新的模板文件
# 例如: _custom_analysis.html

# 在base_template.html中引入
{% include 'sections/_custom_analysis.html' %}
```

### 2. 自定义样式

```css
/* 在base_template.html的<style>标签中添加 */
.custom-section {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 30px;
}
```

### 3. 添加新的图表类型

```javascript
// 在对应的模板文件中添加
function createCustomChart(data) {
    const trace = {
        x: data.x,
        y: data.y,
        type: 'scatter',
        mode: 'markers'
    };
    
    const layout = {
        title: '自定义图表',
        xaxis: { title: 'X轴' },
        yaxis: { title: 'Y轴' }
    };
    
    Plotly.newPlot('custom-chart', [trace], layout);
}
```

## 注意事项

### 1. 数据要求
- 时间序列数据需要pandas.Series格式
- 索引应为datetime类型
- 数值数据应为float类型

### 2. 性能考虑
- 大量因子时建议分批处理
- 时间序列数据过长时考虑采样
- 图表数据过多时考虑分页显示

### 3. 浏览器兼容性
- 支持现代浏览器 (Chrome, Firefox, Safari, Edge)
- 需要JavaScript支持
- 建议使用最新版本的Plotly.js

## 故障排除

### 1. 模板渲染失败
- 检查模板文件路径
- 确认Jinja2语法正确
- 验证数据格式

### 2. 图表不显示
- 检查Plotly.js是否正确加载
- 确认数据不为空
- 查看浏览器控制台错误

### 3. 样式问题
- 检查CSS文件路径
- 确认字体文件可用
- 验证响应式设计

## 更新日志

### v1.0.0 (2025-08-23)
- 初始版本发布
- 支持批量报告和合并报告
- 交互式HTML界面
- 完整的因子分析功能

## 贡献指南

欢迎提交Issue和Pull Request来改进这个报告系统！

## 许可证

本项目采用MIT许可证。
