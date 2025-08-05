# 🚀 股票筛选系统 (Stock Scanner)

一个功能完整的量化交易平台，集成了股票数据获取、策略开发、回测分析、实时监控等功能。

## ✨ 主要特性

- 📊 **多数据源支持**: Baostock + AKShare
- 🎯 **多种策略**: 爆发式选股、均线回踩、波段交易等
- 📈 **完整回测**: 历史数据回测和性能分析
- 🤖 **机器学习**: 集成ML模型预测
- 🚀 **高性能**: 缓存系统和性能监控
- 🌐 **Web界面**: Streamlit前端 + FastAPI后端
- 📱 **实时监控**: 系统健康检查和性能指标
- 🔧 **数据兼容性**: 自动处理无穷大和NaN值，确保JSON序列化兼容

## 🆕 最新更新

### 2025-08-01 - JSON序列化兼容性修复 & 突破策略实现

- ✅ 修复了策略返回数据中包含无穷大(inf)和NaN值时导致的JSON序列化错误
- ✅ 增强了`convert_to_python_types`函数，自动将特殊数值转换为None
- ✅ 确保所有策略（放量上涨策略、均线回踩策略等）都能正常返回JSON响应
- ✅ **新增**: 完整实现了突破策略，包括阻力位识别、突破检测、信号评分等功能
- ✅ 提高了系统的稳定性和数据兼容性

## 🏗️ 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   MySQL         │
│   Frontend      │◄──►│   Backend       │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Redis Cache   │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Baostock      │
                       │   AKShare       │
                       │   Data Sources  │
                       └─────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd stock_scanner

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据库配置

```bash
# 安装MySQL和Redis
# 创建数据库
CREATE DATABASE stock_scanner;
```

### 3. 环境变量配置

创建 `.env` 文件：

```env
# 数据库配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=stock_scanner

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379

# 后端配置
BACKEND_URL=localhost
BACKEND_PORT=8000

# 环境配置
ENVIRONMENT=development
DEBUG=true
```

### 4. 启动服务

```bash
# 启动后端服务
python run_backend.py

# 启动前端服务
python run_frontend.py
```

### 5. 访问应用

- 🌐 **前端界面**: http://localhost:8501
- 📚 **API文档**: http://localhost:8000/docs
- 🔧 **后端API**: http://localhost:8000

## 📚 文档

- [📖 完整功能文档](docs/项目工具功能文档.md)
- [⚡ 快速参考手册](docs/快速参考手册.md)
- [🔌 API接口文档](docs/API接口文档.md)

## 🎯 核心功能

### 1. 数据查看器

- 股票K线图展示
- 技术指标可视化 (MACD, RSI, 布林带等)
- 成交量分析
- 交互式图表

### 2. 策略扫描器

- **爆发式选股策略**: 寻找短期暴涨潜力
- **均线回踩策略**: 价格回踩均线买入机会
- **波段交易策略**: 基于技术指标的波段操作
- **突破策略**: 识别价格突破关键阻力位
- **头肩底形态策略**: 识别经典技术形态

### 3. 回测系统

- 历史数据回测
- 性能指标计算 (收益率、最大回撤、夏普比率等)
- 交易记录分析
- 可视化结果展示

### 4. 机器学习模型

- 集成多种ML算法
- 特征工程和模型训练
- 预测概率计算
- 模型性能评估

## 🛠️ 工具和组件

### 缓存系统

```python
from backend.utils.cache_manager import stock_data_cache

@stock_data_cache(expire=1800)
def get_stock_data(code, start_date, end_date):
    return source.get_stock_data(code, start_date, end_date)
```

### 性能监控

```python
from backend.utils.performance_monitor import get_performance_monitor

monitor = get_performance_monitor()
monitor.start_monitoring()
health = monitor.get_current_health()
```

### 配置管理

```python
from backend.configs.app_config import get_config

config = get_config()
db_config = config.database
strategy_config = config.strategy
```

## 📊 使用示例

### 策略扫描

```python
from backend.business.strategies.explosive_stock import ExplosiveStockStrategy

strategy = ExplosiveStockStrategy()
strategy.set_parameters({
    "signal": 70.0,
    "volume_ratio": 1.5,
    "rsi_range": (45, 65)
})

signal = strategy.generate_signal(stock_data)
print(f"信号强度: {signal['signal']}")
print(f"买入建议: {signal['buy_signal']}")
```

### 回测分析

```python
from backend.services.backtest_service import BacktestService

backtest = BacktestService()
results = await backtest.run_backtest(
    strategy="爆发式选股策略",
    start_date="2024-01-01",
    end_date="2024-12-31",
    backtest_init_params={
        "initial_capital": 100000,
        "max_positions": 4
    }
)

print(f"总收益率: {results['summary']['total_return']}%")
print(f"最大回撤: {results['summary']['max_drawdown']}%")
```

## 🔧 开发指南

### 添加新策略

```python
from backend.business.strategies.base import BaseStrategy


class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="我的策略", description="策略描述")
        self._init_params()

    def _init_params(self):
        self._params = {
            "param1": 10,
            "param2": 20
        }

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 实现策略逻辑
        return signal
```

### 使用缓存装饰器

```python
from backend.utils.cache_manager import strategy_result_cache

@strategy_result_cache(expire=3600)
def run_strategy(strategy_name: str, params: dict):
    # 策略执行逻辑
    return results
```

## 📈 性能优化

- **缓存系统**: Redis + 内存缓存
- **批量处理**: 减少API调用次数
- **异步处理**: 提高并发性能
- **数据压缩**: 减少存储空间

## 🔍 监控和调试

### 系统健康检查

```bash
curl http://localhost:8000/api/system/health
```

### 性能指标

```bash
curl http://localhost:8000/api/system/metrics
```

### 缓存状态

```bash
curl http://localhost:8000/api/cache/status
```

## 🆘 常见问题

### 1. 数据源连接失败

- 检查网络连接
- 验证API密钥
- 查看错误日志

### 2. 数据库连接问题

- 检查MySQL服务状态
- 验证连接参数
- 检查防火墙设置

### 3. 策略执行缓慢

- 使用缓存装饰器
- 优化算法逻辑
- 减少数据量

### 4. JSON序列化错误 (已修复)

- **问题**: 策略返回数据时出现"Out of range float values are not JSON compliant"错误
- **原因**: 数据中包含无穷大(inf)或NaN值，这些值无法被JSON序列化
- **解决方案**: 系统已自动处理，将特殊数值转换为None
- **影响**: 所有策略现在都能正常返回JSON响应

### 5. 突破策略返回None错误 (已修复)

- **问题**: 突破策略使用时出现"策略扫描出来的结果形态不支持：<class 'NoneType'>"错误
- **原因**: 突破策略的`generate_signal`方法没有实现，返回了None
- **解决方案**: 完整实现了突破策略，包括阻力位识别、突破检测、信号评分等功能
- **影响**: 突破策略现在可以正常使用，提供完整的交易信号分析

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目文档: [docs/](docs/)
- 问题反馈: [Issues](../../issues)
- 功能建议: [Discussions](../../discussions)

## 🙏 致谢

感谢以下开源项目的支持：

- [Baostock](http://baostock.com/baostock/index.php)
- [AKShare](https://akshare.akfamily.xyz/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！ 