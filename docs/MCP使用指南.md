# MCP功能使用指南

## 🎉 恭喜！您的MCP功能已经成功配置

### ✅ 已完成配置

1. **MCP服务器**：已创建并测试通过
2. **依赖包**：已安装 `mcp` 包
3. **配置文件**：已配置 `.cursor/mcp_servers.json`
4. **功能测试**：股票数据获取功能已验证

### 🚀 如何在Cursor中启用MCP功能

#### 方法1：重启Cursor（推荐）
1. 完全关闭Cursor
2. 重新打开Cursor
3. 打开您的项目文件夹
4. MCP功能将自动启用

#### 方法2：手动启动MCP服务器
```bash
# 在项目根目录运行
python simple_mcp_server.py
```

### 📋 可用的MCP工具

#### 1. 股票数据获取
- **功能**：获取指定股票的历史交易数据
- **参数**：
  - `stock_code`: 股票代码（如 'sz.301383'）
  - `start_date`: 开始日期（YYYY-MM-DD）
  - `end_date`: 结束日期（可选）

#### 2. 因子计算
- **功能**：计算技术指标和基本面因子
- **参数**：
  - `stock_code`: 股票代码
  - `factors`: 因子列表
  - `start_date`: 开始日期
  - `end_date`: 结束日期（可选）

#### 3. 回测功能
- **功能**：运行量化策略回测
- **参数**：
  - `strategy`: 策略名称
  - `stock_codes`: 股票代码列表
  - `start_date`: 开始日期
  - `end_date`: 结束日期
  - `config`: 策略配置（可选）

#### 4. 股票列表获取
- **功能**：获取指定市场的股票列表
- **参数**：
  - `market`: 市场类型（'sz', 'sh'）

#### 5. 股票分析
- **功能**：分析单只股票
- **参数**：
  - `stock_code`: 股票代码
  - `analysis_type`: 分析类型（'technical', 'fundamental', 'factor'）

### 💬 使用示例

#### 示例1：获取股票数据
```
用户: "帮我获取股票sz.301383最近100天的数据"
AI: 我将使用fetch_stock_data工具为您获取数据...
```

#### 示例2：计算技术指标
```
用户: "计算这只股票的RSI和MACD指标"
AI: 我将使用calculate_factors工具计算技术指标...
```

#### 示例3：运行回测
```
用户: "运行上升通道策略的回测"
AI: 我将使用run_backtest工具执行回测分析...
```

### 🔧 故障排除

#### 问题1：MCP功能无法使用
**解决方案**：
1. 重启Cursor
2. 检查 `.cursor/mcp_servers.json` 文件是否存在
3. 确认 `simple_mcp_server.py` 文件在项目根目录

#### 问题2：工具调用失败
**解决方案**：
1. 检查股票代码格式是否正确
2. 确认日期格式为 YYYY-MM-DD
3. 查看错误日志

#### 问题3：数据获取失败
**解决方案**：
1. 检查网络连接
2. 确认数据源服务正常
3. 验证股票代码是否存在

### 📁 文件结构

```
stock_scanner/
├── .cursor/
│   └── mcp_servers.json          # MCP服务器配置
├── mcp_server.py                 # 完整MCP服务器
├── simple_mcp_server.py          # 简化MCP服务器
├── start_mcp_server.py           # 启动脚本
├── test_mcp_server.py            # 测试脚本
└── docs/
    └── MCP功能使用说明.md         # 详细文档
```

### 🧪 测试功能

运行测试命令验证MCP功能：

```bash
# 测试股票数据获取
python simple_mcp_server.py test sz.301383

# 测试其他功能
python test_mcp_server.py
```

### 🔄 更新和维护

#### 添加新工具
1. 在 `mcp_server.py` 中添加工具定义
2. 实现工具功能
3. 更新文档

#### 修改配置
1. 编辑 `.cursor/mcp_servers.json`
2. 重启Cursor
3. 测试功能

### 📞 技术支持

如果遇到问题，请：
1. 查看错误日志
2. 运行测试脚本
3. 检查配置文件
4. 重启Cursor

### 🎯 下一步

现在您可以：
1. 在Cursor中使用自然语言调用股票数据功能
2. 通过AI助手进行股票分析
3. 运行量化策略回测
4. 获取实时的市场数据

享受您的MCP功能吧！🚀
