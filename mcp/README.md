# MCP 模块

## 概述

MCP (Model Context Protocol) 模块为股票扫描器项目提供了与AI助手交互的标准化接口。

## 文件结构

```
mcp/
├── __init__.py              # 包初始化文件
├── README.md               # 本说明文档
├── mcp_server.py           # 完整的MCP服务器实现
└── simple_mcp_server.py    # 简化的MCP服务器（用于测试）
```

## 功能特性

### 1. 股票数据获取
- 获取指定股票的历史交易数据
- 支持自定义时间范围
- 返回结构化的JSON数据

### 2. 因子计算
- 计算技术指标和基本面因子
- 支持多种因子类型
- 批量处理能力

### 3. 回测功能
- 运行量化策略回测
- 支持多种策略类型
- 生成详细的回测报告

### 4. 股票列表获取
- 获取指定市场的股票列表
- 支持深市、沪市等不同市场

### 5. 股票分析
- 技术分析
- 基本面分析
- 因子分析

## 使用方法

### 启动MCP服务器

在项目根目录运行：
```bash
python run_mcp_server.py
```

### 测试功能

```bash
# 测试股票数据获取
python mcp/simple_mcp_server.py test sz.300773
```

### 在Cursor中使用

1. 重启Cursor
2. 打开命令面板（Cmd + Shift + P）
3. 搜索MCP相关命令
4. 使用自然语言调用各种功能

## 配置

MCP服务器配置文件位于 `.cursor/mcp_servers.json`：

```json
{
  "mcpServers": {
    "stock-scanner": {
      "command": "python",
      "args": ["run_mcp_server.py"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  }
}
```

## 开发

### 添加新工具

1. 在 `mcp_server.py` 中添加工具定义
2. 实现工具功能
3. 更新文档

### 测试

运行测试脚本验证功能：
```bash
python mcp/simple_mcp_server.py test <股票代码>
```

## 依赖

- mcp: MCP协议实现
- pandas: 数据处理
- numpy: 数值计算

## 版本历史

- v1.0.0: 初始版本，支持基本的股票数据获取和分析功能

