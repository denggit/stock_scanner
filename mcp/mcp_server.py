#!/usr/bin/env python3
"""
MCP (Model Context Protocol) 服务器
为股票扫描器项目提供MCP接口，支持股票数据获取、因子计算、回测等功能
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# 导入项目模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.business.data.data_fetcher import StockDataFetcher
from backend.business.factor.core.factor.factor_engine import FactorEngine
from backend.business.backtest_event.core.backtest_engine import BacktestEngine
from backend.business.backtest_event.execution.rising_channel_backtest import RisingChannelBacktest
from backend.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

class StockScannerMCPServer:
    """股票扫描器MCP服务器"""
    
    def __init__(self):
        """初始化MCP服务器"""
        self.server = Server("stock-scanner-mcp")
        self.stock_fetcher = StockDataFetcher()
        self.factor_engine = None
        self.backtest_engine = None
        
        # 注册工具
        self._register_tools()
    
    def _register_tools(self):
        """注册MCP工具"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """列出所有可用工具"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="fetch_stock_data",
                        description="获取股票历史数据",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "stock_code": {
                                    "type": "string",
                                    "description": "股票代码，如'sz.301383'"
                                },
                                "start_date": {
                                    "type": "string",
                                    "description": "开始日期，格式YYYY-MM-DD"
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "结束日期，格式YYYY-MM-DD，可选"
                                }
                            },
                            "required": ["stock_code", "start_date"]
                        }
                    ),
                    Tool(
                        name="calculate_factors",
                        description="计算股票因子",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "stock_code": {
                                    "type": "string",
                                    "description": "股票代码"
                                },
                                "factors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "要计算的因子列表"
                                },
                                "start_date": {
                                    "type": "string",
                                    "description": "开始日期"
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "结束日期"
                                }
                            },
                            "required": ["stock_code", "factors", "start_date"]
                        }
                    ),
                    Tool(
                        name="run_backtest",
                        description="运行回测",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "strategy": {
                                    "type": "string",
                                    "description": "策略名称，如'rising_channel'"
                                },
                                "stock_codes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "股票代码列表"
                                },
                                "start_date": {
                                    "type": "string",
                                    "description": "开始日期"
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "结束日期"
                                },
                                "config": {
                                    "type": "object",
                                    "description": "策略配置参数"
                                }
                            },
                            "required": ["strategy", "stock_codes", "start_date", "end_date"]
                        }
                    ),
                    Tool(
                        name="get_stock_list",
                        description="获取股票列表",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "market": {
                                    "type": "string",
                                    "description": "市场类型，如'sz', 'sh'",
                                    "default": "sz"
                                }
                            }
                        }
                    ),
                    Tool(
                        name="analyze_stock",
                        description="分析单只股票",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "stock_code": {
                                    "type": "string",
                                    "description": "股票代码"
                                },
                                "analysis_type": {
                                    "type": "string",
                                    "description": "分析类型：technical, fundamental, factor",
                                    "default": "technical"
                                }
                            },
                            "required": ["stock_code"]
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """处理工具调用"""
            try:
                if name == "fetch_stock_data":
                    return await self._fetch_stock_data(arguments)
                elif name == "calculate_factors":
                    return await self._calculate_factors(arguments)
                elif name == "run_backtest":
                    return await self._run_backtest(arguments)
                elif name == "get_stock_list":
                    return await self._get_stock_list(arguments)
                elif name == "analyze_stock":
                    return await self._analyze_stock(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"未知工具: {name}")]
                    )
            except Exception as e:
                logger.error(f"工具调用错误: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"错误: {str(e)}")]
                )
    
    async def _fetch_stock_data(self, args: Dict[str, Any]) -> CallToolResult:
        """获取股票数据"""
        stock_code = args["stock_code"]
        start_date = args["start_date"]
        end_date = args.get("end_date")
        
        try:
            df = self.stock_fetcher.fetch_stock_data(stock_code, start_date=start_date)
            
            if end_date:
                df = df[df['trade_date'] <= end_date]
            
            # 转换为JSON格式
            result = {
                "stock_code": stock_code,
                "data_points": len(df),
                "columns": df.columns.tolist(),
                "sample_data": df.head(5).to_dict('records'),
                "date_range": {
                    "start": df['trade_date'].min(),
                    "end": df['trade_date'].max()
                }
            }
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"获取股票数据失败: {str(e)}")]
            )
    
    async def _calculate_factors(self, args: Dict[str, Any]) -> CallToolResult:
        """计算因子"""
        stock_code = args["stock_code"]
        factors = args["factors"]
        start_date = args["start_date"]
        end_date = args.get("end_date")
        
        try:
            if self.factor_engine is None:
                self.factor_engine = FactorEngine()
            
            # 这里需要根据实际的因子引擎接口进行调整
            result = {
                "stock_code": stock_code,
                "factors": factors,
                "status": "因子计算功能待实现",
                "message": "需要根据实际的FactorEngine接口进行实现"
            }
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"计算因子失败: {str(e)}")]
            )
    
    async def _run_backtest(self, args: Dict[str, Any]) -> CallToolResult:
        """运行回测"""
        strategy = args["strategy"]
        stock_codes = args["stock_codes"]
        start_date = args["start_date"]
        end_date = args["end_date"]
        config = args.get("config", {})
        
        try:
            if strategy == "rising_channel":
                backtest = RisingChannelBacktest()
                # 这里需要根据实际的回测接口进行调整
                result = {
                    "strategy": strategy,
                    "stock_codes": stock_codes,
                    "status": "回测功能待实现",
                    "message": "需要根据实际的RisingChannelBacktest接口进行实现"
                }
            else:
                result = {
                    "error": f"不支持的策略: {strategy}"
                }
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"运行回测失败: {str(e)}")]
            )
    
    async def _get_stock_list(self, args: Dict[str, Any]) -> CallToolResult:
        """获取股票列表"""
        market = args.get("market", "sz")
        
        try:
            # 这里需要实现获取股票列表的逻辑
            result = {
                "market": market,
                "status": "获取股票列表功能待实现",
                "message": "需要实现从数据源获取股票列表的功能"
            }
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"获取股票列表失败: {str(e)}")]
            )
    
    async def _analyze_stock(self, args: Dict[str, Any]) -> CallToolResult:
        """分析股票"""
        stock_code = args["stock_code"]
        analysis_type = args.get("analysis_type", "technical")
        
        try:
            result = {
                "stock_code": stock_code,
                "analysis_type": analysis_type,
                "status": "股票分析功能待实现",
                "message": "需要实现技术分析、基本面分析、因子分析等功能"
            }
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"分析股票失败: {str(e)}")]
            )

async def main():
    """主函数"""
    # 创建MCP服务器
    mcp_server = StockScannerMCPServer()
    
    # 启动服务器
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="stock-scanner-mcp",
                server_version="1.0.0",
                capabilities=mcp_server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

def run_simple_server():
    """运行简单的MCP服务器（用于测试）"""
    print("🚀 启动股票扫描器MCP服务器...")
    print("📋 可用工具:")
    print("  - fetch_stock_data: 获取股票历史数据")
    print("  - calculate_factors: 计算股票因子")
    print("  - run_backtest: 运行回测")
    print("  - get_stock_list: 获取股票列表")
    print("  - analyze_stock: 分析股票")
    print("✅ 服务器已启动，等待连接...")
    
    # 保持服务器运行
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())
