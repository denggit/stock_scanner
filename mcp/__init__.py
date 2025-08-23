"""
MCP (Model Context Protocol) 模块
为股票扫描器项目提供MCP服务器功能
"""

__version__ = "1.0.0"
__author__ = "Stock Scanner Team"

from .simple_mcp_server import fetch_stock_data

__all__ = [
    "fetch_stock_data",
]

