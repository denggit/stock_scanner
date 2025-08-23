#!/usr/bin/env python3
"""
MCP服务器启动脚本
用于启动股票扫描器的MCP服务器
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    print("🚀 启动股票扫描器MCP服务器...")
    print(f"📁 项目路径: {project_root}")
    print(f"🐍 Python版本: {sys.version}")
    
    try:
        # 导入并运行MCP服务器
        from mcp.simple_mcp_server import main as run_server
        run_server()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装MCP依赖: pip install mcp")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

