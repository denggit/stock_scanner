#!/usr/bin/env python3
"""
简单的MCP服务器测试版本
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def fetch_stock_data(stock_code: str, start_date: str, end_date: str = None):
    """获取股票数据"""
    try:
        from backend.business.data.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        df = fetcher.fetch_stock_data(stock_code, start_date=start_date)
        
        if end_date:
            df = df[df['trade_date'] <= end_date]
        
        # 处理日期序列化问题
        sample_data = df.head(3).copy()
        sample_data['trade_date'] = sample_data['trade_date'].astype(str)
        
        return {
            "success": True,
            "stock_code": stock_code,
            "data_points": len(df),
            "columns": df.columns.tolist(),
            "sample_data": sample_data.to_dict('records'),
            "date_range": {
                "start": str(df['trade_date'].min()),
                "end": str(df['trade_date'].max())
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """主函数"""
    print("🚀 股票扫描器MCP服务器已启动")
    print("📋 可用功能:")
    print("  - fetch_stock_data: 获取股票数据")
    print("  - 测试命令: python simple_mcp_server.py test sz.301383")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) > 2:
            stock_code = sys.argv[2]
            print(f"\n🧪 测试获取股票数据: {stock_code}")
            result = fetch_stock_data(stock_code, "2024-01-01")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ 请提供股票代码")
    else:
        print("\n✅ 服务器准备就绪，可在Cursor中使用MCP功能")

if __name__ == "__main__":
    main()
