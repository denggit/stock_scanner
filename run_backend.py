#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/27/2025 4:49 PM
@File       : run_backend.py
@Description: 
"""
from dotenv import load_dotenv

load_dotenv()

import os
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import uvicorn

if __name__ == '__main__':
    # 打印当前Python路径，用于调试
    print("Python path:", sys.path)
    print("Current working directory:", os.getcwd())
    print("Root directory:", ROOT_DIR)

    backend_url = os.getenv('BACKEND_URL')
    backend_port = os.getenv('BACKEND_PORT')

    uvicorn.run(
        "backend.app:app",
        host=backend_url,
        port=int(backend_port),
        reload=False,
        reload_dirs=[os.path.join(ROOT_DIR, "backend")],
        log_level="info",  # 添加详细日志
    )
