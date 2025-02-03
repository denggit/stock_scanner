#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 6:02 PM
@File       : logger.py
@Description: 
"""
import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", log_level=logging.INFO, set_root_logger=False) -> logging.Logger:
    """设置日志记录器"""
    # 创建日志目录
    log_dir = os.path.join(log_dir, name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 清除现有的处理器
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # 创建日志文件处理器
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'{name}_{today}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')

    # 创建控制台处理器
    console_handler = logging.StreamHandler()

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if set_root_logger:
        # 设置root logger
        logging.root.addHandler(file_handler)
        logging.root.setLevel(log_level)

    return logger
