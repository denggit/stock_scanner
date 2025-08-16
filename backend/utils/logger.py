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


class LoggerManager:
    """日志记录器管理器，使用单例模式确保每个名称的日志记录器只被初始化一次"""
    
    _instances = {}
    _lock = {}
    
    @classmethod
    def get_logger(cls, name: str, log_dir: str = "logs", log_level=logging.INFO, set_root_logger=False) -> logging.Logger:
        """
        获取或创建日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            log_level: 日志级别
            set_root_logger: 是否设置root logger
            
        Returns:
            logging.Logger: 配置好的日志记录器
        """
        # 如果已经存在该名称的日志记录器，直接返回
        if name in cls._instances:
            return cls._instances[name]
        
        # 创建新的日志记录器
        logger = cls._create_logger(name, log_dir, log_level, set_root_logger)
        cls._instances[name] = logger
        return logger
    
    @classmethod
    def _create_logger(cls, name: str, log_dir: str, log_level: int, set_root_logger: bool) -> logging.Logger:
        """创建新的日志记录器"""
        # 创建日志目录
        log_dir = os.path.join(log_dir, name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # 清除可能存在的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 创建日志文件处理器
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(log_dir, f'{name}_{today}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')

        # 创建控制台处理器
        console_handler = logging.StreamHandler()

        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        if set_root_logger:
            # 设置root logger
            logger.addHandler(console_handler)
            logging.root.addHandler(file_handler)
            logging.root.setLevel(log_level)
        else:
            # 添加处理器
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger


def setup_logger(name: str, log_dir: str = "logs", log_level=logging.INFO, set_root_logger=False) -> logging.Logger:
    """
    设置日志记录器（兼容旧接口）
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        log_level: 日志级别
        set_root_logger: 是否设置root logger
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return LoggerManager.get_logger(name, log_dir, log_level, set_root_logger)
