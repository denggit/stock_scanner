#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 1/29/2025 8:25 PM
@File       : database.py
@Description: 
"""
import os

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    MYSQL_HOST: str = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT: int = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_USER: str = os.getenv('MYSQL_USER', 'user')
    MYSQL_PASSWORD: str = os.getenv('MYSQL_PASSWORD', 'password')
    MYSQL_DATABASE: str = os.getenv('MYSQL_DB', 'stock_db')

    DATA_UPDATE_INTERVAL: int = int(os.getenv('DATA_UPDATE_INTERVAL', 24))
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', 3))
    RETRY_DELAY: int = int(os.getenv('RETRY_DELAY', 5))


config = Config()
