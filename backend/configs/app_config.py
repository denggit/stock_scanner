#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/4/2025 3:45 PM
@File       : app_config.py
@Description: 应用配置管理
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    host: str = Field(default="localhost", env="MYSQL_HOST")
    port: int = Field(default=3306, env="MYSQL_PORT")
    user: str = Field(default="", env="MYSQL_USER")
    password: str = Field(default="", env="MYSQL_PASSWORD")
    database: str = Field(default="", env="MYSQL_DATABASE")
    charset: str = "utf8mb4"
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: int = Field(default=5, env="RETRY_DELAY")


class RedisConfig(BaseSettings):
    """Redis配置"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


class DataSourceConfig(BaseSettings):
    """数据源配置"""
    # Baostock配置
    baostock_max_retries: int = 3
    baostock_retry_delay: int = 2
    baostock_login_interval: int = 300  # 5分钟重新登录
    
    # AKShare配置
    akshare_timeout: int = 30
    akshare_max_retries: int = 3
    
    # 数据更新配置
    update_interval: int = Field(default=24, env="DATA_UPDATE_INTERNAL")  # 小时
    batch_size: int = 100  # 批量处理大小


class BackendConfig(BaseSettings):
    """后端服务配置"""
    url: str = Field(default="localhost", env="BACKEND_URL")
    port: int = Field(default=8000, env="BACKEND_PORT")
    workers: int = 4
    log_level: str = "info"
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]


class StrategyConfig(BaseSettings):
    """策略配置"""
    # 默认策略参数
    default_ma_period: int = 20
    default_rsi_period: int = 14
    default_bb_period: int = 20
    default_bb_std: float = 2.0
    
    # 信号阈值
    min_signal_score: float = 70.0
    min_volume_ratio: float = 1.5
    min_explosion_prob: float = 0.5
    
    # 风险控制
    max_position_size: float = 0.1  # 单只股票最大仓位
    stop_loss: float = 0.07  # 止损比例
    take_profit: float = 0.30  # 止盈比例
    
    # 机器学习模型配置
    use_ml_model: bool = True
    ml_model_fallback: bool = True
    model_update_interval: int = 24  # 小时


class CacheConfig(BaseSettings):
    """缓存配置"""
    # 缓存过期时间（秒）
    stock_data_expire: int = 1800  # 30分钟
    strategy_result_expire: int = 3600  # 1小时
    indicator_expire: int = 7200  # 2小时
    technical_analysis_expire: int = 3600  # 1小时
    
    # 内存缓存限制
    max_memory_cache_size: int = 10000
    memory_cache_cleanup_interval: int = 3600  # 1小时


class LoggingConfig(BaseSettings):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class AppConfig(BaseSettings):
    """应用主配置"""
    
    # 环境配置
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # 子配置
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    backend: BackendConfig = BackendConfig()
    strategy: StrategyConfig = StrategyConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # 应用特定配置
    app_name: str = "Stock Scanner"
    app_version: str = "1.0.0"
    timezone: str = "Asia/Shanghai"
    
    # 文件路径配置
    base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    logs_dir: str = os.path.join(base_dir, "logs")
    models_dir: str = os.path.join(base_dir, "backend", "ml", "models")
    results_dir: str = os.path.join(base_dir, "results")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.models_dir,
            self.results_dir
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        return f"mysql+pymysql://{self.database.user}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
    
    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """判断是否为开发环境"""
        return self.environment.lower() == "development"
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "database": self.database.dict(),
            "redis": self.redis.dict(),
            "data_source": self.data_source.dict(),
            "backend": self.backend.dict(),
            "strategy": self.strategy.dict(),
            "cache": self.cache.dict(),
            "logging": self.logging.dict(),
            "paths": {
                "base_dir": self.base_dir,
                "data_dir": self.data_dir,
                "logs_dir": self.logs_dir,
                "models_dir": self.models_dir,
                "results_dir": self.results_dir
            }
        }


# 全局配置实例
_config = None


def get_config() -> AppConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """重新加载配置"""
    global _config
    _config = AppConfig()
    return _config 