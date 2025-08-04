#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2/4/2025 4:00 PM
@File       : performance_monitor.py
@Description: 性能监控和健康检查系统
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Any

import psutil

from backend.configs.app_config import get_config
from backend.utils.cache_manager import get_cache_manager


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    active_connections: int
    cache_hit_rate: float
    cache_memory_usage: Dict[str, int]
    api_response_time: float
    database_query_time: float
    strategy_execution_time: float


@dataclass
class HealthStatus:
    """健康状态数据类"""
    timestamp: datetime
    status: str  # "healthy", "warning", "critical"
    checks: Dict[str, Dict[str, Any]]
    message: str


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.metrics_history = deque(maxlen=1000)  # 保留最近1000条记录
        self.health_history = deque(maxlen=100)  # 保留最近100条健康检查记录
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 60  # 60秒监控间隔

        # 性能阈值
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'api_response_warning': 2.0,  # 秒
            'api_response_critical': 5.0,
            'db_query_warning': 1.0,
            'db_query_critical': 3.0,
            'cache_hit_warning': 0.7,  # 70%
            'cache_hit_critical': 0.5,  # 50%
        }

        # 统计信息
        self.stats = {
            'api_calls': defaultdict(int),
            'api_response_times': defaultdict(list),
            'db_queries': defaultdict(int),
            'db_query_times': defaultdict(list),
            'strategy_executions': defaultdict(int),
            'strategy_execution_times': defaultdict(list),
        }

        self.lock = threading.Lock()

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("性能监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("性能监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                health = self._check_health(metrics)

                with self.lock:
                    self.metrics_history.append(metrics)
                    self.health_history.append(health)

                # 记录警告和严重问题
                if health.status in ['warning', 'critical']:
                    logging.warning(f"系统健康状态: {health.status} - {health.message}")

                time.sleep(self.monitor_interval)

            except Exception as e:
                logging.error(f"性能监控循环出错: {e}")
                time.sleep(self.monitor_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        try:
            # 系统指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            # 缓存指标
            cache_manager = get_cache_manager()
            cache_stats = cache_manager.get_memory_usage()

            # 计算缓存命中率（简化版本）
            cache_hit_rate = 0.8  # 这里需要实际统计

            # 计算平均响应时间
            api_response_time = self._calculate_average_response_time()
            db_query_time = self._calculate_average_db_query_time()
            strategy_execution_time = self._calculate_average_strategy_time()

            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                active_connections=len(psutil.net_connections()),
                cache_hit_rate=cache_hit_rate,
                cache_memory_usage=cache_stats,
                api_response_time=api_response_time,
                database_query_time=db_query_time,
                strategy_execution_time=strategy_execution_time
            )

        except Exception as e:
            logging.error(f"收集性能指标失败: {e}")
            return self._create_default_metrics()

    def _check_health(self, metrics: PerformanceMetrics) -> HealthStatus:
        """检查系统健康状态"""
        checks = {}
        issues = []

        # CPU检查
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            checks['cpu'] = {'status': 'critical', 'value': metrics.cpu_percent}
            issues.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            checks['cpu'] = {'status': 'warning', 'value': metrics.cpu_percent}
            issues.append(f"CPU使用率较高: {metrics.cpu_percent:.1f}%")
        else:
            checks['cpu'] = {'status': 'healthy', 'value': metrics.cpu_percent}

        # 内存检查
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            checks['memory'] = {'status': 'critical', 'value': metrics.memory_percent}
            issues.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            checks['memory'] = {'status': 'warning', 'value': metrics.memory_percent}
            issues.append(f"内存使用率较高: {metrics.memory_percent:.1f}%")
        else:
            checks['memory'] = {'status': 'healthy', 'value': metrics.memory_percent}

        # 磁盘检查
        if metrics.disk_usage_percent >= self.thresholds['disk_critical']:
            checks['disk'] = {'status': 'critical', 'value': metrics.disk_usage_percent}
            issues.append(f"磁盘使用率过高: {metrics.disk_usage_percent:.1f}%")
        elif metrics.disk_usage_percent >= self.thresholds['disk_warning']:
            checks['disk'] = {'status': 'warning', 'value': metrics.disk_usage_percent}
            issues.append(f"磁盘使用率较高: {metrics.disk_usage_percent:.1f}%")
        else:
            checks['disk'] = {'status': 'healthy', 'value': metrics.disk_usage_percent}

        # API响应时间检查
        if metrics.api_response_time >= self.thresholds['api_response_critical']:
            checks['api_response'] = {'status': 'critical', 'value': metrics.api_response_time}
            issues.append(f"API响应时间过长: {metrics.api_response_time:.2f}s")
        elif metrics.api_response_time >= self.thresholds['api_response_warning']:
            checks['api_response'] = {'status': 'warning', 'value': metrics.api_response_time}
            issues.append(f"API响应时间较慢: {metrics.api_response_time:.2f}s")
        else:
            checks['api_response'] = {'status': 'healthy', 'value': metrics.api_response_time}

        # 数据库查询时间检查
        if metrics.database_query_time >= self.thresholds['db_query_critical']:
            checks['database'] = {'status': 'critical', 'value': metrics.database_query_time}
            issues.append(f"数据库查询时间过长: {metrics.database_query_time:.2f}s")
        elif metrics.database_query_time >= self.thresholds['db_query_warning']:
            checks['database'] = {'status': 'warning', 'value': metrics.database_query_time}
            issues.append(f"数据库查询时间较慢: {metrics.database_query_time:.2f}s")
        else:
            checks['database'] = {'status': 'healthy', 'value': metrics.database_query_time}

        # 缓存命中率检查
        if metrics.cache_hit_rate <= self.thresholds['cache_hit_critical']:
            checks['cache'] = {'status': 'critical', 'value': metrics.cache_hit_rate}
            issues.append(f"缓存命中率过低: {metrics.cache_hit_rate:.2%}")
        elif metrics.cache_hit_rate <= self.thresholds['cache_hit_warning']:
            checks['cache'] = {'status': 'warning', 'value': metrics.cache_hit_rate}
            issues.append(f"缓存命中率较低: {metrics.cache_hit_rate:.2%}")
        else:
            checks['cache'] = {'status': 'healthy', 'value': metrics.cache_hit_rate}

        # 确定整体状态
        if any(check['status'] == 'critical' for check in checks.values()):
            status = 'critical'
        elif any(check['status'] == 'warning' for check in checks.values()):
            status = 'warning'
        else:
            status = 'healthy'

        message = '; '.join(issues) if issues else '系统运行正常'

        return HealthStatus(
            timestamp=datetime.now(),
            status=status,
            checks=checks,
            message=message
        )

    def _create_default_metrics(self) -> PerformanceMetrics:
        """创建默认指标"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_used_mb=0.0,
            disk_usage_percent=0.0,
            network_io={},
            active_connections=0,
            cache_hit_rate=0.0,
            cache_memory_usage={},
            api_response_time=0.0,
            database_query_time=0.0,
            strategy_execution_time=0.0
        )

    def _calculate_average_response_time(self) -> float:
        """计算平均API响应时间"""
        with self.lock:
            all_times = []
            for times in self.stats['api_response_times'].values():
                all_times.extend(times)

            if not all_times:
                return 0.0

            # 只保留最近100次调用的时间
            recent_times = all_times[-100:]
            return sum(recent_times) / len(recent_times)

    def _calculate_average_db_query_time(self) -> float:
        """计算平均数据库查询时间"""
        with self.lock:
            all_times = []
            for times in self.stats['db_query_times'].values():
                all_times.extend(times)

            if not all_times:
                return 0.0

            recent_times = all_times[-100:]
            return sum(recent_times) / len(recent_times)

    def _calculate_average_strategy_time(self) -> float:
        """计算平均策略执行时间"""
        with self.lock:
            all_times = []
            for times in self.stats['strategy_execution_times'].values():
                all_times.extend(times)

            if not all_times:
                return 0.0

            recent_times = all_times[-100:]
            return sum(recent_times) / len(recent_times)

    def record_api_call(self, endpoint: str, response_time: float):
        """记录API调用"""
        with self.lock:
            self.stats['api_calls'][endpoint] += 1
            self.stats['api_response_times'][endpoint].append(response_time)

            # 限制历史记录数量
            if len(self.stats['api_response_times'][endpoint]) > 100:
                self.stats['api_response_times'][endpoint] = self.stats['api_response_times'][endpoint][-100:]

    def record_db_query(self, query_type: str, query_time: float):
        """记录数据库查询"""
        with self.lock:
            self.stats['db_queries'][query_type] += 1
            self.stats['db_query_times'][query_type].append(query_time)

            if len(self.stats['db_query_times'][query_type]) > 100:
                self.stats['db_query_times'][query_type] = self.stats['db_query_times'][query_type][-100:]

    def record_strategy_execution(self, strategy_name: str, execution_time: float):
        """记录策略执行"""
        with self.lock:
            self.stats['strategy_executions'][strategy_name] += 1
            self.stats['strategy_execution_times'][strategy_name].append(execution_time)

            if len(self.stats['strategy_execution_times'][strategy_name]) > 100:
                self.stats['strategy_execution_times'][strategy_name] = self.stats['strategy_execution_times'][
                                                                            strategy_name][-100:]

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None

    def get_current_health(self) -> Optional[HealthStatus]:
        """获取当前健康状态"""
        with self.lock:
            if self.health_history:
                return self.health_history[-1]
            return None

    def get_metrics_history(self, hours: int = 24) -> List[PerformanceMetrics]:
        """获取历史性能指标"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_health_history(self, hours: int = 24) -> List[HealthStatus]:
        """获取历史健康状态"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            return [h for h in self.health_history if h.timestamp >= cutoff_time]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                'api_calls': dict(self.stats['api_calls']),
                'db_queries': dict(self.stats['db_queries']),
                'strategy_executions': dict(self.stats['strategy_executions']),
                'metrics_count': len(self.metrics_history),
                'health_count': len(self.health_history)
            }


# 全局监控器实例
_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def monitor_performance(operation_type: str = "general"):
    """性能监控装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time

                if operation_type == "api":
                    # 从函数名或参数中提取endpoint
                    endpoint = func.__name__
                    monitor.record_api_call(endpoint, execution_time)
                elif operation_type == "database":
                    query_type = func.__name__
                    monitor.record_db_query(query_type, execution_time)
                elif operation_type == "strategy":
                    strategy_name = func.__name__
                    monitor.record_strategy_execution(strategy_name, execution_time)

        return wrapper

    return decorator


# 便捷装饰器
def monitor_api(func):
    """API性能监控装饰器"""
    return monitor_performance("api")(func)


def monitor_database(func):
    """数据库性能监控装饰器"""
    return monitor_performance("database")(func)


def monitor_strategy(func):
    """策略性能监控装饰器"""
    return monitor_performance("strategy")(func)
