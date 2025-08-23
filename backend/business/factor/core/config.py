#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : config.py
@Description: 因子计算系统配置文件
@Author     : Zijun Deng
@Date       : 2025-08-21
"""
import datetime
import os
from datetime import date
from typing import Dict, Any

# =============================================================================
# 基础配置
# =============================================================================

# 默认日期配置
DEFAULT_START_DATE = '2025-01-01'
DEFAULT_END_DATE = date.today().strftime("%Y-%m-%d")

# 默认股票池配置
DEFAULT_STOCK_POOL = 'sz50'
AVAILABLE_STOCK_POOLS = [
    'full', '全量股票',
    'no_st', '非ST股票', 
    'st', 'ST股票',
    'sz50', '上证50',
    'hs300', '沪深300',
    'zz500', '中证500'
]

# =============================================================================
# 回测参数配置
# =============================================================================

# 选股参数
DEFAULT_TOP_N = 10
DEFAULT_N_GROUPS = 5

# 回测时间配置
DEFAULT_BACKTEST_START_DATE = '2020-01-01'
DEFAULT_BACKTEST_END_DATE = datetime.date.today().strftime("%Y-%m-%d")

# 回测频率配置
DEFAULT_REBALANCE_FREQ = 'daily'  # daily, weekly, monthly
DEFAULT_HOLDING_PERIOD = 1  # 持仓期数

# =============================================================================
# 因子计算配置
# =============================================================================

# 因子计算批次大小
DEFAULT_BATCH_SIZE = 10
MAX_BATCH_SIZE = 50

# 并行计算配置
DEFAULT_USE_PARALLEL = True
DEFAULT_MAX_WORKERS = None  # None表示自动检测CPU核心数
MIN_WORKERS = 1
MAX_WORKERS_LIMIT = 16

# 数据获取配置
DEFAULT_OPTIMIZE_DATA_FETCH = True  # 重新启用智能优化
# 对于WorldQuant因子，也使用智能数据获取
DEFAULT_OPTIMIZE_DATA_FETCH_FOR_WORLDQUANT = True
DEFAULT_DATA_FIELDS = [
    'code', 'trade_date', 'open', 'high', 'low', 'close', 'preclose',
    'volume', 'amount', 'turn', 'tradestatus', 'pct_chg', 'pe_ttm',
    'pb_mrq', 'ps_ttm', 'pcf_ncf_ttm', 'is_st', 'vwap'
]

# =============================================================================
# 因子预处理配置
# =============================================================================

# 标准化配置
DEFAULT_STANDARDIZATION_METHOD = 'zscore'  # zscore, minmax, rank
AVAILABLE_STANDARDIZATION_METHODS = ['zscore', 'minmax', 'rank']

# 去极值配置
DEFAULT_WINSORIZE_LIMITS = (0.01, 0.99)  # (lower_percentile, upper_percentile)
DEFAULT_WINSORIZE_METHOD = 'quantile'  # quantile, mad, zscore

# 中性化配置
DEFAULT_NEUTRALIZATION_METHOD = 'industry_market_cap'  # industry, market_cap, industry_market_cap

# =============================================================================
# 数据管理配置
# =============================================================================

# 数据缓存配置
DEFAULT_CACHE_ENABLED = True
DEFAULT_CACHE_EXPIRE_HOURS = 24
DEFAULT_CACHE_DIR = 'storage/cache'

# 数据质量配置
DEFAULT_MIN_DATA_QUALITY_SCORE = 0.8
DEFAULT_MAX_MISSING_RATIO = 0.3
DEFAULT_MIN_STOCK_COUNT = 100

# 财务数据配置
DEFAULT_FINANCIAL_DATA_LOOKBACK_YEARS = 2
DEFAULT_FINANCIAL_DATA_QUARTERLY = True

# =============================================================================
# 报告生成配置
# =============================================================================

# 报告输出配置
DEFAULT_REPORT_OUTPUT_DIR = 'storage/reports'
DEFAULT_REPORT_FORMAT = 'html'  # html, pdf, excel
AVAILABLE_REPORT_FORMATS = ['html', 'pdf', 'excel']

# 报告内容配置
DEFAULT_INCLUDE_CHARTS = True
DEFAULT_INCLUDE_TABLES = True
DEFAULT_INCLUDE_STATISTICS = True

# 图表配置
DEFAULT_CHART_STYLE = 'seaborn'
DEFAULT_CHART_SIZE = (12, 8)
DEFAULT_CHART_DPI = 100

# =============================================================================
# 日志配置
# =============================================================================

# 日志级别配置
DEFAULT_LOG_LEVEL = 'INFO'
AVAILABLE_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# 日志输出配置
DEFAULT_LOG_TO_FILE = True
DEFAULT_LOG_FILE_PATH = 'storage/logs/factor_calculation.log'
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# 性能配置
# =============================================================================

# 内存配置
DEFAULT_MAX_MEMORY_USAGE_GB = 8.0
DEFAULT_CHUNK_SIZE = 10000

# 计算配置
DEFAULT_NUMERIC_PRECISION = 'float64'  # float32, float64
DEFAULT_USE_NUMBA = False  # 是否使用numba加速

# =============================================================================
# 错误处理配置
# =============================================================================

# 错误处理策略
DEFAULT_ERROR_HANDLING_STRATEGY = 'continue'  # continue, stop, retry
DEFAULT_MAX_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 1

# 异常阈值配置
DEFAULT_MAX_FAILURE_RATIO = 0.1  # 最大失败比例
DEFAULT_MIN_SUCCESS_COUNT = 1  # 最小成功数量

# =============================================================================
# 配置验证函数
# =============================================================================

def validate_config() -> Dict[str, Any]:
    """
    验证配置参数的有效性
    
    Returns:
        验证结果字典
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # 验证日期格式
    try:
        from datetime import datetime
        datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d")
        datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d")
    except ValueError as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"日期格式错误: {e}")
    
    # 验证股票池
    if DEFAULT_STOCK_POOL not in AVAILABLE_STOCK_POOLS:
        validation_results['warnings'].append(f"股票池 {DEFAULT_STOCK_POOL} 不在推荐列表中")
    
    # 验证数值参数
    if DEFAULT_TOP_N <= 0:
        validation_results['valid'] = False
        validation_results['errors'].append("TOP_N 必须大于0")
    
    if DEFAULT_N_GROUPS <= 0:
        validation_results['valid'] = False
        validation_results['errors'].append("N_GROUPS 必须大于0")
    
    if DEFAULT_BATCH_SIZE <= 0 or DEFAULT_BATCH_SIZE > MAX_BATCH_SIZE:
        validation_results['valid'] = False
        validation_results['errors'].append(f"BATCH_SIZE 必须在 1-{MAX_BATCH_SIZE} 之间")
    
    # 验证并行配置
    if DEFAULT_MAX_WORKERS is not None:
        if DEFAULT_MAX_WORKERS < MIN_WORKERS or DEFAULT_MAX_WORKERS > MAX_WORKERS_LIMIT:
            validation_results['warnings'].append(f"MAX_WORKERS 建议在 {MIN_WORKERS}-{MAX_WORKERS_LIMIT} 之间")
    
    # 验证标准化方法
    if DEFAULT_STANDARDIZATION_METHOD not in AVAILABLE_STANDARDIZATION_METHODS:
        validation_results['valid'] = False
        validation_results['errors'].append(f"标准化方法 {DEFAULT_STANDARDIZATION_METHOD} 不支持")
    
    # 验证报告格式
    if DEFAULT_REPORT_FORMAT not in AVAILABLE_REPORT_FORMATS:
        validation_results['valid'] = False
        validation_results['errors'].append(f"报告格式 {DEFAULT_REPORT_FORMAT} 不支持")
    
    # 验证日志级别
    if DEFAULT_LOG_LEVEL not in AVAILABLE_LOG_LEVELS:
        validation_results['valid'] = False
        validation_results['errors'].append(f"日志级别 {DEFAULT_LOG_LEVEL} 不支持")
    
    return validation_results

def get_config_summary() -> Dict[str, Any]:
    """
    获取配置摘要
    
    Returns:
        配置摘要字典
    """
    return {
        'dates': {
            'start_date': DEFAULT_START_DATE,
            'end_date': DEFAULT_END_DATE,
            'backtest_start': DEFAULT_BACKTEST_START_DATE,
            'backtest_end': DEFAULT_BACKTEST_END_DATE
        },
        'stock_pool': {
            'default': DEFAULT_STOCK_POOL,
            'available': AVAILABLE_STOCK_POOLS
        },
        'backtest': {
            'top_n': DEFAULT_TOP_N,
            'n_groups': DEFAULT_N_GROUPS,
            'rebalance_freq': DEFAULT_REBALANCE_FREQ,
            'holding_period': DEFAULT_HOLDING_PERIOD
        },
        'factor_calculation': {
            'batch_size': DEFAULT_BATCH_SIZE,
            'use_parallel': DEFAULT_USE_PARALLEL,
            'max_workers': DEFAULT_MAX_WORKERS,
            'optimize_data_fetch': DEFAULT_OPTIMIZE_DATA_FETCH
        },
        'preprocessing': {
            'standardization_method': DEFAULT_STANDARDIZATION_METHOD,
            'winsorize_limits': DEFAULT_WINSORIZE_LIMITS,
            'neutralization_method': DEFAULT_NEUTRALIZATION_METHOD
        },
        'performance': {
            'max_memory_usage_gb': DEFAULT_MAX_MEMORY_USAGE_GB,
            'chunk_size': DEFAULT_CHUNK_SIZE,
            'numeric_precision': DEFAULT_NUMERIC_PRECISION,
            'use_numba': DEFAULT_USE_NUMBA
        },
        'reporting': {
            'output_dir': DEFAULT_REPORT_OUTPUT_DIR,
            'format': DEFAULT_REPORT_FORMAT,
            'include_charts': DEFAULT_INCLUDE_CHARTS,
            'include_tables': DEFAULT_INCLUDE_TABLES
        },
        'logging': {
            'level': DEFAULT_LOG_LEVEL,
            'to_file': DEFAULT_LOG_TO_FILE,
            'file_path': DEFAULT_LOG_FILE_PATH
        }
    }

# =============================================================================
# 环境变量覆盖
# =============================================================================

def load_from_env():
    """从环境变量加载配置"""
    import os
    
    # 日期配置
    if os.getenv('FACTOR_START_DATE'):
        global DEFAULT_START_DATE
        DEFAULT_START_DATE = os.getenv('FACTOR_START_DATE')
    
    if os.getenv('FACTOR_END_DATE'):
        global DEFAULT_END_DATE
        DEFAULT_END_DATE = os.getenv('FACTOR_END_DATE')
    
    # 股票池配置
    if os.getenv('FACTOR_STOCK_POOL'):
        global DEFAULT_STOCK_POOL
        DEFAULT_STOCK_POOL = os.getenv('FACTOR_STOCK_POOL')
    
    # 回测参数配置
    if os.getenv('FACTOR_TOP_N'):
        global DEFAULT_TOP_N
        DEFAULT_TOP_N = int(os.getenv('FACTOR_TOP_N'))
    
    if os.getenv('FACTOR_N_GROUPS'):
        global DEFAULT_N_GROUPS
        DEFAULT_N_GROUPS = int(os.getenv('FACTOR_N_GROUPS'))
    
    # 批次大小配置
    if os.getenv('FACTOR_BATCH_SIZE'):
        global DEFAULT_BATCH_SIZE
        DEFAULT_BATCH_SIZE = int(os.getenv('FACTOR_BATCH_SIZE'))
    
    # 并行配置
    if os.getenv('FACTOR_USE_PARALLEL'):
        global DEFAULT_USE_PARALLEL
        DEFAULT_USE_PARALLEL = os.getenv('FACTOR_USE_PARALLEL').lower() == 'true'
    
    if os.getenv('FACTOR_MAX_WORKERS'):
        global DEFAULT_MAX_WORKERS
        DEFAULT_MAX_WORKERS = int(os.getenv('FACTOR_MAX_WORKERS'))
    
    # 日志配置
    if os.getenv('FACTOR_LOG_LEVEL'):
        global DEFAULT_LOG_LEVEL
        DEFAULT_LOG_LEVEL = os.getenv('FACTOR_LOG_LEVEL').upper()

# 初始化时加载环境变量
load_from_env()

# =============================================================================
# 配置验证
# =============================================================================

# 启动时验证配置
validation_result = validate_config()
if not validation_result['valid']:
    raise ValueError(f"配置验证失败: {validation_result['errors']}")

if validation_result['warnings']:
    import warnings
    for warning in validation_result['warnings']:
        warnings.warn(warning)
