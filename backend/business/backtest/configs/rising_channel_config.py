#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道策略配置文件
包含所有策略相关的常量配置

配置层次说明：
1. BASE_CONFIG: 基础配置（回测框架级别）
2. ENVIRONMENTS: 环境配置（运行环境级别）
3. STRATEGY_PARAMS: 策略参数（策略级别，可被环境配置覆盖）
4. OPTIMIZATION_RANGES: 参数优化范围
5. STRATEGY_VARIANTS: 策略变体配置
"""

import os
from datetime import datetime
from typing import Dict, Any


class RisingChannelConfig:
    """
    上升通道策略配置类
    集中管理所有策略相关的常量
    
    配置优先级（从高到低）：
    1. 用户传入的params参数
    2. ENVIRONMENTS[environment]中的配置
    3. STRATEGY_PARAMS中的默认值
    4. BASE_CONFIG中的基础值
    """

    # ==================== 基础配置 ====================
    BASE_CONFIG = {
        'initial_cash': 200000.0,  # 初始资金20万
        'commission': 0.0003,  # 手续费率
        'stock_pool': 'no_st',  # 股票池：非ST股票
        'start_date': '2020-01-01',  # 开始日期
        'end_date': datetime.today().strftime("%Y-%m-%d"),  # 结束日期
        'min_data_days': 120  # 最小数据天数
    }

    # ==================== 环境配置 ====================
    ENVIRONMENTS = {
        "development": {
            "max_stocks": 500,
            "description": "开发环境 - 快速验证策略逻辑",
            # 环境级别的策略参数覆盖
            "strategy_overrides": {
                "max_positions": 20,  # 覆盖策略默认值
                "min_channel_score": 60.0,  # 覆盖策略默认值
            }
        },
        "optimization": {
            "max_stocks": 1000,
            "description": (
                "优化基线环境（仅在单次回测/对比回测中生效）。\n"
                "注意：当运行'参数优化'流程时，不使用此处的 max_stocks 与 strategy_overrides；\n"
                "优化流程会使用 OPTIMIZATION_CONFIG.max_stocks_for_optimization 控制抽样规模，"
                "并使用 OPTIMIZATION_RANGES 里的参数网格进行穷举。"
            ),
            "strategy_overrides": {
                "max_positions": 20,
                "min_channel_score": 60.0,
            }
        },
        "production": {
            "max_stocks": None,  # 不限制
            "description": "生产环境 - 全量股票回测",
            "strategy_overrides": {
                "max_positions": 20,  # 生产环境默认值
                "min_channel_score": 60.0,
            }
        },
        "full_backtest": {
            "max_stocks": None,
            "description": "完整回测 - 大量股票测试",
            "strategy_overrides": {
                "max_positions": 20,
                "min_channel_score": 60.0,
            }
        }
    }

    # ==================== 策略参数（默认值） ====================
    STRATEGY_PARAMS = {
        # 策略基础参数
        'max_positions': 50,  # 最大持仓数量（默认值，可被环境配置覆盖）
        'min_data_points': 60,  # 最小数据点数
        'min_channel_score': 60.0,  # 最小通道评分（默认值，可被环境配置覆盖）
        'enable_logging': True,  # 是否启用日志

        # 卖出规则参数
        'sell_on_close_breakout': True,  # 是否使用收盘价突破通道上沿作为卖出条件（True=收盘价，False=最高价）

        # 通道分析参数
        'k': 2.0,  # 通道斜率参数
        'L_max': 120,  # 最大回看天数
        'delta_cut': 5,  # 切割参数
        'pivot_m': 3,  # 枢轴参数
        'R2_min': 0.70,  # 最小R²值（用于通道有效性判定）；若在选股阶段想取消下限，可将选股用的 R2_min 设为 None
        'R2_max': 0.95,  # 最大R²值上限（仅用于选股过滤；None 表示不设上限）
        'R2_range': None,  # 参数优化时可传入 [R2_min, R2_max]，两者均可为 None
        'width_pct_min': 0.05,  # 最小通道宽度
        'width_pct_max': 0.12,  # 最大通道宽度
        
        # 数据预处理参数
        'adjust': 1,  # 复权类型：1-后复权，2-前复权，3-不复权
        'logarithm': False,  # 是否使用对数价格计算通道
    }

    # ==================== 参数优化范围 ====================
    OPTIMIZATION_RANGES = {
        # 'max_positions': [20],  # 持仓数量范围
        # 'min_channel_score': [60.0],  # 通道评分范围
        # # 'k': [1.5, 2.0, 2.5],  # 通道斜率范围
        # # 'R2_min': [0.15, 0.20, 0.25],  # 最小R²值范围（用于通道有效性判定）
        # 'width_pct_min': [0.04, 0.05],  # 最小通道宽度范围
        # 'width_pct_max': [0.12, 0.15],  # 最大通道宽度范围
        # 'R2_range': [[0.2, 0.45], [0.45, 0.70], [0.7, 0.95]]
        # 'R2_min': [0.2, 0.35, 0.45]
    }

    # ==================== 策略变体配置 ====================
    STRATEGY_VARIANTS = {
        'conservative': {
            'name': '保守策略',
            'params': {
                'max_positions': 30,  # 较少持仓
                'min_channel_score': 70.0,  # 更高评分要求
                'k': 1.5,  # 较低斜率
                # （已移除 gain_trigger）
                'R2_min': 0.25,  # 更高R²要求
                'width_pct_min': 0.05,  # 更宽通道要求
                'width_pct_max': 0.18,  # 最大通道宽度
                'sell_on_close_breakout': True,  # 保守策略使用收盘价突破，避免盘中波动
            }
        },
        'aggressive': {
            'name': '激进策略',
            'params': {
                'max_positions': 70,  # 更多持仓
                'min_channel_score': 50.0,  # 较低评分要求
                'k': 2.5,  # 较高斜率
                # （已移除 gain_trigger）
                'R2_min': 0.15,  # 较低R²要求
                'width_pct_min': 0.03,  # 较窄通道要求
                'width_pct_max': 0.25,  # 最大通道宽度
                'sell_on_close_breakout': False,  # 激进策略使用最高价突破，快速响应
            }
        }
    }

    # ==================== 距离计算配置 ====================
    DISTANCE_CONFIG = {
        'fallback_distance_invalid': 10.0,  # 下沿无效时的固定距离
        'fallback_distance_no_state': 100.0,  # 无通道状态时的固定距离
        'lower_price_ratio_invalid': 0.9,  # 下沿无效时的价格比例
        'lower_price_ratio_no_state': 0.0,  # 无通道状态时的价格比例
    }

    # ==================== 日志配置 ====================
    LOG_CONFIG = {
        'logger_name': 'backtest',
        'default_level': 'INFO',
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }

    # ==================== 报告配置 ====================
    REPORT_CONFIG = {
        'report_dir': os.path.abspath(os.path.join(os.path.dirname(__file__), '../backtest_reports')),
        'file_prefix': 'rising_channel',
        'excel_engine': 'openpyxl'
    }

    # ==================== 预筛选配置 ====================
    PREFILTER_CONFIG = {
        'enable_prefilter': True,  # 是否启用预筛选
        'min_stocks_for_prefilter': 100,  # 触发预筛选的最小股票数量
        'ma_period': 60,  # 移动平均线周期
        'lookback_days': 20,  # 回看天数
        'volume_threshold': 1.5,  # 成交量放大倍数阈值
        'min_conditions_met': 2,  # 至少满足的条件数量
        'enable_volume_check': False,  # 是否启用成交量检查（可选条件）
    }

    # ==================== 优化配置 ====================
    OPTIMIZATION_CONFIG = {
        'max_stocks_for_optimization': 1000,  # 优化时使用的股票数量
        'target_metric': 'total_return',  # 优化目标指标
    }

    # 说明：
    # - 当调用 run_basic_backtest / run_comparison_backtest 时，将使用 ENVIRONMENTS[env] 中的 max_stocks 与 strategy_overrides。
    # - 当调用 run_parameter_optimization（参数优化）时：
    #   * 股票样本由 OPTIMIZATION_CONFIG['max_stocks_for_optimization'] 控制，而非 ENVIRONMENTS[env]['max_stocks']；
    #   * 参与穷举的策略参数取自 OPTIMIZATION_RANGES 指定的键；未在网格中的键不会自动继承 ENVIRONMENTS 的值，
    #     将回落到策略类默认值（例如 RisingChannelStrategy.params 中的默认值）。

    @classmethod
    def get_environment_config(cls, environment: str, params: dict = None) -> Dict[str, Any]:
        """
        根据环境获取配置
        
        Args:
            environment: 运行环境
            params: 可选，用户自定义参数字典，可覆盖默认配置
            
        Returns:
            环境配置字典
        """
        # 获取基础配置
        base_config = cls.BASE_CONFIG.copy()

        # 获取环境特定配置
        if environment in cls.ENVIRONMENTS:
            env_config = cls.ENVIRONMENTS[environment]
            base_config.update(env_config)
        else:
            # 默认使用开发环境配置
            env_config = cls.ENVIRONMENTS["development"]
            base_config.update(env_config)

        # 如果传入了params参数，则覆盖base_config中的对应项
        if params is not None:
            for k, v in params.items():
                base_config[k] = v

        return base_config

    @classmethod
    def get_strategy_params(cls, environment_overrides: dict = None) -> Dict[str, Any]:
        """
        获取策略参数
        
        Args:
            environment_overrides: 环境级别的策略参数覆盖，如果为None则使用默认值
            
        Returns:
            策略参数字典
        """
        params = cls.STRATEGY_PARAMS.copy()

        # 如果传入了环境覆盖参数，则覆盖默认值
        if environment_overrides is not None:
            for k, v in environment_overrides.items():
                if k in params:  # 只覆盖策略参数中存在的键
                    params[k] = v

        return params

    @classmethod
    def get_optimization_ranges(cls) -> Dict[str, list]:
        """
        获取参数优化范围
        
        Returns:
            参数优化范围字典
        """
        return cls.OPTIMIZATION_RANGES.copy()

    @classmethod
    def get_strategy_variants(cls) -> Dict[str, Dict]:
        """
        获取策略变体配置
        
        Returns:
            策略变体配置字典
        """
        return cls.STRATEGY_VARIANTS.copy()

    @classmethod
    def get_distance_config(cls) -> Dict[str, Any]:
        """
        获取距离计算配置
        
        Returns:
            距离计算配置字典
        """
        return cls.DISTANCE_CONFIG.copy()

    @classmethod
    def get_log_config(cls) -> Dict[str, str]:
        """
        获取日志配置

        Returns:
            日志配置字典
        """
        return cls.LOG_CONFIG.copy()

    @classmethod
    def get_report_config(cls) -> Dict[str, str]:
        """
        获取报告配置
        
        Returns:
            报告配置字典
        """
        return cls.REPORT_CONFIG.copy()

    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        """
        获取优化配置
        
        Returns:
            优化配置字典
        """
        return cls.OPTIMIZATION_CONFIG.copy()

    @classmethod
    def get_prefilter_config(cls) -> Dict[str, Any]:
        """
        获取预筛选配置
        
        Returns:
            预筛选配置字典
        """
        return cls.PREFILTER_CONFIG.copy()

    @classmethod
    def get_channel_analysis_config(cls) -> Dict[str, Any]:
        """
        获取通道分析配置（已废弃，请使用get_strategy_params）
        
        Returns:
            通道分析配置字典
        """
        # 从STRATEGY_PARAMS中提取通道分析相关参数
        strategy_params = cls.get_strategy_params()
        channel_params = {
            'delta_cut': strategy_params.get('delta_cut'),
            'pivot_m': strategy_params.get('pivot_m'),
            'min_data_points': strategy_params.get('min_data_points'),
        }
        return channel_params


# 导出配置实例
config = RisingChannelConfig()
