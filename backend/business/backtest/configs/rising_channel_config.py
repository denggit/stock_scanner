#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上升通道策略配置文件
包含所有策略相关的常量配置
"""

import os
from datetime import datetime
from typing import Dict, Any


class RisingChannelConfig:
    """
    上升通道策略配置类
    集中管理所有策略相关的常量
    """

    # ==================== 环境配置 ====================
    ENVIRONMENTS = {
        "development": {
            "max_stocks": 100,
            "description": "开发环境 - 快速验证策略逻辑",
            "max_positions": 5,
        },
        "optimization": {
            "max_stocks": 1000,
            "description": (
                "优化基线环境（仅在单次回测/对比回测中生效）。\n"
                "注意：当运行'参数优化'流程时，不使用此处的 max_stocks 与 max_positions；\n"
                "优化流程会使用 OPTIMIZATION_CONFIG.max_stocks_for_optimization 控制抽样规模，"
                "并使用 OPTIMIZATION_RANGES 里的参数网格进行穷举。"
            ),
            "max_positions": 20,
        },
        "production": {
            "max_stocks": None,  # 不限制
            "description": "生产环境 - 全量股票回测",
            "max_positions": 20,  # 默认值
        },
        "full_backtest": {
            "max_stocks": None,
            "description": "完整回测 - 大量股票测试",
            "max_positions": 20,
        }
    }

    # ==================== 基础配置 ====================
    BASE_CONFIG = {
        'initial_cash': 200000.0,  # 初始资金20万
        'commission': 0.0003,  # 手续费率
        'stock_pool': 'no_st',  # 股票池：非ST股票
        'start_date': '2024-01-01',  # 开始日期
        'end_date': datetime.today().strftime("%Y-%m-%d"),  # 结束日期
        'min_data_days': 120  # 最小数据天数
    }

    # ==================== 策略参数 ====================
    STRATEGY_PARAMS = {
        'min_channel_score': 60.0,  # 最小通道评分
        'k': 2.0,  # 通道斜率参数
        'L_max': 120,  # 最大通道长度
        'gain_trigger': 0.30,  # 收益触发阈值
        'beta_delta': 0.15,  # Beta变化阈值
        'R2_min': 0.20,  # 最小R²值（用于通道有效性判定）；若在选股阶段想取消下限，可将选股用的 R2_min 设为 None
        'R2_max': 0.4,  # 最大R²值上限（仅用于选股过滤；None 表示不设上限）
        'width_pct_min': 0.04,  # 最小通道宽度
        'width_pct_max': 0.20  # 最大通道宽度 - 调整为更宽松的值
    }

    # ==================== 参数优化范围 ====================
    OPTIMIZATION_RANGES = {
        'max_positions': [10, 15, 20],  # 持仓数量范围
        # 'min_channel_score': [60.0],  # 通道评分范围
        # # 'k': [1.5, 2.0, 2.5],  # 通道斜率范围
        # 'gain_trigger': [0.25, 0.30, 0.35],  # 收益触发阈值范围
        # # 'R2_min': [0.15, 0.20, 0.25],  # 最小R²值范围（用于通道有效性判定）
        # 'width_pct_min': [0.03, 0.04, 0.05],  # 最小通道宽度范围
        # 新增：R2 区间搜索（选股过滤用）。示例：[[0.15, 0.30], [0.20, 0.35], [None, None]]
        'R2_range': [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, None]]
    }

    # ==================== 策略变体配置 ====================
    STRATEGY_VARIANTS = {
        'conservative': {
            'name': '保守策略',
            'params': {
                'max_positions': 30,  # 较少持仓
                'min_channel_score': 70.0,  # 更高评分要求
                'k': 1.5,  # 较低斜率
                'gain_trigger': 0.25,  # 较低收益触发
                'R2_min': 0.25,  # 更高R²要求
                'width_pct_min': 0.05,  # 更宽通道要求
                'width_pct_max': 0.18  # 最大通道宽度
            }
        },
        'aggressive': {
            'name': '激进策略',
            'params': {
                'max_positions': 70,  # 更多持仓
                'min_channel_score': 50.0,  # 较低评分要求
                'k': 2.5,  # 较高斜率
                'gain_trigger': 0.35,  # 较高收益触发
                'R2_min': 0.15,  # 较低R²要求
                'width_pct_min': 0.03,  # 较窄通道要求
                'width_pct_max': 0.25  # 最大通道宽度
            }
        }
    }

    # ==================== 距离计算配置 ====================
    DISTANCE_CONFIG = {
        'fallback_distance_invalid': 10.0,  # 下沿无效时的固定距离
        'fallback_distance_no_state': 100.0,  # 无通道状态时的固定距离
        'lower_price_ratio_invalid': 0.9,  # 下沿无效时的价格比例
        'lower_price_ratio_no_state': 0.0,  # 无通道状态时的价格比例
        'min_distance_below_lower': 0.1,  # 价格低于下沿时的最小距离
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

    # ==================== 优化配置 ====================
    OPTIMIZATION_CONFIG = {
        'max_stocks_for_optimization': None,  # 优化时使用的股票数量
        'target_metric': 'total_return',  # 优化目标指标
    }

    # 说明：
    # - 当调用 run_basic_backtest / run_comparison_backtest 时，将使用 ENVIRONMENTS[env] 中的 max_stocks 与 max_positions。
    # - 当调用 run_parameter_optimization（参数优化）时：
    #   * 股票样本由 OPTIMIZATION_CONFIG['max_stocks_for_optimization'] 控制，而非 ENVIRONMENTS[env]['max_stocks']；
    #   * 参与穷举的策略参数取自 OPTIMIZATION_RANGES 指定的键；未在网格中的键不会自动继承 ENVIRONMENTS 的值，
    #     将回落到策略类默认值（例如 RisingChannelStrategy.params 中的默认值）。

    # ==================== 通道分析配置 ====================
    CHANNEL_ANALYSIS_CONFIG = {
        'delta_cut': 5,  # 切割参数
        'pivot_m': 3,  # 枢轴参数
        'break_days': 3,  # 突破天数
        'reanchor_fail_max': 2,  # 重锚定失败最大次数
        'min_data_points': 60,  # 最小数据点数
    }

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
    def get_strategy_params(cls, max_positions: int = None) -> Dict[str, Any]:
        """
        获取策略参数
        
        Args:
            max_positions: 最大持仓数量，如果为None则使用默认值
            
        Returns:
            策略参数字典
        """
        params = cls.STRATEGY_PARAMS.copy()

        # 添加通道分析配置
        params.update(cls.CHANNEL_ANALYSIS_CONFIG)

        # 如果指定了max_positions，则使用指定值
        if max_positions is not None:
            params['max_positions'] = max_positions

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
    def get_channel_analysis_config(cls) -> Dict[str, Any]:
        """
        获取通道分析配置
        
        Returns:
            通道分析配置字典
        """
        return cls.CHANNEL_ANALYSIS_CONFIG.copy()


# 导出配置实例
config = RisingChannelConfig()
