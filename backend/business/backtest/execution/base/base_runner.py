#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用回测运行器
抽取多策略将共用的回测编排逻辑：数据获取、主数据选择、回测执行、结果打印与报表保存、参数优化与策略对比。
使用方式：各策略只需提供`strategy_class`与`config_cls`，即可快速复用。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from math import ceil
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from backend.business.backtest import DataUtils, ReportUtils
from backend.business.backtest.core import BacktestEngine
from backend.business.backtest.database import ChannelDBAdapter
from backend.utils.logger import setup_logger


class BaseBacktestRunner:
    """
    通用回测运行器基类
    - 设计目标：最小化不同策略Runner间的重复代码，统一回测流程编排
    - 依赖注入：通过`strategy_class`与`config_cls`注入策略与配置，保持高内聚低耦合
    
    Attributes:
        strategy_class: 策略类（必须为backtrader策略类）
        config_cls: 配置类，需提供以下类方法：
            - get_log_config()
            - get_environment_config(environment: str)
            - get_strategy_params(max_positions: Optional[int])
            - get_report_config()
            - get_optimization_config()
            - get_optimization_ranges()
            - get_strategy_variants()  (可选，用于对比回测)
        environment: 运行环境 development/optimization/production/full_backtest
        config: 当前环境下的配置字典
        strategy_params: 当前环境对应的策略参数
        logger: 日志记录器
    """

    def __init__(
            self,
            strategy_class: Type,
            config_cls: Type,
            log_level: int = logging.INFO,
            environment: Optional[str] = None,
    ) -> None:
        """
        初始化通用回测运行器
        
        Args:
            strategy_class: 策略类
            config_cls: 策略配置类
            log_level: 日志级别
            environment: 运行环境，默认读取`BACKTEST_ENV`
        """
        self.strategy_class = strategy_class
        self.config_cls = config_cls

        # 日志初始化
        log_config = self.config_cls.get_log_config()
        self.logger = setup_logger(log_config.get('logger_name', 'backtest'), log_level=log_level)

        # 环境配置
        if environment is None:
            environment = os.getenv('BACKTEST_ENV', 'development')
        self.environment = environment
        self.config = self.config_cls.get_environment_config(environment)

        # 策略参数 - 使用环境配置中的策略覆盖参数
        environment_overrides = self.config.get('strategy_overrides', {})
        self.strategy_params = self.config_cls.get_strategy_params(environment_overrides)

        # 初始化缓存适配器
        self.cache_adapter = self._init_cache_adapter()

        # 记录关键信息
        self.logger.info(f"环境配置: {self.environment}")
        self.logger.info(f"最大股票数量: {self.config.get('max_stocks')}")
        self.logger.info(f"最大持仓数量: {self.strategy_params.get('max_positions')}")
        self.logger.info(f"配置描述: {self.config.get('description')}")
        if environment_overrides:
            self.logger.info(f"环境策略覆盖: {environment_overrides}")

    def _init_cache_adapter(self):
        """
        初始化通道数据库适配器（兼容旧变量名cache_adapter）
        
        Returns:
            ChannelDBAdapter: 通道数据库适配器实例
        """
        try:
            # 检查是否需要启用缓存（可以通过环境变量控制）
            enable_cache = os.getenv('BACKTEST_ENABLE_CACHE', 'true').lower() == 'true'

            if not enable_cache:
                self.logger.info("通道数据库适配器已禁用")
                return None

            # 初始化数据库适配器
            cache_adapter = ChannelDBAdapter()

            self.logger.info("上升通道数据库适配器初始化成功")
            return cache_adapter

        except Exception as e:
            self.logger.warning(f"通道数据库适配器初始化失败，将使用原始计算: {e}")
            return None

    # 预加载功能已移除，改为逐日按需读写

    def _extract_channel_params(self, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取AscendingChannelRegression的纯算法参数
        
        这些参数决定了通道计算的结果，应该作为缓存的键
        不包含策略层面的参数（如min_channel_score等）
        
        Args:
            strategy_params: 策略参数（实际不使用，从算法默认配置获取）
            
        Returns:
            Dict: 通道算法参数
        """
        try:
            # 直接从AscendingChannelRegression获取默认配置
            from backend.business.factor.core.engine.library.channel_analysis.rising_channel import \
                AscendingChannelRegression

            # 创建一个临时实例来获取默认配置
            temp_analyzer = AscendingChannelRegression()
            algorithm_params = temp_analyzer._get_config_dict()

            self.logger.info(f"从AscendingChannelRegression获取算法参数: {algorithm_params}")
            return algorithm_params

        except Exception as e:
            self.logger.warning(f"无法从AscendingChannelRegression获取参数，使用硬编码默认值: {e}")

            # 回退到硬编码的算法参数（与AscendingChannelRegression._get_default_config()一致）
            return {
                'k': 2.0,
                'L_max': 120,
                'delta_cut': 5,
                'pivot_m': 3,
                'gain_trigger': 0.30,
                'beta_delta': 0.15,
                'break_days': 3,
                'reanchor_fail_max': 2,
                'min_data_points': 60,
                'R2_min': 0.20,
                'width_pct_min': 0.04,
                'width_pct_max': 0.12
            }

    # ------------------------ 对外主流程 ------------------------
    def run_basic_backtest(self) -> Optional[Dict[str, Any]]:
        """
        运行基础多股票回测
        
        Returns:
            回测结果字典
        """
        self.logger.info("开始运行多股票回测...")
        try:
            extended_start = self._compute_extended_start_date(self.config['start_date'])
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=extended_start,
                end_date=self.config['end_date'],
                max_stocks=self.config['max_stocks'],
                min_data_days=self.config['min_data_days']
            )
            self.logger.info(f"成功获取 {len(stock_data_dict)} 只股票的数据")
            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            results = self._run_multi_stock_backtest(stock_data_dict)
            self._print_results(results)
            self._save_report(results, "multi_stock_backtest")
            return results
        except Exception as e:
            self.logger.exception(f"多股票回测失败: {e}")
            return None

    def run_parameter_optimization(self) -> Optional[Dict[str, Any]]:
        """
        运行参数优化
        
        Returns:
            优化结果字典
        """
        self.logger.info("开始运行参数优化...")
        try:
            opt_cfg = self.config_cls.get_optimization_config()
            extended_start = self._compute_extended_start_date(self.config['start_date'])
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=extended_start,
                end_date=self.config['end_date'],
                max_stocks=opt_cfg['max_stocks_for_optimization'],
                min_data_days=self.config['min_data_days']
            )
            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            param_ranges = self.config_cls.get_optimization_ranges()
            results = self._run_parameter_optimization(stock_data_dict, param_ranges)
            self._print_optimization_results(results)
            self._save_optimization_report(results)
            return results
        except Exception as e:
            self.logger.exception(f"参数优化失败: {e}")
            return None

    def run_comparison_backtest(self) -> Optional[Dict[str, Any]]:
        """
        运行对比回测（如保守/标准/激进）
        """
        self.logger.info("开始运行对比回测...")
        try:
            extended_start = self._compute_extended_start_date(self.config['start_date'])
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=extended_start,
                end_date=self.config['end_date'],
                max_stocks=self.config.get('max_stocks', 50) or 50,
                min_data_days=self.config['min_data_days']
            )
            if not stock_data_dict:
                self.logger.error("没有获取到有效的股票数据")
                return None

            variants = {}
            if hasattr(self.config_cls, 'get_strategy_variants'):
                variants = self.config_cls.get_strategy_variants()

            strategies: List[Dict[str, Any]] = []
            if 'conservative' in variants:
                strategies.append(variants['conservative'])
            strategies.append({'name': '标准策略', 'params': self.strategy_params})
            if 'aggressive' in variants:
                strategies.append(variants['aggressive'])

            results = self._run_comparison_backtest(stock_data_dict, strategies)
            self._print_comparison_results(results)
            self._save_comparison_report(results)
            return results
        except Exception as e:
            self.logger.exception(f"对比回测失败: {e}")
            return None

    # ------------------------ 具体实现 ------------------------
    def _run_multi_stock_backtest(
            self,
            stock_data_dict: Dict[str, pd.DataFrame],
            strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        多股票回测核心逻辑：
        - 选主数据（时间轴）
        - 添加策略（将全部股票数据通过参数传入策略）
        - 运行并回收策略信息与交易记录
        """
        if strategy_params is None:
            strategy_params = self.strategy_params

        engine = BacktestEngine(
            initial_cash=self.config['initial_cash'],
            commission=self.config['commission']
        )

        main_code = self._select_main_stock(stock_data_dict)
        main_data = stock_data_dict[main_code]
        self.logger.info(f"选择主数据源: {main_code}")

        engine.add_data(main_data, name=main_code)

        engine.add_strategy(
            self.strategy_class,
            stock_data_dict=stock_data_dict,
            cache_adapter=self.cache_adapter,
            effective_start_date=self.config['start_date'],
            **strategy_params
        )

        # 释放大对象引用，降低峰值内存占用（不影响功能/配置）
        stock_data_dict = None  # hint GC

        self.logger.info("开始运行回测...")
        results = engine.run(f"多股票_{self.strategy_class.__name__}")

        strat_instance = self._get_strategy_instance(engine)
        if strat_instance is not None:
            try:
                if hasattr(strat_instance, 'get_strategy_info'):
                    results['strategy_info'] = strat_instance.get_strategy_info()
                if hasattr(strat_instance, 'get_performance_summary'):
                    results['performance'] = strat_instance.get_performance_summary()
                if 'performance' in results and 'trades' in results['performance']:
                    results['trades'] = results['performance']['trades']
                    self.logger.info(f"成功获取 {len(results['trades'])} 笔交易记录")
                elif hasattr(strat_instance, 'trades'):
                    results['trades'] = getattr(strat_instance, 'trades', [])
                    self.logger.info(f"从策略实例获取 {len(results['trades'])} 笔交易记录")
            except Exception as e:
                self.logger.warning(f"附加策略信息失败: {e}")

        return results

    def _select_main_stock(self, stock_data_dict: Dict[str, pd.DataFrame]) -> str:
        """选择数据最长的股票作为主数据源"""
        best_stock = None
        max_length = 0
        for code, df in stock_data_dict.items():
            if len(df) > max_length:
                max_length = len(df)
                best_stock = code
        return best_stock

    def _compute_extended_start_date(self, original_start_date: str) -> str:
        """
        根据策略的 min_data_points 计算向前扩展的开始日期，避免策略内跳过。
        规则：ceil((min_data_points + 20) * 1.5) 天；且不早于 2020-01-01（数据最早可用日）。
        """
        try:
            min_pts = int(self.strategy_params.get('min_data_points', 60))
        except Exception:
            min_pts = 60
        extend_days = ceil((min_pts + 20) * 1.5)
        try:
            base = datetime.strptime(original_start_date, '%Y-%m-%d')
        except Exception:
            base = datetime.now()
        extended = base - timedelta(days=extend_days)
        earliest = datetime(2020, 1, 1)
        if extended < earliest:
            extended = earliest
        return extended.strftime('%Y-%m-%d')

    def _get_strategy_instance(self, engine: BacktestEngine):
        """从引擎取回刚刚运行的策略实例（若可得）"""
        try:
            if hasattr(engine, '_last_run_results') and engine._last_run_results:
                return engine._last_run_results[0]
        except Exception as e:
            self.logger.warning(f"获取策略实例失败: {e}")
        return None

    def _run_parameter_optimization(self, stock_data_dict: Dict[str, pd.DataFrame],
                                    parameter_ranges: Dict[str, List[Any]]):
        """
        穷举参数组合并以总收益率为目标筛选最佳参数
        改进版本：每次基础回测完成后立即保存报告，避免程序中断导致数据丢失
        """
        import itertools
        names = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        combos = list(itertools.product(*values))
        self.logger.info(f"总共 {len(combos)} 个参数组合")

        results: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        best_value = float('-inf')

        # 创建增量报告文件
        report_cfg = self.config_cls.get_report_config()
        report_dir = report_cfg['report_dir']
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incremental_filename = f"{report_dir}/{report_cfg['file_prefix']}_optimization_incremental_{timestamp}.xlsx"

        # 用于跟踪进度的文件
        progress_filename = f"{report_dir}/{report_cfg['file_prefix']}_optimization_progress_{timestamp}.json"

        self.logger.info(f"增量报告文件: {incremental_filename}")
        self.logger.info(f"进度跟踪文件: {progress_filename}")

        for i, combo in enumerate(combos):
            params = dict(zip(names, combo))
            try:
                self.logger.info(f"测试参数组合 {i + 1}/{len(combos)}: {params}")
                result = self._run_multi_stock_backtest(stock_data_dict, params)
                target_value = 0
                if result and 'metrics' in result:
                    target_value = result['metrics'].get('总收益率', 0) or 0

                # 创建结果记录
                result_record = {
                    'parameters': params,
                    'result': result,
                    'target_value': target_value,
                    'combo_index': i,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result_record)

                # 更新最优结果
                if target_value > best_value:
                    best_value = target_value
                    best = {'parameters': params, 'result': result, 'target_value': target_value}
                    self.logger.info(f"发现新的最优参数: {params}, 收益率: {target_value:.2f}%")

                # 每次完成后立即保存增量报告
                self._save_incremental_optimization_report(
                    results, best, incremental_filename, i + 1, len(combos)
                )

                # 保存进度信息
                self._save_optimization_progress(
                    progress_filename, i + 1, len(combos), best, results
                )

            except Exception as e:
                self.logger.warning(f"参数组合 {params} 测试失败: {e}")
                # 即使失败也要保存当前进度
                self._save_optimization_progress(
                    progress_filename, i + 1, len(combos), best, results
                )
                continue

        # 最终保存完整报告
        final_filename = f"{report_dir}/{report_cfg['file_prefix']}_optimization_final_{timestamp}.xlsx"
        self._save_final_optimization_report(results, best, final_filename, len(combos))

        # 清理中间态报告文件
        self._cleanup_intermediate_reports(incremental_filename, progress_filename)

        self.logger.info(f"参数优化完成，最终报告已保存: {final_filename}")
        self.logger.info("中间态报告文件已清理")

        return {
            'optimization_results': results,
            'best_params': best['parameters'] if best else {},
            'best_value': best['target_value'] if best else 0,
            'total_combinations': len(combos),
            'final_report': final_filename
        }

    def _run_comparison_backtest(self, stock_data_dict: Dict[str, pd.DataFrame], strategies: List[Dict[str, Any]]):
        """依次运行不同参数配置的同一策略，汇总比较报告"""
        strategy_results: Dict[str, Dict[str, Any]] = {}
        for cfg in strategies:
            name = cfg['name']
            params = cfg['params']
            self.logger.info(f"运行策略: {name}")
            try:
                strategy_results[name] = self._run_multi_stock_backtest(stock_data_dict, params)
            except Exception as e:
                self.logger.error(f"策略 {name} 运行失败: {e}")
        comparison_report = self._generate_comparison_report(strategy_results)
        return {'strategy_results': strategy_results, 'comparison_report': comparison_report}

    # ------------------------ 输出与报表 ------------------------
    def _print_results(self, results: Optional[Dict[str, Any]]) -> None:
        if not results:
            return
        print("\n" + "=" * 60)
        print("多股票策略回测结果")
        print("=" * 60)
        if 'report' in results:
            print(results['report'])
        metrics = results.get('metrics', {})
        if metrics:
            def safe_get(key, default=0):
                v = metrics.get(key, default)
                return v if v is not None else default

            print("\n关键指标:")
            print(f"  总收益率: {safe_get('总收益率', 0):.2f}%")
            print(f"  夏普比率: {safe_get('夏普比率', 0):.4f}")
            print(f"  最大回撤: {safe_get('最大回撤', 0):.2f}%")
            print(f"  交易次数: {safe_get('交易次数', 0)}")
            print(f"  胜率: {safe_get('胜率', 0):.2f}%")
        if 'strategy_info' in results:
            info = results['strategy_info']
            print(f"\n策略信息:")
            print(f"  当前持仓数量: {info.get('current_status', {}).get('position_count', 0)}")
            print(f"  策略参数: {info.get('parameters', {})}")

    def _print_optimization_results(self, results: Optional[Dict[str, Any]]) -> None:
        if not results:
            return
        print("\n" + "=" * 60)
        print("参数优化结果")
        print("=" * 60)
        print(f"总参数组合数: {results['total_combinations']}")
        print(f"最优参数: {results['best_params']}")
        print(f"最优收益率: {results['best_value']:.2f}%")
        print("\n前5个最优结果:")
        sorted_results = sorted(results['optimization_results'], key=lambda x: x['target_value'], reverse=True)
        for i, r in enumerate(sorted_results[:5]):
            def safe_get_metrics(key, default=0):
                metrics = r['result'].get('metrics', {}) if r.get('result') else {}
                v = metrics.get(key, default)
                return v if v is not None else default

            print(f"{i + 1}. 参数: {r['parameters']}")
            print(f"   收益率: {r['target_value']:.2f}%")
            print(f"   夏普比率: {safe_get_metrics('夏普比率', 0):.4f}")
            print(f"   最大回撤: {safe_get_metrics('最大回撤', 0):.2f}%")
            print(f"   交易次数: {safe_get_metrics('交易次数', 0)}")
            print(f"   胜率: {safe_get_metrics('胜率', 0):.2f}%\n")

    def _print_comparison_results(self, results: Optional[Dict[str, Any]]) -> None:
        if not results:
            return
        print("\n" + "=" * 60)
        print("策略对比结果")
        print("=" * 60)
        if 'comparison_report' in results:
            print(results['comparison_report'])
        strategy_results = results.get('strategy_results', {})
        for name, res in strategy_results.items():
            metrics = res.get('metrics', {})

            def safe_get(key, default=0):
                v = metrics.get(key, default)
                return v if v is not None else default

            print(f"\n{name}:")
            print(f"  总收益率: {safe_get('总收益率', 0):.2f}%")
            print(f"  夏普比率: {safe_get('夏普比率', 0):.4f}")
            print(f"  最大回撤: {safe_get('最大回撤', 0):.2f}%")
            print(f"  交易次数: {safe_get('交易次数', 0)}")
            print(f"  胜率: {safe_get('胜率', 0):.2f}%")

    def _generate_comparison_report(self, strategy_results: Dict[str, Dict[str, Any]]) -> str:
        lines: List[str] = ["=== 策略对比报告 ===", "", "策略名称\t总收益率\t夏普比率\t最大回撤\t交易次数\t胜率",
                            "-" * 80]
        for name, res in strategy_results.items():
            summary = res.get('metrics', {})

            def safe_get(key, default=0):
                v = summary.get(key, default)
                return v if v is not None else default

            lines.append(
                f"{name}\t{safe_get('总收益率', 0):.2f}%\t{safe_get('夏普比率', 0):.4f}\t{safe_get('最大回撤', 0):.2f}%\t{safe_get('交易次数', 0)}\t{safe_get('胜率', 0):.2f}%"
            )
        return "\n".join(lines)

    # ------------------------ 报表保存 ------------------------
    def _save_report(self, results: Optional[Dict[str, Any]], report_type: str) -> None:
        if not results:
            return
        try:
            report_cfg = self.config_cls.get_report_config()
            report_dir = report_cfg['report_dir']
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_dir}/{report_cfg['file_prefix']}_{report_type}_{timestamp}.xlsx"
            ReportUtils.save_report_to_excel(results, filename)
            self.logger.info(f"报告已保存: {filename}")
        except Exception as e:
            self.logger.exception(f"保存报告失败: {e}")

    def _save_optimization_report(self, results: Optional[Dict[str, Any]]) -> None:
        if not results:
            return
        try:
            report_cfg = self.config_cls.get_report_config()
            report_dir = report_cfg['report_dir']
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_dir}/{report_cfg['file_prefix']}_optimization_{timestamp}.xlsx"

            optimization_data: List[Dict[str, Any]] = []
            for r in results['optimization_results']:
                def safe_get_metrics(key, default=0):
                    metrics = r['result'].get('metrics', {}) if r.get('result') else {}
                    v = metrics.get(key, default)
                    return v if v is not None else default

                row = {
                    '参数组合': str(r['parameters']),
                    '收益率': r.get('target_value', 0),
                    '夏普比率': safe_get_metrics('夏普比率', 0),
                    '最大回撤': safe_get_metrics('最大回撤', 0),
                    '交易次数': safe_get_metrics('交易次数', 0),
                    '胜率': safe_get_metrics('胜率', 0)
                }
                optimization_data.append(row)
            df = pd.DataFrame(optimization_data).sort_values('收益率', ascending=False)

            with pd.ExcelWriter(filename, engine=report_cfg.get('excel_engine', 'openpyxl')) as writer:
                df.to_excel(writer, sheet_name='优化结果', index=False)
                best_df = pd.DataFrame([{
                    '最优参数': str(results.get('best_params', {})),
                    '最优收益率': results.get('best_value', 0),
                    '总组合数': results.get('total_combinations', 0)
                }])
                best_df.to_excel(writer, sheet_name='最优结果', index=False)
            self.logger.info(f"优化报告已保存: {filename}")
        except Exception as e:
            self.logger.exception(f"保存优化报告失败: {e}")

    def _save_comparison_report(self, results: Optional[Dict[str, Any]]) -> None:
        if not results:
            return
        try:
            report_cfg = self.config_cls.get_report_config()
            report_dir = report_cfg['report_dir']
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_dir}/{report_cfg['file_prefix']}_comparison_{timestamp}.xlsx"
            ReportUtils.save_report_to_excel(results, filename)
            self.logger.info(f"对比报告已保存: {filename}")
        except Exception as e:
            self.logger.exception(f"保存对比报告失败: {e}")

    def _save_incremental_optimization_report(self, results: List[Dict[str, Any]],
                                              best: Optional[Dict[str, Any]],
                                              filename: str,
                                              current_count: int,
                                              total_count: int):
        """
        保存增量优化报告
        
        Args:
            results: 当前所有结果
            best: 当前最优结果
            filename: 报告文件名
            current_count: 当前完成的组合数
            total_count: 总组合数
        """
        try:
            optimization_data: List[Dict[str, Any]] = []
            for r in results:
                def safe_get_metrics(key, default=0):
                    metrics = r['result'].get('metrics', {}) if r.get('result') else {}
                    v = metrics.get(key, default)
                    return v if v is not None else default

                row = {
                    '序号': r.get('combo_index', 0) + 1,
                    '参数组合': str(r['parameters']),
                    '收益率': r.get('target_value', 0),
                    '夏普比率': safe_get_metrics('夏普比率', 0),
                    '最大回撤': safe_get_metrics('最大回撤', 0),
                    '交易次数': safe_get_metrics('交易次数', 0),
                    '胜率': safe_get_metrics('胜率', 0),
                    '完成时间': r.get('timestamp', '')
                }
                optimization_data.append(row)

            # 按收益率排序
            df = pd.DataFrame(optimization_data).sort_values('收益率', ascending=False)

            # 创建最优结果数据
            best_data = []
            if best:
                def safe_get_metrics(key, default=0):
                    metrics = best['result'].get('metrics', {}) if best.get('result') else {}
                    v = metrics.get(key, default)
                    return v if v is not None else default

                best_data = [{
                    '最优参数': str(best.get('parameters', {})),
                    '最优收益率': best.get('target_value', 0),
                    '夏普比率': safe_get_metrics('夏普比率', 0),
                    '最大回撤': safe_get_metrics('最大回撤', 0),
                    '交易次数': safe_get_metrics('交易次数', 0),
                    '胜率': safe_get_metrics('胜率', 0),
                    '完成进度': f"{current_count}/{total_count}",
                    '更新时间': datetime.now().isoformat()
                }]

            report_cfg = self.config_cls.get_report_config()

            with pd.ExcelWriter(filename, engine=report_cfg.get('excel_engine', 'openpyxl')) as writer:
                # 优化结果表
                df.to_excel(writer, sheet_name='优化结果', index=False)

                # 最优结果表
                if best_data:
                    best_df = pd.DataFrame(best_data)
                    best_df.to_excel(writer, sheet_name='最优结果', index=False)

                # 进度信息表
                progress_df = pd.DataFrame([{
                    '总组合数': total_count,
                    '已完成数': current_count,
                    '完成比例': f"{current_count / total_count * 100:.1f}%",
                    '最优收益率': best.get('target_value', 0) if best else 0,
                    '最后更新': datetime.now().isoformat()
                }])
                progress_df.to_excel(writer, sheet_name='进度信息', index=False)

            self.logger.info(f"增量报告已更新: {filename} (进度: {current_count}/{total_count})")

        except Exception as e:
            self.logger.exception(f"保存增量报告失败: {e}")

    def _save_optimization_progress(self, filename: str, current_count: int, total_count: int,
                                    best: Optional[Dict[str, Any]], results: List[Dict[str, Any]]):
        """
        保存优化进度信息到JSON文件
        
        Args:
            filename: 进度文件名
            current_count: 当前完成的组合数
            total_count: 总组合数
            best: 当前最优结果
            results: 当前所有结果
        """
        try:
            import json

            progress_data = {
                'total_combinations': total_count,
                'completed_count': current_count,
                'completion_percentage': current_count / total_count * 100,
                'best_parameters': best.get('parameters', {}) if best else {},
                'best_value': best.get('target_value', 0) if best else 0,
                'last_update': datetime.now().isoformat(),
                'results_summary': [
                    {
                        'combo_index': r.get('combo_index', 0),
                        'parameters': r.get('parameters', {}),
                        'target_value': r.get('target_value', 0),
                        'timestamp': r.get('timestamp', '')
                    }
                    for r in results
                ]
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.exception(f"保存进度信息失败: {e}")

    def _save_final_optimization_report(self, results: List[Dict[str, Any]],
                                        best: Optional[Dict[str, Any]],
                                        filename: str,
                                        total_count: int):
        """
        保存最终优化报告
        
        Args:
            results: 所有结果
            best: 最优结果
            filename: 报告文件名
            total_count: 总组合数
        """
        try:
            optimization_data: List[Dict[str, Any]] = []
            for r in results:
                def safe_get_metrics(key, default=0):
                    metrics = r['result'].get('metrics', {}) if r.get('result') else {}
                    v = metrics.get(key, default)
                    return v if v is not None else default

                row = {
                    '序号': r.get('combo_index', 0) + 1,
                    '参数组合': str(r['parameters']),
                    '收益率': r.get('target_value', 0),
                    '夏普比率': safe_get_metrics('夏普比率', 0),
                    '最大回撤': safe_get_metrics('最大回撤', 0),
                    '交易次数': safe_get_metrics('交易次数', 0),
                    '胜率': safe_get_metrics('胜率', 0),
                    '完成时间': r.get('timestamp', '')
                }
                optimization_data.append(row)

            # 按收益率排序
            df = pd.DataFrame(optimization_data).sort_values('收益率', ascending=False)

            report_cfg = self.config_cls.get_report_config()

            with pd.ExcelWriter(filename, engine=report_cfg.get('excel_engine', 'openpyxl')) as writer:
                # 优化结果表
                df.to_excel(writer, sheet_name='优化结果', index=False)

                # 最优结果表
                if best:
                    def safe_get_metrics(key, default=0):
                        metrics = best['result'].get('metrics', {}) if best.get('result') else {}
                        v = metrics.get(key, default)
                        return v if v is not None else default

                    best_df = pd.DataFrame([{
                        '最优参数': str(best.get('parameters', {})),
                        '最优收益率': best.get('target_value', 0),
                        '夏普比率': safe_get_metrics('夏普比率', 0),
                        '最大回撤': safe_get_metrics('最大回撤', 0),
                        '交易次数': safe_get_metrics('交易次数', 0),
                        '胜率': safe_get_metrics('胜率', 0),
                        '总组合数': total_count,
                        '完成时间': datetime.now().isoformat()
                    }])
                    best_df.to_excel(writer, sheet_name='最优结果', index=False)

                # 统计信息表
                stats_df = pd.DataFrame([{
                    '总参数组合数': total_count,
                    '成功完成数': len(results),
                    '成功率': f"{len(results) / total_count * 100:.1f}%",
                    '最优收益率': best.get('target_value', 0) if best else 0,
                    '平均收益率': df['收益率'].mean() if not df.empty else 0,
                    '收益率标准差': df['收益率'].std() if not df.empty else 0,
                    '完成时间': datetime.now().isoformat()
                }])
                stats_df.to_excel(writer, sheet_name='统计信息', index=False)

            self.logger.info(f"最终优化报告已保存: {filename}")

        except Exception as e:
            self.logger.exception(f"保存最终报告失败: {e}")

    def _cleanup_intermediate_reports(self, incremental_filename: str, progress_filename: str):
        """
        清理中间态报告文件
        
        Args:
            incremental_filename: 增量报告文件名
            progress_filename: 进度文件文件名
        """
        try:
            # 删除增量报告文件
            if os.path.exists(incremental_filename):
                os.remove(incremental_filename)
                self.logger.info(f"已删除增量报告文件: {incremental_filename}")

            # 删除进度文件
            if os.path.exists(progress_filename):
                os.remove(progress_filename)
                self.logger.info(f"已删除进度文件: {progress_filename}")

        except Exception as e:
            self.logger.warning(f"清理中间态报告文件失败: {e}")
            # 不抛出异常，避免影响主流程
