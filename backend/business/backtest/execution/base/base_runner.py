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
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from backend.business.backtest import DataUtils, ReportUtils
from backend.business.backtest.core import BacktestEngine
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

        # 策略参数
        self.strategy_params = self.config_cls.get_strategy_params(self.config.get('max_positions'))

        # 记录关键信息
        self.logger.info(f"环境配置: {self.environment}")
        self.logger.info(f"最大股票数量: {self.config.get('max_stocks')}")
        self.logger.info(f"最大持仓数量: {self.config.get('max_positions')}")
        self.logger.info(f"配置描述: {self.config.get('description')}")

    # ------------------------ 对外主流程 ------------------------
    def run_basic_backtest(self) -> Optional[Dict[str, Any]]:
        """
        运行基础多股票回测
        
        Returns:
            回测结果字典
        """
        self.logger.info("开始运行多股票回测...")
        try:
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=self.config['start_date'],
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
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=self.config['start_date'],
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
            stock_data_dict = DataUtils.get_stock_data_for_backtest(
                stock_pool=self.config['stock_pool'],
                start_date=self.config['start_date'],
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
        engine.add_strategy(self.strategy_class, stock_data_dict=stock_data_dict, **strategy_params)

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
        """穷举参数组合并以总收益率为目标筛选最佳参数"""
        import itertools
        names = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        combos = list(itertools.product(*values))
        self.logger.info(f"总共 {len(combos)} 个参数组合")

        results: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        best_value = float('-inf')

        for i, combo in enumerate(combos):
            params = dict(zip(names, combo))
            try:
                self.logger.info(f"测试参数组合 {i + 1}/{len(combos)}: {params}")
                result = self._run_multi_stock_backtest(stock_data_dict, params)
                target_value = 0
                if result and 'metrics' in result:
                    target_value = result['metrics'].get('总收益率', 0) or 0
                results.append({'parameters': params, 'result': result, 'target_value': target_value})
                if target_value > best_value:
                    best_value = target_value
                    best = {'parameters': params, 'result': result, 'target_value': target_value}
            except Exception as e:
                self.logger.warning(f"参数组合 {params} 测试失败: {e}")
                continue

        return {
            'optimization_results': results,
            'best_params': best['parameters'] if best else {},
            'best_value': best['target_value'] if best else 0,
            'total_combinations': len(combos)
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
