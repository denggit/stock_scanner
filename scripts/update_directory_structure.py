#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录结构自动更新脚本

该脚本用于自动扫描项目目录结构并更新docs/项目目录结构.md文件。
当项目文件发生变化时，运行此脚本可以自动更新目录结构文档。

作者: AI Assistant
创建时间: 2024-12-19
"""

import re
import sys
from datetime import datetime
from pathlib import Path


class DirectoryStructureUpdater:
    """目录结构更新器类"""

    def __init__(self, project_root: str = "."):
        """
        初始化目录结构更新器
        
        Args:
            project_root (str): 项目根目录路径，默认为当前目录
        """
        self.project_root = Path(project_root).resolve()
        self.output_file = self.project_root / "docs" / "项目目录结构.md"
        self.exclude_dirs = {
            '.git', '__pycache__', '.ipynb_checkpoints',
            'venv', '.idea', 'node_modules', '.vscode'
        }
        self.exclude_files = {
            '.DS_Store', 'Thumbs.db', '.gitignore', '.gitattributes'
        }
        self.simplify_dirs = {'logs', 'results', 'docs'}

    def should_exclude(self, path: Path) -> bool:
        """
        判断是否应该排除该路径
        
        Args:
            path (Path): 文件或目录路径
            
        Returns:
            bool: 是否应该排除
        """
        # 排除隐藏文件和系统文件
        if path.name.startswith('.'):
            return True

        # 排除特定目录
        if path.is_dir() and path.name in self.exclude_dirs:
            return True

        # 排除特定文件
        if path.is_file() and path.name in self.exclude_files:
            return True

        return False

    def get_file_icon(self, path: Path) -> str:
        """
        根据文件类型获取图标
        
        Args:
            path (Path): 文件路径
            
        Returns:
            str: 文件图标
        """
        if path.is_dir():
            return "📁"
        elif path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
            return "📄"
        elif path.suffix in ['.md', '.txt', '.rst']:
            return "📄"
        elif path.suffix in ['.yaml', '.yml', '.json', '.xml']:
            return "📄"
        elif path.suffix in ['.sh', '.bat', '.ps1']:
            return "📄"
        else:
            return "📄"

    def get_description(self, path: Path) -> str:
        """
        获取文件或目录的描述
        
        Args:
            path (Path): 文件或目录路径
            
        Returns:
            str: 描述信息
        """
        name = path.name

        # 目录描述
        if path.is_dir():
            descriptions = {
                'backend': '后端核心代码',
                'pages': '前端页面',
                'scripts': '脚本工具',
                'bin': '可执行脚本',
                'analysis': '分析模块',
                'docs': '文档目录',
                'logs': '日志文件目录',
                'results': '结果输出目录',
                'strategies': '交易策略模块',
                'source': '数据源模块',
                'services': '业务服务层',
                'utils': '工具模块',
                'configs': '配置模块',
                'data': '数据管理模块',
                'quant': '量化交易核心',
                'interface': '接口层',
                'ml': '机器学习模块',
                'backtest': '回测模块',
                'core': '核心引擎',
                'strategy_templates': '策略模板配置',
                'database': '数据库配置',
                'risk_constraints': '风险约束配置',
                'factor_engine': '因子引擎',
                'scoring_model': '评分模型',
                'execution': '执行引擎',
                'data_preprocessing': '数据预处理',
                'monitoring': '监控模块',
                'storage': '存储模块',
                'visualization': '可视化模块',
                'train_sz50': '上证50训练数据',
                'analyze_explosive_20日_30%_5%回撤': '特定分析结果'
            }
            return descriptions.get(name, '')

        # 文件描述
        descriptions = {
            '__init__.py': '包初始化文件',
            'app.py': '主应用入口',
            'base.py': '策略基类',
            'strategy_template.py': '策略模板',
            'explosive_stock.py': '爆发性股票策略',
            'ma_pullback.py': '均线回调策略',
            'continuous_rise.py': '连续上涨策略',
            'swing_trading.py': '波段交易策略',
            'long_term_uptrend.py': '长期上涨趋势策略',
            'double_up.py': '翻倍策略',
            'hs_bottom.py': '底部策略',
            'breakout.py': '突破策略',
            'rising_channel.py': '上升通道策略',
            'baostock_src.py': '宝硕数据源',
            'akshare_src.py': 'AKShare数据源',
            'strategy_service.py': '策略服务',
            'stock_service.py': '股票服务',
            'backtest_service.py': '回测服务',
            'performance_monitor.py': '性能监控',
            'cache_manager.py': '缓存管理',
            'api_response.py': 'API响应处理',
            'logger.py': '日志工具',
            'indicators.py': '技术指标',
            'format_info.py': '格式化工具',
            'file_check.py': '文件检查',
            'app_config.py': '应用配置',
            'pe_validation.yaml': 'PE验证配置',
            'data_manager.py': '数据库操作',
            'data_fetcher.py': '股票数据获取',
            'data_update.py': '数据管理器',
            'strategy_interface.py': '策略接口',
            'backtest_interface.py': '回测接口',
            'stock_interface.py': '股票接口',
            'train.py': '训练脚本',
            'data_collector.py': '数据收集器',
            'model_trainer.py': '模型训练器',
            'explosive_stock_backtest.py': '爆发性股票回测',
            'strategy_scanner.py': '策略扫描页面',
            'backtest.py': '回测页面',
            'data_viewer.py': '数据查看页面',
            'factor_cal.py': '因子计算脚本',
            'factor_analysis.py': '因子分析脚本',
            'validate_factor.py': '因子验证脚本',
            'update_stock_data.py': '股票数据更新脚本',
            'update_financial_data.py': '财务数据更新脚本',
            'update_financial_data.sh': 'Shell脚本',
            'update_database.sh': '数据库更新脚本',
            'update_database.bat': 'Windows批处理脚本',
            'train_explosive_model.sh': '模型训练脚本',
            'API接口文档.md': 'API接口文档',
            '快速参考手册.md': '快速参考手册',
            '项目工具功能文档.md': '项目功能文档',
            '项目目录结构.md': '项目目录结构文档',
            'README.md': '项目说明文档',
            'requirements.txt': 'Python依赖包',
            'run_frontend.py': '前端启动脚本',
            'run_backend.py': '后端启动脚本',
            'debug_rising_channel.py': '上升通道调试脚本',
            'test_rising_channel.py': '上升通道测试脚本',
            'env_list.txt': '环境变量列表',
            '.gitignore': 'Git忽略文件',
            '.gitattributes': 'Git属性文件',
            '.DS_Store': 'macOS系统文件'
        }
        return descriptions.get(name, '')

    def generate_tree(self, path: Path, prefix: str = "", is_last: bool = True) -> str:
        """
        生成目录树结构
        
        Args:
            path (Path): 当前路径
            prefix (str): 前缀字符串
            is_last (bool): 是否为最后一个项目
            
        Returns:
            str: 目录树字符串
        """
        if self.should_exclude(path):
            return ""

        # 获取相对路径
        rel_path = path.relative_to(self.project_root)
        if str(rel_path) == ".":
            name = self.project_root.name
        else:
            name = path.name

        # 获取图标和描述
        icon = self.get_file_icon(path)
        description = self.get_description(path)
        desc_text = f" # {description}" if description else ""

        # 构建当前行
        connector = "└── " if is_last else "├── "
        line = f"{prefix}{connector}{icon} {name}{desc_text}\n"

        # 如果是简化目录，直接返回
        if path.is_dir() and path.name in self.simplify_dirs:
            return line

        # 如果是目录，递归处理子项目
        if path.is_dir():
            try:
                items = sorted([p for p in path.iterdir() if not self.should_exclude(p)])
                if not items:  # 空目录
                    return line

                for i, item in enumerate(items):
                    is_last_item = (i == len(items) - 1)
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    line += self.generate_tree(item, new_prefix, is_last_item)
            except PermissionError:
                # 处理权限错误
                pass

        return line

    def update_documentation(self):
        """
        更新目录结构文档
        """
        print("🔄 正在扫描项目目录结构...")

        # 生成目录树
        tree = self.generate_tree(self.project_root)

        # 读取现有文档模板
        template = self._get_document_template()

        # 替换目录树部分
        tree_start = template.find("```\n")
        tree_end = template.find("\n```", tree_start)

        if tree_start != -1 and tree_end != -1:
            new_content = (
                    template[:tree_start + 4] +
                    tree +
                    template[tree_end:]
            )
        else:
            # 如果找不到模板，创建新的
            new_content = self._create_new_document(tree)

        # 更新时间戳
        new_content = self._update_timestamp(new_content)

        # 写入文件
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ 目录结构已更新到: {self.output_file}")

    def _get_document_template(self) -> str:
        """
        获取文档模板
        
        Returns:
            str: 文档模板内容
        """
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return self._create_new_document("")

    def _create_new_document(self, tree: str) -> str:
        """
        创建新的文档内容
        
        Args:
            tree (str): 目录树字符串
            
        Returns:
            str: 新文档内容
        """
        return f"""# 📁 股票扫描器项目目录结构

```
{tree}
```

## 📋 项目架构说明

这是一个**功能完整的股票扫描和量化交易系统**，采用前后端分离架构：

### 🏗️ **核心架构**
- **后端** (`backend/`): 基于Python的量化交易核心引擎
- **前端** (`pages/`): Streamlit构建的Web界面
- **数据层** (`source/`, `data/`): 多数据源集成和数据库管理
- **策略层** (`strategies/`): 多种交易策略实现
- **服务层** (`services/`): 业务逻辑服务封装

### 🚀 **主要功能模块**
1. **策略扫描** - 多种技术分析策略
2. **回测系统** - 历史数据回测验证
3. **数据管理** - 股票和财务数据获取存储
4. **机器学习** - 预测模型训练
5. **量化引擎** - 因子计算和评分模型
6. **监控系统** - 性能和日志监控

### 🔧 **技术栈**
- **后端**: Python, FastAPI/Flask
- **前端**: Streamlit
- **数据**: SQLite/MySQL, Pandas, NumPy
- **量化**: TA-Lib, Scikit-learn
- **数据源**: AKShare, 宝硕数据

---

**最后更新时间**: {datetime.now().strftime('%Y年%m月%d日')}
**文档版本**: v1.0
"""

    def _update_timestamp(self, content: str) -> str:
        """
        更新时间戳
        
        Args:
            content (str): 文档内容
            
        Returns:
            str: 更新后的内容
        """
        # 更新时间戳
        timestamp_pattern = r'(\*\*最后更新时间\*\*: ).*'
        new_timestamp = f"**最后更新时间**: {datetime.now().strftime('%Y年%m月%d日')}"
        content = re.sub(timestamp_pattern, new_timestamp, content)

        return content


def main():
    """主函数"""
    print("🚀 开始更新项目目录结构...")

    # 创建更新器实例
    updater = DirectoryStructureUpdater()

    try:
        # 更新文档
        updater.update_documentation()
        print("🎉 目录结构更新完成！")
    except Exception as e:
        print(f"❌ 更新失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
