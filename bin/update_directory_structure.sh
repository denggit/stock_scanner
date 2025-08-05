#!/bin/bash
# 目录结构更新脚本
# 用于自动更新项目目录结构文档

echo "🔄 正在更新项目目录结构..."

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 运行Python更新脚本
python scripts/update_directory_structure.py

echo "✅ 目录结构更新完成！" 