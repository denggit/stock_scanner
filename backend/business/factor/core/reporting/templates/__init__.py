#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : __init__.py
@Description: 报告模板管理模块
@Author     : Zijun Deng
@Date       : 2025-08-23
"""

from .html_templates import HTMLTemplateManager
from .base_template import BaseTemplate

__all__ = [
    'HTMLTemplateManager',
    'BaseTemplate'
]
