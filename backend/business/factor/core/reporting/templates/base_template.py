#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File       : base_template.py
@Description: 基础模板类
@Author     : Zijun Deng
@Date       : 2025-08-23
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BaseTemplate(ABC):
    """
    基础模板抽象类
    
    所有报告模板都应该继承此类，实现统一的模板接口
    """
    
    def __init__(self):
        """初始化模板"""
        self.template_data = {}
    
    @abstractmethod
    def render(self, **kwargs) -> str:
        """
        渲染模板
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            渲染后的内容字符串
        """
        pass
    
    def set_template_data(self, data: Dict[str, Any]) -> None:
        """
        设置模板数据
        
        Args:
            data: 模板数据字典
        """
        self.template_data.update(data)
    
    def get_template_data(self) -> Dict[str, Any]:
        """
        获取模板数据
        
        Returns:
            模板数据字典
        """
        return self.template_data.copy()
    
    def format_datetime(self, dt: Optional[datetime] = None) -> str:
        """
        格式化日期时间
        
        Args:
            dt: 日期时间对象，默认为当前时间
            
        Returns:
            格式化后的日期时间字符串
        """
        if dt is None:
            dt = datetime.now()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def format_percentage(self, value: float) -> str:
        """
        格式化百分比
        
        Args:
            value: 数值
            
        Returns:
            格式化后的百分比字符串
        """
        return f"{value:.2%}"
    
    def format_number(self, value: float, decimal_places: int = 2) -> str:
        """
        格式化数字
        
        Args:
            value: 数值
            decimal_places: 小数位数
            
        Returns:
            格式化后的数字字符串
        """
        return f"{value:.{decimal_places}f}"
    
    def escape_html(self, text: str) -> str:
        """
        HTML转义
        
        Args:
            text: 原始文本
            
        Returns:
            转义后的文本
        """
        if not text:
            return ""
        
        escape_map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }
        
        for char, escaped in escape_map.items():
            text = text.replace(char, escaped)
        
        return text
