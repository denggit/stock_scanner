#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通道类策略实现模块
包含基于通道分析的各种交易策略
"""

from .rising_channel import RisingChannelStrategy, create_rising_channel_strategy

__all__ = [
    'RisingChannelStrategy',
    'create_rising_channel_strategy'
]
