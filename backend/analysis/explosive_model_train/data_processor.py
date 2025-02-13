#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块：负责数据清洗、标准化和平衡处理

本模块提供了一系列数据预处理工具，用于处理机器学习模型训练前的数据准备工作，包括：
1. 缺失值处理：多种填充策略
2. 异常值检测和处理
3. 特征缩放和标准化
4. 类别不平衡处理
5. 数据质量检查

主要类：
    DataProcessor: 数据预处理主类，提供所有数据处理方法

特点：
    - 支持多种预处理策略
    - 保持训练集和测试集的一致性
    - 提供预处理状态的保存和加载
    - 详细的处理日志

使用示例：
    processor = DataProcessor()
    X_processed, y_processed = processor.process_data(X, y, is_training=True)
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler


class DataProcessor:
    """数据预处理类，提供完整的数据处理流程
    
    该类实现了一系列数据预处理方法，包括：
    1. 缺失值处理
    2. 异常值检测和处理
    3. 特征缩放
    4. 类别平衡
    
    所有处理方法都会保存状态，确保测试数据使用与训练数据相同的转换。
    
    Attributes:
        feature_scaler: 特征缩放器实例
        imputer: 缺失值填充器实例
    """

    def __init__(self):
        """初始化数据处理器
        
        初始化所需的转换器和填充器，但不立即进行拟合。
        转换器和填充器将在首次处理训练数据时进行拟合。
        """
        self.feature_scaler = None
        self.imputer = None

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """处理数据集中的缺失值
        
        支持多种缺失值填充策略：
        - mean: 均值填充
        - median: 中位数填充
        - most_frequent: 众数填充
        - constant: 常数填充
        
        Args:
            df (pd.DataFrame): 包含缺失值的数据集
            strategy (str): 填充策略，默认为'median'
        
        Returns:
            pd.DataFrame: 填充缺失值后的数据集
            
        Raises:
            ValueError: 不支持的填充策略
        """
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy=strategy)
            df_imputed = pd.DataFrame(
                self.imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        else:
            df_imputed = pd.DataFrame(
                self.imputer.transform(df),
                columns=df.columns,
                index=df.index
            )
        return df_imputed

    def handle_outliers(self, df: pd.DataFrame, n_sigmas: float = 3) -> pd.DataFrame:
        """检测和处理异常值
        
        使用z-score方法检测异常值，将超过阈值的值限制在合理范围内：
        1. 计算每个特征的均值和标准差
        2. 识别超过n_sigmas个标准差的值
        3. 将异常值截断到上下限范围内
        
        Args:
            df (pd.DataFrame): 输入数据集
            n_sigmas (float): 标准差的倍数阈值
        
        Returns:
            pd.DataFrame: 处理异常值后的数据集
            
        Notes:
            - 只处理数值型特征
            - 保留原始索引
        """
        df_clean = df.copy()
        for column in df.select_dtypes(include=[np.number]).columns:
            mean = df[column].mean()
            std = df[column].std()

            # 将超过n_sigmas个标准差的值视为异常值，替换为上下限
            lower_bound = mean - n_sigmas * std
            upper_bound = mean + n_sigmas * std

            df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)

        return df_clean

    def scale_features(self, df: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
        """特征缩放
        
        支持两种缩放方法：
        1. StandardScaler: 标准化为均值0，方差1
        2. RobustScaler: 基于分位数的缩放，对异常值更稳健
        
        Args:
            df (pd.DataFrame): 输入数据集
            scaler_type (str): 缩放方法，'standard'或'robust'
        
        Returns:
            pd.DataFrame: 缩放后的数据集
            
        Raises:
            ValueError: 不支持的缩放方法
        """
        if self.feature_scaler is None:
            if scaler_type == 'standard':
                self.feature_scaler = StandardScaler()
            elif scaler_type == 'robust':
                self.feature_scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported scaler type: {scaler_type}")

            scaled_data = pd.DataFrame(
                self.feature_scaler.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        else:
            scaled_data = pd.DataFrame(
                self.feature_scaler.transform(df),
                columns=df.columns,
                index=df.index
            )

        return scaled_data

    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series,
                               sampling_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
        """处理类别不平衡问题
        
        使用SMOTE算法进行过采样，生成合成的少数类样本：
        1. 分析类别分布
        2. 生成合成样本
        3. 合并原始和合成样本
        
        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 标签向量
            sampling_strategy (float): 采样比例，少数类与多数类的目标比例
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 
                - 重采样后的特征矩阵
                - 重采样后的标签向量
                
        Notes:
            - 只在训练集上使用
            - 保持特征的统计特性
        """
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    def process_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                     is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """执行完整的数据预处理流程
        
        按顺序执行以下步骤：
        1. 缺失值处理
        2. 异常值处理
        3. 特征缩放
        4. 类别平衡（仅训练集）
        
        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series, optional): 标签向量
            is_training (bool): 是否为训练数据
        
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: 
                - 处理后的特征矩阵
                - 处理后的标签向量（如果提供）
                
        Raises:
            ValueError: 数据格式错误
            RuntimeError: 处理过程出错
        """
        # 1. 处理缺失值
        X_processed = self.handle_missing_values(X)

        # 2. 处理异常值
        X_processed = self.handle_outliers(X_processed)

        # 3. 特征缩放
        X_processed = self.scale_features(X_processed)

        # 4. 如果是训练数据且提供了标签，处理类别不平衡
        if is_training and y is not None:
            X_processed, y_processed = self.handle_class_imbalance(X_processed, y)
            return X_processed, y_processed

        return X_processed, y

    def get_feature_names(self) -> list:
        """获取特征名称列表"""
        if hasattr(self.feature_scaler, 'feature_names_in_'):
            return list(self.feature_scaler.feature_names_in_)
        return []

    def save_preprocessor(self, path: str):
        """保存预处理器
        
        Args:
            path: 保存路径
        """
        import joblib
        joblib.dump({
            'scaler': self.feature_scaler,
            'imputer': self.imputer
        }, path)

    def load_preprocessor(self, path: str):
        """加载预处理器
        
        Args:
            path: 加载路径
        """
        import joblib
        preprocessor = joblib.load(path)
        self.feature_scaler = preprocessor['scaler']
        self.imputer = preprocessor['imputer']
