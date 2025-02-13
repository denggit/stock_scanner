#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练模块：负责机器学习模型的训练、评估和预测

本模块提供了完整的模型训练和评估功能，支持：
1. 多种机器学习算法：XGBoost、LightGBM、Random Forest
2. 模型训练和参数调优
3. 交叉验证和性能评估
4. 特征重要性分析
5. 预测和结果输出

主要类：
    ModelTrainer: 模型训练器主类，封装所有模型相关操作

特点：
    - 支持多种主流机器学习算法
    - 提供统一的训练和评估接口
    - 详细的模型评估指标
    - 灵活的参数配置

使用示例：
    trainer = ModelTrainer(model_type='xgboost')
    train_metrics, val_metrics = trainer.train(X, y)
    predictions = trainer.predict(X_new)
"""

import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    """模型训练器类，提供完整的模型训练和评估功能
    
    支持多种机器学习算法，提供统一的接口进行：
    1. 模型初始化和参数配置
    2. 模型训练和验证
    3. 性能评估和指标计算
    4. 预测和结果输出
    
    Attributes:
        model_type (str): 选用的模型类型
        model: 训练好的模型实例
        best_params (dict): 最优参数配置
    """

    def __init__(self, model_type: str = 'xgboost'):
        """初始化模型训练器
        
        Args:
            model_type (str): 模型类型，可选值：
                - 'xgboost': XGBoost模型
                - 'lightgbm': LightGBM模型
                - 'random_forest': 随机森林模型
                
        Raises:
            ValueError: 不支持的模型类型
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        
        # 默认模型参数
        self.default_params = {
            'xgboost': {
                'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': 5,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'scale_pos_weight': 1,
                'random_state': 42
            },
            'lightgbm': {
                'learning_rate': 0.1,
                'n_estimators': 100,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary',
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        }

    def _create_model(self, params: Dict[str, Any] = None) -> Any:
        """创建模型实例
        
        根据指定的模型类型和参数创建相应的模型实例。
        
        Args:
            params (Dict[str, Any], optional): 模型参数字典
                如果为None，使用默认参数
        
        Returns:
            模型实例，可能是XGBoost、LightGBM或RandomForest
            
        Raises:
            ValueError: 不支持的模型类型
        """
        if params is None:
            params = self.default_params[self.model_type]

        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**params)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_size: float = 0.2) -> Tuple[Dict[str, float], Dict[str, float]]:
        """训练模型
        
        完整的模型训练流程：
        1. 数据集划分
        2. 模型训练
        3. 性能评估
        4. 结果记录
        
        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 标签向量
            validation_size (float): 验证集比例
        
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]:
                - 训练集评估指标
                - 验证集评估指标
                
        Notes:
            评估指标包括：准确率、精确率、召回率、F1分数、AUC
        """
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=42, stratify=y
        )

        # 创建并训练模型
        self.model = self._create_model()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )

        # 评估模型
        train_metrics = self.evaluate(X_train, y_train)
        val_metrics = self.evaluate(X_val, y_val)

        return train_metrics, val_metrics

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型性能
        
        计算多个评估指标：
        1. 准确率 (Accuracy)
        2. 精确率 (Precision)
        3. 召回率 (Recall)
        4. F1分数 (F1-score)
        5. AUC值 (Area Under Curve)
        6. 混淆矩阵 (Confusion Matrix)
        
        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 真实标签
            
        Returns:
            Dict[str, float]: 评估指标字典
            
        Raises:
            ValueError: 模型未训练
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # 获取预测结果
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # 计算各种评估指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba)
        }

        # 计算混淆矩阵
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm

        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5) -> Dict[str, List[float]]:
        """执行交叉验证
        
        使用K折交叉验证评估模型性能：
        1. 数据集划分为K份
        2. 迭代训练和评估
        3. 计算平均性能
        
        Args:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 标签向量
            cv (int): 折数
            
        Returns:
            Dict[str, List[float]]: 各指标的交叉验证结果
        """
        model = self._create_model()
        
        # 计算不同指标的交叉验证分数
        metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=metric
            )
            metrics[metric] = scores.tolist()

        return metrics

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预测新数据
        
        对新数据进行预测，返回：
        1. 预测的类别
        2. 预测的概率
        
        Args:
            X (pd.DataFrame): 待预测的特征矩阵
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - 预测的类别（0或1）
                - 预测为正类的概率
                
        Raises:
            ValueError: 模型未训练
            RuntimeError: 预测过程出错
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性排序
        
        分析模型中各特征的重要性：
        1. 计算特征重要性分数
        2. 排序并返回结果
        
        Returns:
            pd.DataFrame: 特征重要性DataFrame，包含：
                - feature: 特征名称
                - importance: 重要性分数
                
        Raises:
            ValueError: 模型不支持特征重要性分析
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if self.model_type in ['xgboost', 'lightgbm']:
            importance = self.model.feature_importances_
            feature_names = self.model.feature_names_in_
        elif self.model_type == 'random_forest':
            importance = self.model.feature_importances_
            feature_names = self.model.feature_names_in_
        else:
            raise ValueError(f"Feature importance not supported for {self.model_type}")

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save_model(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """加载模型
        
        Args:
            path: 模型路径
        """
        import joblib
        self.model = joblib.load(path) 