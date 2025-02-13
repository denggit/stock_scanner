#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练流水线模块：组织和管理整个模型训练流程

本模块实现了一个完整的股票暴涨预测模型训练流水线，包括：
1. 数据准备和特征工程
2. 数据预处理和清洗
3. 模型训练和评估
4. 结果记录和模型保存

主要类：
    ExplosiveStockPredictor: 股票暴涨预测器类，整合了整个预测流程

主要功能：
    - 从原始数据构建训练特征
    - 处理数据不平衡和异常值
    - 训练机器学习模型
    - 评估模型性能
    - 保存训练好的模型
    - 预测新数据

使用示例：
    predictor = ExplosiveStockPredictor('config/model_config.yaml')
    X, y = predictor.prepare_training_data(price_data, financial_data, explosion_data)
    results = predictor.train_model(X, y)
    predictor.save_model('models')
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import yaml

from .feature_engineering import FeatureEngineering
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer


class ExplosiveStockPredictor:
    """股票暴涨预测器，整合特征工程、数据处理和模型训练的完整流程
    
    该类提供了端到端的股票暴涨预测模型训练和预测功能，包括：
    1. 配置管理：通过YAML配置文件管理模型参数
    2. 特征工程：构建技术指标和财务指标特征
    3. 数据预处理：处理缺失值、异常值和类别不平衡
    4. 模型训练：支持多种机器学习模型
    5. 模型评估：多种评估指标和交叉验证
    6. 结果记录：详细的训练过程日志
    7. 模型持久化：保存训练好的模型和预处理器
    
    Attributes:
        config (dict): 模型配置参数
        feature_engineering (FeatureEngineering): 特征工程实例
        data_processor (DataProcessor): 数据预处理实例
        model_trainer (ModelTrainer): 模型训练实例
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化预测器
        
        Args:
            config_path (str, optional): 配置文件路径，用于加载模型参数
                配置文件应为YAML格式，包含模型类型、参数等配置信息
                如果为None，将使用默认配置
        
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: 配置文件格式错误
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.feature_engineering = FeatureEngineering()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer(
            model_type=self.config.get('model_type', 'xgboost')
        )
        
        # 设置日志
        self._setup_logging()

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """加载YAML格式的配置文件
        
        Args:
            config_path (str): 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置参数字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: 配置文件格式错误
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """配置日志系统
        
        设置日志记录的详细配置，包括：
        1. 创建日志目录
        2. 设置日志格式
        3. 配置文件和控制台输出
        4. 设置日志级别
        
        日志文件名包含时间戳，便于追踪不同的训练运行。
        
        Raises:
            PermissionError: 无法创建日志目录或文件
            IOError: 日志文件操作错误
        """
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def prepare_training_data(self, 
                            price_data: pd.DataFrame,
                            financial_data: pd.DataFrame,
                            explosion_data: pd.DataFrame) -> tuple:
        """准备模型训练数据
        
        该方法执行完整的数据准备流程，包括：
        1. 特征构建：技术指标、财务指标等
        2. 数据清洗：处理缺失值和异常值
        3. 特征缩放：标准化或归一化
        4. 类别平衡：处理样本不平衡问题
        
        Args:
            price_data (pd.DataFrame): 股票价格数据，包含：
                - 日期索引
                - OHLCV数据
                - 其他价格相关指标
            financial_data (pd.DataFrame): 财务数据，包含：
                - 财务比率
                - 成长指标
                - 盈利能力指标
            explosion_data (pd.DataFrame): 历史暴涨数据，包含：
                - 暴涨起始日期
                - 暴涨结束日期
                - 涨幅信息
            
        Returns:
            tuple: (X_processed, y_processed)
                X_processed (pd.DataFrame): 处理后的特征矩阵
                y_processed (pd.Series): 处理后的标签向量
                
        Raises:
            ValueError: 输入数据格式错误或缺少必要字段
        """
        logging.info("Preparing training data...")
        
        # 构建特征
        X, y = self.feature_engineering.prepare_features(
            price_data=price_data,
            financial_data=financial_data,
            forward_days=self.config.get('forward_days', 20),
            return_threshold=self.config.get('return_threshold', 0.3)
        )
        
        # 数据预处理
        X_processed, y_processed = self.data_processor.process_data(X, y, is_training=True)
        
        logging.info(f"Prepared {len(X_processed)} samples with {X_processed.shape[1]} features")
        logging.info(f"Positive samples: {sum(y_processed == 1)}, "
                    f"Negative samples: {sum(y_processed == 0)}")
        
        return X_processed, y_processed

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """训练股票暴涨预测模型
        
        执行完整的模型训练流程，包括：
        1. 模型初始化和参数设置
        2. 训练集和验证集划分
        3. 模型训练和验证
        4. 交叉验证评估
        5. 特征重要性分析
        
        Args:
            X (pd.DataFrame): 处理后的特征矩阵，包含：
                - 技术指标特征
                - 财务指标特征
                - 市场情绪特征
            y (pd.Series): 二分类标签，其中：
                - 1 表示未来出现暴涨
                - 0 表示未来未出现暴涨
        
        Returns:
            Dict[str, Any]: 训练结果字典，包含：
                - train_metrics: 训练集评估指标
                - val_metrics: 验证集评估指标
                - cv_metrics: 交叉验证结果
                - feature_importance: 特征重要性排序
                
        Raises:
            ValueError: 特征或标签数据格式错误
            RuntimeError: 模型训练过程出错
        """
        logging.info("Starting model training...")
        
        # 训练模型
        train_metrics, val_metrics = self.model_trainer.train(
            X, y, validation_size=self.config.get('validation_size', 0.2)
        )
        
        # 交叉验证
        cv_metrics = self.model_trainer.cross_validate(
            X, y, cv=self.config.get('cv_folds', 5)
        )
        
        # 获取特征重要性
        feature_importance = self.model_trainer.get_feature_importance()
        
        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'cv_metrics': cv_metrics,
            'feature_importance': feature_importance
        }
        
        self._log_training_results(results)
        
        return results

    def _log_training_results(self, results: Dict[str, Any]):
        """记录模型训练结果
        
        详细记录模型训练过程中的各项指标，包括：
        1. 训练集和验证集的性能指标
        2. 交叉验证结果
        3. 重要特征排名
        4. 混淆矩阵
        
        Args:
            results (Dict[str, Any]): 训练结果字典，包含：
                - train_metrics: 训练集评估指标
                - val_metrics: 验证集评估指标
                - cv_metrics: 交叉验证结果
                - feature_importance: 特征重要性
        """
        logging.info("\nTraining Results:")
        logging.info("\nTraining Metrics:")
        for metric, value in results['train_metrics'].items():
            if metric != 'confusion_matrix':
                logging.info(f"{metric}: {value:.4f}")

        logging.info("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            if metric != 'confusion_matrix':
                logging.info(f"{metric}: {value:.4f}")

        logging.info("\nTop 10 Important Features:")
        for _, row in results['feature_importance'].head(10).iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")

    def save_model(self, model_dir: str):
        """保存训练好的模型和预处理器
        
        将模型和相关组件保存到指定目录，包括：
        1. 训练好的模型
        2. 特征预处理器
        3. 模型配置信息
        
        保存的文件包含时间戳，便于版本管理。
        
        Args:
            model_dir (str): 模型保存目录路径
                如果目录不存在，将自动创建
        
        Raises:
            IOError: 保存过程中出现错误
            PermissionError: 没有写入权限
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = os.path.join(model_dir, f'model_{timestamp}.joblib')
        self.model_trainer.save_model(model_path)
        
        # 保存预处理器
        preprocessor_path = os.path.join(model_dir, f'preprocessor_{timestamp}.joblib')
        self.data_processor.save_preprocessor(preprocessor_path)
        
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Preprocessor saved to {preprocessor_path}")

    def predict(self, X: pd.DataFrame) -> tuple:
        """使用训练好的模型进行预测
        
        完整的预测流程包括：
        1. 数据预处理
        2. 特征转换
        3. 模型预测
        4. 预测结果后处理
        
        Args:
            X (pd.DataFrame): 待预测的特征数据，需要包含：
                - 与训练数据相同的特征列
                - 相同的数据格式和单位
        
        Returns:
            tuple: (predictions, probabilities)
                predictions (np.ndarray): 预测类别（0或1）
                probabilities (np.ndarray): 预测为正类的概率
        
        Raises:
            ValueError: 输入数据格式错误
            RuntimeError: 模型未训练或预测过程出错
        """
        # 数据预处理
        X_processed, _ = self.data_processor.process_data(X, is_training=False)
        
        # 预测
        predictions, probabilities = self.model_trainer.predict(X_processed)
        
        return predictions, probabilities


def main():
    """主函数：执行完整的模型训练流程
    
    完整的执行流程包括：
    1. 加载配置文件
    2. 初始化预测器
    3. 加载训练数据
    4. 准备特征和标签
    5. 训练模型
    6. 评估结果
    7. 保存模型
    
    使用示例：
        $ python train_pipeline.py
    
    注意：
        - 需要确保配置文件存在且格式正确
        - 需要准备好训练数据
        - 需要有足够的磁盘空间保存模型和日志
    """
    # 加载配置
    predictor = ExplosiveStockPredictor('config/model_config.yaml')
    
    # 加载数据
    # 这里需要实现数据加载逻辑
    price_data = pd.DataFrame()  # 替换为实际的数据加载
    financial_data = pd.DataFrame()  # 替换为实际的数据加载
    explosion_data = pd.DataFrame()  # 替换为实际的数据加载
    
    # 准备数据
    X, y = predictor.prepare_training_data(price_data, financial_data, explosion_data)
    
    # 训练模型
    results = predictor.train_model(X, y)
    
    # 保存模型
    predictor.save_model('models')


if __name__ == '__main__':
    main() 