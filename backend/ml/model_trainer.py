import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from multiprocessing import cpu_count


class ExplosiveStockModelTrainer:
    """爆发式股票预测模型训练器"""

    def __init__(self):
        # 获取 CPU 核心数
        n_jobs = cpu_count() - 2  # 留出2个核心给系统使用
        
        self.models = {
            'gbdt': GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.01,
                max_depth=6,         # 适度增加深度
                subsample=0.9,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt'
            ),
            'rf': RandomForestClassifier(
                random_state=42,
                n_jobs=n_jobs,
                n_estimators=400,
                max_depth=10,        # 增加深度
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                class_weight={0: 1, 1: 1.8}  # 微调权重
            ),
            'xgb': XGBClassifier(
                random_state=42,
                n_jobs=n_jobs,
                n_estimators=200,
                max_depth=6,         # 适度增加深度
                learning_rate=0.01,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=4,  # 减小以增加灵活性
                scale_pos_weight=1.8,  # 微调权重
                tree_method='hist',
                gamma=0.12          # 减小以增加灵活性
            ),
            # 'lr': None  # 移除表现不佳的逻辑回归模型
        }
        self.weights = {
            'gbdt': 0.35,  # GBDT的precision很高
            'rf': 0.45,    # RF整体表现最好
            'xgb': 0.20    # 降低XGB权重
        }
        self.scaler = StandardScaler()
        self.trained_models = {}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        try:
            # 数据验证
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)

            # 检查并处理无效值
            logging.info("检查训练数据...")
            invalid_cols = []
            for col in X_train.columns:
                invalid_count = (~np.isfinite(X_train[col])).sum()
                if invalid_count > 0:
                    logging.warning(f"列 '{col}' 包含 {invalid_count} 个无效值")
                    invalid_cols.append(col)

            if invalid_cols:
                logging.info("正在处理无效值...")
                # 对于每个包含无效值的列，使用该列的均值填充
                for col in invalid_cols:
                    col_mean = X_train[col][np.isfinite(X_train[col])].mean()
                    X_train[col] = X_train[col].replace([np.inf, -np.inf, np.nan], col_mean)
                logging.info("无效值处理完成")

            # 再次验证数据
            if not np.isfinite(X_train.values).all():
                raise ValueError("处理后的数据仍然包含无效值")

            # 标准化
            logging.info("开始标准化数据...")
            X_train_scaled = self.scaler.fit_transform(X_train)

            # 验证标准化后的数据
            if not np.isfinite(X_train_scaled).all():
                raise ValueError("标准化后的数据包含无效值")

            # 训练模型
            self.trained_models = {}
            for name, model in self.models.items():
                logging.info(f"开始训练 {name} 模型...")
                self.trained_models[name] = model
                model.fit(X_train_scaled, y_train)
                logging.info(f"{name} 模型训练完成")

            if X_val is not None and y_val is not None:
                logging.info("开始验证模型...")
                X_val_scaled = self.scaler.transform(X_val)
                for name, model in self.trained_models.items():
                    val_score = model.score(X_val_scaled, y_val)
                    logging.info(f"{name} 验证集得分: {val_score:.4f}")

            logging.info("所有模型训练完成")

        except Exception as e:
            logging.exception(f"模型训练失败: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> float:
        """集成预测"""
        try:
            features_scaled = self.scaler.transform(features)
            predictions = []

            for name, model in self.trained_models.items():
                pred = model.predict_proba(features_scaled)[0][1]
                weighted_pred = pred * self.weights[name]
                predictions.append(weighted_pred)
                logging.debug(f"{name} 模型预测概率: {pred:.4f}, 权重: {self.weights[name]}")

            final_prediction = sum(predictions)
            logging.debug(f"最终集成预测概率: {final_prediction:.4f}")

            return final_prediction

        except Exception as e:
            logging.exception(f"预测失败: {e}")
            return 0.0

    def save_models(self, base_path: str):
        """保存所有模型和scaler"""
        try:
            # 确保基础目录存在
            base_dir = os.path.dirname(base_path)
            os.makedirs(base_dir, exist_ok=True)

            # 保存每个模型
            for name, model in self.trained_models.items():
                # 构建模型文件路径（不要在base_path后面加斜杠）
                model_path = f"{base_path}_{name}.joblib"
                joblib.dump(model, model_path)
                logging.info(f"模型 {name} 已保存到: {model_path}")

            # 保存scaler
            scaler_path = f"{base_path}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logging.info(f"Scaler已保存到: {scaler_path}")

        except Exception as e:
            logging.exception(f"保存模型失败: {e}")
            raise

    def load_models(self, models: dict, scaler_path: str):
        """加载所有模型和scaler"""
        try:
            for name, model_path in models.items():
                self.trained_models[name] = joblib.load(model_path)
                logging.debug(f"模型 {name} 已加载: {model_path}")

            self.scaler = joblib.load(scaler_path)
            logging.debug(f"Scaler已加载: {scaler_path}")

        except Exception as e:
            logging.exception(f"加载模型失败: {e}")
            raise

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """评估所有模型的性能

        - 准确率 (Accuracy)：模型预测的正确比例。但要注意，如果数据中大部分样本是某一类（比如不上涨的股票），模型可能会倾向于预测多数类，导致准确率虚高。
        - 精确率 (Precision)：模型预测为"上涨"的股票中，实际真的上涨的比例。
        - 召回率 (Recall)：实际"上涨"的股票中，模型预测正确的比例。召回率高，说明模型能捕捉到大部分上涨的股票。但高召回率可能意味着模型过于激进，把很多不上涨的股票也预测为上涨。
        - F1分数 (F1)：精确率和召回率的平衡值，越高越好。
        - AUC值 (AUC)：模型区分"上涨"和"不上涨"股票的能力，0.5是随机猜测，1是完美预测。
        
        最完美的是每一项值都很高

        - 混淆矩阵 (confusion_matrix)：
        预测值 ->          0    1
        实际值 ->      0   X    X
        实际值 ->      1   X    X


        """
        try:
            results = {}
            X_test_scaled = self.scaler.transform(X_test)

            for name, model in self.trained_models.items():
                # 获取预测结果
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)

                # 计算各种评估指标
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary'),
                    'f1': f1_score(y_test, y_pred, average='binary'),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'auc': roc_auc_score(y_test, y_pred_proba[:, 1])
                }

                # 输出评估报告
                logging.info(f"\n{name} 模型评估结果：")
                logging.info(f"准确率: {results[name]['accuracy']:.4f}")
                logging.info(f"精确率: {results[name]['precision']:.4f}")
                logging.info(f"召回率: {results[name]['recall']:.4f}")
                logging.info(f"F1分数: {results[name]['f1']:.4f}")
                if results[name]['auc'] is not None:
                    logging.info(f"AUC分数: {results[name]['auc']:.4f}")

                # 输出混淆矩阵
                cm = results[name]['confusion_matrix']
                logging.info("\n混淆矩阵：")
                logging.info("预测值 ->      0        1")
                logging.info("实际值  0: {:>6}   {:>6}".format(cm[0][0], cm[0][1]))
                logging.info("实际值  1: {:>6}   {:>6}".format(cm[1][0], cm[1][1]))

            return results

        except Exception as e:
            logging.exception(f"模型评估失败: {e}")
            return {}
