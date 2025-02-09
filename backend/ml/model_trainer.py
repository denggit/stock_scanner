import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.utils.logger import setup_logger

logger = setup_logger("train_model", set_root_logger=True)


class ExplosiveStockModelTrainer:
    """爆发式股票预测模型训练器"""

    def __init__(self):
        self.models = {
            'gbdt': GradientBoostingClassifier(random_state=42),
            'rf': RandomForestClassifier(random_state=42),
            'xgb': XGBClassifier(random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        self.weights = {
            'gbdt': 0.4,
            'rf': 0.3,
            'xgb': 0.2,
            'lr': 0.1
        }
        self.scaler = StandardScaler()
        self.trained_models = {}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        try:
            # 数据验证
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)

            # 确保数据有效
            if not np.isfinite(X_train.values).all():
                raise ValueError("训练数据包含无效值")

            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)

            # 再次验证标准化后的数据
            if not np.isfinite(X_train_scaled).all():
                raise ValueError("标准化后的数据包含无效值")

            # 训练模型
            self.trained_models = self.models.copy()
            for name, model in self.trained_models.items():
                logger.info(f"开始训练 {name} 模型...")
                model.fit(X_train_scaled, y_train)

            logger.info("模型训练完成")

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                for name, model in self.trained_models.items():
                    val_score = model.score(X_val_scaled, y_val)
                    logger.info(f"{name} 验证集得分: {val_score:.4f}")

        except Exception as e:
            logger.exception(f"模型训练失败: {e}")
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
                logger.debug(f"{name} 模型预测概率: {pred:.4f}, 权重: {self.weights[name]}")

            final_prediction = sum(predictions)
            logger.debug(f"最终集成预测概率: {final_prediction:.4f}")

            return final_prediction

        except Exception as e:
            logger.exception(f"预测失败: {e}")
            return 0.0

    def save_models(self, base_path: str):
        """保存所有模型和scaler"""
        try:
            for name, model in self.trained_models.items():
                model_path = f"{base_path}/{name}_model.joblib"
                joblib.dump(model, model_path)
                logger.info(f"模型 {name} 已保存到: {model_path}")

            scaler_path = f"{base_path}/scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler已保存到: {scaler_path}")

        except Exception as e:
            logger.exception(f"保存模型失败: {e}")

    def load_models(self, base_path: str):
        """加载所有模型和scaler"""
        try:
            for name in self.models.keys():
                model_path = f"{base_path}/{name}_model.joblib"
                self.trained_models[name] = joblib.load(model_path)
                logger.info(f"模型 {name} 已加载: {model_path}")

            scaler_path = f"{base_path}/scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler已加载: {scaler_path}")

        except Exception as e:
            logger.exception(f"加载模型失败: {e}")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """评估所有模型的性能"""
        try:
            results = {}
            X_test_scaled = self.scaler.transform(X_test)

            for name, model in self.trained_models.items():
                # 获取预测结果
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                # 计算各种评估指标
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }

                # 输出评估报告
                logger.info(f"\n{name} 模型评估结果：")
                logger.info(f"准确率: {results[name]['accuracy']:.4f}")
                logger.info(f"精确率: {results[name]['precision']:.4f}")
                logger.info(f"召回率: {results[name]['recall']:.4f}")
                logger.info(f"F1分数: {results[name]['f1']:.4f}")
                logger.info(f"AUC分数: {results[name]['auc']:.4f}")
                logger.info("\n混淆矩阵：\n" + str(results[name]['confusion_matrix']))

            return results

        except Exception as e:
            logger.exception(f"模型评估失败: {e}")
            return {}
