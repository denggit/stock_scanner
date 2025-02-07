import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.utils.logger import setup_logger

logger = setup_logger("train_model")


class ExplosiveStockModelTrainer:
    """爆发式股票预测模型训练器"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """
        训练模型
        
        Args:
            features: 特征DataFrame
            labels: 标签Series
        """
        try:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )

            # 特征标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 训练模型
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )

            self.model.fit(X_train_scaled, y_train)

            # 评估模型
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            logger.info(f"模型训练完成：\n"
                        f"- 训练集准确率: {train_score:.4f}\n"
                        f"- 测试集准确率: {test_score:.4f}")

        except Exception as e:
            logger.exception(f"模型训练失败: {e}")

    def save_model(self, model_path: str, scaler_path: str):
        """保存模型和标准化器"""
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"模型已保存到: {model_path}")
        except Exception as e:
            logger.exception(f"保存模型失败: {e}")

    def load_model(self, model_path: str, scaler_path: str):
        """加载模型和标准化器"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"模型已加载: {model_path}")
        except Exception as e:
            logger.exception(f"加载模型失败: {e}")
