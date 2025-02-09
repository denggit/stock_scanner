import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

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

            # 使用SMOTE处理不平衡数据
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # 训练模型
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )

            self.model.fit(X_train_balanced, y_train_balanced)

            # 添加更详细的模型评估
            y_pred = self.model.predict(X_test_scaled)
            logger.info("\n分类报告：\n" + classification_report(y_test, y_pred))
            logger.info("\n混淆矩阵：\n" + str(confusion_matrix(y_test, y_pred)))

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
            logger.debug(f"模型已加载: {model_path}")
        except Exception as e:
            logger.exception(f"加载模型失败: {e}")

    def analyze_feature_importance(self, feature_names):
        """分析特征重要性"""
        if self.model is None:
            logger.warning("模型未训练，无法分析特征重要性")
            return
            
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        logger.info("\n特征重要性排名：\n" + str(importance))
        return importance

    def cross_validate(self, features: pd.DataFrame, labels: pd.Series, cv=5):
        """执行交叉验证"""
        try:
            X_scaled = self.scaler.fit_transform(features)
            scores = cross_val_score(self.model, X_scaled, labels, cv=cv)
            logger.info(f"\n交叉验证结果：\n"
                       f"平均准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            return scores
        except Exception as e:
            logger.exception(f"交叉验证失败: {e}")

    def optimize_parameters(self, features: pd.DataFrame, labels: pd.Series):
        """使用网格搜索优化模型参数"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        X_scaled = self.scaler.fit_transform(features)
        grid_search.fit(X_scaled, labels)
        
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳得分: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
