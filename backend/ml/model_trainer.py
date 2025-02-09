import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.utils.logger import setup_logger

logger = setup_logger("train_model")


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

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """训练所有模型"""
        try:
            # 数据清理：替换无穷大值和异常值
            features = features.replace([np.inf, -np.inf], np.nan)

            # 计算每列的均值和标准差
            means = features.mean()
            stds = features.std()

            # 处理异常值：将超过3个标准差的值限制在范围内
            for column in features.columns:
                upper_limit = means[column] + 3 * stds[column]
                lower_limit = means[column] - 3 * stds[column]
                features[column] = features[column].clip(lower_limit, upper_limit)

            # 填充剩余的NaN值
            features = features.fillna(features.mean())

            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )

            # 特征标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 训练每个模型
            for name, model in self.models.items():
                logger.info(f"训练模型: {name}")
                model.fit(X_train_scaled, y_train)
                self.trained_models[name] = model

                # 评估模型
                y_pred = model.predict(X_test_scaled)
                logger.info(f"\n{name} 模型评估报告：")
                logger.info("\n分类报告：\n" + classification_report(y_test, y_pred))
                logger.info("\n混淆矩阵：\n" + str(confusion_matrix(y_test, y_pred)))

        except Exception as e:
            logger.exception(f"模型训练失败: {e}")
            raise  # 重新抛出异常，以便上层函数知道训练失败

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

    def analyze_feature_importance(self, feature_names):
        """分析特征重要性"""
        try:
            if 'gbdt' not in self.trained_models:
                logger.warning("GBDT模型未训练，无法分析特征重要性")
                return pd.DataFrame()

            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.trained_models['gbdt'].feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)

            logger.info("\n特征重要性排名：\n" + str(importance))
            return importance
        except Exception as e:
            logger.warning(f"特征重要性分析失败: {e}")
            return pd.DataFrame()

    def cross_validate(self, features: pd.DataFrame, labels: pd.Series, cv=5):
        """执行交叉验证"""
        try:
            X_scaled = self.scaler.fit_transform(features)
            scores = cross_val_score(self.models['gbdt'], X_scaled, labels, cv=cv)
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

    def select_features(self, features: pd.DataFrame, importance_threshold: float = 0.01):
        """根据重要性筛选特征"""
        try:
            # 获取特征重要性
            importance = self.analyze_feature_importance(features)

            # 筛选重要特征
            important_features = importance[
                importance['avg_importance'] > importance_threshold
                ].index.tolist()

            logger.info(f"\n筛选出 {len(important_features)} 个重要特征")
            logger.info("\n重要特征列表：\n" + str(important_features))

            return important_features

        except Exception as e:
            logger.exception(f"特征筛选失败: {e}")
            return features.columns.tolist()

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
