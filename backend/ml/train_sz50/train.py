import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(root_dir))
os.chdir(root_dir)

import dotenv
import os
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from backend.data.stock_data_fetcher import StockDataFetcher
from backend.utils.logger import setup_logger
from data_collector import ExplosiveStockDataCollector
from model_trainer import ExplosiveStockModelTrainer

dotenv.load_dotenv()
logger = setup_logger("train_model", set_root_logger=True)


def train_model(model_save_path: str, scaler_save_path: str, stock_pool: str = 'full'):
    """
    训练爆发式股票预测模型
    
    Args:
        model_save_path: 模型保存路径
        scaler_save_path: 标准化器保存路径
    """
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

        # 1. 初始化数据收集器和数据获取器
        collector = ExplosiveStockDataCollector()
        data_fetcher = StockDataFetcher()

        # 2. 获取股票列表
        stock_list = data_fetcher.get_stock_list(pool_name=stock_pool)
        logger.info(f"获取到 {len(stock_list)} 只股票")

        # 3. 设置时间范围（使用近3年数据）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=366 * 3)).strftime("%Y-%m-%d")

        # 4. 收集所有股票的训练数据
        all_features = []
        all_labels = []

        for _, stock in tqdm(stock_list.iterrows(), total=len(stock_list), desc="收集训练数据"):
            try:
                stock_data = data_fetcher.fetch_stock_data(
                    code=stock['code'],
                    start_date=start_date,
                    end_date=end_date
                )

                if len(stock_data) < 60:  # 数据太少的股票跳过
                    logger.warning(f"该股票数据太少，跳过: {stock['code']}")
                    continue

                # 确保数值类型是float32
                for col in stock_data.select_dtypes(include=[np.number]).columns:
                    stock_data[col] = stock_data[col].astype(np.float32)

                # 收集训练数据
                features, labels = collector.collect_training_data(stock_data)

                if len(features) > 0 and len(labels) > 0:
                    all_features.append(features)
                    all_labels.append(labels)

            except Exception as e:
                logger.warning(f"处理股票 {stock['code']} 时出错: {str(e)}")
                continue

        # 5. 合并所有数据
        if not all_features or not all_labels:
            raise ValueError("没有收集到有效的训练数据")

        features_df = pd.concat(all_features, axis=0)
        labels_series = pd.concat(all_labels, axis=0)

        # 确保特征和标签的索引匹配
        logger.info("对齐特征和标签数据...")
        common_index = features_df.index.intersection(labels_series.index)
        features_df = features_df.loc[common_index]
        labels_series = labels_series.loc[common_index]

        logger.info(f"收集到的训练数据大小：{len(features_df)} 行")
        logger.info(f"正样本比例：{(labels_series == 1).mean():.2%}")  # 会上涨30%的样本
        logger.info(f"负样本比例：{(labels_series == 0).mean():.2%}")  # 不会上涨30%的样本

        # 6. 数据质量检查
        logger.info("检查数据质量...")
        invalid_features = features_df.columns[~np.isfinite(features_df).all()].tolist()
        if invalid_features:
            logger.warning(f"以下特征包含无效值: {invalid_features}")
            logger.info("正在移除包含无效值的行...")
            valid_mask = np.isfinite(features_df).all(axis=1)
            features_df = features_df[valid_mask]
            labels_series = labels_series[valid_mask]
            logger.info(f"清理后的数据大小：{len(features_df)} 行")

        # 验证数据对齐
        if len(features_df) != len(labels_series):
            raise ValueError(f"特征和标签数量不匹配: 特征={len(features_df)}, 标签={len(labels_series)}")
        if not (features_df.index == labels_series.index).all():
            raise ValueError("特征和标签的索引不匹配")

        # 7. 特征分析，很多高相似度的特征对感觉都有用，暂时不移除他们
        high_corr_features = collector.analyze_feature_correlation(features_df)

        # 8. 保存特征和标签的统计信息
        feature_stats = features_df.describe()
        logger.info(f"\n特征统计信息:\n{feature_stats}")
        logger.info(f"标签分布:\n{labels_series.value_counts()}")

        # 9. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels_series,
            test_size=0.2,
            random_state=42,
            stratify=labels_series  # 确保训练集和测试集的标签分布一致
        )

        # 10. 在数据预处理后添加特征选择
        selector = SelectFromModel(
            estimator=XGBClassifier(n_estimators=100, random_state=42),
            prefit=False
        )

        # 应用特征选择
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # 在特征选择之后添加采样策略
        logger.info("开始进行样本平衡处理...")

        # 使用 SMOTE 进行过采样，进一步降低采样比例
        smote = SMOTE(
            sampling_strategy=0.4,  # 从0.2提高到0.4，增加正样本比例
            random_state=42,
            k_neighbors=5  # 减少邻居数量，避免过度平滑
        )

        # 应用 SMOTE
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

        # 输出平衡后的样本比例
        logger.info(f"平衡后的训练集大小：{len(X_train_balanced)} 行")
        logger.info(f"平衡后的正样本比例：{(y_train_balanced == 1).mean():.2%}")
        logger.info(f"平衡后的负样本比例：{(y_train_balanced == 0).mean():.2%}")

        # 使用平衡后的数据训练模型
        trainer = get_adjusted_trainer()
        trainer.train(X_train_balanced, y_train_balanced)

        # 12. 模型评估
        evaluation_results = trainer.evaluate_models(X_test_selected, y_test)
        # 输出模型评估结果
        log_evaluation_results(evaluation_results)

        # 13. 保存模型
        trainer.save_models(model_save_path)

    except Exception as e:
        logger.exception(f"模型训练过程出错: {e}")
        raise


def log_evaluation_results(evaluation_results):
    """打印模型评估结果"""
    logger.info("模型评估结果：")
    # 提取除 confusion_matrix 之外的指标名称
    metric_names = [name for name in next(iter(evaluation_results.values())) if name != 'confusion_matrix']

    # 准备表格数据
    table_data = []
    for model_name, metrics in evaluation_results.items():
        row = [model_name]
        for metric_name in metric_names:
            metric_value = metrics[metric_name]
            if isinstance(metric_value, (float, np.float64)):
                row.append(f"{metric_value:.4f}")
            else:
                row.append(metric_value)
        table_data.append(row)

    # 输出表格
    logger.info("模型评估指标对比表格：")
    logger.info("\n" + tabulate(table_data, headers=['模型'] + metric_names, tablefmt='grid'))

    # 单独输出每个模型的混淆矩阵
    for model_name, metrics in evaluation_results.items():
        confusion_matrix = metrics['confusion_matrix']
        headers = ["实际值\预测值", "不会上涨", "会上涨"]
        data = [
            ["不会上涨"] + confusion_matrix[0],
            ["会上涨"] + confusion_matrix[1]
        ]
        logger.info(f"\n模型 {model_name} 的混淆矩阵：")
        logger.info("\n" + tabulate(data, headers=headers, tablefmt='grid'))


def select_important_features(X_train, y_train, X_test, threshold=0.01):
    """选择重要特征"""
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    
    # 获取特征重要性
    importance_scores = pd.Series(xgb.feature_importances_, index=X_train.columns)
    important_features = importance_scores[importance_scores > threshold].index
    
    logger.info(f"选择的重要特征数量: {len(important_features)}")
    logger.info("\n特征重要性排名:")
    for feat, score in importance_scores.nlargest(10).items():
        logger.info(f"{feat}: {score:.4f}")
    
    return X_train[important_features], X_test[important_features], important_features


def get_adjusted_trainer():
    trainer = ExplosiveStockModelTrainer()
    
    # 调整模型参数
    trainer.models['gbdt'] = GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight={0: 1, 1: 2}  # 增加正样本权重
    )
    
    trainer.models['rf'] = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        n_estimators=500,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight={0: 1, 1: 3}  # 进一步增加正样本权重
    )
    
    trainer.models['xgb'] = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=3,  # 增加正样本权重
        gamma=0.1
    )
    
    return trainer


if __name__ == "__main__":
    pool_name = "full"
    # 修改保存路径的格式
    model_base_path = f"backend/ml/models/explosive_stock_model_{pool_name}"

    train_model(
        model_save_path=model_base_path,  # 不需要添加.joblib后缀
        scaler_save_path=model_base_path,  # 使用相同的基础路径
        stock_pool=pool_name
    )
