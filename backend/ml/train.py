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

from backend.data.stock_data_fetcher import StockDataFetcher
from backend.utils.logger import setup_logger
from data_collector import ExplosiveStockDataCollector
from model_trainer import ExplosiveStockModelTrainer

dotenv.load_dotenv()
logger = setup_logger("train_model", set_root_logger=True)


def train_model(model_save_path: str, scaler_save_path: str):
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
        stock_list = data_fetcher.get_stock_list()
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

        logger.info(f"收集到的训练数据大小：{len(features_df)} 行")
        logger.info(f"正样本比例：{labels_series.mean():.2%}")

        # 6. 特征分析
        high_corr_features = collector.analyze_feature_correlation(features_df)

        # 7. 保存特征和标签的统计信息
        feature_stats = features_df.describe()
        logger.info(f"\n特征统计信息:\n{feature_stats}")
        logger.info(f"标签分布:\n{labels_series.value_counts()}")

        # 8. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels_series,
            test_size=0.2,
            random_state=42,
            stratify=labels_series  # 确保训练集和测试集的标签分布一致
        )

        # 9. 训练模型
        trainer = ExplosiveStockModelTrainer()
        trainer.train(X_train, y_train)

        # 10. 模型评估
        evaluation_results = trainer.evaluate_models(X_test, y_test)
        logger.info("模型评估结果：")
        logger.info(evaluation_results)

        # 11. 保存模型
        trainer.save_models(model_save_path)

    except Exception as e:
        logger.exception(f"模型训练过程出错: {e}")
        raise


if __name__ == "__main__":
    train_model(
        model_save_path="backend/ml/models/explosive_stock_model.joblib",
        scaler_save_path="backend/ml/models/explosive_stock_scaler.joblib"
    )
