from datetime import datetime, timedelta
from typing import Dict

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.data.stock_data_fetcher import StockDataFetcher
from backend.utils.indicators import CalIndicators

# 设置中文字体
font_path = '/System/Library/Fonts/STHeiti Light.ttc'  # 这是macOS上的一个中文字体路径
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


# 加载数据
def load_data(file_path):
    """
    加载股票暴涨数据

    :param file_path: str, 数据文件的路径
    :return: DataFrame, 股票数据
    """
    return pd.read_excel(file_path, sheet_name='扫描结果')


class ExplosiveStockAnalyzer:
    """
    暴涨股票特征分析器
    
    主要分析维度：
    1. 量价特征：成交量变化、价格变化趋势
    2. 技术指标：MACD、KDJ、RSI等
    3. 波动特征：振幅、换手率
    4. 资金流向：主力资金、散户资金
    """

    def __init__(self):
        self.data_fetcher = StockDataFetcher()

    def analyze_volume_price_pattern(self,
                                     code: str,
                                     start_date: str,
                                     end_date: str) -> Dict:
        """
        分析量价关系特征
        
        Args:
            code: 股票代码
            start_date: 分析起始日期
            end_date: 分析结束日期
            
        Returns:
            Dict: 包含以下特征
                - volume_increase: 成交量增长率
                - price_volume_correlation: 量价相关性
                - volume_concentration: 成交量集中度
        """
        # 获取分析周期前20个交易日的数据
        analysis_start = (datetime.strptime(start_date, '%Y-%m-%d') -
                          timedelta(days=30)).strftime('%Y-%m-%d')

        df = self.data_fetcher.fetch_stock_data(
            code=code, start_date=analysis_start, end_date=end_date
        ).iloc[-20:]

        # 计算特征
        features = {
            'volume_increase': (df['volume'][-5:].mean() /
                                df['volume'][:-5].mean() - 1),
            'price_volume_correlation': df['close'].corr(df['volume']),
            'volume_concentration': (df['volume'][-5:].sum() /
                                     df['volume'].sum())
        }

        return features

    def analyze_technical_indicators(self,
                                     code: str,
                                     start_date: str,
                                     end_date: str) -> Dict:
        """
        分析技术指标特征
        
        Args:
            code: 股票代码
            start_date: 分析起始日期
            end_date: 分析结束日期
            
        Returns:
            Dict: 包含以下特征
                - macd_pattern: MACD形态
                - kdj_pattern: KDJ形态
                - rsi_pattern: RSI形态
        """
        analysis_start = (datetime.strptime(start_date, '%Y-%m-%d') -
                          timedelta(days=30)).strftime('%Y-%m-%d')

        df = self.data_fetcher.fetch_stock_data(
            code=code, start_date=analysis_start, end_date=end_date
        ).iloc[-20:]

        # 计算MACD指标
        macd, macd_signal, macd_hist = CalIndicators.macd(df, fast_period=12, slow_period=26, signal_period=9)

        # 计算KDJ指标
        k, d, j = CalIndicators.kdj(df)

        # 计算RSI指标
        rsi = CalIndicators.rsi(df)

        features = {
            'macd': macd.iloc[-1],
            'signal': macd_signal.iloc[-1],
            'hist': macd_hist.iloc[-1],
            'k': k.iloc[-1],
            'd': d.iloc[-1],
            'j': j.iloc[-1],
            'rsi': rsi.iloc[-1]
        }
        return features

    def analyze_volatility(self,
                           code: str,
                           start_date: str,
                           end_date: str) -> Dict:
        """
        分析波动特征
        
        Args:
            code: 股票代码
            start_date: 分析起始日期
            end_date: 分析结束日期
            
        Returns:
            Dict: 包含以下特征
                - amplitude: 振幅
                - turn_rate: 换手率
                - price_range: 价格区间
        """
        df = self.data_fetcher.fetch_stock_data(
            code=code, start_date=start_date, end_date=end_date
        ).iloc[-20:]

        features = {
            'amplitude': (df['high'].max() - df['low'].min()) / df['low'].min(),
            'turn_rate': df['turn'].mean(),
            'price_range': df.close.std() / df['close'].mean()
        }

        return features


def analyze_features(data: pd.DataFrame) -> None:
    """
    分析暴涨股票的前置特征
    """
    analyzer = ExplosiveStockAnalyzer()

    # 存储所有股票的特征
    all_features = []

    for _, row in data.iterrows():
        code = row['code']
        start_date = row['start_date']
        end_date = row['end_date']

        # 获取各维度特征
        volume_price_features = analyzer.analyze_volume_price_pattern(
            code, start_date, end_date
        )
        technical_features = analyzer.analyze_technical_indicators(
            code, start_date, end_date
        )
        volatility_features = analyzer.analyze_volatility(
            code, start_date, end_date
        )

        # 合并特征
        features = {
            # 'code': code,
            'return': row['max_return'],
            **volume_price_features,
            **technical_features,
            **volatility_features
        }

        all_features.append(features)

    # 转换为DataFrame进行分析
    features_df = pd.DataFrame(all_features)

    # 特征分析和可视化
    _plot_feature_analysis(features_df)
    _generate_analysis_report(features_df)


def _plot_feature_analysis(features_df: pd.DataFrame) -> None:
    """
    绘制特征分析图表
    """
    plt.figure(figsize=(15, 10))

    # 相关性热力图
    plt.subplot(2, 2, 1)
    sns.heatmap(features_df.corr(), annot=True, cmap='coolwarm')
    plt.title('特征相关性分析')

    # 收益率与成交量增长率的散点图
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=features_df, x='volume_increase', y='return')
    plt.title('收益率vs成交量增长率')

    plt.tight_layout()
    plt.savefig('feature_analysis.png')
    plt.close()


def _generate_analysis_report(features_df: pd.DataFrame) -> None:
    """
    生成分析报告
    """
    report = pd.DataFrame({
        '特征': features_df.columns,
        '均值': features_df.mean(),
        '中位数': features_df.median(),
        '标准差': features_df.std()
    })

    report.to_excel('analysis_report.xlsx', index=False)


if __name__ == "__main__":
    data = load_data('扫描翻倍股_30%涨幅_5%回撤+20日.xlsx')
    analyze_features(data)
