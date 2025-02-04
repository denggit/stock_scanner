import os


def run_backtest(strategy: str, params: dict, start_date: str, end_date: str, initical_capital: float):
    """运行回测"""
    backend_url = os.getenv('BACKEND_URL')
    backend_port = os.getenv('BACKEND_PORT')
