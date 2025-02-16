import logging
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))
os.chdir(root_dir)

import dotenv

dotenv.load_dotenv()

from backend.data.data_manager import DataUpdateManager
from backend.utils.logger import setup_logger

logger = setup_logger("update_financial_data", log_level=logging.INFO, set_root_logger=True)

if __name__ == "__main__":
    start_year = 2020
    end_year = 2024
    data_manager = DataUpdateManager()

    try:
        logger.info("Starting financial data update...")

        data_manager.update_all_financial_data(
            start_year=start_year,
            end_year=end_year,
        )

        logger.info("Financial data update completed successfully")

    except Exception as e:
        logger.error(f"Error updating financial data: {str(e)}")
        raise
