import pandas as pd
from src.logger import logger
def load_data(path):
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)
