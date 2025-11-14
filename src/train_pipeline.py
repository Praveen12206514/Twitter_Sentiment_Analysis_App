from src.utils import load_data
from src.data_prep import preprocess_df
from src.eda import run_eda
from src.modeling import train_and_save
from src.config import Config
from src.logger import logger

def run():
    logger.info("Starting training pipeline")
    df = load_data(Config.RAW_CSV)
    run_eda(df)
    df_clean = preprocess_df(df)
    clf, vect = train_and_save(df_clean, Config.MODEL_PATH, Config.VECTORIZER_PATH)
    logger.info("Training pipeline finished")

if __name__=="__main__":
    run()
