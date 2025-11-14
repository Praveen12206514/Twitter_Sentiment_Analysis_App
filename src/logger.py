import logging, os
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("twitter_sentiment")

def log_prediction(text: str, label: str, confidence: float, model_name: str = "logreg_tfidf"):
    logger.info(f"PREDICTION | model={model_name} | label={label} | confidence={confidence:.4f} | text={text}")
