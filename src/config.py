import os
BASE_DIR = os.getcwd()
class Config:
    RAW_CSV = os.path.join(BASE_DIR, "data", "Tweets.csv")
    PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
    LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
