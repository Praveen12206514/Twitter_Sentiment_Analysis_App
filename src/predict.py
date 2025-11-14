import joblib, os
from src.logger import logger, log_prediction
from src.config import Config
from src.data_prep import clean_text

def load_model_and_vectorizer(model_path=None, vect_path=None):
    model_path = model_path or Config.MODEL_PATH
    vect_path = vect_path or Config.VECTORIZER_PATH
    model = joblib.load(model_path)
    vect = joblib.load(vect_path)
    logger.info("Loaded model and vectorizer")
    return model, vect

def predict_single(model, vect, text):
    cleaned = clean_text(text)
    X = vect.transform([cleaned])
    pred = model.predict(X)[0]
    proba = max(model.predict_proba(X)[0]) if hasattr(model, "predict_proba") else 1.0
    log_prediction(text, pred, proba, model_name="logreg_tfidf")
    return pred, proba