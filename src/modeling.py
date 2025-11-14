import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.logger import logger

def train_and_save(df, model_path, vect_path):
    logger.info("Training model (TF-IDF + LogisticRegression)")
    texts = df["clean_text"].fillna("").tolist()
    y = df["airline_sentiment"].tolist()
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    X = vect.fit_transform(texts)
    clf = LogisticRegression(max_iter=400)
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    joblib.dump(vect, vect_path)
    logger.info(f"Saved model to {model_path} and vectorizer to {vect_path}")
    return clf, vect
