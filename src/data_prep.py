import re
from src.logger import logger

def clean_text(text: str) -> str:
    if text is None: return ""
    t = str(text)
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocess_df(df):
    logger.info("Starting preprocessing")
    df = df.copy()
    if "text" in df.columns:
        df["clean_text"] = df["text"].apply(clean_text)
    else:
        df["clean_text"] = ""
    logger.info("Preprocessing complete")
    return df
