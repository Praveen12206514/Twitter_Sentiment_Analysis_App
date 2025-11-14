import os, json
import matplotlib.pyplot as plt
from src.logger import logger

def run_eda(df):
    logger.info("Running EDA")
    eda_dir = os.path.join(os.getcwd(), "eda")
    charts = os.path.join(eda_dir, "charts")
    os.makedirs(charts, exist_ok=True)
    if "airline_sentiment" in df.columns:
        vc = df["airline_sentiment"].value_counts()
        plt.figure(figsize=(8,4))
        vc.plot(kind="bar")
        plt.title("Sentiment Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(charts, "label_distribution.png"))
        plt.close()
    if "text" in df.columns:
        df["length"] = df["text"].astype(str).str.len()
        plt.figure(figsize=(8,4))
        df["length"].hist(bins=40)
        plt.title("Tweet length distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(charts, "length_hist.png"))
        plt.close()
    summary = {
        "rows": int(len(df)),
        "sentiment_counts": df["airline_sentiment"].value_counts().to_dict() if "airline_sentiment" in df.columns else {}
    }
    with open(os.path.join(eda_dir, "eda_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("EDA complete")
