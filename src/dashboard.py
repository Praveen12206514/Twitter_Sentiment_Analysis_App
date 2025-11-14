# Streamlit dashboard mirroring requested UI
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st, pandas as pd, joblib
import plotly.express as px   # <-- Added for interactive chart

from src.config import Config
from src.utils import load_data
from src.eda import run_eda
from src.data_prep import preprocess_df, clean_text
from src.predict import load_model_and_vectorizer, predict_single

st.set_page_config(page_title="US Airline Twitter Sentiment", layout="wide")
st.title("✈ US Airline Twitter Sentiment — Dashboard")

# paths
MODEL_PATH = Config.MODEL_PATH
VECT_PATH = Config.VECTORIZER_PATH
CSV_PATH = Config.RAW_CSV


@st.cache_data
def load_processed():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df
    return pd.DataFrame()


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        return load_model_and_vectorizer(MODEL_PATH, VECT_PATH)
    return None, None


# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📊 Overview", "🔮 Live Prediction"])


# -------------------------------------------------
# 📌 TAB 1 — OVERVIEW
# -------------------------------------------------
with tab1:
    df = load_processed()

    if df.empty:
        st.warning("Processed data not found. Run training pipeline first.")

    else:
        # ------- METRICS -------
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Total Tweets", int(len(df)))
        with col2:
            st.metric("Airlines", int(df["airline"].nunique()) if "airline" in df.columns else 0)
        with col3:
            if "tweet_created" in df.columns:
                ts = pd.to_datetime(df["tweet_created"], errors="coerce")
                st.metric("Time Range", f"{ts.min().date()} → {ts.max().date()}")
            else:
                st.metric("Time Range", "N/A")

        # ------- SENTIMENT DISTRIBUTION -------
        st.subheader("Sentiment Distribution")
        if "airline_sentiment" in df.columns:
            counts = df["airline_sentiment"].value_counts()
            st.bar_chart(counts, use_container_width=True)

        # ------- SENTIMENT BY AIRLINE (INTERACTIVE) -------
        st.subheader("Sentiment by Airline")

        if "airline" in df.columns and "airline_sentiment" in df.columns:

            grouped = (
                df.groupby(["airline", "airline_sentiment"])
                  .size()
                  .reset_index(name="count")
            )

            sentiment_order = ["negative", "neutral", "positive"]

            fig = px.bar(
                grouped,
                x="airline",
                y="count",
                color="airline_sentiment",
                barmode="stack",
                category_orders={"airline_sentiment": sentiment_order},
                color_discrete_map={
                    "negative": "#ff7f7f",
                    "neutral": "#1f77b4",
                    "positive": "#9ecae1",
                },
                hover_data={"count": True, "airline": True, "airline_sentiment": True},
            )

            fig.update_layout(
                xaxis_title="Airline",
                yaxis_title="Tweet Count",
                legend_title="Sentiment",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=14),
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

        # # OPTIONAL static image
        # chart_path = os.path.join(os.getcwd(), "eda", "charts", "label_distribution.png")
        # if os.path.exists(chart_path):
        #     st.image(chart_path, use_container_width=True)


# -------------------------------------------------
# 📌 TAB 2 — LIVE PREDICTION
# -------------------------------------------------
with tab2:
    model, vect = load_model()
    
    if model is None:
        st.info("Model not trained yet. Run training script to create model and vectorizer.")
    else:
        st.success("✅ Model loaded successfully. Ready for predictions!")

    txt = st.text_area("Enter a tweet to analyze:", "My flight was amazing! Great service from the airline.")

    if st.button("Predict Sentiment"):
        if model is None:
            st.error("Model not found.")
        elif not txt.strip():
            st.warning("Please enter a tweet.")
        else:
            label, conf = predict_single(model, vect, txt)
            conf_pct = conf * 100
            emoji = "✅" if label.lower() == "positive" else ("❌" if label.lower() == "negative" else "⚪")

            st.markdown(f"### {emoji} Prediction: *{label.capitalize()}*")
            st.write(f"*Confidence:* {conf_pct:.2f}%")