# dashboard.py
import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -----------------------------
# 🎨 NETFLIX UI STYLE
# -----------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #141414;
        color: white;
    }
    h1 {
        color: #E50914;
        text-align: center;
        font-size: 42px;
    }
    h2, h3 {
        color: #ffffff;
    }
    div.stButton > button {
        background-color: #E50914;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #b20710;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Netflix Trend Predictor", layout="centered")

st.markdown("<h1>NETFLIX TREND ANALYZER 🎬</h1>", unsafe_allow_html=True)
st.write("Predict trends and explore top trending Netflix content using ML.")

st.markdown("---")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "netflix_trend_model.pkl")
data_path = os.path.join(BASE_DIR, "..", "data", "netflix_features.csv")

# -----------------------------
# Load model
# -----------------------------
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Model loaded successfully ✅")
else:
    st.error("Model file NOT found ❌")

# -----------------------------
# Load dataset
# -----------------------------
df = None
if os.path.exists(data_path):
    df = pd.read_csv(data_path)

# -----------------------------
# 🔍 Filters
# -----------------------------
st.markdown("### 🔍 Filters")

genre = None
if df is not None and 'genre' in df.columns:
    genre = st.selectbox("Select Genre", ["All"] + list(df['genre'].dropna().unique()))

search = st.text_input("Search by Title")

st.markdown("---")

# -----------------------------
# 🎯 Input Section
# -----------------------------
st.markdown("### 🎯 Enter Content Details")

release_year = st.number_input(
    "Release Year",
    min_value=1950,
    max_value=datetime.now().year,
    value=2022
)

duration = st.number_input(
    "Duration (minutes)",
    min_value=1,
    max_value=500,
    value=90
)

# Content Age
content_age = datetime.now().year - release_year
st.number_input("Content Age (years)", value=content_age, disabled=True)

st.markdown("---")

# -----------------------------
# 🔮 Prediction
# -----------------------------
if st.button("Predict Trend"):

    if model is not None:
        try:
            expected_cols = list(model.feature_names_in_)

            X_new = pd.DataFrame({
                'release_year': [release_year],
                'content_age': [content_age],
                'duration_minutes': [duration],
                'duration': [duration]
            })

            X_new = X_new[expected_cols]

            prediction = model.predict(X_new)[0]

            st.write("Prediction Value:", prediction)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0][1]
                st.progress(int(proba * 100))
                st.write(f"Confidence: {proba*100:.2f}%")

            if prediction == 1:
                st.success("🔥 This content is likely to TREND on Netflix!")
            else:
                st.info("ℹ️ This content is NOT likely to trend yet.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")

# -----------------------------
# 🔥 Top 10 Trending
# -----------------------------
st.markdown("### 🔥 Top 10 Predicted Trending Content")

if st.button("Show Top 10 Trending"):

    if df is not None and model is not None:

        try:
            year_df = df[df['release_year'] == release_year].copy()

            # Apply filters
            if genre and genre != "All":
                year_df = year_df[year_df['genre'] == genre]

            if search:
                year_df = year_df[
                    year_df['title'].str.contains(search, case=False, na=False)
                ]

            if year_df.empty:
                st.warning("No matching data found.")
                st.stop()

            # Feature engineering
            year_df['content_age'] = datetime.now().year - year_df['release_year']

            if 'duration_minutes' not in year_df.columns:
                if 'duration' in year_df.columns:
                    year_df['duration_minutes'] = year_df['duration']

            expected_cols = list(model.feature_names_in_)
            X = year_df[expected_cols]

            probs = model.predict_proba(X)[:, 1]
            year_df['trend_score'] = probs

            top10 = year_df.sort_values(by='trend_score', ascending=False).head(10)

            st.success(f"Top 10 Trending Content in {release_year}")

            display_cols = [col for col in ['title', 'genre', 'duration_minutes', 'trend_score'] if col in top10.columns]
            st.dataframe(top10[display_cols])

            # 📊 Chart
            if 'title' in top10.columns:
                st.subheader("📊 Trending Score Chart")
                st.bar_chart(top10.set_index('title')['trend_score'])

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")

# -----------------------------
# 🔍 Search Results
# -----------------------------
if search and df is not None:
    st.markdown("### 🔍 Search Results")
    result = df[df['title'].str.contains(search, case=False, na=False)]
    st.dataframe(result.head(10))

# -----------------------------
# Dataset preview
# -----------------------------
if st.checkbox("Show Dataset Sample"):
    if df is not None:
        st.dataframe(df.head(10))
    else:
        st.info("Dataset not available.")