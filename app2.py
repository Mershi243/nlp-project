import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# ✅ Fix: Download missing NLTK resources
nltk.download("punkt")
nltk.download("vader_lexicon")

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    tokens = word_tokenize(text)  # ✅ Fixed missing Punkt tokenizer issue
    sentiment_score = sia.polarity_scores(" ".join(tokens))["compound"]
    sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    return sentiment_score, sentiment_label

# Streamlit App Title
st.title("📊 NLP-Based Sentiment Analysis App")

# 📂 File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Detect text column dynamically
    text_column = next((col for col in df.columns if df[col].dtype == "object"), None)

    if text_column:
        # Apply sentiment analysis
        df[["Sentiment_Score", "Sentiment_Label"]] = df[text_column].apply(analyze_sentiment).apply(pd.Series)
        st.dataframe(df[[text_column, "Sentiment_Label"]])
    else:
        st.warning("No valid text column found!")

# ✍️ Real-time Text Analysis
user_text = st.text_area("Enter text for sentiment analysis:")
if user_text:
    score, label = analyze_sentiment(user_text)
    st.write(f"Sentiment Score: {score}")
    st.write(f"Sentiment Label: {label}")
