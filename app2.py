import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# 🔽 Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# 📌 Initialize NLP tools
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# 🔎 Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# 🏷️ Sentiment Analysis Function
def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    sentiment_score = sia.polarity_scores(processed_text)["compound"]
    sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    return processed_text, sentiment_score, sentiment_label

# 🎯 Streamlit UI
st.title("📊 NLP-Based Sentiment Analysis App")

# 📂 File Upload Section
st.header("📥 Upload a CSV or TXT file for Sentiment Analysis")
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file:
    # 📑 Handle File Upload
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        content = uploaded_file.getvalue().decode("utf-8")
        df = pd.DataFrame({"Text": content.splitlines()})

    st.write("### 📌 Dataset Preview")
    st.dataframe(df.head())

    # 🎯 Select Text Column for Analysis
    text_columns = df.select_dtypes(include="object").columns
    if not text_columns.empty:
        text_column = st.selectbox("Select the text column:", text_columns)

        if st.button("Analyze Sentiment"):
            df["Processed_Text"], df["Sentiment_Score"], df["Sentiment_Label"] = zip(*df[text_column].apply(analyze_sentiment))

            st.write("### ✅ Sentiment Analysis Results")
            st.dataframe(df[[text_column, "Processed_Text", "Sentiment_Label"]])

            # 📊 Sentiment Distribution
            st.write("### 🔍 Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Sentiment_Label", data=df, palette={"Positive": "green", "Negative": "red", "Neutral": "gray"}, ax=ax)
            st.pyplot(fig)

            # 📥 Download Results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", data=csv, file_name="sentiment_results.csv", mime="text/csv")
    else:
        st.warning("⚠️ No valid text column found in the uploaded file!")

# ✍️ **Real-time Text Sentiment Analysis**
st.header("📝 Real-time Text Sentiment Analysis")
user_text = st.text_area("Enter text for sentiment analysis:")

if user_text:
    processed_text, sentiment_score, sentiment_label = analyze_sentiment(user_text)
    st.write(f"**Processed Text:** {processed_text}")
    st.write(f"**Sentiment Score:** {sentiment_score}")
    st.write(f"**Sentiment Label:** {sentiment_label}")

# 🎤 **Real-time Speech Sentiment Analysis**
st.header("🎤 Real-time Speech Sentiment Analysis")

if st.button("🎙️ Start Recording"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now!")
        audio = recognizer.listen(source)

        try:
            speech_text = recognizer.recognize_google(audio)
            st.success(f"**Recognized Speech:** {speech_text}")

            processed_text, sentiment_score, sentiment_label = analyze_sentiment(speech_text)
            st.write(f"**Processed Text:** {processed_text}")
            st.write(f"**Sentiment Score:** {sentiment_score}")
            st.write(f"**Sentiment Label:** {sentiment_label}")

        except sr.UnknownValueError:
            st.error("⚠️ Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"⚠️ Could not request results from Google Speech Recognition service; {e}")
