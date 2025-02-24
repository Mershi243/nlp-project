import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import string

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize required NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""  # Return empty string if text is missing

    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return " ".join(tokens)

# Function to analyze sentiment
def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    sentiment_score = sia.polarity_scores(processed_text)["compound"]

    if sentiment_score >= 0.05:
        sentiment_label = "Positive"
    elif sentiment_score <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return processed_text, sentiment_score, sentiment_label

# Streamlit App UI
st.title("Sentiment Analysis App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Let user select text column dynamically
    text_column = st.selectbox("Select the column containing text", df.columns)

    if st.button("Analyze Sentiment"):
        try:
            df["Processed_Text"], df["Sentiment_Score"], df["Sentiment_Label"] = zip(*df[text_column].apply(analyze_sentiment))
            st.write(df)  # Display DataFrame
        except Exception as e:
            st.error(f"Error processing sentiment: {e}")

    st.download_button("Download Results", df.to_csv(index=False), file_name="sentiment_results.csv", mime="text/csv")
