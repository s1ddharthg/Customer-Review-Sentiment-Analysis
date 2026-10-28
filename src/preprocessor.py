import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_dataset(df):

    # Remove rows with missing values
    df = df.dropna()

    # Convert text to lowercase
    df["Comments"] = df["Comments"].astype(str).str.lower()

    # Remove punctuation and special characters
    df["Comments"] = df["Comments"].str.replace(r"[^\w\s]", "", regex=True)

    df["Rating"] = df["Rating"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    # Remove stopwords
    df["Comments"] = df["Comments"].astype(str).apply(
        lambda x: " ".join(word for word in x.split() if word not in stop_words)
    )

    df["sentiment"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

    return df