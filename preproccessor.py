import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv(dataset_file, encoding="latin-1", header=None)

def preprocess_dataset(df):
    # Remove rows with missing values
    df = df.dropna()

    # Convert text to lowercase
    df[1] = df[1].astype(str).str.lower()

    # Remove punctuation and special characters
    df[1] = df[1].str.replace(r"[^\w\s]", "", regex=True)

    df[2] = df[2].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

    # Remove stopwords
    df[5] = df[5].astype(str).apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

    df["sentiment"] = df[2].apply(lambda x: 1 if x >= 4 else 0)

    return df