import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from .preprocessor import preprocess_dataset
from .download import download_dataset
from .config import TrainConfig

config = TrainConfig()
dataset_path = download_dataset()
dataset_file = os.path.join(dataset_path, config.dataset_file)

if not os.path.isfile(dataset_file):
    raise FileNotFoundError(f"Expected dataset file not found: {dataset_file}")

df = pd.read_csv(dataset_file, encoding="latin-1")
df_cleaned = preprocess_dataset(df)

vectorizer = TfidfVectorizer(config.max_features)
X = vectorizer.fit_transform(df_cleaned['review'])
y = df_cleaned['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, config.test_size, config.random_state)

model = DecisionTreeClassifier(config.random_state)
model.train(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)