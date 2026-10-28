import nltk
import os
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from download import download_dataset
from config import TrainConfig

config = TrainConfig()
dataset_path = download_dataset()
dataset_file = os.path.join(dataset_path, config.dataset_file)

if not os.path.isfile(dataset_file):
    raise FileNotFoundError(f"Expected dataset file not found: {dataset_file}")

df = pd.read_csv(dataset_file, encoding="cp1252")
print(df.head())
