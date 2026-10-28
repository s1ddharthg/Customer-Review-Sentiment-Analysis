import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from download import download_dataset

# file_path = '/content/flipkart_data.csv'
# df = pd.read_csv(file_path)


# df.head()

download_dataset()
