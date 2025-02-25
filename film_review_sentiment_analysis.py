import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("IMDB Dataset.csv")
# print(df.head())  # shows the structure of the csv

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

important_words = {"not", "no", "never"}  # stopwords to keep

# clean data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words or word in important_words])
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

# convert sentiment labels to nums
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

# convert text to vectors
vectorizer = TfidfVectorizer(max_features=5000)  # limit: 5000 most important words
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
    
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")