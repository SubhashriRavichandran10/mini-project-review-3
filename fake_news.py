import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("news.csv")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_news(news):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    result = model.predict(vector)
    return "Real News" if result[0] == 1 else "Fake News"

# User input
news_input = input("\nEnter News Text: ")
print("Prediction:", predict_news(news_input))
