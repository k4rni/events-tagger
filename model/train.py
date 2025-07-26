import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

# Load events.json training data file
with open(Path(__file__).parent / "events.json") as f:
    data = json.load(f)

# Extracts texts (title of events) and tags (labeled tags)
texts = [item["text"] for item in data]
tags = [item["tags"] for item in data]

# Converts the raw text into a numeric matrix
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# Converts lists of tags into a binary matrix (column = tag, row = tags that apply to that text)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(tags)

# Use OneVsRestClassifier to train one Logistic Regression model per tag
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X, Y)

# Saves the trained components
# - vectorizer: Converting new text into features
# - clf: The trained classifier
# - mlb: Converting predicted tag indices back to tag names
joblib.dump((vectorizer, clf, mlb), Path(__file__).parent / "event_tagger.pkl")
print("✅ Model trained and saved.")
