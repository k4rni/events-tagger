import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

# Load training data
with open(Path(__file__).parent / "events.json") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
tags = [item["tags"] for item in data]

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# Binarize tags
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(tags)

# Train
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X, Y)

# Save everything
joblib.dump((vectorizer, clf, mlb), Path(__file__).parent / "event_tagger.pkl")
print("✅ Model trained and saved.")
