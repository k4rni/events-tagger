from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

# Loads the pretrained model
vectorizer, clf, mlb = joblib.load(Path(__file__).parent / "model/event_tagger.pkl")


class EventInput(BaseModel):
    text: str


# Receives a POST request to /tag with JSON
@app.post("/tag")
async def tag_event(event: EventInput):
    # Transforms the text into a numerical TF-IDF vector
    X = vectorizer.transform([event.text])
    # Uses the classifier to predict which tags apply
    Y = clf.predict(X)
    # Converts the binary label vector back into human-readable tags
    tags = mlb.inverse_transform(Y)[0]

    # If no tag was predicted, return the tag with the highest probability
    if not tags:
        probs = clf.predict_proba(X)[0]
        top_index = probs.argmax()
        fallback_tag = mlb.classes_[top_index]
        tags = [fallback_tag]

    # Returns the result as JSON
    return {"tags": tags}
