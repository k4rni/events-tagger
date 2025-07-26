from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

vectorizer, clf, mlb = joblib.load(Path(__file__).parent / "model/event_tagger.pkl")

class EventInput(BaseModel):
    text: str

@app.post("/tag")
async def tag_event(event: EventInput):
    X = vectorizer.transform([event.text])
    Y = clf.predict(X)
    tags = mlb.inverse_transform(Y)[0]

    if not tags:
        probs = clf.predict_proba(X)[0]
        top_index = probs.argmax()
        fallback_tag = mlb.classes_[top_index]
        tags = [fallback_tag]

    return {"tags": tags}
