from fastapi import FastAPI, Request
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
    return {"tags": tags}
