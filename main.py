from fastapi import FastAPI
from pydantic import BaseModel
from transformers.pipelines import pipeline

app = FastAPI()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class Input(BaseModel):
    description: str
    labels: list[str] = [
        "family", "nature", "music", "fitness", "food", "animals", "educational", "sports"
    ]

@app.post("/predict")
async def predict(input: Input):
    result = classifier(input.description, input.labels)
    return {
        "labels": result["labels"],
        "scores": result["scores"]
    }

@app.get("/")
def home():
    return {"message": "Zero-shot classifier is live!"}
