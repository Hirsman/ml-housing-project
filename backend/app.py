from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()


def get_latest_model():
    model_path = Path("artifacts/model_latest.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Model not found")
    return joblib.load(model_path)


model = get_latest_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
