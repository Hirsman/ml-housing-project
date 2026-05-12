import time

import pandas as pd
from fastapi import FastAPI

from backend.services.ab_router import choose_variant
from backend.services.experiment_logger import log_prediction
from backend.services.model_registry import get_models

app = FastAPI()

models = get_models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):

    start_time = time.time()

    # user id
    user_id = data.pop("user_id", "anonymous")

    # mapping frontend -> modèle
    rename_map = {
        "average_rooms": "AveRooms",
        "average_bedrooms": "AveBedrms",
        "average_occupancy": "AveOccup",
        "housing_median_age": "HouseAge",
        "median_income": "MedInc",
        "population": "Population",
        "latitude": "Latitude",
        "longitude": "Longitude",
    }

    # normalisation
    normalized_data = {}

    for k, v in data.items():
        normalized_key = rename_map.get(k, k)
        normalized_data[normalized_key] = v

    data = normalized_data

    # features attendues par sklearn
    expected_features = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    # garantir ordre + colonnes
    data = {k: data.get(k, 0) for k in expected_features}

    # dataframe
    df = pd.DataFrame([data])

    # A/B routing
    variant = choose_variant(user_id)
    model = models[variant]

    # prediction
    prediction = float(model.predict(df)[0])

    # latence
    latency_ms = (time.time() - start_time) * 1000

    # logging expérimentation
    log_prediction(
        {
            "user_id": user_id,
            "variant": variant,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "features": data,
        }
    )

    # réponse API
    return {
        "prediction": prediction,
        "variant": variant,
        "latency_ms": latency_ms,
    }