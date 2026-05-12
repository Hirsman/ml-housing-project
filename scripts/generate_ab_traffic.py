import time
import pandas as pd
from fastapi import FastAPI

from backend.services.ab_router import choose_variant
from backend.services.experiment_logger import log_prediction
from backend.services.model_registry import get_models

app = FastAPI()

# Charger les modèles A/B
models = get_models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):

    # 🔥 mesure du temps de réponse
    start_time = time.time()

    # 🔥 extraction user_id
    user_id = data.pop("user_id", "anonymous")

    # 🔥 mapping des features frontend → modèle
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

    # normalisation des noms de colonnes
    data = {rename_map.get(k, k): v for k, v in data.items()}

    # features attendues par le modèle
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

    # sécurisation (valeurs manquantes → 0)
    data = {k: data.get(k, 0) for k in expected_features}

    # dataframe pour sklearn
    df = pd.DataFrame([data])

    # 🔥 A/B testing routing
    variant = choose_variant(user_id)
    model = models[variant]

    # prédiction
    prediction = float(model.predict(df)[0])

    # latence
    latency_ms = (time.time() - start_time) * 1000

    # logging expérimentation
    log_prediction({
        "user_id": user_id,
        "variant": variant,
        "prediction": prediction,
        "latency_ms": latency_ms,
        "features": data,
    })

    # réponse API
    return {
        "prediction": prediction,
        "variant": variant,
        "latency_ms": latency_ms,
    }