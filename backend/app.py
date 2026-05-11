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

    # Extraire user_id
    user_id = data.pop("user_id", "anonymous")

    # Construire dataframe
    df = pd.DataFrame([data])

    # Choisir variante
    variant = choose_variant(user_id)

    # Sélectionner modèle
    model = models[variant]

    # Prédiction
    prediction = float(model.predict(df)[0])

    # Logging expérimentation
    log_prediction({
        "user_id": user_id,
        "variant": variant,
        "prediction": prediction,
        "features": data
    })

    # Réponse API
    return {
        "prediction": prediction,
        "variant": variant
    }