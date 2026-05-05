from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any

from ml_project.config import RANDOM_STATE, MODELS, ARTIFACTS_DIR, BEST_MODEL_NAME


# =========================
# 1. FACTORY DE MODÈLES
# =========================
def get_model(name: str):
    if name == "linear":
        return LinearRegression()

    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif name == "gb":
        return GradientBoostingRegressor()

    else:
        raise ValueError(f"Unknown model: {name}")


# =========================
# 2. TRAIN
# =========================
def train_model(X_train, y_train, model_name="rf"):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    return model


# =========================
# 3. EVALUATION
# =========================
def evaluate_model(model, X_test, y_test) -> Dict[str, float]:

    preds = model.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }


# =========================
# 4. PIPELINE MULTI-MODELES
# =========================
def run_all_models(X_train, X_test, y_train, y_test, artifacts_dir=None):

    if artifacts_dir is None:
        artifacts_dir = ARTIFACTS_DIR

    results: Dict[str, Dict[str, float]] = {}

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)

    best_model = None
    best_score = -float("inf")
    best_name = None

    for name in MODELS:
        print(f"\nTraining model: {name}")

        model = train_model(X_train, y_train, name)
        metrics = evaluate_model(model, X_test, y_test)

        results[name] = metrics

        joblib.dump(model, artifacts_path / f"{name}_model.joblib")

        print(f"{name} -> R2: {metrics['r2']:.4f}")

        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_model = model
            best_name = name

    joblib.dump(best_model, artifacts_path / BEST_MODEL_NAME)

    print(f"\nBest model: {best_name} (R2={best_score:.4f})")

    return results, best_name