import json
from pathlib import Path

import joblib

from ml_project.config import MODELS
from ml_project.data import load_housing_data
from ml_project.evaluate import evaluate_model
from ml_project.features import split_features_target, split_train_test
from ml_project.train import train_model


def run_pipeline(artifacts_dir: str = "artifacts") -> dict:

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # ======================
    # 1. DATA
    # ======================
    df = load_housing_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # ======================
    # 2. MODELES
    # ======================
    results = {}
    best_model = None
    best_score = -float("inf")
    best_name = None

    print("MODELS USED:", MODELS)

    # ======================
    # 3. TRAIN + EVAL
    # ======================
    for name in MODELS:
        print(f"Training: {name}")

        model = train_model(X_train, y_train, model_name=name)
        metrics = evaluate_model(model, X_test, y_test)

        results[name] = metrics

        joblib.dump(model, artifacts_path / f"{name}_model.joblib")

        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_model = model
            best_name = name

    # ======================
    # 4. BEST MODEL
    # ======================
    joblib.dump(best_model, artifacts_path / "best_model.joblib")

    # ======================
    # 5. METRICS
    # ======================
    with open(artifacts_path / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "results": results,
                "best_model": best_name,
                "best_score": best_score,
            },
            file,
            indent=2,
        )

    return {
        "results": results,
        "best_model": best_name,
        "best_score": best_score,
    }