from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from pathlib import Path


# =========================
# 1. FACTORY DE MODÈLES
# =========================
def get_model(name: str):
    """Retourne un modèle ML selon son nom."""

    if name == "linear":
        return LinearRegression()

    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    elif name == "gb":
        return GradientBoostingRegressor()

    else:
        raise ValueError(f"Unknown model: {name}")


# =========================
# 2. TRAIN D'UN MODÈLE
# =========================
def train_model(X_train, y_train, model_name="rf"):
    """Entraîne un modèle choisi."""
    model = get_model(model_name)
    model.fit(X_train, y_train)
    return model


# =========================
# 3. ÉVALUATION
# =========================
def evaluate_model(model, X_test, y_test):
    """Calcule les métriques principales."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }

    return metrics


# =========================
# 4. PIPELINE MULTI-MODELES
# =========================
def run_all_models(X_train, X_test, y_train, y_test, artifacts_dir="artifacts"):
    """Entraîne plusieurs modèles et compare leurs performances."""

    models = ["linear", "rf", "gb"]
    results = {}
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)

    best_model = None
    best_score = -float("inf")
    best_name = None

    for name in models:
        print(f"\nTraining model: {name}")

        model = train_model(X_train, y_train, name)
        metrics = evaluate_model(model, X_test, y_test)

        results[name] = metrics

        # sauvegarde modèle
        joblib.dump(model, artifacts_path / f"{name}_model.joblib")

        print(f"{name} -> R2: {metrics['r2']:.4f}")

        # sélection du meilleur modèle
        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_model = model
            best_name = name

    # sauvegarde du meilleur modèle
    joblib.dump(best_model, artifacts_path / "best_model.joblib")

    print(f"\nBest model: {best_name} (R2={best_score:.4f})")

    return results, best_name