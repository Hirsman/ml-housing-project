import pandas as pd

REQUIRED_COLUMNS = ["age", "bmi", "bp"]


def validate_columns(df: pd.DataFrame) -> bool:
    """Vérifie que le DataFrame contient les colonnes minimales attendues."""
    return all(col in df.columns for col in REQUIRED_COLUMNS)


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes contenant au moins une valeur manquante."""
    return df.dropna()
