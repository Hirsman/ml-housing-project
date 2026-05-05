import pandas as pd

from ml_project.data import load_housing_data


def test_target_column_exists_and_not_empty():
    df = load_housing_data()

    target_column = "MedHouseVal"

    # 1. la colonne existe
    assert target_column in df.columns, "Colonne cible absente"

    # 2. pas entièrement vide
    assert df[target_column].notna().any(), "Colonne cible entièrement vide"
