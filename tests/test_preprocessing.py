# tests/test_preprocessing.py
import pandas as pd

from src.ml_project.preprocessing import remove_missing_values, validate_columns


def test_validate_columns_success():
    df = pd.DataFrame({"age": [50], "bmi": [0.06], "bp": [0.02]})
    assert validate_columns(df) is True


def test_validate_columns_failure():
    df = pd.DataFrame({"age": [50], "bmi": [0.06]})
    assert validate_columns(df) is False


def test_remove_missing_values():
    df = pd.DataFrame({"age": [50, None], "bmi": [0.06, 0.02], "bp": [0.02, 0.01]})
    cleaned = remove_missing_values(df)
    assert len(cleaned) == 1
