import sys
from pathlib import Path

from ml_project.data import load_housing_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_load_housing_data_not_empty():
    df = load_housing_data()
    assert not df.empty


def test_target_column_exists():
    df = load_housing_data()
    assert "MedHouseVal" in df.columns
