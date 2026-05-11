import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def split_features_target(df: pd.DataFrame):
    """Sépare les variables explicatives X et la cible y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(X, y):
    """Découpe les données en train/test."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
