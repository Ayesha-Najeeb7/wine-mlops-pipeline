"""Data loading, validation and splitting for the Wine dataset."""
from typing import Optional, Tuple

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
EXPECTED_FEATURES = 13
TEST_SIZE = 0.2


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Return the Wine features (178 x 13) and integer class labels (0, 1, 2)."""
    bunch = load_wine(as_frame=True)
    return bunch.data, bunch.target


def validate_data(X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
    """Raise ValueError if data has nulls or the wrong number of features."""
    if X.isnull().values.any():
        raise ValueError("Validation failed: null values found in features.")
    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Validation failed: expected {EXPECTED_FEATURES} features, got {X.shape[1]}."
        )
    if y is not None and pd.Series(y).isnull().any():
        raise ValueError("Validation failed: null values found in target.")
    return True


def get_train_test_split(test_size: float = TEST_SIZE, seed: int = SEED):
    """Validated, stratified 80/20 split. Returns X_train, X_test, y_train, y_test."""
    X, y = load_data()
    validate_data(X, y)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
