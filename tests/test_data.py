import numpy as np
import pytest

from src.data import EXPECTED_FEATURES, get_train_test_split, load_data, validate_data


def test_dataset_shape():
    X, y = load_data()
    assert X.shape == (178, EXPECTED_FEATURES)
    assert set(y.unique()) == {0, 1, 2}


def test_validation_passes_on_clean_data():
    X, y = load_data()
    assert validate_data(X, y) is True


def test_validation_rejects_nulls():
    X, y = load_data()
    X = X.copy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="null"):
        validate_data(X, y)


def test_validation_rejects_wrong_feature_count():
    X, y = load_data()
    with pytest.raises(ValueError, match="features"):
        validate_data(X.iloc[:, :12], y)


def test_split_is_80_20_and_stratified():
    X_train, X_test, y_train, y_test = get_train_test_split()
    assert len(X_train) + len(X_test) == 178
    assert abs(len(X_test) / 178 - 0.2) < 0.01
    _, y = load_data()
    full = y.value_counts(normalize=True).sort_index().values
    assert np.allclose(y_train.value_counts(normalize=True).sort_index().values, full, atol=0.02)
    assert np.allclose(y_test.value_counts(normalize=True).sort_index().values, full, atol=0.03)


def test_split_is_reproducible():
    a = get_train_test_split()
    b = get_train_test_split()
    assert a[0].equals(b[0]) and a[3].equals(b[3])
