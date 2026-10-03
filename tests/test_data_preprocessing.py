import pandas as pd
import pytest

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    clean,
    get_feature_target,
)


@pytest.fixture
def sample_df():
    data = {
        "age": [63, 37, None],
        "sex": [1, 1, 0],
        "cp": [3, 2, 1],
        "trestbps": [145, 130, 120],
        "chol": [233, 250, 200],
        "fbs": [1, 0, 0],
        "restecg": [0, 1, 0],
        "thalach": [150, 187, 172],
        "exang": [0, 0, 0],
        "oldpeak": [2.3, 3.5, 1.4],
        "slope": [0, 0, 2],
        "ca": [0.0, 0.0, None],
        "thal": [1.0, 2.0, 2.0],
        "target": [1, 1, 0],
    }
    return pd.DataFrame(data)


def test_clean_drops_duplicates(sample_df):
    dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned = clean(dup)
    assert len(cleaned) == len(sample_df)


def test_clean_coerces_numeric(sample_df):
    cleaned = clean(sample_df)
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert pd.api.types.is_numeric_dtype(cleaned[col])


def test_get_feature_target_shapes(sample_df):
    cleaned = clean(sample_df)
    X, y = get_feature_target(cleaned)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(X) == len(y) == len(cleaned)
    assert set(y.unique()).issubset({0, 1})


def test_preprocessor_handles_missing_values(sample_df):
    cleaned = clean(sample_df)
    X, y = get_feature_target(cleaned)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    # No NaNs should remain after imputation.
    import numpy as np

    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(arr).any()
    assert arr.shape[0] == len(X)
