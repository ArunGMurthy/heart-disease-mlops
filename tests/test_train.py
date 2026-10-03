import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor
from src.train import evaluate


@pytest.fixture
def toy_classification_data():
    rng = np.random.RandomState(0)
    n = 100
    X = pd.DataFrame(
        {
            "age": rng.randint(30, 80, n),
            "trestbps": rng.randint(100, 180, n),
            "chol": rng.randint(150, 300, n),
            "thalach": rng.randint(100, 200, n),
            "oldpeak": rng.uniform(0, 4, n),
            "ca": rng.randint(0, 4, n),
            "sex": rng.randint(0, 2, n),
            "cp": rng.randint(0, 4, n),
            "fbs": rng.randint(0, 2, n),
            "restecg": rng.randint(0, 3, n),
            "exang": rng.randint(0, 2, n),
            "slope": rng.randint(0, 3, n),
            "thal": rng.randint(0, 3, n),
        }
    )
    y = pd.Series(rng.randint(0, 2, n))
    return X, y


def test_pipeline_trains_and_evaluates(toy_classification_data):
    X, y = toy_classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

    for key in ("accuracy", "precision", "recall", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_pipeline_predict_proba_shape(toy_classification_data):
    X, y = toy_classification_data
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
