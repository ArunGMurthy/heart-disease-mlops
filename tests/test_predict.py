import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor
from src.predict import predict

PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 1,
    "ca": 0,
    "thal": 3,
}


@pytest.fixture
def toy_model():
    X = pd.DataFrame([PATIENT, {**PATIENT, "age": 45, "cp": 1, "thal": 7, "ca": 2}] * 10)
    y = pd.Series([0, 1] * 10)
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    return model.fit(X, y)


def test_predict_returns_label_and_confidence(toy_model):
    results = predict(toy_model, [PATIENT, PATIENT])
    assert len(results) == 2
    for r in results:
        assert r["prediction"] in (0, 1)
        assert r["label"] == ("Disease" if r["prediction"] == 1 else "No Disease")
        assert 0.5 <= r["confidence"] <= 1.0


def test_predict_rejects_invalid_record(toy_model):
    with pytest.raises(ValidationError):
        predict(toy_model, [{**PATIENT, "sex": 5}])
