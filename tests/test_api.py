import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"

SAMPLE_PATIENT = {
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


@pytest.fixture(scope="module", autouse=True)
def ensure_model_exists():
    """Train and save a tiny dummy model if none exists, so API tests are
    self-contained and don't require a full training run."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if not MODEL_PATH.exists():
        import numpy as np
        import pandas as pd

        rng = np.random.RandomState(0)
        n = 60
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
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))]
        )
        pipeline.fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        created = True
    yield
    if created:
        MODEL_PATH.unlink(missing_ok=True)


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint_returns_valid_response(client):
    resp = client.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("Disease", "No Disease")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_endpoint_rejects_invalid_input(client):
    bad_payload = dict(SAMPLE_PATIENT)
    bad_payload["sex"] = 5  # out of allowed range [0, 1]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
