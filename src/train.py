"""
Train and evaluate classification models for heart disease prediction.

Trains Logistic Regression and Random Forest, performs cross-validation
and hyperparameter tuning (GridSearchCV), logs everything (params,
metrics, plots, model artifacts) to MLflow, and saves the best full
pipeline (preprocessing + model) as a pickle for reproducible inference.

Usage:
    python src/train.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data_preprocessing import (  # noqa: E402
    build_preprocessor,
    clean,
    get_feature_target,
    load_raw,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MLFLOW_DB_PATH = ROOT / "mlflow.db"

RANDOM_STATE = 42

MODEL_GRID = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10],
        },
    },
}


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_and_log(model_name: str, cfg: dict, X_train, X_test, y_train, y_test, preprocessor):
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", cfg["estimator"])])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipeline, cfg["params"], cv=cv, scoring="roc_auc", n_jobs=-1)

    with mlflow.start_run(run_name=model_name):
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        metrics = evaluate(best_pipeline, X_test, y_test)

        mlflow.log_param("model_type", model_name)
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ROC curve artifact
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax)
        ax.set_title(f"ROC Curve - {model_name}")
        fig.tight_layout()
        roc_path = FIG_DIR / f"roc_{model_name}.png"
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(roc_path))

        mlflow.sklearn.log_model(
            best_pipeline, artifact_path="model", serialization_format="pickle"
        )

        print(f"[{model_name}] best_params={search.best_params_} metrics={metrics}")
        return best_pipeline, metrics, search.best_score_


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("heart-disease-classification")

    df = clean(load_raw())
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}
    best_overall = None
    best_score = -1.0
    best_name = None

    for name, cfg in MODEL_GRID.items():
        preprocessor = build_preprocessor()
        pipeline, metrics, cv_score = train_and_log(
            name, cfg, X_train, X_test, y_train, y_test, preprocessor
        )
        results[name] = metrics
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_overall = pipeline
            best_name = name

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_overall, f)

    metadata = {
        "best_model": best_name,
        "metrics": results[best_name],
        "all_results": results,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {best_name} (test ROC-AUC={best_score:.4f})")
    print(f"Saved pipeline to {model_path}")


if __name__ == "__main__":
    main()
