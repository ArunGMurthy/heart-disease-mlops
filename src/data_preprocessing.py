"""
Data cleaning and preprocessing for the Heart Disease UCI dataset.

Builds a scikit-learn ColumnTransformer pipeline so the exact same
transformations used in training are reusable at inference time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "heart_disease.csv"
PROCESSED_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "heart_disease_clean.csv"
)

TARGET = "target"

# Numeric (continuous) features -> impute median + scale.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# Categorical features -> impute most frequent + one-hot encode.
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicate rows, coerce dtypes."""
    df = df.drop_duplicates().reset_index(drop=True)
    # 'ca' and 'thal' sometimes load as object due to the '?' NA marker.
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    save_processed(cleaned)
    print(f"Cleaned data saved: {cleaned.shape[0]} rows x {cleaned.shape[1]} cols")
