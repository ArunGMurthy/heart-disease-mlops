"""
Download the Heart Disease UCI dataset (Cleveland subset) and save a raw CSV.

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/45/heart+disease

Usage:
    python data/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column names per the UCI dataset documentation.
COLUMNS = [
    "age",
    "sex",
    "cp",        # chest pain type
    "trestbps",  # resting blood pressure
    "chol",      # serum cholesterol
    "fbs",       # fasting blood sugar > 120 mg/dl
    "restecg",   # resting ECG results
    "thalach",   # max heart rate achieved
    "exang",     # exercise induced angina
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of the peak exercise ST segment
    "ca",        # number of major vessels colored by fluoroscopy
    "thal",      # thalassemia
    "target",    # diagnosis of heart disease (angiographic status)
]

OUT_PATH = Path(__file__).resolve().parent / "raw" / "heart_disease.csv"


def download(out_path: Path = OUT_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {RAW_URL} ...")
    df = pd.read_csv(RAW_URL, header=None, names=COLUMNS, na_values="?")

    # Binarize target: 0 = no disease, 1..4 = disease presence -> 1
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to {out_path}")
    return out_path


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:  # pragma: no cover
        print(f"Download failed ({exc}).", file=sys.stderr)
        print(
            "If you have no internet access, manually download "
            "processed.cleveland.data from the UCI repository "
            "(https://archive.ics.uci.edu/dataset/45/heart+disease), "
            "name it 'raw_cleveland.data' and place it in data/raw/, "
            "then re-run this script with --from-local.",
            file=sys.stderr,
        )
        sys.exit(1)
