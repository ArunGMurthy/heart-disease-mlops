"""
Batch/offline inference with the packaged model pipeline.

Loads models/model.pkl (preprocessing + classifier) and predicts for one
patient (JSON object) or many (JSON array of objects), without needing the
API to be running.

Usage:
    python -m src.predict sample_request.json
    python -m src.predict patients.json --output predictions.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from api.schemas import PatientFeatures

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, records: list[dict]) -> list[dict]:
    """Validate records against the API schema and return prediction + confidence."""
    rows = [PatientFeatures(**r).model_dump() for r in records]
    df = pd.DataFrame(rows)
    preds = model.predict(df)
    probas = model.predict_proba(df)
    return [
        {
            "prediction": int(p),
            "label": "Disease" if p == 1 else "No Disease",
            "confidence": round(float(proba[p]), 4),
        }
        for p, proba in zip(preds, probas)
    ]


def main():
    parser = argparse.ArgumentParser(description="Predict heart disease risk from a JSON file.")
    parser.add_argument("input", type=Path, help="JSON object or array of patient records")
    parser.add_argument("--output", type=Path, help="optional path to write predictions JSON")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    records = data if isinstance(data, list) else [data]
    results = predict(load_model(), records)

    out = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(out)
        print(f"Wrote {len(results)} predictions to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
