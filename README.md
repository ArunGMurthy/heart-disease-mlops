# Heart Disease Risk Prediction — End-to-End MLOps Pipeline

A reproducible, CI/CD-automated, containerized, and monitored ML pipeline
that predicts heart disease risk from patient health data (UCI Heart
Disease dataset) and serves predictions via a FastAPI `/predict` endpoint.

## Architecture

```
                 ┌──────────────┐
                 │ UCI Dataset  │
                 │ (download_   │
                 │  data.py)    │
                 └──────┬───────┘
                        ▼
           ┌─────────────────────────┐
           │  Preprocessing Pipeline │  src/data_preprocessing.py
           │  (impute, scale, OHE)   │
           └──────────┬──────────────┘
                       ▼
        ┌───────────────────────────────┐
        │  Model Training + Tuning      │  src/train.py
        │  LogisticRegression, RF       │
        │  + GridSearchCV + CV metrics  │
        └──────────────┬─────────────────┘
                        │  logs params/metrics/artifacts
                        ▼
                 ┌─────────────┐
                 │   MLflow    │  mlflow.db + mlruns/
                 └──────┬──────┘
                        ▼
              models/model.pkl (best pipeline)
                        │
                        ▼
        ┌───────────────────────────────┐
        │  FastAPI serving app          │  api/main.py
        │  /predict  /health  /metrics  │
        └──────────────┬─────────────────┘
                        ▼
                 ┌─────────────┐
                 │   Docker    │  Dockerfile
                 └──────┬──────┘
                        ▼
      ┌─────────────────────────────────┐
      │ Kubernetes (Minikube/GKE/EKS/AKS)│  k8s/*.yaml
      │ Deployment + LoadBalancer Service│
      └─────────────────┬─────────────────┘
                        ▼
          Prometheus (scrape /metrics) + Grafana
          monitoring/, docker-compose.monitoring.yml

GitHub Actions (.github/workflows/ci.yml): lint → unit tests → train →
build/smoke-test Docker image, on every push/PR.
```

## Project Structure

```
.
├── data/
│   ├── download_data.py        # fetches raw UCI dataset
│   ├── raw/                    # downloaded CSV (gitignored)
│   └── processed/              # cleaned CSV (tracked)
├── src/
│   ├── data_preprocessing.py   # cleaning + sklearn ColumnTransformer
│   ├── eda.py                  # EDA plots -> reports/figures/
│   └── train.py                # training, tuning, MLflow logging
├── api/
│   ├── main.py                 # FastAPI app (/predict, /health, /metrics)
│   └── schemas.py              # pydantic request/response models
├── tests/                      # pytest unit tests
├── k8s/                        # Deployment + Service manifests
├── monitoring/                 # Prometheus config
├── .github/workflows/ci.yml    # CI/CD pipeline
├── reports/figures/            # EDA & evaluation plots
├── screenshots/                # deployment/CI screenshots for report
├── Dockerfile
├── docker-compose.monitoring.yml
├── requirements.txt
├── sample_request.json         # example /predict payload
└── RUNBOOK.md                  # step-by-step execution guide
```

## Setup / Install Instructions

Requires **Python 3.13** (the pinned `numpy==2.5.3` needs Python ≥ 3.12;
the Dockerfile and CI also use 3.13).

All commands are run from the project root. Scripts inside `src/` must
be run as modules (`python -m src.train`, not `python src/train.py`),
otherwise the `src` package import fails.

### 1. Clone and create environment
```bash
git clone <repo-url>
cd heart-disease-mlops
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data
```bash
python data/download_data.py
```

### 3. Run EDA
```bash
python -m src.eda
# figures saved to reports/figures/
```

### 4. Preprocess (optional standalone step)
```bash
python -m src.data_preprocessing
# writes data/processed/heart_disease_clean.csv
```

### 5. Train models (logs to MLflow, saves best pipeline)
```bash
python -m src.train

# View experiment tracking UI:
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000
```

### 6. Run tests
```bash
pytest tests/ -v
```

### 7. Run the API locally
```bash
uvicorn api.main:app --reload
# docs: http://localhost:8000/docs
```

Example request (payload in `sample_request.json`; works in bash,
PowerShell and Command Prompt — on Windows use `curl.exe`, since
`curl` in PowerShell is an alias for `Invoke-WebRequest`):
```bash
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
# -> {"prediction":0,"label":"No Disease","confidence":0.8188,"model_version":"logistic_regression"}
```

### 8. Build & run with Docker
```bash
docker build -t heart-disease-api:latest .
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
curl.exe http://localhost:8000/health
docker rm -f heart-disease-api   # stop before starting the monitoring stack (same port)
```

### 9. Deploy to Kubernetes
See [k8s/README.md](k8s/README.md). With Docker Desktop's built-in
Kubernetes, the LoadBalancer service is exposed at `http://localhost/`
(port 80):
```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods
curl.exe -X POST http://localhost/predict -H "Content-Type: application/json" -d @sample_request.json
```

### 10. Local monitoring stack (Prometheus + Grafana)
```bash
docker compose -f docker-compose.monitoring.yml up --build -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin) — add Prometheus
#             (http://prometheus:9090) as a data source and build a
#             dashboard from the heart-disease-api /metrics series.
```

## Dataset

**Heart Disease UCI Dataset** (Cleveland subset), source:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
14 features (age, sex, chest pain type, resting blood pressure,
cholesterol, etc.) and a binary target indicating presence of heart
disease. Downloaded automatically by `data/download_data.py`.

## Models

Two classifiers are trained and tuned via `GridSearchCV` with 5-fold
stratified cross-validation, optimizing ROC-AUC:
- **Logistic Regression** (`C` ∈ {0.01, 0.1, 1, 10} tuned)
- **Random Forest** (`n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 5, 10} tuned)

Metrics logged per run: accuracy, precision, recall, ROC-AUC, plus an
ROC curve plot artifact. The best model (by test ROC-AUC) is saved to
`models/model.pkl` and served by the API.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.885 | 0.839 | 0.929 | 0.961 |
| Random Forest | 0.869 | 0.833 | 0.893 | 0.947 |

## CI/CD Pipeline

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. Install dependencies
2. Lint (`flake8`)
3. Unit tests (`pytest`, JUnit report uploaded as artifact)
4. Download dataset + train model (fails the pipeline on error)
5. Upload trained model, MLflow runs, and figures as workflow artifacts
6. Build the Docker image and smoke-test the `/health` endpoint

## Monitoring & Logging

- All API requests are logged (method, path, status code, latency) via
  a FastAPI middleware (`api/main.py`).
- `/metrics` exposes Prometheus-format metrics (request counts,
  latencies) via `prometheus-fastapi-instrumentator`.
- `docker-compose.monitoring.yml` + `monitoring/prometheus.yml` spin up
  a local Prometheus + Grafana stack scraping the API.

