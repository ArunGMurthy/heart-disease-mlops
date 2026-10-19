# Heart Disease MLOps Pipeline — Runbook

---

## Prerequisites

- Terminal open at project root with `.venv` activated
- Docker Desktop running
- Kubernetes enabled in Docker Desktop

---

## Step 1 — Repo Structure

```bash
ls
```

Key folders:
- `src/` — data preprocessing, EDA, training
- `api/` — FastAPI serving app
- `tests/` — unit tests
- `k8s/` — Kubernetes manifests
- `monitoring/` — Prometheus config
- `.github/workflows/ci.yml` — CI/CD pipeline

---

## Step 2 — Download Data and Train

```bash
python data/download_data.py
python -m src.train
```

Open MLflow UI:
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000`:
- Open `heart-disease-classification` experiment
- Click each run — show params, metrics (accuracy, F1, ROC-AUC), ROC curve artifact
- Logistic Regression wins with ROC-AUC 0.961

---

## Step 3 — Unit Tests

```bash
pytest tests/ -v
```

9 tests passing across:
- `test_data_preprocessing.py` — cleaning, imputation
- `test_train.py` — pipeline trains, metrics shape
- `test_api.py` — health, predict, invalid input rejection

---

## Step 4 — API Locally

```bash
uvicorn api.main:app --reload
```

In a second terminal:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Open `http://localhost:8000/docs` in browser.

Stop the server (Ctrl+C) before next step.

---

## Step 5 — Docker

```bash
docker build -t heart-disease-api:latest .
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
docker rm -f heart-disease-api
```

---

## Step 6 — Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods
kubectl get svc
curl -X POST http://localhost/predict -H "Content-Type: application/json" -d @sample_request.json
```

---

## Step 7 — Monitoring

```bash
docker compose -f docker-compose.monitoring.yml up --build -d
```

Open in browser:
- `http://localhost:9090` → Status → Targets — verify `heart-disease-api` is **UP**
- `http://localhost:3000` (admin/admin) — open Grafana dashboard

Generate traffic:
```bash
for i in {1..10}; do curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json; done
```

Shut down:
```bash
docker compose -f docker-compose.monitoring.yml down
```

---

## Step 8 — GitHub Actions

Open GitHub repo → **Actions** tab → latest workflow run.

Two jobs:
- `lint-test-train` — flake8, pytest, train, upload artifacts
- `docker-build` — build image, smoke-test `/health`
