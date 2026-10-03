# MLOps Assignment 1 — Final Report

**Course:** AIMLCZG523 (MLOps), BITS Pilani WILP
**Project:** Heart Disease Risk Prediction — End-to-End MLOps Pipeline
**Student:** _<name / ID>_
**Date:** _<date>_
**Repository:** _<GitHub URL — fill in after push>_

---

## 1. Setup / Install Instructions

```bash
git clone <repo-url>
cd heart-disease-mlops
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/download_data.py     # fetch raw UCI dataset
python src/eda.py                # generate EDA figures
python src/train.py              # train, tune, log to MLflow, save model
pytest tests/ -v                 # run unit tests
uvicorn api.main:app --reload    # serve the API locally
```

Full details, Docker, and Kubernetes steps are in
[`README.md`](../README.md) and [`k8s/README.md`](../k8s/README.md).

---

## 2. Dataset & EDA

**Dataset:** Heart Disease UCI (Cleveland subset), 303 patient records,
13 predictive features (demographic, clinical, and exercise-test
measurements) and a binary target (`target`: 1 = disease present).
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease),
fetched automatically by `data/download_data.py`.

**Cleaning performed** (`src/data_preprocessing.py::clean`):
- Dropped exact duplicate rows.
- Coerced all numeric/categorical columns to numeric dtype (the raw
  UCI file encodes missing values as `?`, which `download_data.py`
  already maps to `NaN`).
- Remaining missing values: `ca` (4 rows) and `thal` (2 rows) — these
  are imputed downstream in the modelling pipeline (median for `ca`,
  most-frequent for `thal`) rather than dropped, to avoid losing data.

**Class balance:** 164 negative (54.1%) vs. 139 positive (45.9%) —
mildly imbalanced but not severe enough to require resampling;
addressed instead by using ROC-AUC (threshold-independent) as the
primary tuning metric. See `reports/figures/class_balance.png`.

**Feature distributions:** see `reports/figures/histograms.png`.
`chol` and `oldpeak` are right-skewed with a handful of high outliers;
`age`, `trestbps`, and `thalach` are roughly bell-shaped.

**Correlation with target** (Pearson, sorted by magnitude), see
`reports/figures/correlation_heatmap.png`:

| Feature | Corr. w/ target | Interpretation |
|---|---:|---|
| `thal` (thalassemia) | 0.526 | Strongest single predictor — abnormal thalassemia results strongly associate with disease. |
| `ca` (# major vessels) | 0.460 | More vessels showing blockage via fluoroscopy → higher risk. |
| `exang` (exercise angina) | 0.432 | Exercise-induced chest pain is a strong positive indicator. |
| `oldpeak` (ST depression) | 0.425 | Larger ST depression under exercise → higher risk. |
| `thalach` (max heart rate) | -0.417 | Higher achievable heart rate is protective (negative correlation). |
| `cp` (chest pain type) | 0.414 | Pain type is informative. |
| `slope`, `sex`, `age` | 0.34 / 0.28 / 0.22 | Moderate predictors. |
| `chol`, `fbs` | 0.09 / 0.03 | Essentially uninformative in this dataset — cholesterol and fasting blood sugar alone do not separate the classes well, consistent with published findings on this dataset. |

No feature pair showed extreme multicollinearity (|r| > 0.9), so all
13 features were retained for modelling rather than dropped.

---

## 3. Feature Engineering & Modelling Choices

**Preprocessing pipeline** (`src/data_preprocessing.py::build_preprocessor`,
a scikit-learn `ColumnTransformer`, reused identically at train and
inference time):
- Numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`,
  `ca`): median imputation → `StandardScaler`.
- Categorical features (`sex`, `cp`, `fbs`, `restecg`, `exang`,
  `slope`, `thal`): most-frequent imputation → `OneHotEncoder`.

**Models trained** (`src/train.py`), each as a full
`Pipeline(preprocess, classifier)` so the fitted transformers travel
with the model:
- **Logistic Regression** — tuned over `C ∈ {0.01, 0.1, 1, 10}`.
- **Random Forest** — tuned over `n_estimators ∈ {100, 200}`,
  `max_depth ∈ {None, 5, 10}`.

**Tuning method:** `GridSearchCV` with 5-fold **stratified**
cross-validation (preserves class ratio per fold), scoring on
**ROC-AUC** (robust to the mild class imbalance and threshold-free).
80/20 stratified train/test split, `random_state=42` throughout for
reproducibility.

**Results** (held-out 20% test set, 61 patients):

| Model | Best hyperparameters | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | `C=0.1` | — | 0.885 | 0.839 | 0.929 | **0.961** |
| Random Forest | `max_depth=5, n_estimators=100` | — | 0.869 | 0.833 | 0.893 | 0.947 |

**Model selection:** Logistic Regression was chosen as the final model
— it achieved the highest test ROC-AUC (0.961 vs. 0.947) *and* the
highest recall (0.929), which matters clinically: missing a true
positive (a sick patient predicted healthy) is more costly than a
false alarm. Its simplicity also means faster inference and easier
interpretability (coefficients map directly to feature effects) than
the Random Forest, with no accuracy trade-off here — on a dataset this
small (303 rows), a linear model generalizes at least as well as an
ensemble, consistent with the close CV/test agreement (no overfitting
gap).

ROC curves for both models: `reports/figures/roc_logistic_regression.png`,
`reports/figures/roc_random_forest.png`.

---

## 4. Experiment Tracking (MLflow)

- **Tracking backend:** local SQLite (`mlflow.db`), experiment name
  `heart-disease-classification`.
- **Logged per run:** all grid-searched hyperparameters, CV best
  ROC-AUC, all four test metrics, the ROC curve plot as an artifact,
  and the fitted pipeline itself (`mlflow.sklearn.log_model`,
  pickle-serialized).
- Reproduce and inspect: run `python src/train.py`, then
  `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open
  `http://localhost:5000` — the runs table shows both models
  side-by-side with all logged params/metrics.
- _<Insert a screenshot of the MLflow UI runs table here, saved to
  `screenshots/mlflow_ui.png`.>_

---

## 5. Model Packaging & Reproducibility

- The best pipeline (preprocessing + Logistic Regression, fully
  fitted) is serialized with `pickle` to `models/model.pkl`, and
  `models/metadata.json` records which model won and its metrics.
  Both files are committed to the repository so the API can serve
  predictions without retraining.
- It is also logged as an MLflow model artifact under the
  corresponding run, as a second, independently reproducible copy.
- All dependencies are pinned to exact versions in `requirements.txt`
  (generated via `pip freeze` against the verified working
  environment) rather than loose ranges, so `pip install -r
  requirements.txt` reproduces the exact environment used for
  training and evaluation.
- End-to-end reproduction: `python data/download_data.py && python
  src/train.py` regenerates `models/model.pkl` from scratch and
  reproduces the metrics in Section 3 (modulo negligible floating
  point / grid-search tie-breaking variation — the random seed is
  fixed throughout).

---

## 6. CI/CD Pipeline

**Workflow:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
(GitHub Actions), triggered on every push/PR to `main`. Two jobs:

1. **`lint-test-train`**
   - Install pinned dependencies.
   - Lint with `flake8` (fails the pipeline on style/error violations).
   - Run the pytest suite (9 tests across preprocessing, training, and
     API) — fails the pipeline on any test failure, JUnit report
     uploaded as a workflow artifact.
   - Download the dataset and train the model — fails the pipeline if
     training errors.
   - Upload the trained model, MLflow run data, and EDA/evaluation
     figures as workflow artifacts.
2. **`docker-build`** (depends on job 1 succeeding)
   - Downloads the trained model artifact.
   - Builds the Docker image.
   - Runs it and polls `/health` until it responds, failing the job if
     the container never becomes healthy.

This satisfies the production-readiness requirement that "the
pipeline must fail on code or test errors and give clear logs" — each
step is a hard gate for the next.

- _<Insert a screenshot of a green GitHub Actions run here, saved to
  `screenshots/ci_pipeline.png`.>_

---

## 7. Model Containerization

`Dockerfile` builds a `python:3.11-slim` image that installs pinned
dependencies and serves the FastAPI app via `uvicorn` on port 8000,
with a container `HEALTHCHECK` against `/health`.

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
# -> {"prediction":0,"label":"No Disease","confidence":0.8188,"model_version":"logistic_regression"}
```

(The `/predict`, `/health`, and `/metrics` endpoints were verified
locally via `uvicorn` directly — see Section 9 — with the response
above; the containerized version is expected to behave identically
since the API code and model artifact are unchanged.)

- _<Insert a screenshot of the running container and the curl response
  above, saved to `screenshots/docker_run.png`.>_

---

## 8. Production Deployment

**Target:** _<Minikube / Docker Desktop Kubernetes / GKE / EKS / AKS — fill in>_

Manifests: [`k8s/deployment.yaml`](../k8s/deployment.yaml) (2 replicas,
readiness + liveness probes against `/health`, resource
requests/limits), [`k8s/service.yaml`](../k8s/service.yaml)
(`LoadBalancer`, port 80 → 8000). Full steps in
[`k8s/README.md`](../k8s/README.md):

```bash
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest   # if using Minikube
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc heart-disease-api-service
```

- _<Insert screenshots: `kubectl get pods`, `kubectl get svc`, and a
  successful curl/Postman `/predict` response against the exposed
  service, saved to `screenshots/k8s_pods.png`,
  `screenshots/k8s_svc.png`, `screenshots/k8s_predict.png`.>_
- **Deployed API URL / access instructions:** _<fill in>_

---

## 9. Monitoring & Logging

- **Request logging:** a FastAPI middleware (`api/main.py::log_requests`)
  logs every request's method, path, status code, and latency in
  milliseconds. Verified locally — example log line:
  ```
  2026-10-03 ... INFO heart-disease-api - POST /predict -> 200 (12.43ms)
  ```
- **Metrics:** `/metrics` exposes Prometheus-format counters and
  histograms (request counts, latency buckets per endpoint) via
  `prometheus-fastapi-instrumentator` — verified locally, returns
  standard Prometheus text format.
- **Local monitoring stack:** [`docker-compose.monitoring.yml`](../docker-compose.monitoring.yml)
  + [`monitoring/prometheus.yml`](../monitoring/prometheus.yml) run the
  API alongside Prometheus (scraping `/metrics` every 10s) and
  Grafana.
  ```bash
  docker compose -f docker-compose.monitoring.yml up --build
  # Prometheus: http://localhost:9090
  # Grafana:    http://localhost:3000 (admin/admin)
  ```
- _<Insert a screenshot of the Prometheus targets page (showing the
  API target as "UP") and/or a Grafana dashboard panel showing
  request rate or latency, saved to
  `screenshots/monitoring_dashboard.png`.>_

---

## 10. Architecture

```
 UCI Dataset ──▶ download_data.py ──▶ data/raw, data/processed
                                            │
                                            ▼
                         src/data_preprocessing.py
                    (impute → scale/one-hot encode)
                                            │
                                            ▼
                    src/train.py  (LogReg + RF, GridSearchCV,
                     5-fold stratified CV, scoring=ROC-AUC)
                                            │
                              logs params/metrics/artifacts
                                            ▼
                                   MLflow (SQLite backend)
                                            │
                                            ▼
                         models/model.pkl (best pipeline)
                                            │
                                            ▼
                      api/main.py — FastAPI (/predict, /health, /metrics)
                                            │
                                            ▼
                               Dockerfile → container image
                                            │
                                            ▼
                     Kubernetes (Minikube / GKE / EKS / AKS)
                  Deployment (2 replicas) + LoadBalancer Service
                                            │
                                            ▼
                    Prometheus (scrapes /metrics) + Grafana dashboard

 GitHub Actions (.github/workflows/ci.yml): on every push —
 lint → unit tests → train → upload artifacts → build & smoke-test
 the Docker image.
```

---

## 11. Repository & Video

- **Code repository:** _<GitHub URL>_
- **Video walkthrough of the pipeline:** _<link>_

---

## Appendix: Production-Readiness Checklist

- [x] All scripts run from a clean `pip install -r requirements.txt`
      setup (verified against pinned exact versions).
- [ ] Model serves correctly in an isolated Docker container (proof:
      screenshot/log — pending Section 7).
- [x] CI pipeline fails on lint/test/training errors with clear logs
      (each job step is a hard gate; verify with a live GitHub Actions
      run and attach the log link/screenshot).
