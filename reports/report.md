# MLOps Assignment 1 — Final Report

**Course:** AIMLCZG523 (MLOps) | **Student:** _<name/ID>_ | **Date:** _<date>_

> Fill in each section below as you complete the corresponding pipeline
> stage. Export to PDF/DOCX for final submission (target ~10 pages).

## 1. Setup / Install Instructions

See [README.md](../README.md#setup--install-instructions) for full
steps. Summarize environment, Python version, and any platform-specific
notes here.

## 2. Dataset & EDA

- Dataset: Heart Disease UCI (Cleveland subset), `data/download_data.py`.
- Cleaning steps performed (missing-value handling, dtype coercion):
  `src/data_preprocessing.py::clean`.
- Key EDA findings (insert figures from `reports/figures/`):
  - Class balance: `reports/figures/class_balance.png`
  - Feature distributions: `reports/figures/histograms.png`
  - Correlations: `reports/figures/correlation_heatmap.png`
  - _<Write 2-3 paragraphs of insight here — which features correlate
    most with target, any skew/outliers observed, class imbalance
    ratio, etc.>_

## 3. Feature Engineering & Modelling Choices

- Numeric features scaled with `StandardScaler` after median imputation.
- Categorical features one-hot encoded after most-frequent imputation.
- Models trained: Logistic Regression, Random Forest.
- Hyperparameter tuning: `GridSearchCV`, 5-fold stratified CV, scoring
  = ROC-AUC.
- _<Insert the best hyperparameters and cross-validated scores found by
  `src/train.py`, and justify the final model choice.>_

| Model               | Accuracy | Precision | Recall | ROC-AUC |
|---------------------|---------:|----------:|-------:|--------:|
| Logistic Regression | _TBD_    | _TBD_     | _TBD_  | _TBD_   |
| Random Forest       | _TBD_    | _TBD_     | _TBD_  | _TBD_   |

ROC curves: `reports/figures/roc_logistic_regression.png`,
`reports/figures/roc_random_forest.png`.

## 4. Experiment Tracking (MLflow)

- Tracking URI: local `mlruns/` (file-based).
- Experiment: `heart-disease-classification`.
- _<Insert a screenshot of the MLflow UI (`mlflow ui`) showing the runs
  table with logged params/metrics, saved to `screenshots/mlflow_ui.png`.>_

## 5. Model Packaging & Reproducibility

- Best pipeline (preprocessing + model) serialized to `models/model.pkl`
  via `pickle`; also logged as an MLflow model artifact
  (`mlflow.sklearn.log_model`).
- Full dependency pinning in `requirements.txt`.
- Reproduction: `python data/download_data.py && python src/train.py`.

## 6. CI/CD Pipeline

- Workflow: `.github/workflows/ci.yml` (GitHub Actions).
- Stages: lint (flake8) → unit tests (pytest) → train model → upload
  artifacts → build & smoke-test Docker image.
- _<Insert screenshot of a green GitHub Actions run, saved to
  `screenshots/ci_pipeline.png`.>_

## 7. Model Containerization

- `Dockerfile` builds a slim Python 3.11 image serving the FastAPI app
  via `uvicorn` on port 8000.
- Local verification:
  ```bash
  docker build -t heart-disease-api:latest .
  docker run -p 8000:8000 heart-disease-api:latest
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
  ```
- _<Insert screenshot of the container running and a successful
  `/predict` response, saved to `screenshots/docker_run.png`.>_

## 8. Production Deployment

- Target: _<Minikube / Docker Desktop Kubernetes / GKE / EKS / AKS>_
- Manifests: `k8s/deployment.yaml` (2 replicas, readiness/liveness
  probes), `k8s/service.yaml` (LoadBalancer).
- Steps: see [k8s/README.md](../k8s/README.md).
- _<Insert screenshots: `kubectl get pods`, `kubectl get svc`, and a
  successful curl/Postman response against the exposed service, saved
  to `screenshots/k8s_pods.png`, `screenshots/k8s_svc.png`,
  `screenshots/k8s_predict.png`.>_
- Deployed API URL / access instructions: _<fill in>_

## 9. Monitoring & Logging

- Request logging via FastAPI middleware (method, path, status,
  latency) — visible in container logs (`docker logs`).
- Prometheus metrics exposed at `/metrics`
  (`prometheus-fastapi-instrumentator`).
- Local stack: `docker-compose.monitoring.yml` (Prometheus + Grafana).
- _<Insert screenshot of Prometheus targets page and/or a Grafana
  dashboard showing request rate/latency, saved to
  `screenshots/monitoring_dashboard.png`.>_

## 10. Architecture Diagram

See [README.md](../README.md#architecture) for the full pipeline
diagram (data → preprocessing → training/MLflow → packaging → API →
Docker → Kubernetes → monitoring, automated by GitHub Actions CI/CD).

## 11. Repository & Video

- Code repository: _<GitHub URL>_
- Short video walkthrough of the pipeline: _<link>_

## Appendix: Production-Readiness Checklist

- [ ] All scripts run from a clean `pip install -r requirements.txt` setup.
- [ ] Model serves correctly in an isolated Docker container (proof: screenshot/log).
- [ ] CI pipeline fails loudly on lint/test/training errors with clear logs.
