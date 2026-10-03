# Kubernetes Deployment (Minikube / Docker Desktop)

## 1. Build the image
```bash
docker build -t heart-disease-api:latest .
```

## 2. Make the image available to the cluster
```bash
# Minikube
minikube image load heart-disease-api:latest

# Docker Desktop Kubernetes: no extra step needed, it shares the local Docker daemon.
```

## 3. Apply manifests
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 4. Verify
```bash
kubectl get pods
kubectl get svc heart-disease-api-service

# Minikube: get a reachable URL for the LoadBalancer service
minikube service heart-disease-api-service --url
```

## 5. Test the endpoint
```bash
curl -X POST <SERVICE_URL>/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
```

Take screenshots of `kubectl get pods`, `kubectl get svc`, and the curl
response for the assignment report, and save them to `screenshots/`.

## Scaling
```bash
kubectl scale deployment heart-disease-api --replicas=3
```

## Cleanup
```bash
kubectl delete -f k8s/service.yaml -f k8s/deployment.yaml
```
