# Implicit Feedback Recommender System

![Architecture](./docs/architecture.png)

## Table of Contents

1. [Project Overview](#project-overview)  
2. [How It Works](#how-it-works)  
3. [Quick Start](#quick-start)  
4. [Local Development](#local-development)  
5. [Docker](#docker)  
6. [Google Cloud Deployment](#google-cloud-deployment)  
7. [CI/CD](#cicd)  
8. [Monitoring & Logging](#monitoring--logging)  
9. [Custom Domain & SSL](#custom-domain--ssl)  
10. [Post-Deployment](#post-deployment)  

---

## Project Overview

This is a microservice for generating item recommendations based on implicit feedback (e.g. views, likes) using collaborative filtering (ALS and LightFM). It:


---

## How It Works

1. **Data & Model**  
   - We preprocess the MovieLens 100k dataset into a user×item sparse matrix.  
   - We train Alternating Least Squares (implicit) and LightFM models.  
   - We serialize to `model_bundle.pkl`, containing the model, `user2idx`, `item2idx`, and interaction matrix.

2. **API**  
   - `GET /` → Health check: returns `{"status":"ok"}`.  
   - `GET /recommend/{user_id}?N={N}` → Returns top-N recommendations for `user_id`.

3. **Docker**  
   - Base image: `python:3.11-slim`.  
   - Installs build deps (`build-essential`, `libopenblas-dev`, `libomp-dev`).  
   - Copies code & `model_bundle.pkl`.  
   - Listens on port **8080**.

4. **Cloud Run**  
   - Image stored in Google Container Registry (GCR).  
   - Managed by Cloud Run: auto-scaling, pay-per-use, free SSL, custom domains.

---

## Quick Start

```bash
git clone https://github.com/nikitafg/implicit-feedback-recsys.git
cd implicit-feedback-recsys
```

Local Development:
1)Create virtual environment and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2)Train & evaluate:
```bash
python src/train_model.py      # trains and saves model_bundle.pkl
python src/evaluate.py         # prints Precision@10
```

Docker
Build and run locally:
```bash
docker build -t implicit-feedback-local:8080 .
docker run -d --rm -p 8080:8080 implicit-feedback-local:8080

# Health check:
curl http://localhost:8080/ 
# → {"status":"ok"}

# Recommendations:
curl "http://localhost:8080/recommend/1?N=5"
```


Google Cloud Deployment
1. Configure GCP
```bash
gcloud auth login
gcloud config set project micro-rigging-409523
gcloud config set run/region europe-west3
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
```

2. Build & Push to GCR
```bash
gcloud builds submit \
  --tag gcr.io/micro-rigging-409523/implicit-feedback-recsys:latest \
  .
```

3. Deploy to Cloud Run
```bash
gcloud run deploy implicit-feedback-recsys \
  --image gcr.io/micro-rigging-409523/implicit-feedback-recsys:latest \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --concurrency 10
```

4. Verify
```bash
URL=$(gcloud run services describe implicit-feedback-recsys \
  --platform=managed --region=europe-west3 \
  --format="value(status.url)")

curl "$URL/"
curl "$URL/recommend/1?N=5"
```

CI/CD
GitHub Actions (.github/workflows/docker-publish.yml):
Lints & tests Python code.
Builds Docker image.
Pushes image to Docker Hub and/or GCR.
Optionally, add a step to auto-deploy to Cloud Run on main branch.

Monitoring & Logging
Cloud Monitoring for service metrics (CPU, memory, requests).
Cloud Logging (Stackdriver) captures stdout/stderr logs.
Can integrate Prometheus/Grafana via Google Managed Service for Prometheus.


Custom Domain & SSL
In GCP Console → Cloud Run → Domain mappings → Add mapping.
Point your DNS (CNAME or A/AAAA records) to the provided endpoints.
GCP automatically provisions and renews a free SSL certificate.
Access your service at https://your-domain.com.



Author & Contacts
Nikita Marshchonok
GitHub: https://github.com/NikitaMarshchonok
Email: n.marshchonok@gmail.com
