# AI/ML Sentiment Analysis — CI/CD Pipeline

> A production-grade, end-to-end MLOps project: train a binary sentiment
> classifier, serve it via a FastAPI REST API, containerise it with Docker,
> and ship it automatically with a GitHub Actions CI/CD pipeline.

[![CI/CD](https://github.com/Hansaka2001/ai-ml-cicd-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Hansaka2001/ai-ml-cicd-pipeline/actions/workflows/ci.yml)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Model Details](#4-model-details)
5. [Local Setup & Quickstart](#5-local-setup--quickstart)
6. [API Reference](#6-api-reference)
7. [Testing](#7-testing)
8. [Code Quality](#8-code-quality)
9. [Docker](#9-docker)
10. [CI/CD Pipeline](#10-cicd-pipeline)
11. [Deployment (Render)](#11-deployment-render)

---

## 1. Project Overview

This project demonstrates a **complete MLOps workflow** built from scratch:

| Phase | What was built |
|---|---|
| **ML Training** | TF-IDF + Logistic Regression trained on 50k IMDb reviews |
| **REST API** | FastAPI app with `/` and `/predict` endpoints |
| **Testing** | 69 unit + integration tests with pytest and mocks |
| **Code Quality** | Black (formatting) + Ruff (linting) enforced project-wide |
| **Containerisation** | Docker image with health check, runtime-only deps |
| **CI** | GitHub Actions — lint → test → docker build + smoke test |
| **CD** | Automatic deployment to Render on every push to `main` |

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| ML | scikit-learn (TfidfVectorizer + LogisticRegression) |
| Dataset | Hugging Face `stanfordnlp/imdb` (50,000 reviews) |
| Serialisation | joblib |
| Validation | Pydantic v2 |
| Testing | pytest + httpx + unittest.mock |
| Formatting | Black |
| Linting | Ruff |
| Containerisation | Docker (python:3.12-slim) |
| CI/CD | GitHub Actions |
| Registry | GitHub Container Registry (ghcr.io) |
| Deployment | Render |

---

## 3. Project Structure

```
ai-ml-cicd-pipeline/
│
├── app/                            # FastAPI application
│   ├── __init__.py
│   ├── main.py                     # App factory, routes, error handlers
│   ├── model_loader.py             # Singleton loader for joblib artifacts
│   ├── predictor.py                # Pure inference logic (decoupled from HTTP)
│   └── schemas.py                  # Pydantic request / response models
│
├── training/                       # Model training
│   ├── __init__.py
│   ├── train.py                    # End-to-end training pipeline
│   └── model/                      # Saved artifacts (committed to repo)
│       ├── tfidf_vectorizer.joblib
│       └── logistic_regression.joblib
│
├── tests/                          # Full test suite (69 tests)
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures & mock objects
│   ├── test_api.py                 # Integration tests (endpoints + validation)
│   ├── test_model_loader.py        # Unit tests for model_loader.py
│   ├── test_predictor.py           # Unit tests for predictor.py
│   └── test_schemas.py             # Unit tests for Pydantic schemas
│
├── reports/                        # CI test report output
│   └── .gitkeep
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # 4-job CI/CD pipeline
│
├── Dockerfile                      # Production container definition
├── .dockerignore                   # Exclude dev/test files from image
├── requirements.txt                # All dependencies (dev + runtime)
├── requirements-app.txt            # Runtime-only deps (used in Docker)
├── pyproject.toml                  # pytest, Black, and Ruff configuration
└── README.md
```

---

## 4. Model Details

| Property | Value |
|---|---|
| **Dataset** | `stanfordnlp/imdb` — 25k train / 25k test reviews |
| **Vectoriser** | `TfidfVectorizer` — 50,000 features, unigrams + bigrams |
| **Classifier** | `LogisticRegression` — lbfgs solver, C=5.0 |
| **Accuracy** | **90.4%** on 25k test reviews |
| **ROC-AUC** | **0.97** |
| **Serialisation** | `joblib` → `training/model/*.joblib` |

Training report:
```
              precision    recall  f1-score   support
    Negative       0.90      0.90      0.90     12500
    Positive       0.90      0.90      0.90     12500
    accuracy                           0.90     25000
```

---

## 5. Local Setup & Quickstart

### Prerequisites
- Python 3.12+
- pip

### Step 1 — Clone and install

```bash
git clone https://github.com/Hansaka2001/ai-ml-cicd-pipeline.git
cd ai-ml-cicd-pipeline
pip install -r requirements.txt
```

### Step 2 — Train the model

Downloads the IMDb dataset (~84 MB) and saves the model artifacts.
Takes approximately 3–4 minutes on the first run.

```bash
python training/train.py
```

### Step 3 — Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4 — Make predictions

```bash
# Positive review
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely incredible!"}'

# Negative review
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Terrible film. Worst movie I have ever seen."}'
```

### Step 5 — Open the interactive docs

Go to **http://localhost:8000/docs** for the Swagger UI.

---

## 6. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome message + API version |
| `POST` | `/predict` | Classify text as Positive or Negative |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### `POST /predict`

**Request body:**
```json
{ "text": "Your movie review here (1–10,000 characters)" }
```

**Response body:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.9712
}
```

- `sentiment` — `"Positive"` or `"Negative"`
- `confidence` — probability score between `0.0` and `1.0`

### Error responses

| Status | Cause |
|---|---|
| `422` | Missing, empty, or whitespace-only `text` field |
| `503` | Model artifacts missing — run `python training/train.py` first |
| `500` | Unexpected server error |

---

## 7. Testing

The project has **69 tests** across 4 test files. Unit tests use mocks and never
require real model artifacts, so they always run — even in CI.

```bash
# Run all tests
pytest tests/ -v

# Run without the live model tests
pytest tests/ -v -m "not live"
```

### Test coverage

| File | What it tests |
|---|---|
| `test_api.py` | HTTP endpoints (mock-backed + live), validation, 503 error |
| `test_model_loader.py` | Singleton caching, FileNotFoundError handling |
| `test_predictor.py` | Prediction labels, confidence values, rounding |
| `test_schemas.py` | Pydantic validation — boundaries, required fields |

---

## 8. Code Quality

```bash
# Check formatting (no files are modified)
black . --check --diff

# Run linter
ruff check .

# Auto-fix safe linting issues
ruff check . --fix
```

All rules are configured in `pyproject.toml`.

---

## 9. Docker

### Prerequisites
- Docker Desktop running
- Model trained (`python training/train.py`)

### Build & run

```bash
# Build the image
docker build -t sentiment-api .

# Run the container
docker run -p 8000:8000 --name sentiment-test sentiment-api
```

### Test the running container

```bash
# Health check
curl http://localhost:8000/

# Prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "A genuinely moving and beautifully crafted film."}'
```

### Check container health status

```bash
docker ps
# STATUS column: (healthy) = ✅  (unhealthy) = ❌
```

### Stop and clean up

```bash
docker stop sentiment-test && docker rm sentiment-test
```

The image is automatically built and pushed to **GitHub Container Registry**
(`ghcr.io`) on every push to `main`.

---

## 10. CI/CD Pipeline

The GitHub Actions pipeline at `.github/workflows/ci.yml` runs on every
**push** and **pull request** to `main`.

### Job graph

```
push / PR to main
        │
  ┌─────┴──────┐
  ▼            ▼
lint          test            ← run in parallel
(Black+Ruff)  (pytest + XML)
  └─────┬──────┘
        │ both must pass
        ▼
  docker-build                ← build image → smoke test → push to GHCR
        │  (push to main only)
        ▼
    deploy                    ← trigger Render deploy hook
```

### Job summary

| Job | Runs on | What it does |
|---|---|---|
| `lint` | Every push + PR | Black format check + Ruff lint |
| `test` | After lint passes | Full pytest suite + JUnit XML artifact |
| `docker-build` | After both pass | Build image, smoke test GET /, push to GHCR |
| `deploy` | Push to main only | POST to Render deploy hook |

### Performance features
- **Concurrency cancellation** — stale runs are cancelled when a new commit is pushed
- **Pip caching** — shared cache key across `lint` and `test` jobs
- **Docker layer caching** — via GitHub Actions cache backend
- **Test artifact** — JUnit XML uploaded for 14 days for debugging

---

## 11. Deployment (Render)

### One-time setup

1. Create a new **Web Service** on [Render](https://render.com)
2. Point it to this repository
3. Set **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. Go to **Settings → Deploy Hooks** → copy the URL
5. In GitHub: **Settings → Secrets → Actions → New secret**
   - Name: `RENDER_DEPLOY_HOOK_URL`
   - Value: the URL from step 4

After this, every push to `main` that passes all CI checks will automatically
trigger a new deployment on Render.

### Environment variables (on Render)

No extra env vars are required. The app uses only local files (the model
artifacts baked into the Docker image).

---

## License

MIT