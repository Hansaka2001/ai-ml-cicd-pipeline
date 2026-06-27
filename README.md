# Sentiment Analysis REST API

A clean, production-style binary sentiment classifier built with **FastAPI** and **scikit-learn**, trained on the Hugging Face **IMDb** dataset.

---

## Project Structure

```
ai-ml-cicd-pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application & route definitions
│   ├── model_loader.py   # Singleton loader for trained artifacts
│   ├── predictor.py      # Pure inference logic
│   └── schemas.py        # Pydantic request / response models
├── training/
│   ├── __init__.py
│   ├── model/            # Auto-created after training
│   │   ├── tfidf_vectorizer.joblib
│   │   └── logistic_regression.joblib
│   └── train.py          # End-to-end training pipeline
├── tests/
│   └── test_api.py       # Integration tests (pytest + httpx)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Downloads the IMDb dataset (~84 MB) and trains the classifier.  
Artifacts are saved to `training/model/`.

```bash
python training/train.py
```

Expected output (after ~1–2 minutes):

```
Accuracy : 0.8942
ROC-AUC  : 0.9582
              precision    recall  f1-score   support
    Negative       0.90      0.89      0.89     12500
    Positive       0.89      0.90      0.89     12500
```

### 3. Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely incredible!"}'
```

**Response:**

```json
{
  "sentiment": "Positive",
  "confidence": 0.9712
}
```

---

## API Reference

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Welcome message + links to docs      |
| POST   | `/predict` | Classify text as Positive / Negative |
| GET    | `/docs`    | Interactive Swagger UI               |
| GET    | `/redoc`   | ReDoc documentation                  |

### POST `/predict`

**Request body:**

```json
{ "text": "string (1–10,000 characters)" }
```

**Response body:**

```json
{
  "sentiment": "Positive | Negative",
  "confidence": 0.0
}
```

### Error responses

| Status | Meaning                                          |
|--------|--------------------------------------------------|
| 422    | Validation error — empty or missing `text` field |
| 503    | Model artifacts missing — run training first     |
| 500    | Unexpected server error                          |

---

## Running Tests

```bash
pytest tests/ -v
```

> Tests are **automatically skipped** when model artifacts are missing.

---

## Model Details

| Component            | Choice                        |
|----------------------|-------------------------------|
| Dataset              | Hugging Face `imdb` (50k)     |
| Vectoriser           | `TfidfVectorizer` (50k vocab) |
| Classifier           | `LogisticRegression` (lbfgs)  |
| Feature engineering  | Unigrams + bigrams            |
| Serialisation        | `joblib`                      |