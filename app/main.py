"""
main.py
-------
FastAPI application entrypoint.

Endpoints:
  GET  /         — Welcome message and links to API docs.
  POST /predict  — Accept a JSON body and return sentiment + confidence.

Error handling:
  - 422 Unprocessable Entity: automatic Pydantic validation errors.
  - 503 Service Unavailable: model artifacts not found (training not run).
  - 500 Internal Server Error: unexpected inference failures.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.predictor import predict
from app.schemas import PredictRequest, PredictResponse, WelcomeResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "A production-style REST API that classifies movie review text as "
        "**Positive** or **Negative** using a Logistic Regression model "
        "trained on the IMDb dataset."
    ),
    version="1.0.0",
    contact={
        "name": "AI/ML CI-CD Pipeline",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Return a 503 when model artifacts are missing."""
    log.error("Model artifacts not found: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Model is not available. Please run `python training/train.py` "
                "to train and save the model before starting the API."
            )
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that prevents raw tracebacks leaking to clients."""
    log.exception("Unexpected error during request to %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/",
    response_model=WelcomeResponse,
    summary="Welcome",
    tags=["Health"],
)
async def root() -> WelcomeResponse:
    """Return a welcome message and pointers to the API documentation.

    This endpoint can also serve as a lightweight health-check probe.
    """
    return WelcomeResponse(
        message="Welcome to the Sentiment Analysis API!",
        version="1.0.0",
        docs_url="/docs",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Sentiment",
    tags=["Prediction"],
    responses={
        200: {"description": "Sentiment prediction returned successfully."},
        422: {"description": "Validation error — invalid request body."},
        503: {"description": "Model artifacts not found; training required."},
        500: {"description": "Unexpected server error."},
    },
)
async def predict_sentiment(request: PredictRequest) -> PredictResponse:
    """Classify the provided text as **Positive** or **Negative**.

    **Request body**
    ```json
    { "text": "This movie was absolutely amazing!" }
    ```

    **Response body**
    ```json
    { "sentiment": "Positive", "confidence": 0.9712 }
    ```

    - `sentiment` — `"Positive"` if the review is favourable, `"Negative"` otherwise.
    - `confidence` — Probability score between 0.0 and 1.0 for the predicted class.
    """
    try:
        result: PredictResponse = predict(request.text)
    except FileNotFoundError:
        # Re-raise so the registered handler can produce the 503 response.
        raise
    except Exception as exc:
        log.exception("Prediction failed for input: %.80s", request.text)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from exc

    return result
