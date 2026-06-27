# =============================================================================
# Dockerfile
# Packages the FastAPI Sentiment Analysis API for production deployment.
#
# Build stages:
#   (single-stage for clarity at this phase; can be converted to multi-stage
#    once build times need further optimisation)
#
# Prerequisites — run locally BEFORE building:
#   python training/train.py
#
# The trained model files (training/model/*.joblib) must exist locally before
# running `docker build`. They are copied into the image at build time.
# The .dockerignore file ensures only the model directory (not the training
# source code) is included in the image.
#
# Build:
#   docker build -t sentiment-api .
#
# Run:
#   docker run -p 8000:8000 sentiment-api
#
# Test (in a second terminal):
#   curl http://localhost:8000/
#   curl -X POST http://localhost:8000/predict \
#        -H "Content-Type: application/json" \
#        -d '{"text": "This movie was fantastic!"}'
# =============================================================================

# ---------------------------------------------------------------------------
# Base image: official Python 3.12 slim variant.
# `slim` removes most system packages, giving a much smaller image than `full`
# while still being sufficient for scikit-learn and FastAPI.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# ---------------------------------------------------------------------------
# System-level setup.
# ---------------------------------------------------------------------------

# Prevent Python from writing .pyc files (reduces image size).
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure stdout/stderr are unbuffered so logs appear in real time.
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# ---------------------------------------------------------------------------
# Install runtime Python dependencies.
# This layer is cached as long as requirements-app.txt doesn't change,
# making subsequent rebuilds much faster.
# ---------------------------------------------------------------------------
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# ---------------------------------------------------------------------------
# Copy the application source code.
# ---------------------------------------------------------------------------
COPY app/ ./app/

# ---------------------------------------------------------------------------
# Copy the trained model artifacts.
# model_loader.py will raise FileNotFoundError (→ HTTP 503) if these are
# absent, so the container remains functional but returns 503 on /predict
# until a valid model is present.
#
# NOTE: If training/model/ does not exist yet, run:
#   python training/train.py
# before executing `docker build`.
# ---------------------------------------------------------------------------
COPY training/model/ ./training/model/

# ---------------------------------------------------------------------------
# Expose the port uvicorn listens on.
# (This is documentation metadata; the actual binding is set in CMD.)
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check — Docker will probe GET / every 30 s.
# The container is marked "healthy" when the endpoint returns HTTP 200.
# This also powers health checks on Render, Railway, and other PaaS platforms.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" \
    || exit 1

# ---------------------------------------------------------------------------
# Start the API server.
# --host 0.0.0.0  — accept connections from outside the container.
# --port 8000     — must match EXPOSE above.
# --workers 1     — single worker is fine for this stage; increase for prod.
# ---------------------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
