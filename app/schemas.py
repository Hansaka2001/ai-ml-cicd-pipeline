"""
schemas.py
----------
Pydantic models that define the shape of API request and response payloads.

Using explicit Pydantic models (rather than raw dicts) provides:
  - Automatic JSON serialisation / deserialisation.
  - Self-documenting OpenAPI / Swagger schema generation.
  - Input validation with clear error messages.
"""

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for the POST /predict endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="The movie review text to classify.",
        examples=["This movie was absolutely amazing! A masterpiece."],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject strings that are whitespace-only."""
        if not value.strip():
            raise ValueError("'text' must not be blank or whitespace-only.")
        return value.strip()


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class PredictResponse(BaseModel):
    """Response body returned by the POST /predict endpoint."""

    sentiment: str = Field(
        ...,
        description="Predicted sentiment label: 'Positive' or 'Negative'.",
        examples=["Positive"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Model's confidence in its prediction, expressed as a probability "
            "between 0.0 (completely uncertain) and 1.0 (completely certain)."
        ),
        examples=[0.9712],
    )


# ---------------------------------------------------------------------------
# Welcome
# ---------------------------------------------------------------------------


class WelcomeResponse(BaseModel):
    """Response body for the GET / health-check endpoint."""

    message: str = Field(
        ...,
        description="A short welcome message.",
        examples=["Welcome to the Sentiment Analysis API!"],
    )
    version: str = Field(
        ...,
        description="Current API version string.",
        examples=["1.0.0"],
    )
    docs_url: str = Field(
        ...,
        description="URL to the interactive Swagger documentation.",
        examples=["/docs"],
    )
