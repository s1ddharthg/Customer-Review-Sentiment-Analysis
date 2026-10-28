from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# 1. Validate incoming requests
class SentimentInput(BaseModel):
    review: str = Field(
        min_length=1,
        description="The customer review text to analyze",
        examples=[
            "The product quality is excellent and shipping was fast!"
        ],
    )

    @field_validator("review")
    @classmethod
    def validate_review(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Review cannot be empty or whitespace.")

        return value


# 2. Load model artifacts
def load_models():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return classifier, vectorizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.classifier, app.state.vectorizer = load_models()
        yield
    finally:
        app.state.classifier = None
        app.state.vectorizer = None


# 3. Initialize the API
app = FastAPI(
    title="Customer Review Sentiment API",
    description="Predicts whether a customer review is positive or negative.",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "status": "Online",
        "endpoint": "/predict_sentiment",
    }


# 4. Predict sentiment
@app.post("/predict_sentiment")
def predict_sentiment(data: SentimentInput):
    classifier = app.state.classifier
    vectorizer = app.state.vectorizer

    if classifier is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model assets are not available.",
        )

    try:
        # Convert text into the numerical features expected by the model
        processed_text = vectorizer.transform([data.review])

        prediction = classifier.predict(processed_text)[0]

        # Calculate confidence if supported by the model
        confidence = None

        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(processed_text)[0]
            classes = list(classifier.classes_)
            predicted_index = classes.index(prediction)
            confidence = float(probabilities[predicted_index])

        sentiment = int(prediction)

        if sentiment not in (0, 1):
            raise ValueError("Model returned an unexpected class.")

        return {
            "review": data.review,
            "sentiment": sentiment,
            "label": "Positive" if sentiment == 1 else "Negative",
            "confidence": confidence,
        }

    except Exception as exc:
        # Log the full error for debugging; don't expose internals to clients
        app.logger.exception("Sentiment prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while predicting sentiment.",
        ) from exc
