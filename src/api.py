import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Define the input data structure (Accepts a string review)
class SentimentInput(BaseModel):
    review: str = Field(
        ..., 
        description="The customer review text to analyze",
        examples=["The product quality is excellent and shipping was fast!"]
    )

ml_models = {}

# 2. Load the model and vectorizer/tokenizer at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with open("sentiment_model.pkl", "rb") as f:
            ml_models["classifier"] = pickle.load(f)
            
        with open("tfidf_vectorizer.pkl", "rb") as f:
            ml_models["vectorizer"] = pickle.load(f)
            
    except FileNotFoundError:
        print("Warning: Model artifacts not found. Please ensure files exist.")
    
    yield
    ml_models.clear()

# 3. Initialize FastAPI
app = FastAPI(
    title="Customer Review Sentiment API",
    description="API that predicts sentiment (1 for Positive, 0 for Negative) from text reviews.",
    lifespan=lifespan
)

@app.get("/")
def home():
    return {"status": "Online", "endpoint": "/predict_sentiment"}

# 4. Sentiment Prediction Endpoint
@app.post("/predict_sentiment")
def predict_sentiment(data: SentimentInput):
    if "classifier" not in ml_models:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")

    try:
        text = [data.review]
        if "vectorizer" in ml_models and ml_models["vectorizer"] is not None:
            processed_text = ml_models["vectorizer"].transform(text)
        else:
            processed_text = text
        prediction = ml_models["classifier"].predict(processed_text)[0]
        
        try:
            probabilities = ml_models["classifier"].predict_proba(processed_text)[0]
            confidence = float(probabilities[prediction])
        except AttributeError:
            confidence = None

        return {
            "review": data.review,
            "sentiment": int(prediction),  # 0 or 1
            "label": "Positive" if prediction == 1 else "Negative",
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
