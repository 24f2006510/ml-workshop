from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

app = FastAPI(
    title="Movie Review Sentiment Analysis API",
    description="A minimal prediction API built for the workshop.",
    version="1.0"
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

class ReviewRequest(BaseModel):
    review: str = Field(..., example="This movie was fantastic! I loved it.")

class ReviewResponse(BaseModel):
    sentiment: str = Field(..., example="positive")
    confidence: float = Field(..., example=0.95)

@app.get("/")
async def root():
    return {"message": "Welcome to the Movie Review Sentiment Analysis API!"}

@app.get("/health")
def health():
    """Basic health check endpoint — useful for deployment platforms & load balancers."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=ReviewResponse)
def predict(request: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the server logs.")

    # Preprocess the input review
    review_text = request.review
    # Assuming the model expects a list of reviews
    input_data = [review_text]

    # Make prediction
    prediction = model.predict(input_data)
    confidence = np.max(model.predict_proba(input_data))

    sentiment = "positive" if prediction[0] == 1 else "negative"

    return ReviewResponse(sentiment=sentiment, confidence=float(confidence))