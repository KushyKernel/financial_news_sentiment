
# deployment/api/main.py - FastAPI Production Service
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Initialize FastAPI app
app = FastAPI(title="Financial Sentiment API", version="1.0.0")

# Load production model
class ModelLoader:
    def __init__(self):
        self.model = None
        self.feature_selector = None
        self.metadata = None
        self.load_model()

    def load_model(self):
        try:
            model_path = "../models/financial_sentiment_v1.0_20250902"
            self.model = joblib.load(f"{model_path}_model.pkl")
            self.feature_selector = joblib.load(f"{model_path}_features.pkl")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {e}")

# Request/Response models
class NewsRequest(BaseModel):
    headline: str
    content: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    timestamp: str

# Initialize model loader
model_loader = ModelLoader()

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: NewsRequest):
    """Predict sentiment for financial news"""
    try:
        # Process text (would use your actual preprocessing pipeline)
        # This is a placeholder - you'd implement your actual feature extraction

        # Make prediction
        prediction = model_loader.model.predict([[0.5] * 56])[0]  # Placeholder
        probability = model_loader.model.predict_proba([[0.5] * 56])[0]

        return PredictionResponse(
            sentiment="positive" if prediction == 1 else "negative",
            confidence=float(max(probability)),
            probability_positive=float(probability[1]),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_loader.model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
