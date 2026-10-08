"""FastAPI server for phishing email detection."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

app = FastAPI(
    title="PhishGuard API",
    description="Phishing email detection using NLP & Transformers",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmailRequest(BaseModel):
    subject: str = ""
    body: str


class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    label: str


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": False}


@app.post("/predict", response_model=PredictionResponse)
async def predict(email: EmailRequest):
    """Classify an email as phishing or legitimate."""
    # TODO: Load model and run inference
    raise HTTPException(
        status_code=503,
        detail="Model not loaded yet. Train the model first.",
    )
