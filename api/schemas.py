"""Validation and data schemas for PhishGuard API endpoints."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class EmailPredictionRequest(BaseModel):
    subject: str = Field(..., description="Subject line of the email", example="Urgent: Your account has been suspended")
    body: str = Field(..., description="Full text body of the email", example="Please click here to verify your banking credentials.")
    sender: Optional[str] = Field(None, description="Sender email address", example="support@secure-update-notice.com")


class BatchPredictionRequest(BaseModel):
    emails: List[EmailPredictionRequest] = Field(..., description="Batch of email objects to analyze")


class FeatureContribution(BaseModel):
    feature_name: str
    contribution_score: float


class EmailPredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    threat_level: str
    probabilities: Dict[str, float]
    top_signals: List[FeatureContribution] = []


class BatchPredictionResponse(BaseModel):
    total_processed: int
    phishing_detected_count: int
    results: List[EmailPredictionResponse]
