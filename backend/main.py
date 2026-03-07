"""
Fraud Detection API – FastAPI backend
PROVIDED to students – do not modify this file.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for financial transactions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/fraud_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
model = None


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded from {MODEL_PATH}")
    else:
        print(f"[startup] WARNING: no model found at {MODEL_PATH}. Train first.")


# ── Schemas ────────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in €")
    hour: int = Field(..., ge=0, le=23, description="Hour of the transaction (0-23)")
    merchant_category: int = Field(..., ge=0, le=9, description="Merchant category code")
    distance_from_home: float = Field(..., ge=0, description="Distance from cardholder home (km)")
    num_transactions_last_24h: int = Field(..., ge=0, description="Number of transactions in last 24h")

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 150.0,
                "hour": 14,
                "merchant_category": 2,
                "distance_from_home": 5.0,
                "num_transactions_last_24h": 3,
            }
        }


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_version: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python model/train.py first.",
        )

    features = np.array([[
        tx.amount,
        tx.hour,
        tx.merchant_category,
        tx.distance_from_home,
        tx.num_transactions_last_24h,
    ]])

    proba = float(model.predict_proba(features)[0][1])
    is_fraud = proba > 0.5

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(proba, 4),
        model_version=MODEL_VERSION,
    )
