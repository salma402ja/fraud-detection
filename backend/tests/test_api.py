"""
Deployment tests – run before every production deploy.
These tests validate that the API is alive and returns valid responses.
Run: pytest tests/ -v
"""
import sys
import os

# Allow imports from backend root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_TX = {
    "amount": 150.0,
    "hour": 14,
    "merchant_category": 2,
    "distance_from_home": 5.0,
    "num_transactions_last_24h": 3,
}

FRAUD_TX = {
    "amount": 9500.0,
    "hour": 2,
    "merchant_category": 8,
    "distance_from_home": 450.0,
    "num_transactions_last_24h": 18,
}


# ── Test 1: Health check ───────────────────────────────────────────────────

def test_health_returns_200():
    """API must respond with 200 on /health."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_is_ok():
    """Health endpoint must return status 'ok'."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"


# ── Test 2: Prediction endpoint ────────────────────────────────────────────

def test_predict_returns_200_or_503():
    """Predict must return 200 (model loaded) or 503 (model missing)."""
    response = client.post("/predict", json=VALID_TX)
    assert response.status_code in [200, 503]


def test_predict_response_schema():
    """If model is loaded, response must match expected schema."""
    response = client.post("/predict", json=VALID_TX)
    if response.status_code == 200:
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "model_version" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_probability_in_range():
    """Fraud probability must always be between 0 and 1."""
    response = client.post("/predict", json=VALID_TX)
    if response.status_code == 200:
        proba = response.json()["fraud_probability"]
        assert 0.0 <= proba <= 1.0


# ── Test 3: Input validation ───────────────────────────────────────────────

def test_predict_invalid_input_returns_422():
    """API must reject malformed input with 422 Unprocessable Entity."""
    bad_payload = {"amount": "not_a_number", "hour": 99}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422():
    """API must reject requests with missing required fields."""
    incomplete = {"amount": 100.0}  # missing all other fields
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_negative_amount_rejected():
    """Amount must be strictly positive (> 0)."""
    bad = {**VALID_TX, "amount": -50.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_hour_rejected():
    """Hour must be between 0 and 23."""
    bad = {**VALID_TX, "hour": 25}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
