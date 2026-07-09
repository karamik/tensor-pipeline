#!/usr/bin/env python3
"""
Tensor Pipeline — Gateway Integration Tests
Run with: pytest tests/test_gateway.py -v
"""
import pytest
import httpx

GATEWAY_URL = "http://localhost:8080"

@pytest.fixture
def client():
    return httpx.Client(base_url=GATEWAY_URL, timeout=30)

def test_health_endpoint(client):
    """Test liveness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_ready_endpoint(client):
    """Test readiness probe."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_predict_endpoint(client):
    """Test single prediction."""
    payload = {
        "inputs": [[[0.1] * 224 * 224 * 3]],  # 1 sample, flattened 224x224x3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predictions" in data
    assert "inference_time_ms" in data

def test_batch_predict_endpoint(client):
    """Test batch prediction."""
    payload = {
        "inputs": [
            [[0.1] * 224 * 224 * 3],
            [[0.2] * 224 * 224 * 3],
        ],
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["batch_size"] == 2

def test_metrics_endpoint(client):
    """Test Prometheus metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "gateway_predictions_total" in response.text
