#!/usr/bin/env python3
"""
Tensor Pipeline — Load Benchmark (Locust)
Run: locust -f tests/benchmark.py --host=http://localhost:8080
"""
from locust import HttpUser, task, between
import random

class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        self.model_name = "resnet50"
    
    @task(3)
    def predict_single(self):
        """Single prediction."""
        payload = {
            "inputs": [[[random.random() for _ in range(224*224*3)]]],
        }
        self.client.post("/predict", json=payload)
    
    @task(1)
    def predict_batch(self):
        """Batch prediction (4 samples)."""
        payload = {
            "inputs": [
                [[random.random() for _ in range(224*224*3)]]
                for _ in range(4)
            ],
        }
        self.client.post("/predict/batch", json=payload)
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
