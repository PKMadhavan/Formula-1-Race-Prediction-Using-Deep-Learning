"""
Integration tests for the FastAPI endpoints.
Uses TestClient — no live server needed.
Requires fixtures (run scripts/create_test_fixtures.py first).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "F1" in r.json()["message"]

    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert "models_loaded" in body
        assert "version" in body

    def test_health_models_loaded_keys(self):
        r = client.get("/health")
        models = r.json()["models_loaded"]
        assert "lap_time_lstm"  in models
        assert "pit_stop_fcnn"  in models
        assert "position_mlp"   in models


class TestLapTimePrediction:
    VALID_PAYLOAD = {
        "recent_laps": [
            {"lap_time_sec": 90.1, "lap": 1},
            {"lap_time_sec": 89.8, "lap": 2},
            {"lap_time_sec": 90.3, "lap": 3},
            {"lap_time_sec": 89.5, "lap": 4},
            {"lap_time_sec": 90.0, "lap": 5},
        ],
        "driver_id":  0,
        "circuit_id": 0,
    }

    def test_valid_request(self):
        r = client.post("/predict/lap-time", json=self.VALID_PAYLOAD)
        assert r.status_code in (200, 503)   # 503 if model not trained yet

    def test_response_fields_when_loaded(self):
        r = client.post("/predict/lap-time", json=self.VALID_PAYLOAD)
        if r.status_code == 200:
            body = r.json()
            assert "predicted_lap_time_sec" in body
            assert isinstance(body["predicted_lap_time_sec"], float)

    def test_too_few_laps_rejected(self):
        payload = dict(self.VALID_PAYLOAD)
        payload["recent_laps"] = payload["recent_laps"][:3]
        r = client.post("/predict/lap-time", json=payload)
        assert r.status_code == 422

    def test_negative_lap_time_rejected(self):
        payload = dict(self.VALID_PAYLOAD)
        payload["recent_laps"][0]["lap_time_sec"] = -5.0
        r = client.post("/predict/lap-time", json=payload)
        assert r.status_code == 422


class TestPitStopPrediction:
    VALID_PAYLOAD = {
        "lap": 20,
        "cumulative_pits": 1,
        "prev_lap_ms": 90500.0,
        "curr_lap_ms": 91800.0,
    }

    def test_valid_request(self):
        r = client.post("/predict/pit-stop", json=self.VALID_PAYLOAD)
        assert r.status_code in (200, 503)

    def test_response_fields_when_loaded(self):
        r = client.post("/predict/pit-stop", json=self.VALID_PAYLOAD)
        if r.status_code == 200:
            body = r.json()
            assert "will_pit" in body
            assert "probability" in body
            assert "threshold_used" in body
            assert 0.0 <= body["probability"] <= 1.0

    def test_zero_lap_rejected(self):
        payload = {**self.VALID_PAYLOAD, "lap": 0}
        r = client.post("/predict/pit-stop", json=payload)
        assert r.status_code == 422


class TestPositionPrediction:
    VALID_PAYLOAD = {
        "grid": 5,
        "laps": 57,
        "race_time_ms": 5_400_000.0,
        "fastest_lap_sec": 88.5,
    }

    def test_valid_request(self):
        r = client.post("/predict/position", json=self.VALID_PAYLOAD)
        assert r.status_code in (200, 503)

    def test_response_fields_when_loaded(self):
        r = client.post("/predict/position", json=self.VALID_PAYLOAD)
        if r.status_code == 200:
            body = r.json()
            assert "predicted_position" in body
            assert "predicted_position_rounded" in body
            assert isinstance(body["predicted_position_rounded"], int)

    def test_grid_zero_rejected(self):
        payload = {**self.VALID_PAYLOAD, "grid": 0}
        r = client.post("/predict/position", json=payload)
        assert r.status_code == 422
