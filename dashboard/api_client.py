"""
Thin HTTP client that wraps the FastAPI prediction endpoints.
The dashboard calls these functions instead of hitting requests directly.
"""
from __future__ import annotations

import os
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT  = 10


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API not reachable. Is the server running?"}
    except requests.exceptions.HTTPError as e:
        return {"error": str(e)}


def get_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
        return r.json()
    except Exception:
        return {"status": "unreachable", "models_loaded": {}}


def predict_lap_time(recent_laps: list[dict], driver_id: int, circuit_id: int) -> dict | None:
    return _post("/predict/lap-time", {
        "recent_laps": recent_laps,
        "driver_id":   driver_id,
        "circuit_id":  circuit_id,
    })


def predict_pit_stop(lap: int, cumulative_pits: int, prev_lap_ms: float, curr_lap_ms: float) -> dict | None:
    return _post("/predict/pit-stop", {
        "lap":             lap,
        "cumulative_pits": cumulative_pits,
        "prev_lap_ms":     prev_lap_ms,
        "curr_lap_ms":     curr_lap_ms,
    })


def predict_position(grid: int, laps: int, race_time_ms: float, fastest_lap_sec: float) -> dict | None:
    return _post("/predict/position", {
        "grid":            grid,
        "laps":            laps,
        "race_time_ms":    race_time_ms,
        "fastest_lap_sec": fastest_lap_sec,
    })
