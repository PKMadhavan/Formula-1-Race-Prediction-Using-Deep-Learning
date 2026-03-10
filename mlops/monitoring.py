"""
Model monitoring utilities.
Logs prediction latency and basic drift indicators.
"""
from __future__ import annotations
import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# In-memory rolling window (last 500 predictions per model)
_history: dict[str, deque] = {
    "lap_time": deque(maxlen=500),
    "pit_stop": deque(maxlen=500),
    "position": deque(maxlen=500),
}


def log_prediction(model_name: str, inputs: dict, output: Any, latency_ms: float) -> None:
    _history[model_name].append({
        "inputs": inputs, "output": output, "latency_ms": latency_ms,
        "ts": time.time(),
    })
    if len(_history[model_name]) % 100 == 0:
        avg_latency = sum(r["latency_ms"] for r in _history[model_name]) / len(_history[model_name])
        logger.info("[monitor] %s — avg latency %.1f ms over last %d preds",
                    model_name, avg_latency, len(_history[model_name]))


def get_stats(model_name: str) -> dict:
    records = list(_history.get(model_name, []))
    if not records:
        return {"count": 0}
    latencies = [r["latency_ms"] for r in records]
    return {
        "count": len(records),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
    }
