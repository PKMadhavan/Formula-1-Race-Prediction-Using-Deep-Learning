"""
Prediction endpoints for all three F1 models.
"""
from __future__ import annotations

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from src.api.model_loader import store
from src.api.schemas import (
    LapTimePredictRequest, LapTimePredictResponse,
    PitStopPredictRequest, PitStopPredictResponse,
    PositionPredictRequest, PositionPredictResponse,
)

router = APIRouter(prefix="/predict", tags=["Predictions"])


# ─── Lap Time ─────────────────────────────────────────────────────────────────

@router.post("/lap-time", response_model=LapTimePredictResponse)
def predict_lap_time(req: LapTimePredictRequest):
    if store.lap_time_model is None:
        raise HTTPException(503, "Lap time model not loaded. Run training first.")

    # Build scaled sequence (1, 5, 2)
    seq = np.array(
        [[lap.lap_time_sec, lap.lap] for lap in req.recent_laps],
        dtype=np.float32,
    )
    seq_scaled = store.lap_time_scaler_x.transform(seq).reshape(1, 5, 2)

    x  = torch.tensor(seq_scaled, dtype=torch.float32).to(store.device)
    d  = torch.tensor([req.driver_id],  dtype=torch.long).to(store.device)
    c  = torch.tensor([req.circuit_id], dtype=torch.long).to(store.device)

    with torch.no_grad():
        pred_scaled = store.lap_time_model(x, d, c).cpu().numpy().reshape(-1, 1)

    pred_sec = float(store.lap_time_scaler_y.inverse_transform(pred_scaled)[0, 0])

    return LapTimePredictResponse(
        predicted_lap_time_sec=round(pred_sec, 3),
        driver_id=req.driver_id,
        circuit_id=req.circuit_id,
    )


# ─── Pit Stop ─────────────────────────────────────────────────────────────────

@router.post("/pit-stop", response_model=PitStopPredictResponse)
def predict_pit_stop(req: PitStopPredictRequest):
    if store.pit_stop_model is None:
        raise HTTPException(503, "Pit stop model not loaded. Run training first.")

    x_raw = np.array(
        [[req.lap, req.cumulative_pits, req.prev_lap_ms, req.curr_lap_ms]],
        dtype=np.float32,
    )
    x_scaled = store.pit_stop_scaler.transform(x_raw)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(store.device)

    with torch.no_grad():
        prob = float(store.pit_stop_model(x_tensor).cpu().item())

    will_pit = prob >= store.pit_stop_threshold

    return PitStopPredictResponse(
        will_pit=will_pit,
        probability=round(prob, 4),
        threshold_used=store.pit_stop_threshold,
    )


# ─── Final Position ───────────────────────────────────────────────────────────

@router.post("/position", response_model=PositionPredictResponse)
def predict_position(req: PositionPredictRequest):
    if store.position_model is None:
        raise HTTPException(503, "Position model not loaded. Run training first.")

    x_raw = np.array(
        [[req.grid, req.laps, req.race_time_ms, req.fastest_lap_sec]],
        dtype=np.float32,
    )
    x_scaled = store.position_scaler.transform(x_raw)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(store.device)

    with torch.no_grad():
        pred = float(store.position_model(x_tensor).cpu().item())

    return PositionPredictResponse(
        predicted_position=round(pred, 2),
        predicted_position_rounded=max(1, round(pred)),
    )
