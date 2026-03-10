"""
Pydantic request / response schemas for all three prediction endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ─── Lap Time ────────────────────────────────────────────────────────────────

class LapEntry(BaseModel):
    lap_time_sec: float = Field(..., gt=0, description="Lap time in seconds")
    lap: int            = Field(..., gt=0, description="Lap number")

class LapTimePredictRequest(BaseModel):
    recent_laps: list[LapEntry] = Field(
        ..., min_length=5, max_length=5,
        description="Exactly 5 most recent laps in chronological order"
    )
    driver_id: int   = Field(..., description="Encoded driver index")
    circuit_id: int  = Field(..., description="Encoded circuit index")

class LapTimePredictResponse(BaseModel):
    predicted_lap_time_sec: float
    driver_id: int
    circuit_id: int


# ─── Pit Stop ────────────────────────────────────────────────────────────────

class PitStopPredictRequest(BaseModel):
    lap: int                = Field(..., gt=0)
    cumulative_pits: int    = Field(..., ge=0)
    prev_lap_ms: float      = Field(..., gt=0, description="Previous lap time in milliseconds")
    curr_lap_ms: float      = Field(..., gt=0, description="Current lap time in milliseconds")

class PitStopPredictResponse(BaseModel):
    will_pit: bool
    probability: float
    threshold_used: float


# ─── Final Position ──────────────────────────────────────────────────────────

class PositionPredictRequest(BaseModel):
    grid: int               = Field(..., ge=1, description="Starting grid position")
    laps: int               = Field(..., gt=0, description="Laps completed")
    race_time_ms: float     = Field(..., gt=0, description="Total race time in milliseconds")
    fastest_lap_sec: float  = Field(..., gt=0, description="Fastest lap time in seconds")

class PositionPredictResponse(BaseModel):
    predicted_position: float
    predicted_position_rounded: int


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    version: str
