"""
Data ingestion layer.

Supports two sources:
  1. Kaggle CSV files  — historical data (1950-2024)
  2. FastF1 API       — live / recent session data

Place Kaggle CSVs in data/raw/ before running training.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_kaggle_data(raw_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Load the six Kaggle F1 CSVs into a dict of DataFrames."""
    raw = Path(raw_dir)
    files = {
        "lap_times": "lap_times.csv",
        "races":     "races.csv",
        "drivers":   "drivers.csv",
        "circuits":  "circuits.csv",
        "pit_stops": "pit_stops.csv",
        "results":   "results.csv",
    }
    data: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = raw / filename
        if not path.exists():
            logger.warning("Missing file: %s — some models may not train.", path)
            continue
        data[key] = pd.read_csv(path)
        logger.info("Loaded %s  (%d rows)", filename, len(data[key]))
    return data


def setup_fastf1_cache(cache_dir: str = "data/fastf1_cache") -> None:
    try:
        import fastf1
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
    except ImportError:
        logger.warning("fastf1 not installed — live data unavailable.")


def load_live_session(
    year: int,
    round_number: int,
    session_type: str = "R",
    cache_dir: str = "data/fastf1_cache",
) -> Optional[object]:
    """Load a FastF1 session (R=Race, Q=Qualifying, FP1/FP2/FP3=Practice)."""
    try:
        import fastf1
        setup_fastf1_cache(cache_dir)
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        logger.info("Loaded FastF1 session: %s round %s %s", year, round_number, session_type)
        return session
    except Exception as exc:
        logger.error("FastF1 load failed: %s", exc)
        return None


def fastf1_laps_to_dataframe(session) -> pd.DataFrame:
    """Extract lap-level features from a FastF1 session into a flat DataFrame."""
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()
    laps["lap_time_sec"] = laps["LapTime"].dt.total_seconds()
    laps = laps.rename(columns={
        "DriverNumber": "driver_number",
        "Driver":       "driver_code",
        "LapNumber":    "lap",
        "Stint":        "stint",
        "Compound":     "tire_compound",
        "TyreLife":     "tire_life",
    })
    keep = ["driver_number", "driver_code", "lap", "lap_time_sec",
            "stint", "tire_compound", "tire_life"]
    return laps[[c for c in keep if c in laps.columns]].reset_index(drop=True)


def get_current_schedule(year: int) -> pd.DataFrame:
    """Return the event schedule for a given season."""
    try:
        import fastf1
        setup_fastf1_cache()
        schedule = fastf1.get_event_schedule(year)
        return schedule[["RoundNumber", "EventName", "EventDate", "Country"]].copy()
    except Exception as exc:
        logger.error("Could not fetch schedule: %s", exc)
        return pd.DataFrame()
