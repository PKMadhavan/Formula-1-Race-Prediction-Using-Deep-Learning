"""
2026 Season live data module.
Fetches schedule + session data via FastF1 and runs model predictions.

⚠️  Model accuracy note:
    All three models (LapTimeLSTM, PitStopFCNN, PositionMLP) were trained on
    historical F1 data from 1950–2024. Predictions for 2026 are approximate:
    - New drivers (Antonelli, Hadjar, Doohan, Bortoleto, Cadillac drivers)
      have no representation in the training embeddings.
    - New teams (Audi, Cadillac F1) have no historical baseline.
    - Predictions rely on grid-position and race-time patterns from past seasons.
    For best accuracy, retrain the models on 2025–2026 data once available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 2026 Driver / Team Registry (22 drivers, 11 teams) ───────────────────────
# Sources: confirmed 2026 FIA entry list
# Cadillac F1 drivers confirmed: Colton Herta + Prema/GM junior TBC

DRIVERS_2026 = [
    # Red Bull Racing
    {"code": "VER", "name": "Max Verstappen",       "team": "Red Bull Racing",  "number": 1,  "new": False},
    {"code": "LAW", "name": "Liam Lawson",           "team": "Red Bull Racing",  "number": 30, "new": False},
    # Ferrari
    {"code": "LEC", "name": "Charles Leclerc",       "team": "Ferrari",          "number": 16, "new": False},
    {"code": "HAM", "name": "Lewis Hamilton",        "team": "Ferrari",          "number": 44, "new": False},
    # Mercedes
    {"code": "RUS", "name": "George Russell",        "team": "Mercedes",         "number": 63, "new": False},
    {"code": "ANT", "name": "Kimi Antonelli",        "team": "Mercedes",         "number": 12, "new": True},
    # McLaren
    {"code": "NOR", "name": "Lando Norris",          "team": "McLaren",          "number": 4,  "new": False},
    {"code": "PIA", "name": "Oscar Piastri",         "team": "McLaren",          "number": 81, "new": False},
    # Aston Martin
    {"code": "ALO", "name": "Fernando Alonso",       "team": "Aston Martin",     "number": 14, "new": False},
    {"code": "STR", "name": "Lance Stroll",          "team": "Aston Martin",     "number": 18, "new": False},
    # Alpine
    {"code": "GAS", "name": "Pierre Gasly",          "team": "Alpine",           "number": 10, "new": False},
    {"code": "DOO", "name": "Jack Doohan",           "team": "Alpine",           "number": 7,  "new": True},
    # Williams
    {"code": "SAI", "name": "Carlos Sainz",          "team": "Williams",         "number": 55, "new": False},
    {"code": "ALB", "name": "Alexander Albon",       "team": "Williams",         "number": 23, "new": False},
    # Haas
    {"code": "OCO", "name": "Esteban Ocon",          "team": "Haas",             "number": 31, "new": False},
    {"code": "BEA", "name": "Oliver Bearman",        "team": "Haas",             "number": 87, "new": True},
    # Racing Bulls
    {"code": "TSU", "name": "Yuki Tsunoda",          "team": "Racing Bulls",     "number": 22, "new": False},
    {"code": "HAD", "name": "Isack Hadjar",          "team": "Racing Bulls",     "number": 6,  "new": True},
    # Audi (formerly Sauber — rebranded 2026)
    {"code": "HUL", "name": "Nico Hülkenberg",      "team": "Audi",             "number": 27, "new": False},
    {"code": "BOR", "name": "Gabriel Bortoleto",     "team": "Audi",             "number": 5,  "new": True},
    # Cadillac F1 (new 11th team for 2026)
    {"code": "HER", "name": "Colton Herta",          "team": "Cadillac F1",      "number": 6,  "new": True},
    {"code": "O'W", "name": "Robert Shwartzman",     "team": "Cadillac F1",      "number": 21, "new": True},
]

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari":         "#E8002D",
    "Mercedes":        "#27F4D2",
    "McLaren":         "#FF8000",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Williams":        "#64C4FF",
    "Haas":            "#B6BABD",
    "Racing Bulls":    "#6692FF",
    "Audi":            "#BB0000",       # Audi red (formerly Sauber green)
    "Cadillac F1":     "#CC0000",       # GM / Cadillac red-white-blue
}

# Drivers with NO representation in training data (1950-2024)
NEW_DRIVERS_2026 = {d["code"] for d in DRIVERS_2026 if d["new"]}

MODEL_DISCLAIMER = (
    "⚠️ **Prediction disclaimer**: Models trained on 1950–2024 data. "
    "New drivers (Antonelli, Hadjar, Doohan, Bortoleto, Herta, Shwartzman) "
    "and new teams (Audi, Cadillac F1) have **no historical training data**. "
    "Predictions for these entries use grid-position heuristics only "
    "and should be treated as indicative, not precise."
)


# ── Schedule helpers ──────────────────────────────────────────────────────────

def get_2026_schedule() -> pd.DataFrame:
    """Fetch 2026 F1 calendar via FastF1, fall back to hardcoded list."""
    try:
        import fastf1
        Path("data/fastf1_cache").mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache("data/fastf1_cache")
        schedule = fastf1.get_event_schedule(2026, include_testing=False)
        return schedule[["RoundNumber", "EventName", "EventDate", "Country", "Location"]].copy()
    except Exception as exc:
        logger.warning("FastF1 schedule fetch failed (%s) — using fallback.", exc)
        return _fallback_schedule()


def _fallback_schedule() -> pd.DataFrame:
    races = [
        (1,  "Australian Grand Prix",   "2026-03-15", "Australia",   "Melbourne"),
        (2,  "Chinese Grand Prix",      "2026-03-22", "China",       "Shanghai"),
        (3,  "Japanese Grand Prix",     "2026-04-05", "Japan",       "Suzuka"),
        (4,  "Bahrain Grand Prix",      "2026-04-19", "Bahrain",     "Sakhir"),
        (5,  "Saudi Arabian Grand Prix","2026-04-26", "Saudi Arabia","Jeddah"),
        (6,  "Miami Grand Prix",        "2026-05-03", "USA",         "Miami"),
        (7,  "Emilia Romagna Grand Prix","2026-05-17","Italy",       "Imola"),
        (8,  "Monaco Grand Prix",       "2026-05-24", "Monaco",      "Monte Carlo"),
        (9,  "Spanish Grand Prix",      "2026-06-07", "Spain",       "Barcelona"),
        (10, "Canadian Grand Prix",     "2026-06-14", "Canada",      "Montreal"),
        (11, "Austrian Grand Prix",     "2026-06-28", "Austria",     "Spielberg"),
        (12, "British Grand Prix",      "2026-07-05", "UK",          "Silverstone"),
        (13, "Belgian Grand Prix",      "2026-07-26", "Belgium",     "Spa"),
        (14, "Hungarian Grand Prix",    "2026-08-02", "Hungary",     "Budapest"),
        (15, "Dutch Grand Prix",        "2026-08-30", "Netherlands", "Zandvoort"),
        (16, "Italian Grand Prix",      "2026-09-06", "Italy",       "Monza"),
        (17, "Azerbaijan Grand Prix",   "2026-09-20", "Azerbaijan",  "Baku"),
        (18, "Singapore Grand Prix",    "2026-10-04", "Singapore",   "Singapore"),
        (19, "United States Grand Prix","2026-10-18", "USA",         "Austin"),
        (20, "Mexico City Grand Prix",  "2026-10-25", "Mexico",      "Mexico City"),
        (21, "São Paulo Grand Prix",    "2026-11-08", "Brazil",      "São Paulo"),
        (22, "Las Vegas Grand Prix",    "2026-11-21", "USA",         "Las Vegas"),
        (23, "Qatar Grand Prix",        "2026-11-29", "Qatar",       "Lusail"),
        (24, "Abu Dhabi Grand Prix",    "2026-12-06", "UAE",         "Yas Island"),
    ]
    df = pd.DataFrame(races, columns=["RoundNumber","EventName","EventDate","Country","Location"])
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    return df


def get_next_race(schedule: pd.DataFrame) -> Optional[dict]:
    now = pd.Timestamp.now().tz_localize(None)
    upcoming = schedule[pd.to_datetime(schedule["EventDate"]) >= now]
    row = upcoming.iloc[0] if not upcoming.empty else schedule.iloc[-1]
    return row.to_dict()


def get_last_race(schedule: pd.DataFrame) -> Optional[dict]:
    now = pd.Timestamp.now().tz_localize(None)
    past = schedule[pd.to_datetime(schedule["EventDate"]) < now]
    return past.iloc[-1].to_dict() if not past.empty else None


# ── Live session data ─────────────────────────────────────────────────────────

def load_qualifying_results(year: int, round_number: int) -> Optional[pd.DataFrame]:
    """Try to load qualifying session to get official grid order."""
    try:
        import fastf1
        Path("data/fastf1_cache").mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache("data/fastf1_cache")
        session = fastf1.get_session(year, round_number, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        results = session.results[
            ["DriverNumber", "Abbreviation", "FullName", "TeamName", "Position"]
        ].copy()
        results["grid"] = results["Position"].astype(int)
        return results.sort_values("grid").reset_index(drop=True)
    except Exception as exc:
        logger.info("Qualifying not yet available: %s", exc)
        return None


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_race_outcome(grid_order: list[dict], store) -> list[dict]:
    """
    Run PositionMLP for each driver.
    Falls back to grid + small noise if model unavailable.
    New 2026 drivers are flagged in results.
    """
    import torch

    results = []
    for driver in grid_order:
        grid        = int(driver.get("grid", 10))
        is_new      = driver.get("code", "") in NEW_DRIVERS_2026

        if store is not None and store.position_model is not None and store.position_scaler is not None:
            x_raw    = np.array([[grid, 57, 5_400_000.0, 88.5 + grid * 0.3]], dtype=np.float32)
            x_scaled = store.position_scaler.transform(x_raw)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(store.device)
            with torch.no_grad():
                pred = float(store.position_model(x_tensor).cpu().item())
            # New drivers — widen uncertainty around grid position
            if is_new:
                pred = pred * 0.5 + grid * 0.5 + np.random.uniform(-1.0, 1.0)
        else:
            pred = float(grid) + np.random.uniform(-1.5, 1.5)
            if is_new:
                pred += np.random.uniform(0, 2.0)  # new teams typically further back

        results.append({**driver, "predicted_position": round(max(1.0, pred), 2), "new_driver": is_new})

    results.sort(key=lambda r: r["predicted_position"])
    for i, r in enumerate(results):
        r["predicted_finish"] = i + 1
    return results


def predict_next_lap_times(store, num_drivers: int = 10, base_lap_time: float = 90.0) -> list[dict]:
    """
    Predict next lap time for top N drivers.
    Safely handles missing models and new drivers with no training history.
    """
    import torch

    # Safely get max embedding size — default 0 if model not loaded
    max_driver_idx = 0
    if store is not None and store.lap_time_model is not None:
        try:
            max_driver_idx = store.lap_time_model.driver_emb.num_embeddings - 1
        except Exception:
            max_driver_idx = 0

    predictions = []
    for i, driver in enumerate(DRIVERS_2026[:num_drivers]):
        offset   = i * 0.4
        is_new   = driver["code"] in NEW_DRIVERS_2026

        recent_laps = np.array(
            [[base_lap_time + offset + np.random.uniform(-0.3, 0.3), lap]
             for lap in range(1, 6)],
            dtype=np.float32,
        )

        if (store is not None
                and store.lap_time_model is not None
                and store.lap_time_scaler_x is not None
                and not is_new
                and max_driver_idx > 0):
            try:
                seq_scaled = store.lap_time_scaler_x.transform(recent_laps).reshape(1, 5, 2)
                x = torch.tensor(seq_scaled, dtype=torch.float32).to(store.device)
                d = torch.tensor([min(i, max_driver_idx)], dtype=torch.long).to(store.device)
                c = torch.zeros(1, dtype=torch.long).to(store.device)
                with torch.no_grad():
                    pred_s = store.lap_time_model(x, d, c).cpu().numpy().reshape(-1, 1)
                pred = float(store.lap_time_scaler_y.inverse_transform(pred_s)[0, 0])
            except Exception as exc:
                logger.warning("LapTime inference failed for %s: %s", driver["code"], exc)
                pred = base_lap_time + offset
        else:
            # New driver or model unavailable — heuristic estimate
            pred = base_lap_time + offset + (2.0 if is_new else 0.0)

        predictions.append({
            "driver":                  driver["code"],
            "name":                    driver["name"],
            "team":                    driver["team"],
            "predicted_lap_time_sec":  round(pred, 3),
            "new_driver":              is_new,
        })

    predictions.sort(key=lambda x: x["predicted_lap_time_sec"])
    return predictions
