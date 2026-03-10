"""
Feature engineering and preprocessing pipelines for all three models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 5
RANDOM_SEED = 42


def build_lap_time_dataset(
    data: dict[str, pd.DataFrame],
    scalers_dir: str = "artifacts/scalers",
    seq_len: int = SEQUENCE_LENGTH,
) -> dict[str, Any]:
    lap_times = data["lap_times"].copy()
    races = data["races"][["raceId", "circuitId"]].copy()

    lap_times = lap_times.merge(races, on="raceId", how="left")
    lap_times["lap_time_sec"] = lap_times["milliseconds"] / 1000.0
    lap_times = lap_times.dropna(subset=["lap_time_sec"])
    lap_times = lap_times[
        (lap_times["lap_time_sec"] >= 40) & (lap_times["lap_time_sec"] <= 200)
    ]
    lap_times.sort_values(["raceId", "driverId", "lap"], inplace=True)

    lap_times["driver_enc"],  driver_index  = pd.factorize(lap_times["driverId"])
    lap_times["circuit_enc"], circuit_index = pd.factorize(lap_times["circuitId"])

    X, y, driver_ids, circuit_ids = [], [], [], []
    for (_, _), group in lap_times.groupby(["raceId", "driverId"]):
        times = group["lap_time_sec"].values
        laps  = group["lap"].values
        d_enc = group["driver_enc"].values
        c_enc = group["circuit_enc"].values
        if len(times) > seq_len:
            for i in range(len(times) - seq_len):
                X.append(np.stack([times[i:i+seq_len], laps[i:i+seq_len]], axis=1))
                y.append(times[i + seq_len])
                driver_ids.append(d_enc[i])
                circuit_ids.append(c_enc[i])

    X, y = np.array(X), np.array(y)
    driver_ids, circuit_ids = np.array(driver_ids), np.array(circuit_ids)

    X_tr, X_tmp, y_tr, y_tmp, d_tr, d_tmp, c_tr, c_tmp = train_test_split(
        X, y, driver_ids, circuit_ids, test_size=0.30, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test, d_val, d_test, c_val, c_test = train_test_split(
        X_tmp, y_tmp, d_tmp, c_tmp, test_size=0.50, random_state=RANDOM_SEED
    )

    T, F = X.shape[1], X.shape[2]
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train = scaler_x.fit_transform(X_tr.reshape(-1, F)).reshape(-1, T, F)
    y_train = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()
    X_val_s = scaler_x.transform(X_val.reshape(-1, F)).reshape(-1, T, F)
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    X_test_s = scaler_x.transform(X_test.reshape(-1, F)).reshape(-1, T, F)
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    Path(scalers_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_x, f"{scalers_dir}/lap_time_scaler_x.pkl")
    joblib.dump(scaler_y, f"{scalers_dir}/lap_time_scaler_y.pkl")
    joblib.dump(driver_index.tolist(),  f"{scalers_dir}/driver_index.pkl")
    joblib.dump(circuit_index.tolist(), f"{scalers_dir}/circuit_index.pkl")

    return dict(
        X_train=X_train, X_val=X_val_s, X_test=X_test_s,
        y_train=y_train, y_val=y_val_s, y_test=y_test_s,
        d_train=d_tr, d_val=d_val, d_test=d_test,
        c_train=c_tr, c_val=c_val, c_test=c_test,
        num_drivers=len(driver_index), num_circuits=len(circuit_index),
        scaler_x=scaler_x, scaler_y=scaler_y,
    )


def build_pit_stop_dataset(
    data: dict[str, pd.DataFrame],
    scalers_dir: str = "artifacts/scalers",
) -> dict[str, Any]:
    lap_times = data["lap_times"].copy()
    pit_stops = data["pit_stops"][["raceId", "driverId", "lap"]].copy()

    lap_times = lap_times.merge(
        pit_stops, on=["raceId", "driverId", "lap"], how="left", indicator="pitted"
    )
    lap_times["pitted"] = (lap_times["pitted"] == "both").astype(int)
    lap_times.sort_values(["raceId", "driverId", "lap"], inplace=True)

    features, labels = [], []
    for (_, _), group in lap_times.groupby(["raceId", "driverId"]):
        group = group.sort_values("lap").copy()
        group["cumulative_pits"] = group["pitted"].cumsum()
        for i in range(1, len(group) - 1):
            prev, curr, nxt = group.iloc[i-1], group.iloc[i], group.iloc[i+1]
            features.append([
                curr["lap"], curr["cumulative_pits"],
                prev["milliseconds"] if pd.notna(prev["milliseconds"]) else 0,
                curr["milliseconds"] if pd.notna(curr["milliseconds"]) else 0,
            ])
            labels.append(int(nxt["pitted"] == 1))

    features = np.array(features, dtype=np.float32)
    labels   = np.array(labels,   dtype=np.int64)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        features, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_tmp
    )

    rng = np.random.default_rng(RANDOM_SEED)
    X_pos, X_neg = X_tr[y_tr == 1], X_tr[y_tr == 0]
    n = max(len(X_pos), len(X_neg))
    X_train_os = np.vstack([X_pos[rng.integers(0, len(X_pos), n)],
                             X_neg[rng.integers(0, len(X_neg), n)]])
    y_train_os = np.array([1]*n + [0]*n, dtype=np.int64)
    idx = rng.permutation(len(X_train_os))
    X_train_os, y_train_os = X_train_os[idx], y_train_os[idx]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train_os)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    Path(scalers_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, f"{scalers_dir}/pit_stop_scaler.pkl")

    return dict(
        X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
        y_train=y_train_os, y_val=y_val, y_test=y_test,
        scaler=scaler,
    )


def _parse_lap_time(t: Any) -> float:
    try:
        m, s = str(t).split(":")
        return float(m) * 60 + float(s)
    except Exception:
        return float("nan")


def build_position_dataset(
    data: dict[str, pd.DataFrame],
    scalers_dir: str = "artifacts/scalers",
) -> dict[str, Any]:
    results = data["results"].copy()
    races   = data["races"][["raceId", "year", "circuitId"]].copy()
    drivers = data["drivers"][["driverId", "driverRef"]].copy()

    results = results.merge(races,   on="raceId",   how="left")
    results = results.merge(drivers, on="driverId",  how="left")
    results = results.dropna(subset=["positionOrder"])
    results["milliseconds"]     = pd.to_numeric(results["milliseconds"], errors="coerce")
    results["fastestLapTimeSec"] = results["fastestLapTime"].apply(_parse_lap_time)

    feat_cols = ["grid", "laps", "milliseconds", "fastestLapTimeSec"]
    clean = results[feat_cols + ["positionOrder"]].dropna()
    X = clean[feat_cols].values.astype(np.float32)
    y = clean["positionOrder"].values.astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_tr)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    Path(scalers_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, f"{scalers_dir}/position_scaler.pkl")

    return dict(
        X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
        y_train=y_tr, y_val=y_val, y_test=y_test,
        scaler=scaler,
    )
