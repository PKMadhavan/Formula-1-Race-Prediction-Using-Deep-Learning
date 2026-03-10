"""
Creates dummy trained models + scalers so the API can be tested
without running the full training pipeline.

Usage:
    python scripts/create_test_fixtures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.models.lap_time_model import LapTimeLSTM
from src.models.pit_stop_model import PitStopFCNN
from src.models.position_model import PositionMLP

MODELS_DIR  = Path("artifacts/models")
SCALERS_DIR = Path("artifacts/scalers")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCALERS_DIR.mkdir(parents=True, exist_ok=True)

NUM_DRIVERS  = 50
NUM_CIRCUITS = 20

# ── Lap Time ──────────────────────────────────────────────────────────────────
lap_model = LapTimeLSTM(num_drivers=NUM_DRIVERS, num_circuits=NUM_CIRCUITS)
torch.save(lap_model.state_dict(), MODELS_DIR / "lap_time_lstm.pt")

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
dummy_x = np.random.uniform(40, 150, (100, 2)).astype(np.float32)
dummy_y = np.random.uniform(40, 150, (100, 1)).astype(np.float32)
scaler_x.fit(dummy_x)
scaler_y.fit(dummy_y)
joblib.dump(scaler_x, SCALERS_DIR / "lap_time_scaler_x.pkl")
joblib.dump(scaler_y, SCALERS_DIR / "lap_time_scaler_y.pkl")
joblib.dump(list(range(NUM_DRIVERS)),  SCALERS_DIR / "driver_index.pkl")
joblib.dump(list(range(NUM_CIRCUITS)), SCALERS_DIR / "circuit_index.pkl")
print("✓ Lap time model + scalers")

# ── Pit Stop ──────────────────────────────────────────────────────────────────
pit_model = PitStopFCNN(input_dim=4)
torch.save(pit_model.state_dict(), MODELS_DIR / "pit_stop_fcnn.pt")

pit_scaler = MinMaxScaler()
pit_scaler.fit(np.random.uniform(0, 200_000, (100, 4)))
joblib.dump(pit_scaler, SCALERS_DIR / "pit_stop_scaler.pkl")
print("✓ Pit stop model + scaler")

# ── Position ──────────────────────────────────────────────────────────────────
pos_model = PositionMLP(input_dim=4, hidden_dim=64, dropout=0.2)
torch.save(pos_model.state_dict(), MODELS_DIR / "position_mlp.pt")

pos_scaler = MinMaxScaler()
pos_scaler.fit(np.random.uniform(0, 10_000_000, (100, 4)))
joblib.dump(pos_scaler, SCALERS_DIR / "position_scaler.pkl")
print("✓ Position model + scaler")

print("\nAll fixtures created. Run: uvicorn src.api.main:app --reload --port 8000")
