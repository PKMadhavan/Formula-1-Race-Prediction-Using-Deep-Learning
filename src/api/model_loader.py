"""
Singleton model loader.
Loads all three trained PyTorch models + scalers once at startup
and keeps them in memory for fast inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import torch
import yaml

from src.models.lap_time_model import LapTimeLSTM
from src.models.pit_stop_model import PitStopFCNN
from src.models.position_model import PositionMLP

logger = logging.getLogger(__name__)


class ModelStore:
    """Holds loaded models and scalers."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.lap_time_model:  Optional[LapTimeLSTM] = None
        self.pit_stop_model:  Optional[PitStopFCNN] = None
        self.position_model:  Optional[PositionMLP]  = None

        # Scalers / encoders
        self.lap_time_scaler_x = None
        self.lap_time_scaler_y = None
        self.driver_index:  Optional[list] = None
        self.circuit_index: Optional[list] = None
        self.pit_stop_scaler = None
        self.position_scaler = None

        # Config
        self.pit_stop_threshold: float = 0.75

    def load_all(self, config_path: str = "configs/config.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models_dir  = cfg["artifacts"]["models_dir"]
        scalers_dir = cfg["artifacts"]["scalers_dir"]
        self.pit_stop_threshold = cfg["models"]["pit_stop"]["threshold"]

        self._load_lap_time(models_dir, scalers_dir, cfg)
        self._load_pit_stop(models_dir, scalers_dir, cfg)
        self._load_position(models_dir, scalers_dir, cfg)

    def _load_lap_time(self, models_dir, scalers_dir, cfg) -> None:
        mc = cfg["models"]["lap_time"]
        try:
            self.driver_index  = joblib.load(f"{scalers_dir}/driver_index.pkl")
            self.circuit_index = joblib.load(f"{scalers_dir}/circuit_index.pkl")
            self.lap_time_scaler_x = joblib.load(f"{scalers_dir}/lap_time_scaler_x.pkl")
            self.lap_time_scaler_y = joblib.load(f"{scalers_dir}/lap_time_scaler_y.pkl")

            model = LapTimeLSTM(
                hidden_size=mc["hidden_size"],
                num_layers=mc["num_layers"],
                dropout=mc["dropout"],
                num_drivers=len(self.driver_index),
                num_circuits=len(self.circuit_index),
                driver_embed_dim=mc["driver_embed_dim"],
                circuit_embed_dim=mc["circuit_embed_dim"],
            )
            model.load_state_dict(
                torch.load(f"{models_dir}/lap_time_lstm.pt", map_location=self.device)
            )
            model.eval().to(self.device)
            self.lap_time_model = model
            logger.info("LapTimeLSTM loaded ✓")
        except FileNotFoundError as e:
            logger.warning("LapTimeLSTM not loaded: %s", e)

    def _load_pit_stop(self, models_dir, scalers_dir, cfg) -> None:
        try:
            self.pit_stop_scaler = joblib.load(f"{scalers_dir}/pit_stop_scaler.pkl")
            model = PitStopFCNN(input_dim=4)
            model.load_state_dict(
                torch.load(f"{models_dir}/pit_stop_fcnn.pt", map_location=self.device)
            )
            model.eval().to(self.device)
            self.pit_stop_model = model
            logger.info("PitStopFCNN loaded ✓")
        except FileNotFoundError as e:
            logger.warning("PitStopFCNN not loaded: %s", e)

    def _load_position(self, models_dir, scalers_dir, cfg) -> None:
        try:
            self.position_scaler = joblib.load(f"{scalers_dir}/position_scaler.pkl")
            model = PositionMLP(
                input_dim=4,
                hidden_dim=cfg["models"]["position"]["hidden_dims"][0],
                dropout=cfg["models"]["position"]["dropout"],
            )
            model.load_state_dict(
                torch.load(f"{models_dir}/position_mlp.pt", map_location=self.device)
            )
            model.eval().to(self.device)
            self.position_model = model
            logger.info("PositionMLP loaded ✓")
        except FileNotFoundError as e:
            logger.warning("PositionMLP not loaded: %s", e)

    @property
    def models_status(self) -> dict[str, bool]:
        return {
            "lap_time_lstm":  self.lap_time_model  is not None,
            "pit_stop_fcnn":  self.pit_stop_model  is not None,
            "position_mlp":   self.position_model   is not None,
        }


# Single global instance shared across the app
store = ModelStore()
