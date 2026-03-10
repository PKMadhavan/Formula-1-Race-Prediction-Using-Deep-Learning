"""Training script for LapTimeLSTM. Usage: python -m src.training.train_lap_time"""
from __future__ import annotations
import logging, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

from src.data.ingestion import load_kaggle_data
from src.data.preprocessing import build_lap_time_dataset
from src.models.lap_time_model import LapTimeLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class LapDataset(Dataset):
    def __init__(self, X, y, d, c):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.d = torch.tensor(d, dtype=torch.long)
        self.c = torch.tensor(c, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.d[i], self.c[i]


def train(config_path: str = "configs/config.yaml") -> None:
    with open(config_path) as f: cfg = yaml.safe_load(f)
    mc = cfg["models"]["lap_time"]
    artifacts_dir = cfg["artifacts"]["models_dir"]
    scalers_dir   = cfg["artifacts"]["scalers_dir"]
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        use_mlflow = True
    except Exception:
        use_mlflow = False
        logger.warning("MLflow not available — skipping tracking.")

    logger.info("Loading raw data …")
    raw = load_kaggle_data(cfg["data"]["raw_dir"])
    ds  = build_lap_time_dataset(raw, scalers_dir=scalers_dir)

    train_loader = DataLoader(LapDataset(ds["X_train"], ds["y_train"], ds["d_train"], ds["c_train"]), batch_size=mc["batch_size"], shuffle=True)
    val_loader   = DataLoader(LapDataset(ds["X_val"],   ds["y_val"],   ds["d_val"],   ds["c_val"]),   batch_size=mc["batch_size"])
    test_loader  = DataLoader(LapDataset(ds["X_test"],  ds["y_test"],  ds["d_test"],  ds["c_test"]),  batch_size=mc["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LapTimeLSTM(hidden_size=mc["hidden_size"], num_layers=mc["num_layers"], dropout=mc["dropout"],
                         num_drivers=ds["num_drivers"], num_circuits=ds["num_circuits"],
                         driver_embed_dim=mc["driver_embed_dim"], circuit_embed_dim=mc["circuit_embed_dim"]).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=mc["learning_rate"])

    run_ctx = mlflow.start_run(run_name="lap_time_lstm") if use_mlflow else __import__("contextlib").nullcontext()
    with run_ctx:
        for epoch in range(mc["epochs"]):
            model.train(); tl = 0
            for xb, yb, db, cb in train_loader:
                xb, yb, db, cb = xb.to(device), yb.to(device), db.to(device), cb.to(device)
                optimizer.zero_grad(); loss = loss_fn(model(xb, db, cb), yb); loss.backward(); optimizer.step()
                tl += loss.item() * xb.size(0)
            tl /= len(train_loader.dataset)
            model.eval(); vl = 0
            with torch.no_grad():
                for xb, yb, db, cb in val_loader:
                    xb, yb, db, cb = xb.to(device), yb.to(device), db.to(device), cb.to(device)
                    vl += loss_fn(model(xb, db, cb), yb).item() * xb.size(0)
            vl /= len(val_loader.dataset)
            logger.info("Epoch %d/%d | train=%.6f val=%.6f", epoch+1, mc["epochs"], tl, vl)
            if use_mlflow: mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

        model.eval(); all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb, db, cb in test_loader:
                all_p.extend(model(xb.to(device), db.to(device), cb.to(device)).cpu().numpy())
                all_t.extend(yb.numpy())
        rmse = float(np.sqrt(mean_squared_error(
            ds["scaler_y"].inverse_transform(np.array(all_t).reshape(-1,1)).flatten(),
            ds["scaler_y"].inverse_transform(np.array(all_p).reshape(-1,1)).flatten(),
        )))
        logger.info("Test RMSE: %.3f seconds", rmse)
        if use_mlflow: mlflow.log_metric("test_rmse_seconds", rmse)

        save_path = f"{artifacts_dir}/lap_time_lstm.pt"
        torch.save(model.state_dict(), save_path)
        logger.info("Model saved → %s", save_path)


if __name__ == "__main__": train()
