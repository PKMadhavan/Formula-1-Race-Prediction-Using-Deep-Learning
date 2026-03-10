"""Training script for PositionMLP. Usage: python -m src.training.train_position"""
from __future__ import annotations
import logging, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.ingestion import load_kaggle_data
from src.data.preprocessing import build_position_dataset
from src.models.position_model import PositionMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(config_path: str = "configs/config.yaml") -> None:
    with open(config_path) as f: cfg = yaml.safe_load(f)
    mc = cfg["models"]["position"]
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

    logger.info("Loading raw data …")
    raw = load_kaggle_data(cfg["data"]["raw_dir"])
    ds  = build_position_dataset(raw, scalers_dir=scalers_dir)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(ds["X_train"], dtype=torch.float32).to(device)
    y_train = torch.tensor(ds["y_train"], dtype=torch.float32).to(device)
    X_val   = torch.tensor(ds["X_val"],   dtype=torch.float32).to(device)
    y_val   = torch.tensor(ds["y_val"],   dtype=torch.float32).to(device)
    X_test  = torch.tensor(ds["X_test"],  dtype=torch.float32)

    model     = PositionMLP(input_dim=ds["X_train"].shape[1], hidden_dim=mc["hidden_dims"][0], dropout=mc["dropout"]).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=mc["learning_rate"])

    run_ctx = mlflow.start_run(run_name="position_mlp") if use_mlflow else __import__("contextlib").nullcontext()
    with run_ctx:
        for epoch in range(mc["epochs"]):
            model.train(); optimizer.zero_grad()
            loss = loss_fn(model(X_train), y_train); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad(): vl = loss_fn(model(X_val), y_val).item()
            if (epoch+1) % 10 == 0:
                logger.info("Epoch %d/%d | train=%.4f val=%.4f", epoch+1, mc["epochs"], loss.item(), vl)
            if use_mlflow: mlflow.log_metrics({"train_loss": loss.item(), "val_loss": vl}, step=epoch)

        model.eval()
        with torch.no_grad(): preds = model(X_test.to(device)).cpu().numpy()
        mse  = float(mean_squared_error(ds["y_test"], preds))
        mae  = float(mean_absolute_error(ds["y_test"], preds))
        logger.info("Test MSE=%.2f MAE=%.2f RMSE=%.2f", mse, mae, mse**0.5)
        if use_mlflow: mlflow.log_metrics({"test_mse": mse, "test_mae": mae, "test_rmse": mse**0.5})

        save_path = f"{artifacts_dir}/position_mlp.pt"
        torch.save(model.state_dict(), save_path)
        logger.info("Model saved → %s", save_path)


if __name__ == "__main__": train()
