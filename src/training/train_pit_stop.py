"""Training script for PitStopFCNN. Usage: python -m src.training.train_pit_stop"""
from __future__ import annotations
import logging, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.data.ingestion import load_kaggle_data
from src.data.preprocessing import build_pit_stop_dataset
from src.models.pit_stop_model import PitStopFCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train(config_path: str = "configs/config.yaml") -> None:
    with open(config_path) as f: cfg = yaml.safe_load(f)
    mc = cfg["models"]["pit_stop"]
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
    ds  = build_pit_stop_dataset(raw, scalers_dir=scalers_dir)

    train_loader = DataLoader(PitDataset(ds["X_train"], ds["y_train"]), batch_size=mc["batch_size"], shuffle=True)
    val_loader   = DataLoader(PitDataset(ds["X_val"],   ds["y_val"]),   batch_size=mc["batch_size"])
    test_loader  = DataLoader(PitDataset(ds["X_test"],  ds["y_test"]),  batch_size=mc["batch_size"])

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = mc["threshold"]
    model     = PitStopFCNN(input_dim=ds["X_train"].shape[1]).to(device)
    loss_fn   = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=mc["learning_rate"])

    run_ctx = mlflow.start_run(run_name="pit_stop_fcnn") if use_mlflow else __import__("contextlib").nullcontext()
    with run_ctx:
        for epoch in range(mc["epochs"]):
            model.train(); tl = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
                tl += loss.item() * xb.size(0)
            tl /= len(train_loader.dataset)
            model.eval(); vl = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += loss_fn(model(xb), yb).item() * xb.size(0)
            vl /= len(val_loader.dataset)
            logger.info("Epoch %d/%d | train=%.4f val=%.4f", epoch+1, mc["epochs"], tl, vl)
            if use_mlflow: mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

        model.eval(); all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                all_p.extend(model(xb.to(device)).cpu().numpy()); all_t.extend(yb.numpy())
        all_p, all_t = np.array(all_p), np.array(all_t)
        preds = (all_p >= threshold).astype(int)
        metrics = {
            "test_accuracy":  float(accuracy_score(all_t, preds)),
            "test_precision": float(precision_score(all_t, preds, zero_division=0)),
            "test_recall":    float(recall_score(all_t, preds, zero_division=0)),
            "test_f1":        float(f1_score(all_t, preds, zero_division=0)),
            "test_auc":       float(roc_auc_score(all_t, all_p)),
        }
        for k, v in metrics.items(): logger.info("%s: %.4f", k, v)
        if use_mlflow: mlflow.log_metrics(metrics)

        save_path = f"{artifacts_dir}/pit_stop_fcnn.pt"
        torch.save(model.state_dict(), save_path)
        logger.info("Model saved → %s", save_path)


if __name__ == "__main__": train()
