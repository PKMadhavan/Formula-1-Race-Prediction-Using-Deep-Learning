#!/bin/bash
# Run full training pipeline then launch services
set -e

echo "=== F1 Race Predictor — Full Training Pipeline ==="

# 1. Place Kaggle CSVs in data/raw/ before running this script
if [ ! -f "data/raw/lap_times.csv" ]; then
  echo "ERROR: data/raw/lap_times.csv not found."
  echo "Download the Kaggle F1 dataset and place CSVs in data/raw/"
  exit 1
fi

# 2. Train all models
echo ""
echo "--- Training all models ---"
python -m src.training.train_all

# 3. Verify artifacts exist
echo ""
echo "--- Checking artifacts ---"
for f in artifacts/models/lap_time_lstm.pt \
          artifacts/models/pit_stop_fcnn.pt \
          artifacts/models/position_mlp.pt; do
  if [ -f "$f" ]; then
    echo "  ✓ $f"
  else
    echo "  ✗ MISSING: $f"
    exit 1
  fi
done

echo ""
echo "=== Training complete. Starting services with Docker Compose ==="
docker compose -f docker/docker-compose.yml up --build -d

echo ""
echo "Services running:"
echo "  API       → http://localhost:8000"
echo "  API docs  → http://localhost:8000/docs"
echo "  Dashboard → http://localhost:8501"
echo "  MLflow    → http://localhost:5000"
