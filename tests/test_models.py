"""Unit tests for all three PyTorch models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.models.lap_time_model import LapTimeLSTM
from src.models.pit_stop_model import PitStopFCNN
from src.models.position_model import PositionMLP


class TestLapTimeLSTM:
    def setup_method(self):
        self.model = LapTimeLSTM(num_drivers=10, num_circuits=5)
        self.model.eval()

    def test_output_shape_single(self):
        x = torch.rand(1, 5, 2)
        d = torch.zeros(1, dtype=torch.long)
        c = torch.zeros(1, dtype=torch.long)
        out = self.model(x, d, c)
        assert out.shape == (1,)

    def test_output_shape_batch(self):
        x = torch.rand(32, 5, 2)
        d = torch.randint(0, 10, (32,))
        c = torch.randint(0, 5, (32,))
        out = self.model(x, d, c)
        assert out.shape == (32,)

    def test_output_is_float(self):
        x = torch.rand(4, 5, 2)
        d = torch.zeros(4, dtype=torch.long)
        c = torch.zeros(4, dtype=torch.long)
        out = self.model(x, d, c)
        assert out.dtype == torch.float32

    def test_different_drivers_give_different_outputs(self):
        x = torch.rand(1, 5, 2)
        c = torch.zeros(1, dtype=torch.long)
        out1 = self.model(x, torch.tensor([0]), c)
        out2 = self.model(x, torch.tensor([1]), c)
        assert not torch.isclose(out1, out2).all()


class TestPitStopFCNN:
    def setup_method(self):
        self.model = PitStopFCNN(input_dim=4)
        self.model.eval()

    def test_output_shape_single(self):
        x = torch.rand(1, 4)
        out = self.model(x)
        assert out.shape == (1,)

    def test_output_shape_batch(self):
        x = torch.rand(64, 4)
        out = self.model(x)
        assert out.shape == (64,)

    def test_output_in_probability_range(self):
        x = torch.rand(100, 4)
        out = self.model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_predict_returns_binary(self):
        x = torch.rand(10, 4)
        preds, probs = self.model.predict(x, threshold=0.75)
        assert set(preds.tolist()).issubset({0, 1})
        assert (probs >= 0).all() and (probs <= 1).all()


class TestPositionMLP:
    def setup_method(self):
        self.model = PositionMLP(input_dim=4, hidden_dim=64)
        self.model.eval()

    def test_output_shape_single(self):
        x = torch.rand(1, 4)
        out = self.model(x)
        assert out.shape == (1,)

    def test_output_shape_batch(self):
        x = torch.rand(16, 4)
        out = self.model(x)
        assert out.shape == (16,)

    def test_output_is_continuous(self):
        x = torch.rand(50, 4)
        out = self.model(x)
        # Should not be all same value
        assert out.std().item() > 0
