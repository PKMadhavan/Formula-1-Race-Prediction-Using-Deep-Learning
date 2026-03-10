import torch
import torch.nn as nn


class PositionMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
