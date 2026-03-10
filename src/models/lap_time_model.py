import torch
import torch.nn as nn


class LapTimeLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, dropout=0.3,
                 num_drivers=1000, driver_embed_dim=8, num_circuits=100, circuit_embed_dim=4):
        super().__init__()
        self.driver_emb = nn.Embedding(num_drivers, driver_embed_dim)
        self.circuit_emb = nn.Embedding(num_circuits, circuit_embed_dim)
        lstm_input = input_size + driver_embed_dim + circuit_embed_dim
        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, driver_id, circuit_id):
        B, T, _ = x.shape
        d = self.driver_emb(driver_id).unsqueeze(1).expand(B, T, -1)
        c = self.circuit_emb(circuit_id).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x, d, c], dim=-1)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)
