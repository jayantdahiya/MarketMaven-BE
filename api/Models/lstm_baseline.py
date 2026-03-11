"""
Production LSTM baseline: LayerNorm → LSTM → Dropout → Linear+GELU → Linear.
"""
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Input [B, T, F], output [B, H]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 1,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.gelu = nn.GELU()
        self.fc_out = nn.Linear(hidden_dim // 2, horizon)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden_size = self.lstm.hidden_size
                param.data[hidden_size : 2 * hidden_size].fill_(1.0)
        for layer in [self.fc1, self.fc_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input must be 3D [B, T, F]")
        if not torch.isfinite(x).all():
            raise ValueError("Input contains non-finite values")
        x = self.input_norm(x)
        h_seq, _ = self.lstm(x)
        h_last = h_seq[:, -1, :]
        h_last = self.dropout(h_last)
        z = self.gelu(self.fc1(h_last))
        return self.fc_out(z)
