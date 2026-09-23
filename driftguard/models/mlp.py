"""
models/mlp.py -- Phase 1 baseline: plain MLP fraud classifier on tabular data.
"""
import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits
