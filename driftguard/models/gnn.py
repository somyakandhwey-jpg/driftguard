"""
models/gnn.py -- Phase 2: GraphSAGE node classifier for the transaction graph.

Also exposes `.embed()` so the same encoder backbone can be reused as the
denoiser backbone for the Phase 5 diffusion model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 1)

    def embed(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index):
        h = self.embed(x, edge_index)
        return self.classifier(h).squeeze(-1)  # raw logits
