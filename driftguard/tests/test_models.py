"""tests/test_models.py -- forward-pass shape/sanity checks for every model."""
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mlp import FraudMLP
from models.gnn import FraudGNN
from models.diffusion import GraphDiffusionModel, GraphDenoiser, cosine_beta_schedule


def test_mlp_forward_shape():
    model = FraudMLP(in_dim=29, hidden_dim=16)
    x = torch.randn(8, 29)
    out = model(x)
    assert out.shape == (8,)


def test_gnn_forward_shape():
    model = FraudGNN(in_dim=10, hidden_dim=16, num_layers=2)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (6,)


def test_gnn_embed_shape():
    model = FraudGNN(in_dim=10, hidden_dim=16, num_layers=2)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    h = model.embed(x, edge_index)
    assert h.shape == (6, 16)


def test_cosine_beta_schedule_shape_and_range():
    betas = cosine_beta_schedule(timesteps=100)
    assert betas.shape == (100,)
    assert (betas > 0).all() and (betas < 1).all()


def test_graph_denoiser_forward_shape():
    model = GraphDenoiser(feat_dim=10, hidden_dim=16, num_layers=2)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    t = torch.randint(0, 50, (6,))
    out = model(x, edge_index, t)
    assert out.shape == (6, 10)


def test_diffusion_train_step_and_anomaly_score():
    device = "cpu"
    diffusion = GraphDiffusionModel(feat_dim=8, timesteps=50, device=device)
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]], dtype=torch.long)
    normal_mask = torch.ones(10, dtype=torch.bool)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

    loss = diffusion.train_step(x, edge_index, normal_mask, optimizer)
    assert isinstance(loss, float)

    scores = diffusion.anomaly_score(x, edge_index, t_probe=10, n_repeats=2)
    assert scores.shape == (10,)
    assert (scores >= 0).all()
