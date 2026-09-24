"""tests/test_continual.py -- checks for models/continual.py (EWC, ReplayBuffer)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn import FraudGNN
from models.continual import EWC, ReplayBuffer


def test_ewc_penalty_zero_before_any_task_registered():
    model = FraudGNN(in_dim=5, hidden_dim=8, num_layers=2)
    ewc = EWC(model, ewc_lambda=100.0)
    penalty = ewc.penalty()
    assert penalty.item() == 0.0


def test_ewc_register_task_and_penalty_grows_with_drift():
    model = FraudGNN(in_dim=5, hidden_dim=8, num_layers=2)
    ewc = EWC(model, ewc_lambda=100.0)

    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float)
    mask = torch.ones(6, dtype=torch.bool)

    def loss_fn():
        logits = model(x, edge_index)
        return F.binary_cross_entropy_with_logits(logits[mask], y[mask])

    ewc.register_task(loss_fn)
    assert len(ewc.fisher) > 0

    penalty_before = ewc.penalty().item()
    assert penalty_before == 0.0  # params haven't moved yet

    # perturb the model's weights to simulate further training
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)

    penalty_after = ewc.penalty().item()
    assert penalty_after > 0.0  # moving important weights should cost something


def test_replay_buffer_reservoir_sampling():
    buf = ReplayBuffer(capacity=5, seed=0)
    buf.add(list(range(100)), chunk_id=0)
    assert len(buf) == 5  # capped at capacity even though 100 items were added

    sample = buf.sample(3)
    assert len(sample) == 3
    assert all(0 <= idx < 100 for idx in sample)


def test_replay_buffer_sample_more_than_available():
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add([1, 2, 3], chunk_id=0)
    sample = buf.sample(100)  # asking for more than exists
    assert len(sample) == 3
