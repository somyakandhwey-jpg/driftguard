"""tests/test_data.py -- sanity checks for the dataset loader/generator."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import generate_synthetic_elliptic, load_creditcard


def test_synthetic_elliptic_shapes():
    data = generate_synthetic_elliptic(num_timesteps=5, nodes_per_step=20, feat_dim=10)
    n_nodes = 5 * 20
    assert data.x.shape == (n_nodes, 10)
    assert data.y.shape == (n_nodes,)
    assert data.time_step.shape == (n_nodes,)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.max().item() < n_nodes


def test_synthetic_elliptic_time_steps():
    data = generate_synthetic_elliptic(num_timesteps=5, nodes_per_step=20)
    assert data.time_step.min().item() == 1
    assert data.time_step.max().item() == 5


def test_synthetic_elliptic_labels_are_binary_or_unknown():
    data = generate_synthetic_elliptic(num_timesteps=3, nodes_per_step=30)
    unique_labels = set(data.y.unique().tolist())
    assert unique_labels.issubset({-1, 0, 1})


def test_synthetic_elliptic_labeled_mask_consistent():
    data = generate_synthetic_elliptic(num_timesteps=3, nodes_per_step=30)
    # every node marked unlabeled should have y == -1, and vice versa
    assert (data.y[~data.labeled_mask] == -1).all()
    assert (data.y[data.labeled_mask] != -1).all()


def test_load_creditcard_synthetic_fallback():
    X, y = load_creditcard(csv_path="does/not/exist.csv", synthetic_n=1000, fraud_ratio=0.05)
    assert X.shape[0] == 1000
    assert y.shape[0] == 1000
    assert set(y.tolist()).issubset({0, 1})
    # roughly the requested fraud ratio (allow slack since it's random)
    assert 0.01 < y.mean() < 0.15
