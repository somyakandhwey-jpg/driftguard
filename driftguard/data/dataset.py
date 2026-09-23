"""
data/dataset.py

Loaders for:
  1. Credit-card tabular fraud dataset (Phase 1 baseline)
  2. Elliptic bitcoin transaction graph dataset (Phase 2+)

Both loaders fall back to a synthetic generator with the SAME schema when the
real CSVs aren't present, so every training script in this repo runs
out-of-the-box (e.g. on a fresh Colab runtime) before you've downloaded
anything. Swap in the real data by dropping files into data/raw/ as described
in data/download_elliptic.py.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


# --------------------------------------------------------------------------
# Phase 1: tabular credit-card transactions
# --------------------------------------------------------------------------
def load_creditcard(csv_path=None, synthetic_n=50000, fraud_ratio=0.0017, seed=0):
    """
    Returns (X, y) as numpy arrays.
    Expects Kaggle's `creditcard.csv` (Time, V1..V28, Amount, Class) if csv_path
    is given and exists. Otherwise generates a synthetic dataset with the same
    shape/imbalance so Phase 1 can be built and tested immediately.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        y = df["Class"].values.astype(np.int64)
        X = df.drop(columns=["Class"]).values.astype(np.float32)
        return X, y

    print(f"[dataset] creditcard.csv not found at '{csv_path}'. "
          f"Generating synthetic tabular data ({synthetic_n} rows, "
          f"~{fraud_ratio:.3%} fraud) instead.")
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(synthetic_n * fraud_ratio))
    n_normal = synthetic_n - n_fraud

    # Normal transactions: tight cluster around 0
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 29))
    # Fraud: shifted mean + heavier tails -> separable but not trivial
    fraud = rng.normal(loc=1.5, scale=2.5, size=(n_fraud, 29))

    X = np.vstack([normal, fraud]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(np.int64)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# --------------------------------------------------------------------------
# Phase 2+: Elliptic transaction graph
# --------------------------------------------------------------------------
def load_elliptic(data_dir=None):
    """
    Expects the three standard Elliptic CSVs in data_dir:
      elliptic_txs_features.csv   (txId, time_step, 165 features...)
      elliptic_txs_classes.csv    (txId, class in {'1'=illicit,'2'=licit,'unknown'})
      elliptic_txs_edgelist.csv   (txId1, txId2)

    Returns a torch_geometric.data.Data object with:
      x            [N, F] node features (time_step included as feature 0)
      edge_index   [2, E]
      y            [N] in {0=licit, 1=illicit, -1=unknown}
      time_step    [N] the Elliptic time step (1..49) each node belongs to
      labeled_mask [N] bool, True where y != -1
    Falls back to a synthetic graph with the same schema + injected concept
    drift if the real CSVs aren't found.
    """
    data_dir = data_dir or os.path.join(RAW_DIR, "elliptic")
    feat_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edge_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    if all(os.path.exists(p) for p in (feat_path, class_path, edge_path)):
        return _load_elliptic_real(feat_path, class_path, edge_path)

    print(f"[dataset] Elliptic CSVs not found in '{data_dir}'. "
          f"Generating synthetic drifted transaction graph instead. "
          f"See data/download_elliptic.py to fetch the real dataset.")
    return generate_synthetic_elliptic()


def _load_elliptic_real(feat_path, class_path, edge_path):
    feats = pd.read_csv(feat_path, header=None)
    feats.columns = ["txId", "time_step"] + [f"f_{i}" for i in range(feats.shape[1] - 2)]

    classes = pd.read_csv(class_path)
    classes.columns = ["txId", "class"]
    class_map = {"1": 1, "2": 0, "unknown": -1}
    classes["y"] = classes["class"].map(class_map)

    merged = feats.merge(classes, on="txId", how="left")
    id_to_idx = {tx_id: i for i, tx_id in enumerate(merged["txId"].values)}

    x = torch.tensor(merged.drop(columns=["txId", "class", "y"]).values, dtype=torch.float)
    y = torch.tensor(merged["y"].fillna(-1).values, dtype=torch.long)
    time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)

    edges = pd.read_csv(edge_path)
    edges.columns = ["txId1", "txId2"]
    src = edges["txId1"].map(id_to_idx).dropna().astype(int).values
    dst = edges["txId2"].map(id_to_idx).dropna().astype(int).values
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    # make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_step
    data.labeled_mask = y != -1
    return data


def generate_synthetic_elliptic(num_timesteps=49, nodes_per_step=400, feat_dim=166,
                                 fraud_ratio=0.1, drift_start_step=30, seed=0):
    """
    Builds a synthetic node-classification graph shaped like Elliptic, with an
    engineered concept drift: illicit-node feature distribution shifts after
    `drift_start_step`, so a model trained only on early steps degrades on
    later ones -- exactly the failure mode Phase 3/4 are meant to expose/fix.
    """
    rng = np.random.default_rng(seed)
    all_x, all_y, all_time, edges = [], [], [], []
    node_offset = 0

    for t in range(1, num_timesteps + 1):
        n = nodes_per_step
        n_fraud = max(1, int(n * fraud_ratio))
        n_normal = n - n_fraud

        normal_feats = rng.normal(0.0, 1.0, size=(n_normal, feat_dim))

        if t < drift_start_step:
            fraud_feats = rng.normal(2.0, 1.0, size=(n_fraud, feat_dim))
        else:
            # concept drift: fraud pattern rotates to a different region of
            # feature space and gets noisier -> old decision boundary fails
            drift_progress = min(1.0, (t - drift_start_step) / 10.0)
            mean_shift = 2.0 + drift_progress * 3.0
            fraud_feats = rng.normal(mean_shift, 1.5 + drift_progress, size=(n_fraud, feat_dim))

        x = np.vstack([normal_feats, fraud_feats])
        y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        perm = rng.permutation(n)
        x, y = x[perm], y[perm]

        # random sparse edges within this time step (transactions connect)
        n_edges = n * 2
        e_src = rng.integers(0, n, size=n_edges) + node_offset
        e_dst = rng.integers(0, n, size=n_edges) + node_offset
        edges.append(np.vstack([e_src, e_dst]))

        all_x.append(x)
        all_y.append(y)
        all_time.append(np.full(n, t))
        node_offset += n

    x = torch.tensor(np.vstack(all_x), dtype=torch.float)
    y = torch.tensor(np.concatenate(all_y), dtype=torch.long)
    time_step = torch.tensor(np.concatenate(all_time), dtype=torch.long)
    edge_index = torch.tensor(np.hstack(edges), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # simulate ~20% of nodes being unlabeled, like real Elliptic
    labeled_mask = torch.rand(len(y)) > 0.2
    y_masked = y.clone()
    y_masked[~labeled_mask] = -1

    data = Data(x=x, edge_index=edge_index, y=y_masked)
    data.time_step = time_step
    data.labeled_mask = labeled_mask
    data.y_true = y  # kept for evaluation only, never used in training
    return data


if __name__ == "__main__":
    d = generate_synthetic_elliptic()
    print(d)
    print("Timesteps:", d.time_step.min().item(), "-", d.time_step.max().item())
    print("Fraud rate (labeled):", d.y[d.labeled_mask].float().mean().item())
