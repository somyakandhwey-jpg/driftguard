"""
training/train_phase3_drift_sim.py -- Phase 3: simulate concept drift and show
that a model trained once on early time steps degrades on later ones. This is
the "problem" the rest of the project (Phase 4) solves.

Splits time steps into chunks, trains a single static GNN on the FIRST chunk
only, then evaluates that same frozen model on every subsequent chunk.
Produces a drift report where performance visibly drops after the injected
drift point.

Run:
    python training/train_phase3_drift_sim.py
    python training/train_phase3_drift_sim.py --num_chunks 7
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_elliptic
from models.gnn import FraudGNN
from eval.metrics import binary_metrics, drift_report


def chunk_indices(time_step, labeled_mask, num_chunks):
    """Split the full time range into num_chunks contiguous, roughly equal
    chunks and return (labeled) node indices per chunk."""
    t_min, t_max = time_step.min().item(), time_step.max().item()
    edges = torch.linspace(t_min, t_max + 1, num_chunks + 1)
    chunks = []
    for i in range(num_chunks):
        lo, hi = edges[i].item(), edges[i + 1].item()
        mask = (time_step >= lo) & (time_step < hi) & labeled_mask
        chunks.append(mask.nonzero(as_tuple=True)[0])
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_elliptic(args.data_dir).to(device)
    chunks = chunk_indices(data.time_step, data.labeled_mask, args.num_chunks)
    print(f"Chunk sizes: {[len(c) for c in chunks]}")

    # --- train ONLY on chunk 0 ---
    train_idx = chunks[0]
    model = FraudGNN(in_dim=data.x.size(1)).to(device)
    y_train = data.y[train_idx].float()
    pos_weight = (len(y_train) - y_train.sum().clamp(min=1)) / y_train.sum().clamp(min=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    print(f"\nTraining STATIC model on chunk 0 only ({len(train_idx)} nodes)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_idx], data.y[train_idx].float())
        loss.backward()
        optimizer.step()
    print(f"Final train loss: {loss.item():.4f}")

    # --- evaluate the SAME frozen model on every chunk, including future ones ---
    model.eval()
    with torch.no_grad():
        scores_all = torch.sigmoid(model(data.x, data.edge_index)).cpu().numpy()

    per_chunk_metrics = []
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        y_true = data.y[idx].cpu().numpy()
        scores = scores_all[idx.cpu().numpy()]
        m = binary_metrics(y_true, scores)
        per_chunk_metrics.append(m)
        tag = "(train chunk)" if i == 0 else ""
        print(f"chunk {i}: f1={m['f1']:.3f} precision={m['precision']:.3f} "
              f"recall={m['recall']:.3f} roc_auc={m['roc_auc']:.3f} {tag}")

    drift_report(per_chunk_metrics, metric_key="f1")
    print("\nThis performance drop across chunks IS the concept-drift problem. "
          "Phase 4 (train_phase4_continual.py) fixes it via EWC + replay "
          "without retraining from scratch.")


if __name__ == "__main__":
    main()
