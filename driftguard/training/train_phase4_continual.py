"""
training/train_phase4_continual.py -- Phase 4: continual learning so the
model adapts to concept drift WITHOUT forgetting old patterns and WITHOUT
retraining from scratch on the full history every time.

For each time chunk (in order):
  1. Fine-tune the current model on the new chunk, with an EWC penalty that
     discourages forgetting important weights from previous chunks, plus a
     few replayed nodes from a small buffer of past data.
  2. Snapshot Fisher importance for the new chunk (EWC.register_task).
  3. Evaluate the CURRENT model on every chunk seen so far, to report both
     plasticity (does it learn the new chunk?) and backward transfer / lack
     of forgetting (does old-chunk performance hold up?).

Also runs a "naive" baseline (plain fine-tuning, no EWC/replay) side by side
so you can see the difference continual learning makes.

Run:
    python training/train_phase4_continual.py
    python training/train_phase4_continual.py --num_chunks 7 --ewc_lambda 2000
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_elliptic
from models.gnn import FraudGNN
from models.continual import EWC, ReplayBuffer
from eval.metrics import binary_metrics, drift_report
from training.train_phase3_drift_sim import chunk_indices


def evaluate_all_seen(model, data, chunks, upto):
    model.eval()
    with torch.no_grad():
        scores_all = torch.sigmoid(model(data.x, data.edge_index)).cpu().numpy()
    metrics = []
    for i in range(upto + 1):
        idx = chunks[i]
        if len(idx) == 0:
            metrics.append({"precision": 0, "recall": 0, "f1": 0, "roc_auc": float("nan"), "pr_auc": float("nan")})
            continue
        y_true = data.y[idx].cpu().numpy()
        scores = scores_all[idx.cpu().numpy()]
        metrics.append(binary_metrics(y_true, scores))
    return metrics


def train_one_chunk(model, data, train_idx, replay_idx, epochs, lr, ewc=None):
    combined_idx = torch.cat([train_idx, replay_idx]) if len(replay_idx) else train_idx
    y = data.y[combined_idx].float()
    pos_weight = (len(y) - y.sum().clamp(min=1)) / y.sum().clamp(min=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[combined_idx], data.y[combined_idx].float())
        if ewc is not None:
            loss = loss + ewc.penalty()
        loss.backward()
        optimizer.step()
    return loss.item()


def run_strategy(name, data, chunks, use_continual, epochs, lr, ewc_lambda, replay_capacity, device):
    print(f"\n{'='*60}\nStrategy: {name}\n{'='*60}")
    model = FraudGNN(in_dim=data.x.size(1)).to(device)
    ewc = EWC(model, ewc_lambda=ewc_lambda) if use_continual else None
    buffer = ReplayBuffer(capacity=replay_capacity) if use_continual else None

    final_metrics_per_chunk = None
    for i, train_idx in enumerate(chunks):
        if len(train_idx) == 0:
            continue
        replay_idx = torch.tensor(buffer.sample(replay_capacity // 4), dtype=torch.long, device=device) \
            if (buffer is not None and len(buffer) > 0) else torch.tensor([], dtype=torch.long, device=device)

        loss = train_one_chunk(model, data, train_idx, replay_idx, epochs, lr, ewc=ewc)

        if ewc is not None:
            def _loss_fn(idx=train_idx):
                logits = model(data.x, data.edge_index)
                return nn.functional.binary_cross_entropy_with_logits(
                    logits[idx], data.y[idx].float()
                )
            ewc.register_task(_loss_fn)
        if buffer is not None:
            buffer.add(train_idx.cpu().tolist(), chunk_id=i)

        seen_metrics = evaluate_all_seen(model, data, chunks, upto=i)
        avg_f1 = sum(m["f1"] for m in seen_metrics) / len(seen_metrics)
        print(f"chunk {i} | train loss={loss:.4f} | avg f1 over chunks 0-{i}: {avg_f1:.3f}")
        final_metrics_per_chunk = evaluate_all_seen(model, data, chunks, upto=len(chunks) - 1)

    return final_metrics_per_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40, help="epochs PER chunk")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)
    parser.add_argument("--replay_capacity", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_elliptic(args.data_dir).to(device)
    chunks = chunk_indices(data.time_step, data.labeled_mask, args.num_chunks)
    print(f"Chunk sizes: {[len(c) for c in chunks]}")

    naive_final = run_strategy(
        "Naive fine-tuning (no EWC, no replay)", data, chunks,
        use_continual=False, epochs=args.epochs, lr=args.lr,
        ewc_lambda=0, replay_capacity=0, device=device,
    )
    continual_final = run_strategy(
        "Continual learning (EWC + replay)", data, chunks,
        use_continual=True, epochs=args.epochs, lr=args.lr,
        ewc_lambda=args.ewc_lambda, replay_capacity=args.replay_capacity, device=device,
    )

    print("\n\n############ FINAL COMPARISON (all chunks, after last update) ############")
    print("\n--- Naive fine-tuning ---")
    drift_report(naive_final, metric_key="f1")
    print("\n--- Continual (EWC + replay) ---")
    drift_report(continual_final, metric_key="f1")

    naive_avg = sum(m["f1"] for m in naive_final) / len(naive_final)
    continual_avg = sum(m["f1"] for m in continual_final) / len(continual_final)
    print(f"\nAverage F1 across ALL chunks -> naive: {naive_avg:.3f} | "
          f"continual: {continual_avg:.3f} | improvement: {continual_avg - naive_avg:+.3f}")
    print("(Naive fine-tuning typically forgets early chunks as it adapts to "
          "drift; continual learning should retain higher average F1 across "
          "the full timeline -- that's the novelty this project is testing.)")


if __name__ == "__main__":
    main()
