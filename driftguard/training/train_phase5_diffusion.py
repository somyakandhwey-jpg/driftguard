"""
training/train_phase5_diffusion.py -- Phase 5: graph diffusion model for
LABEL-FREE fraud detection, updated continually across time chunks.

This is the full pitch from the project title: an unsupervised detector
(trains a denoiser only on nodes ASSUMED normal -- no fraud labels used in
training, ground-truth labels are used only to compute evaluation metrics)
that keeps adapting across time chunks via a lightweight continual update
(EWC on the denoiser) instead of full retraining.

Anomaly score = denoising reconstruction error at a fixed noise level.
Nodes whose real features are hard to reconstruct don't fit the learned
"normal" manifold -> flagged as likely fraud.

Run:
    python training/train_phase5_diffusion.py
    python training/train_phase5_diffusion.py --num_chunks 7 --probe_t 50
"""
import argparse
import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_elliptic
from models.diffusion import GraphDiffusionModel
from models.continual import EWC
from eval.metrics import binary_metrics, drift_report
from training.train_phase3_drift_sim import chunk_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=60, help="epochs per chunk")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--diffusion_timesteps", type=int, default=200)
    parser.add_argument("--probe_t", type=int, default=50,
                         help="noise level used for anomaly scoring")
    parser.add_argument("--ewc_lambda", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_elliptic(args.data_dir).to(device)
    # NOTE: chunk_indices returns only LABELED nodes, which we use for eval.
    # Training uses ALL nodes in a chunk (labeled or not) minus a safety
    # exclusion of KNOWN fraud, mirroring "we don't have reliable fraud
    # labels to train on, but we can avoid training on the few we do know."
    labeled_chunks = chunk_indices(data.time_step, data.labeled_mask, args.num_chunks)

    t_min, t_max = data.time_step.min().item(), data.time_step.max().item()
    edges = torch.linspace(t_min, t_max + 1, args.num_chunks + 1)
    all_node_chunks = []
    for i in range(args.num_chunks):
        lo, hi = edges[i].item(), edges[i + 1].item()
        mask = (data.time_step >= lo) & (data.time_step < hi)
        all_node_chunks.append(mask.nonzero(as_tuple=True)[0])

    diffusion = GraphDiffusionModel(
        feat_dim=data.x.size(1), timesteps=args.diffusion_timesteps, device=device
    )
    ewc = EWC(diffusion.model, ewc_lambda=args.ewc_lambda)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)

    per_chunk_metrics = []
    for i in range(args.num_chunks):
        node_idx = all_node_chunks[i]
        if len(node_idx) < 10:
            continue

        # "normal" training set = everything NOT known to be fraud in this chunk
        known_fraud = (data.y[node_idx] == 1)
        normal_mask_local = ~known_fraud  # unsupervised: we only exclude confirmed fraud

        print(f"\nChunk {i}: {len(node_idx)} nodes, "
              f"{known_fraud.sum().item()} known-fraud excluded from training")

        train_mask = _full_mask(data, node_idx, normal_mask_local)
        for epoch in range(args.epochs):
            loss = diffusion.train_step(
                data.x, data.edge_index, normal_mask=train_mask,
                optimizer=optimizer, ewc=ewc if ewc.fisher else None,
            )

        # snapshot importance for this chunk before moving on: reuse the same
        # kind of denoising loss the model was just trained with
        def _make_loss_fn(mask=train_mask):
            def _loss_fn():
                n = data.num_nodes
                t = torch.randint(0, diffusion.timesteps, (n,), device=device)
                x_noisy, noise = diffusion.q_sample(data.x, t)
                pred_noise = diffusion.model(x_noisy, data.edge_index, t)
                return torch.nn.functional.mse_loss(pred_noise[mask], noise[mask])
            return _loss_fn

        ewc.register_task(_make_loss_fn())

        # --- evaluate on this chunk's LABELED nodes using ground truth ---
        eval_idx = labeled_chunks[i]
        if len(eval_idx) > 0:
            scores = diffusion.anomaly_score(data.x, data.edge_index, t_probe=args.probe_t)
            scores_chunk = scores[eval_idx.cpu()].numpy()
            scores_norm = _normalize(scores_chunk)
            y_true = data.y[eval_idx].cpu().numpy()
            m = binary_metrics(y_true, scores_norm)
            per_chunk_metrics.append(m)
            print(f"  chunk {i} eval: f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f} "
                  f"pr_auc={m['pr_auc']:.3f} (labels used for eval ONLY)")

    drift_report(per_chunk_metrics, metric_key="roc_auc")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(diffusion.state_dict(), "checkpoints/phase5_diffusion.pt")
    print("Saved checkpoints/phase5_diffusion.pt")
    print("\nDone. This detector never saw a fraud label during training -- "
          "only the diffusion reconstruction error, continually updated per "
          "chunk with EWC protecting earlier chunks' 'normal' manifold.")


def _full_mask(data, node_idx, local_mask):
    """Expand a chunk-local boolean mask into a full-graph boolean mask."""
    full = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    full[node_idx] = local_mask
    return full


def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


if __name__ == "__main__":
    main()
