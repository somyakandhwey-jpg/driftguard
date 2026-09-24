"""
inference/predict.py -- load a trained checkpoint and score transactions for
fraud risk. Works with either the Phase 2 supervised GNN or the Phase 5
unsupervised diffusion detector.

This is the "demo" entry point: point it at a checkpoint + a graph, get back
a ranked list of the most suspicious transactions.

Examples:
    # Score the built-in (synthetic or real, whichever is available) graph
    # with the Phase 2 supervised GNN checkpoint:
    python inference/predict.py --phase 2 --checkpoint checkpoints/phase2_gnn.pt --top_k 15

    # Same, but with the Phase 5 unsupervised diffusion detector:
    python inference/predict.py --phase 5 --checkpoint checkpoints/phase5_diffusion.pt --top_k 15

If the checkpoint doesn't exist yet, this script tells you exactly which
training script to run first instead of failing silently.
"""
import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_elliptic
from models.gnn import FraudGNN
from models.diffusion import GraphDiffusionModel


TRAIN_SCRIPT_HINT = {
    2: "training/train_phase2_gnn.py",
    5: "training/train_phase5_diffusion.py",
}


def load_phase2(checkpoint_path, data, device):
    model = FraudGNN(in_dim=data.x.size(1)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(data.x, data.edge_index)).cpu()
    return scores


def load_phase5(checkpoint_path, data, device, probe_t=50):
    diffusion = GraphDiffusionModel(feat_dim=data.x.size(1), device=device)
    diffusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
    scores = diffusion.anomaly_score(data.x, data.edge_index, t_probe=probe_t)
    # normalize to [0, 1] so both phases print comparably
    lo, hi = scores.min(), scores.max()
    if hi - lo > 1e-9:
        scores = (scores - lo) / (hi - lo)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[2, 5], required=True,
                         help="Which trained model to use: 2 (supervised GNN) or 5 (unsupervised diffusion)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10, help="How many most-suspicious nodes to print")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint found at '{args.checkpoint}'.")
        print(f"Train one first with:\n    python {TRAIN_SCRIPT_HINT[args.phase]}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_elliptic(args.data_dir).to(device)

    if args.phase == 2:
        scores = load_phase2(args.checkpoint, data, device)
    else:
        scores = load_phase5(args.checkpoint, data, device)

    top_idx = torch.argsort(scores, descending=True)[: args.top_k]

    print(f"\nTop {args.top_k} most suspicious transactions (phase {args.phase} model):")
    print(f"{'node_idx':>10} {'time_step':>10} {'fraud_score':>12} {'true_label':>11}")
    for idx in top_idx.tolist():
        true_label = data.y[idx].item()
        label_str = {1: "fraud", 0: "legit", -1: "unknown"}[true_label]
        print(f"{idx:>10} {data.time_step[idx].item():>10} {scores[idx].item():>12.4f} {label_str:>11}")

    print("\n(true_label is shown for reference only -- the model never sees "
          "it at inference time; for phase 5 it never saw it during training either.)")


if __name__ == "__main__":
    main()
