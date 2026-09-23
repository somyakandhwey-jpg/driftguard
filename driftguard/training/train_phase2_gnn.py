"""
training/train_phase2_gnn.py -- Phase 2: convert transactions into a graph
and train a GNN (GraphSAGE) node classifier. Trained/evaluated only on
labeled nodes (Elliptic has ~23% labeled), same as the standard benchmark
protocol.

Run:
    python training/train_phase2_gnn.py
    python training/train_phase2_gnn.py --data_dir data/raw/elliptic
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_elliptic
from models.gnn import FraudGNN
from eval.metrics import binary_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_elliptic(args.data_dir).to(device)
    labeled_idx = data.labeled_mask.nonzero(as_tuple=True)[0]

    # time-based split: train on earlier time steps, test on later ones
    # (mirrors the standard Elliptic protocol and previews Phase 3's drift setup)
    split_t = int(data.time_step[labeled_idx].float().quantile(0.7).item())
    train_idx = labeled_idx[data.time_step[labeled_idx] <= split_t]
    test_idx = labeled_idx[data.time_step[labeled_idx] > split_t]

    model = FraudGNN(in_dim=data.x.size(1), hidden_dim=args.hidden).to(device)

    y_train = data.y[train_idx].float()
    n_pos = y_train.sum().clamp(min=1)
    pos_weight = (len(y_train) - n_pos) / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    print(f"Nodes: {data.num_nodes} | labeled: {len(labeled_idx)} | "
          f"train: {len(train_idx)} | test: {len(test_idx)} | device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_idx], data.y[train_idx].float())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(data.x, data.edge_index))[test_idx].cpu().numpy()
            m = binary_metrics(data.y[test_idx].cpu().numpy(), scores)
            print(f"epoch {epoch:>3} loss={loss.item():.4f}", end="  ")
            print_metrics("test (future timesteps)", m)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/phase2_gnn.pt")
    print("Saved checkpoints/phase2_gnn.pt")


if __name__ == "__main__":
    main()
