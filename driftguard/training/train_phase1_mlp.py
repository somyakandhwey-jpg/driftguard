"""
training/train_phase1_mlp.py -- Phase 1 MVP: simple supervised fraud
classifier on tabular transactions. Establishes a baseline before we move to
graphs (Phase 2), drift (Phase 3), continual learning (Phase 4) and the
unsupervised diffusion detector (Phase 5).

Run:
    python training/train_phase1_mlp.py
    python training/train_phase1_mlp.py --data data/raw/creditcard.csv
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_creditcard
from models.mlp import FraudMLP
from eval.metrics import binary_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                         help="Path to Kaggle creditcard.csv; omit to use synthetic data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_creditcard(csv_path=args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=args.seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    model = FraudMLP(in_dim=X_train.shape[1], hidden_dim=args.hidden).to(device)

    # class imbalance is extreme (~0.17% fraud) -> weight the positive class
    n_pos = y_train_t.sum().clamp(min=1)
    pos_weight = (len(y_train_t) - n_pos) / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training on {len(X_train)} rows ({device}), "
          f"fraud rate: {y_train.mean():.4%}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(X_test_t)).cpu().numpy()
            m = binary_metrics(y_test, scores)
            print(f"epoch {epoch:>3} loss={loss.item():.4f}", end="  ")
            print_metrics("test", m)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/phase1_mlp.pt")
    print("Saved checkpoints/phase1_mlp.pt")


if __name__ == "__main__":
    main()
