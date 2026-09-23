# DriftGuard

**Continual Graph Diffusion Models for Adaptive Financial Fraud Detection Under Concept Drift**

An unsupervised fraud detector using graph diffusion models that adapts over
time via continual learning — no fraud labels needed for training, and no
full retraining needed as fraud patterns shift.

- **Problem:** fraud detection needs labeled fraud data (scarce), and fraud
  patterns change over time — concept drift makes static models go stale.
- **Idea:** model what *normal* transactions look like with a graph diffusion
  model, and flag anything that doesn't reconstruct well.
- **Novelty:** continual learning (EWC + replay) lets the model adapt to new
  behavior without forgetting old patterns or retraining from scratch.
- **Cost:** $0 — Elliptic dataset (free), Colab free T4 GPU, PyTorch
  Geometric, all open source.

## Project structure

```
driftguard/
├── configs/       per-phase YAML hyperparameters
├── data/          dataset loaders + Kaggle download helper
├── eval/          shared metrics (precision/recall/F1/AUC + drift report)
├── models/        MLP, GNN, EWC/replay buffer, graph diffusion model
├── notebooks/      Colab quickstart notebook
├── training/       one training script per phase
└── checkpoints/    saved model weights (gitignored)
```

## Build order (MVP-first)

| Phase | What it does | Script |
|---|---|---|
| 1 | Simple supervised MLP fraud classifier on tabular transactions (baseline) | `training/train_phase1_mlp.py` |
| 2 | Convert transactions into a graph, train a GraphSAGE GNN classifier | `training/train_phase2_gnn.py` |
| 3 | Simulate concept drift, show a static model's performance degrade over time | `training/train_phase3_drift_sim.py` |
| 4 | Continual learning (EWC + replay buffer) so the model adapts without forgetting | `training/train_phase4_continual.py` |
| 5 | Graph diffusion model for **label-free** anomaly scoring, updated continually | `training/train_phase5_diffusion.py` |

Every script runs **out of the box on synthetic data** (generated on the fly,
same schema as the real dataset, with concept drift baked in for phases 3-5)
so you can build and demo the whole pipeline before downloading anything.

## Setup

```bash
git clone https://github.com/<your-username>/driftguard.git
cd driftguard
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

On Colab, open `notebooks/colab_quickstart.ipynb`, set the runtime to a T4
GPU, and run the cells top to bottom.

## Using the real Elliptic dataset (optional, still free)

1. Get a Kaggle API token (Account → Create New API Token → downloads
   `kaggle.json`).
2. `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. `pip install kaggle`
4. `python data/download_elliptic.py`

This drops the three Elliptic CSVs into `data/raw/elliptic/`. Every script
auto-detects them via `--data_dir data/raw/elliptic` (or just leave the
default — the loader checks that path automatically).

## Running each phase

```bash
# Phase 1: tabular MLP baseline
python training/train_phase1_mlp.py

# Phase 2: GNN on the transaction graph
python training/train_phase2_gnn.py

# Phase 3: watch a static model degrade under simulated drift
python training/train_phase3_drift_sim.py --num_chunks 7

# Phase 4: continual learning fixes it (compared against naive fine-tuning)
python training/train_phase4_continual.py --num_chunks 7 --ewc_lambda 1000

# Phase 5: full pitch — unsupervised graph diffusion detector, updated
# continually across time chunks, no fraud labels used in training
python training/train_phase5_diffusion.py --num_chunks 7
```

Each script prints per-time-chunk precision/recall/F1/ROC-AUC/PR-AUC and a
"drift report" so you can see performance over time directly in the
terminal/notebook output.

## How the pieces fit together

- `data/dataset.py` — loads Elliptic (or generates a synthetic drop-in
  replacement with injected concept drift) as a single `torch_geometric.data.Data`
  graph with a `time_step` per node.
- `models/gnn.py` — GraphSAGE encoder + classifier, used for the supervised
  Phases 2-4, and shares its embedding logic conceptually with the diffusion
  denoiser in Phase 5.
- `models/continual.py` — `EWC` (penalizes drift in important weights) and
  `ReplayBuffer` (reservoir sample of past nodes), used in Phases 4 and 5.
- `models/diffusion.py` — `GraphDiffusionModel`: a denoising diffusion model
  over node features, conditioned on graph structure via message passing.
  Trained only on data *assumed normal* (no fraud labels). Anomaly score =
  reconstruction error at a fixed noise level.
- `eval/metrics.py` — shared metric + drift-report utilities used by every
  script above.

## Roadmap / stretch goals

- [ ] Swap GraphSAGE for a graph-transformer encoder
- [ ] Try edge-level diffusion in addition to node-feature diffusion
- [ ] Add a Slack/webhook alert hook for high anomaly-score transactions
- [ ] Benchmark against IEEE-CIS if compute budget allows

## License

MIT (or update to match your repo's existing `.gitignore`/license setup).
