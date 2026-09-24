"""
eval/plot_drift.py -- turn per-chunk metrics into charts for the results
writeup. Called from a training script or a notebook after you've collected
a list of per-chunk metric dicts (what eval.metrics.binary_metrics returns).

Usage from Python:
    from eval.plot_drift import plot_drift_curve, plot_naive_vs_continual

    plot_drift_curve(per_chunk_metrics, metric_key="f1",
                      title="Phase 3: static model under concept drift",
                      out_path="results/phase3_drift.png")

    plot_naive_vs_continual(naive_metrics, continual_metrics, metric_key="f1",
                             out_path="results/phase4_comparison.png")
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless-safe (works in CI, Colab, scripts without a display)
import matplotlib.pyplot as plt


def plot_drift_curve(per_chunk_metrics, metric_key="f1", title=None, out_path="results/drift.png"):
    values = [m[metric_key] for m in per_chunk_metrics]
    chunks = list(range(len(values)))

    plt.figure(figsize=(8, 4.5))
    plt.plot(chunks, values, marker="o", linewidth=2)
    plt.xlabel("Time chunk")
    plt.ylabel(metric_key.upper())
    plt.title(title or f"{metric_key.upper()} across time chunks")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    _save(out_path)


def plot_naive_vs_continual(naive_metrics, continual_metrics, metric_key="f1",
                             title=None, out_path="results/naive_vs_continual.png"):
    naive_vals = [m[metric_key] for m in naive_metrics]
    continual_vals = [m[metric_key] for m in continual_metrics]
    chunks = list(range(len(naive_vals)))

    plt.figure(figsize=(8, 4.5))
    plt.plot(chunks, naive_vals, marker="o", linewidth=2, label="Naive fine-tuning")
    plt.plot(chunks, continual_vals, marker="s", linewidth=2, label="Continual (EWC + replay)")
    plt.xlabel("Time chunk")
    plt.ylabel(metric_key.upper())
    plt.title(title or f"Naive fine-tuning vs continual learning ({metric_key.upper()})")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    _save(out_path)


def _save(out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")