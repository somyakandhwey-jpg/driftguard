"""
eval/metrics.py -- shared metrics used by every training script.
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)


def binary_metrics(y_true, scores, threshold=0.5):
    """
    y_true: 0/1 ground-truth labels (numpy array)
    scores: model output in [0, 1] (probabilities or normalized anomaly scores)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC metrics need both classes present
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def print_metrics(name, m):
    print(f"[{name}] "
          f"precision={m['precision']:.3f} recall={m['recall']:.3f} "
          f"f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}")


def drift_report(per_chunk_metrics, metric_key="f1"):
    """
    per_chunk_metrics: list of dicts (one per time chunk / task), each from
    binary_metrics(). Prints how performance changes over time -- this is the
    core evidence for "concept drift degrades a static model" (Phase 3) and
    "continual learning recovers it" (Phase 4).
    """
    print(f"\n=== Drift report ({metric_key}) ===")
    values = [m[metric_key] for m in per_chunk_metrics]
    for i, v in enumerate(values):
        bar = "#" * int(max(v, 0) * 40)
        print(f"chunk {i:>2}: {v:.3f} {bar}")
    if len(values) > 1:
        print(f"first-chunk {metric_key}: {values[0]:.3f} | "
              f"last-chunk {metric_key}: {values[-1]:.3f} | "
              f"delta: {values[-1] - values[0]:+.3f}")
    return values
