"""tests/test_metrics.py -- checks for eval/metrics.py."""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.metrics import binary_metrics, drift_report


def test_binary_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.9, 0.8]
    m = binary_metrics(y_true, scores, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_binary_metrics_worst_case():
    y_true = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.1, 0.2]  # completely inverted
    m = binary_metrics(y_true, scores, threshold=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_binary_metrics_single_class_no_crash():
    # only one class present -> AUC undefined, should be NaN not a crash
    y_true = [0, 0, 0, 0]
    scores = [0.1, 0.4, 0.2, 0.3]
    m = binary_metrics(y_true, scores)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])


def test_drift_report_returns_values_in_order():
    metrics = [{"f1": 0.8}, {"f1": 0.6}, {"f1": 0.3}]
    values = drift_report(metrics, metric_key="f1")
    assert values == [0.8, 0.6, 0.3]
