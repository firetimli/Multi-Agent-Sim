"""Metric computation.

TRUSTED INFRASTRUCTURE -- this module is the sole definition of "performance".
It imports nothing from src.agents or src.orchestrator, and nothing in Version 0
computes a metric on the test split.

Primary metric: AUROC. Secondary: AUPRC, F1.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

PRIMARY_METRIC = "auroc"


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype="float64")
    n, n_pos = int(len(y_true)), int(y_true.sum())
    out: dict[str, float | int | None] = {
        "n": n,
        "n_positive": n_pos,
        "positive_rate": round(n_pos / n, 6) if n else None,
    }
    if n == 0 or n_pos == 0 or n_pos == n:
        # degenerate split: AUROC undefined. Recorded, not crashed.
        out.update({"auroc": None, "auprc": None, "f1": None,
                    "best_threshold": None, "logloss": None})
        return out

    f1, thr = best_f1(y_true, y_score)
    out.update({
        "auroc": round(float(roc_auc_score(y_true, y_score)), 6),
        "auprc": round(float(average_precision_score(y_true, y_score)), 6),
        "f1": round(float(f1), 6),
        "best_threshold": round(float(thr), 6),
        "logloss": round(float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))), 6),
    })
    return out


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Max F1 over all thresholds, and the threshold achieving it.

    V0 selects the threshold on the same split being reported. When test
    evaluation is added, the threshold must be taken from validation.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    thr = float(thresholds[idx]) if len(thresholds) else 0.5
    return float(f1[idx]), thr


def overfit_gap(train_metrics: dict, val_metrics: dict) -> float | None:
    """train AUROC - val AUROC, surfaced to the Critic as evidence."""
    a, b = train_metrics.get("auroc"), val_metrics.get("auroc")
    if a is None or b is None:
        return None
    return round(a - b, 6)
