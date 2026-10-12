"""Evaluation metrics and threshold tuning utilities for phishing classification."""

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    auc,
)


def compute_roc_pr_curves(
    y_true: List[int],
    y_scores: List[float]
) -> Dict[str, Any]:
    """Compute ROC and PR curve coordinate points and AUC values."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist(), "auc": float(pr_auc)},
    }


def find_optimal_threshold(
    y_true: List[int],
    y_scores: List[float],
    min_precision: float = 0.95
) -> Dict[str, float]:
    """Find the threshold that maximizes recall given a minimum target precision constraint."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Filter where precision meets requirement
    valid_indices = np.where(precisions[:-1] >= min_precision)[0]
    if len(valid_indices) == 0:
        # Fallback to max F1 threshold
        f1_scores = [f1_score(y_true, [1 if s >= t else 0 for s in y_scores]) for t in thresholds]
        best_idx = int(np.argmax(f1_scores))
        return {
            "optimal_threshold": float(thresholds[best_idx]),
            "achieved_precision": float(precisions[best_idx]),
            "achieved_recall": float(recalls[best_idx]),
            "constraint_met": False,
        }

    # Best recall under precision constraint
    best_idx = valid_indices[np.argmax(recalls[valid_indices])]
    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "achieved_precision": float(precisions[best_idx]),
        "achieved_recall": float(recalls[best_idx]),
        "constraint_met": True,
    }


def compute_confusion_stats(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Calculate TP, FP, TN, FN counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
