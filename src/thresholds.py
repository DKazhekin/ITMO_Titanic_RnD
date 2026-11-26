"""Choose one threshold by the equally weighted mean of held-out fold scores."""

import numpy as np
import pandas as pd

from src.metrics import binary_arrays


def predict_at_threshold(probabilities, threshold: float) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Probabilities must be a finite nonempty vector")
    if ((scores < 0) | (scores > 1)).any() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Probabilities and threshold must be in [0, 1]")
    return (scores > threshold).astype(int)


def threshold_table(truth, probabilities, folds, n_folds: int = 5) -> pd.DataFrame:
    """Score every decision boundary, retaining individual fold scores."""
    scores = np.asarray(probabilities, dtype=float)
    target, _ = binary_arrays(truth, predict_at_threshold(scores, 0.5))
    membership = np.asarray(folds)
    if membership.shape != target.shape or set(membership) != set(range(n_folds)):
        raise ValueError("Every declared fold must be present and aligned with the labels")
    candidates = np.unique(np.concatenate((scores, [0.0, 0.5, 1.0])))
    predictions = scores[:, None] > candidates[None, :]
    result = {"threshold": candidates}
    values = []
    for fold in range(n_folds):
        mask = membership == fold
        if set(target[mask]) != {0, 1}:
            raise ValueError("Every validation fold must contain both classes")
        recalls = [(predictions[mask & (target == cls)] == cls).mean(axis=0) for cls in (0, 1)]
        balanced = (recalls[0] + recalls[1]) / 2
        result[f"fold_{fold}_balanced_accuracy"] = balanced
        values.append(balanced)
    result["mean_balanced_accuracy"] = np.mean(values, axis=0)
    result["std_balanced_accuracy"] = np.std(values, axis=0, ddof=0)
    return pd.DataFrame(result)


def select_threshold(table: pd.DataFrame) -> float:
    """Break exact score ties by distance to 0.5, then the lower threshold."""
    columns = table[["threshold", "mean_balanced_accuracy"]].to_numpy(dtype=float)
    if not len(columns) or not np.isfinite(columns).all():
        raise ValueError("Threshold selection requires finite candidate scores")
    return float(min(columns, key=lambda row: (-row[1], abs(row[0] - 0.5), row[0]))[0])
