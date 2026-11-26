"""Binary diagnostics and paired resampling of complete ticket groups."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def binary_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or prediction.shape != truth.shape or not truth.size:
        raise ValueError("Metrics need nonempty, aligned one-dimensional arrays")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Labels must be binary")
    return truth.astype(int), prediction.astype(int)


def classification_metrics(y_true, y_pred, p_survived) -> dict:
    truth, prediction = binary_arrays(y_true, y_pred)
    scores = np.asarray(p_survived, dtype=float)
    if scores.shape != truth.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite, aligned and between zero and one")
    matrix = np.bincount(2 * truth + prediction, minlength=4).reshape(2, 2)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    recalls = [float(matrix[c, c] / support[c]) if support[c] else None for c in (0, 1)]
    precisions = [float(matrix[c, c] / predicted[c]) if predicted[c] else None for c in (0, 1)]
    both = bool((support > 0).all())
    harmonic = None
    if both:
        harmonic = 2 * recalls[0] * recalls[1] / sum(recalls) if sum(recalls) else 0.0
    return {
        "n": int(truth.size), "n_0": int(support[0]), "n_1": int(support[1]),
        "errors": int((truth != prediction).sum()),
        "accuracy": float((truth == prediction).mean()),
        "balanced_accuracy": float(sum(recalls) / 2) if both else None,
        "recall_0": recalls[0], "recall_1": recalls[1],
        "precision_0": precisions[0], "precision_1": precisions[1],
        "recall_hmean": harmonic,
        "roc_auc": float(roc_auc_score(truth, scores)) if both else None,
        "confusion_matrix": matrix.tolist(),
    }


def paired_group_bootstrap(y_true, model_prediction, control_prediction, groups,
                           n_resamples: int, confidence_level: float, seed: int) -> tuple[dict, pd.DataFrame]:
    """Estimate a percentile interval for the paired balanced-accuracy difference.

    Models stay fixed. Each draw samples the original number of groups with
    replacement and retains every member and every selected multiplicity.
    An undefined replicate aborts the interval instead of silently conditioning
    on a different resampling distribution.
    """
    truth, model = binary_arrays(y_true, model_prediction)
    _, control = binary_arrays(y_true, control_prediction)
    groups = np.asarray(groups)
    if groups.shape != truth.shape or pd.isna(groups).any():
        raise ValueError("Every observation needs a nonmissing group")
    if n_resamples < 2 or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap settings")
    _, membership = np.unique(groups, return_inverse=True)
    n_groups = int(membership.max()) + 1
    # Count correct and total predictions for each class within each group.
    counts = np.zeros((n_groups, 6), dtype=int)
    for cls in (0, 1):
        mask = truth == cls
        np.add.at(counts[:, cls], membership[mask], 1)
        np.add.at(counts[:, 2 + cls], membership[mask & (model == cls)], 1)
        np.add.at(counts[:, 4 + cls], membership[mask & (control == cls)], 1)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_resamples, 2))
    for repetition in range(n_resamples):
        selected = rng.integers(0, n_groups, size=n_groups)
        total = counts[selected].sum(axis=0)
        if not (total[:2] > 0).all():
            raise ValueError(
                f"Bootstrap replicate {repetition} lacks a true class; "
                "no interval was computed and no replicates were discarded"
            )
        samples[repetition] = [
            np.mean(total[2:4] / total[:2]), np.mean(total[4:6] / total[:2]),
        ]
    differences = samples[:, 0] - samples[:, 1]
    tail = (1 - confidence_level) / 2
    bounds = np.quantile(differences, [tail, 1 - tail], method="linear")
    totals = counts.sum(axis=0)
    point = float(np.mean(totals[2:4] / totals[:2]) - np.mean(totals[4:6] / totals[:2]))
    summary = {
        "method": "paired_ticket_group_percentile", "confidence_level": confidence_level,
        "n_resamples": n_resamples, "n_groups": n_groups, "seed": seed,
        "invalid_resamples": 0, "delta_balanced_accuracy": point,
        "lower": float(bounds[0]), "upper": float(bounds[1]),
    }
    draws = pd.DataFrame({
        "replicate": np.arange(n_resamples), "model_balanced_accuracy": samples[:, 0],
        "control_balanced_accuracy": samples[:, 1], "delta_balanced_accuracy": differences,
    })
    return summary, draws
