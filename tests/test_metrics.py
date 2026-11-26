"""Check metric denominators and the sampling unit with independent small examples."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.metrics import classification_metrics, paired_group_bootstrap


def test_metrics_use_equal_class_importance_and_continuous_scores():
    truth = np.array([0, 0, 0, 0, 1, 1])
    prediction = np.array([0, 0, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.9])
    result = classification_metrics(truth, prediction, scores)
    assert result["confusion_matrix"] == [[3, 1], [1, 1]]
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2) / 2)
    assert result["recall_hmean"] == pytest.approx(0.6)
    assert result["precision_0"] == pytest.approx(3 / 4)
    assert result["precision_1"] == pytest.approx(1 / 2)
    assert result["roc_auc"] == pytest.approx(7 / 8)
    assert result["roc_auc"] != roc_auc_score(truth, prediction)


def test_majority_and_undefined_metrics_are_distinct_from_zero():
    control = classification_metrics([0, 0, 1], [0, 0, 0], [0, 0, 0])
    assert control["balanced_accuracy"] == 0.5
    assert control["roc_auc"] == 0.5
    assert control["recall_hmean"] == 0
    assert control["precision_1"] is None
    assert control["recall_1"] == 0
    one_class = classification_metrics([1, 1], [0, 0], [0.1, 0.2])
    for name in ("balanced_accuracy", "recall_0", "recall_hmean", "roc_auc", "precision_1"):
        assert one_class[name] is None
    assert one_class["precision_0"] == 0
    wrong = classification_metrics([0, 1], [1, 0], [1, 0])
    assert wrong["recall_hmean"] == 0


@pytest.mark.parametrize("truth,prediction,scores", [
    ([], [], []), ([0, 1], [0], [0.1, 0.9]), ([0, 2], [0, 1], [0.1, 0.9]),
    ([0, 1], [0, 1], [0.1, np.nan]), ([0, 1], [0, 1], [0.1, 1.1]),
])
def test_invalid_metric_inputs_fail(truth, prediction, scores):
    with pytest.raises(ValueError):
        classification_metrics(truth, prediction, scores)


def test_cluster_bootstrap_matches_explicit_row_expansion_with_multiplicities():
    groups = np.array(["A", "A", "B", "B", "B", "C", "C", "C", "C"])
    truth = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1])
    model = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0])
    control = np.zeros(len(truth), dtype=int)
    summary, draws = paired_group_bootstrap(truth, model, control, groups, 100, 0.95, 42)
    rng = np.random.default_rng(42)
    expected = []
    for _ in range(100):
        selected_groups = np.unique(groups)[rng.integers(0, 3, size=3)]
        selected_rows = np.concatenate([np.flatnonzero(groups == group) for group in selected_groups])
        expected.append(balanced_accuracy_score(truth[selected_rows], model[selected_rows])
                        - balanced_accuracy_score(truth[selected_rows], control[selected_rows]))
    np.testing.assert_allclose(draws.delta_balanced_accuracy, expected)
    np.testing.assert_allclose([summary["lower"], summary["upper"]], np.quantile(expected, [0.025, 0.975]))
    assert draws.control_balanced_accuracy.eq(0.5).all()
    assert summary["delta_balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, model) - 0.5)
    repeated, repeated_draws = paired_group_bootstrap(truth, model, control, groups, 100, 0.95, 42)
    assert repeated == summary
    pd.testing.assert_frame_equal(draws, repeated_draws)


def test_one_class_resample_is_not_silently_dropped():
    with pytest.raises(ValueError, match="no replicates were discarded"):
        paired_group_bootstrap([0, 1], [0, 1], [0, 0], ["A", "B"], 100, 0.95, 42)
