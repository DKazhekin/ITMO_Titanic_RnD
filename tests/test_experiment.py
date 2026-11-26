"""Protect group boundaries, train-only preprocessing and validation selection."""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from src.experiment import (
    CATEGORICAL, NUMERIC, build_pipeline, feature_frame, make_splits, read_settings,
    rows_by_id, validate_splits,
)
from src.train import fit_converged, select_candidate, summarize_candidate


@pytest.fixture
def config():
    return read_settings(Path(__file__).resolve().parents[1] / "configs/baseline.yaml")


@pytest.fixture
def passenger_frame():
    rows = []
    for group in range(30):
        for member in range(3):
            number = group * 3 + member + 1
            rows.append({
                "PassengerId": number, "Survived": int(member == 2),
                "Pclass": group % 3 + 1, "Name": f"Person {number}",
                "Sex": "female" if (group + member) % 3 == 0 else "male",
                "Age": float(10 + number) if number % 13 else np.nan,
                "SibSp": group % 2, "Parch": member % 2,
                "Ticket": f"000{group}", "Fare": float(group + 1),
                "Cabin": np.nan, "Embarked": ["S", "C", "Q"][group % 3],
            })
    return pd.DataFrame(rows)


def test_group_split_matches_fixed_outer_fold_and_fresh_inner_folds(config, passenger_frame):
    frame = passenger_frame
    splits = make_splits(frame, config)
    assert splits == make_splits(frame, config)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    development, test = next(outer.split(frame, frame.Survived, frame.Ticket))
    assert splits["test_ids"] == frame.iloc[test].PassengerId.tolist()
    dev = frame.iloc[development]
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    observed_validation = []
    for actual, (left, right) in zip(splits["inner_folds"], inner.split(dev, dev.Survived, dev.Ticket)):
        assert actual["train_ids"] == dev.iloc[left].PassengerId.tolist()
        assert actual["validation_ids"] == dev.iloc[right].PassengerId.tolist()
        assert set(actual["train_ids"] + actual["validation_ids"]).isdisjoint(splits["test_ids"])
        assert set(rows_by_id(frame, actual["train_ids"]).Ticket).isdisjoint(
            rows_by_id(frame, actual["validation_ids"]).Ticket
        )
        observed_validation.extend(actual["validation_ids"])
    assert sorted(observed_validation) == sorted(splits["development_ids"])


def test_corrupted_split_cannot_reuse_a_test_passenger(config, passenger_frame):
    splits = make_splits(passenger_frame, config)
    splits["inner_folds"][0]["train_ids"][0] = splits["test_ids"][0]
    with pytest.raises(ValueError, match="cover each passenger"):
        validate_splits(passenger_frame, splits)


def test_pipeline_uses_training_statistics_and_handles_unseen_valid_categories(config):
    raw = pd.DataFrame({
        "Age": [20, np.nan, 40, 60], "Fare": [1, 3, np.nan, 9],
        "SibSp": [0, 1, 0, 1], "Parch": [0, 0, 0, 0],
        "Sex": ["male", "female", "male", "female"], "Pclass": [1, 2, 1, 2],
        "Embarked": ["S", pd.NA, "S", "C"], "Survived": [0, 1, 0, 1],
        "Ticket": ["A", "B", "C", "D"], "PassengerId": [1, 2, 3, 4],
    })
    training = feature_frame(raw)
    assert training.columns.tolist() == NUMERIC + CATEGORICAL
    model = fit_converged(build_pipeline(config, 1, "balanced"), training, raw.Survived.to_numpy())
    prepared = model.named_steps["prepare"]
    numeric = prepared.named_transformers_["numeric"]
    categorical = prepared.named_transformers_["categorical"]
    np.testing.assert_allclose(numeric.named_steps["impute"].statistics_, [40, 3, 0.5, 0])
    np.testing.assert_allclose(numeric.named_steps["scale"].mean_, [40, 4, 0.5, 0])
    np.testing.assert_allclose(numeric.named_steps["scale"].scale_, [np.sqrt(200), 3, 0.5, 1])
    assert categorical.named_steps["impute"].statistics_[2] == "S"
    categories = categorical.named_steps["encode"].categories_
    assert [list(values) for values in categories] == [["female", "male"], [1, 2], ["C", "S"]]
    validation = raw.iloc[:1].copy()
    validation.loc[:, "Age"] = 10000
    validation.loc[:, "Fare"] = 1000000
    validation.loc[:, "Pclass"] = 3
    validation.loc[:, "Embarked"] = "Q"
    validation = feature_frame(validation)
    transformed = prepared.transform(validation)
    assert transformed.shape == (1, 10)
    np.testing.assert_array_equal(transformed[0, 4:], [0, 1, 0, 0, 0, 0])
    assert numeric.named_steps["scale"].n_samples_seen_ == 4
    np.testing.assert_allclose(numeric.named_steps["scale"].mean_, [40, 4, 0.5, 0])
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    restored = joblib.load(io.BytesIO(buffer.getvalue()))
    np.testing.assert_array_equal(model.predict(validation), restored.predict(validation))
    np.testing.assert_array_equal(model.predict_proba(validation), restored.predict_proba(validation))
    assert model.named_steps["model"].l1_ratio == 0
    assert model.named_steps["model"].fit_intercept
    # At exactly zero log-odds, sklearn's binary predict selects class zero.
    model.named_steps["model"].coef_[:] = 0
    model.named_steps["model"].intercept_[:] = 0
    assert model.predict(validation).tolist() == [0]
    assert model.predict_proba(validation)[0, 1] == 0.5


def test_selection_is_unweighted_and_ties_use_c_then_no_weights():
    def summary(c, weight, scores):
        rows = [dict(balanced_accuracy=s, recall_0=s, recall_1=s, n_validation=n)
                for s, n in zip(scores, [1, 100])]
        return summarize_candidate(rows, "logistic_regression", c, weight)

    first, second = summary(1, "balanced", [0.9, 0.5]), summary(10, None, [0.6, 0.7])
    assert first["mean_balanced_accuracy"] == pytest.approx(0.7)
    assert first["std_balanced_accuracy"] == pytest.approx(0.2)
    assert select_candidate([second, first]) == first
    tied = [summary(10, None, [0.8, 0.8]), summary(1, "balanced", [0.8, 0.8]),
            summary(1, None, [0.8, 0.8])]
    assert select_candidate(tied) == tied[2]


def test_nonconvergence_is_an_error(config, monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.pipeline import Pipeline

    def nonconvergent(*args, **kwargs):
        warnings.warn("No convergence", ConvergenceWarning)

    monkeypatch.setattr(Pipeline, "fit", nonconvergent)
    with pytest.raises(ConvergenceWarning):
        fit_converged(build_pipeline(config, 1, None), pd.DataFrame({"Age": [1, 2]}), np.array([0, 1]))
