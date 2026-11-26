"""Protect interaction, threshold boundaries, fold weighting and local tracking."""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from src.experiment import build_pipeline, read_settings, write_csv
from src.features import AgeMaleInteraction
from src.hw2 import load_series, read_csv, read_hw2_settings, save_manifest, source_hashes, validation_metrics, versions
from src.thresholds import predict_at_threshold, select_threshold, threshold_table
from src.tracking import LocalTracking


ROOT = Path(__file__).resolve().parents[1]


def test_interaction_imputes_raw_age_using_only_training_rows():
    train = pd.DataFrame({"Age": [10.0, 20.0, 30.0, np.nan],
                          "Sex": ["male", "female", "male", "female"]})
    transformer = AgeMaleInteraction().fit(train)
    np.testing.assert_array_equal(transformer.transform(train).ravel(), [10, 0, 30, 0])
    unseen = pd.DataFrame({"Age": [np.nan, np.nan, 1000], "Sex": ["male", "female", "male"]})
    np.testing.assert_array_equal(transformer.transform(unseen).ravel(), [20, 0, 1000])
    assert transformer.age_median_ == 20
    assert np.isnan(train.Age.iloc[3])
    with pytest.raises(ValueError, match="finite training"):
        AgeMaleInteraction().fit(train.assign(Age=np.nan))


def test_added_branch_preserves_original_features_and_serializes():
    config = read_settings(ROOT / "configs/baseline.yaml")
    frame = pd.DataFrame({
        "Age": [10.0, np.nan, 30.0, 60.0], "Fare": [1, 3, 5, 7],
        "SibSp": [0, 1, 0, 1], "Parch": [0, 0, 0, 0],
        "Sex": ["male", "female", "male", "female"], "Pclass": [1, 2, 1, 2],
        "Embarked": ["S", "S", "S", "C"],
    })
    basic = build_pipeline(config, 1, None).fit(frame, [0, 1, 0, 1])
    extended = build_pipeline(config, 1, None, interaction=True).fit(frame, [0, 1, 0, 1])
    a = basic.named_steps["prepare"].transform(frame)
    b = extended.named_steps["prepare"].transform(frame)
    np.testing.assert_array_equal(a, b[:, :-1])
    branch = extended.named_steps["prepare"].named_transformers_["age_male"]
    np.testing.assert_array_equal(branch.named_steps["interact"].transform(frame).ravel(), [10, 0, 30, 0])
    assert branch.named_steps["scale"].mean_[0] == 10
    buffer = io.BytesIO()
    joblib.dump(extended, buffer)
    restored = joblib.load(io.BytesIO(buffer.getvalue()))
    np.testing.assert_array_equal(extended.predict_proba(frame), restored.predict_proba(frame))


def test_threshold_candidates_cover_strict_boundaries_without_rounding():
    probabilities = np.array([0, 0.5, np.nextafter(0.5, 1), 1, 0.2, 0.8])
    truth, folds = [0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1]
    table = threshold_table(truth, probabilities, folds, n_folds=2)
    np.testing.assert_array_equal(table.threshold, np.unique(np.r_[probabilities, 0, 0.5, 1]))
    np.testing.assert_array_equal(predict_at_threshold(probabilities, 0.5), [0, 0, 1, 1, 0, 1])
    for threshold in table.threshold:
        expected = []
        for fold in (0, 1):
            mask = np.array(folds) == fold
            expected.append(balanced_accuracy_score(np.array(truth)[mask], probabilities[mask] > threshold))
        row = table.loc[table.threshold == threshold].iloc[0]
        assert row.mean_balanced_accuracy == np.mean(expected)


def test_fold_means_are_not_pooled_or_weighted_by_size():
    frame = pd.DataFrame({"Survived": [0, 1, 0, 0, 0, 1, 1, 1],
                          "probability": [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
                          "fold": [0, 0, 1, 1, 1, 1, 1, 1]})
    table = threshold_table(frame.Survived, frame.probability, frame.fold, n_folds=2)
    assert table.loc[table.threshold == 0.5, "mean_balanced_accuracy"].item() == 0.5
    assert balanced_accuracy_score(frame.Survived, frame.probability > 0.5) == 0.25
    assert validation_metrics(frame, 0.5)["summary"]["balanced_accuracy_mean"] == 0.5


def test_ties_use_exact_score_then_distance_then_lower_threshold():
    table = pd.DataFrame({"threshold": [0.75, 0.25, 0.5], "mean_balanced_accuracy": [0.8, 0.8, 0.8]})
    assert select_threshold(table) == 0.5
    assert select_threshold(table.iloc[:2]) == 0.25
    table.loc[0, "mean_balanced_accuracy"] = np.nextafter(0.8, 1)
    assert select_threshold(table) == 0.75


@pytest.mark.parametrize("scores,threshold", [([np.nan], 0.5), ([1.1], 0.5), ([0.1], np.nan)])
def test_invalid_scores_are_rejected(scores, threshold):
    with pytest.raises(ValueError):
        predict_at_threshold(scores, threshold)


def test_missing_fold_or_class_is_rejected():
    with pytest.raises(ValueError, match="declared fold"):
        threshold_table([0, 1], [0.1, 0.9], [0, 0])
    with pytest.raises(ValueError, match="both classes"):
        threshold_table([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], n_folds=2)


def test_csv_round_trip_preserves_threshold_decision_boundaries(tmp_path):
    scores = np.array([np.nextafter(0.5, 0), 0.5, np.nextafter(0.5, 1), 1.0])
    path = tmp_path / "probabilities.csv"
    write_csv(path, pd.DataFrame({"probability": scores, "Ticket": ["001", "001", "002", "003"]}))
    restored = read_csv(path)
    np.testing.assert_array_equal(restored.probability, scores)
    assert restored.Ticket.tolist() == ["001", "001", "002", "003"]
    np.testing.assert_array_equal(predict_at_threshold(restored.probability, 0.5), [0, 0, 1, 1])


def test_series_rejects_changed_result_before_loading_models(tmp_path):
    from src.experiment import digest

    manifest = {"lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
                "versions": versions(), "source_sha256": source_hashes()}
    (tmp_path / "model.joblib").write_bytes(b"unloaded fitted object")
    save_manifest(tmp_path, manifest)
    assert load_series(tmp_path)["files"] == manifest["files"]
    (tmp_path / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_series(tmp_path)


# MLflow 3.18 still uses SQLAlchemy's deprecated relationship loader.
# Keep this specific upstream warning visible; all other warnings stay errors.
@pytest.mark.filterwarnings(
    "default:The ``noload`` loader strategy is deprecated:sqlalchemy.exc.SADeprecationWarning"
)
def test_local_tracking_preserves_series_and_marks_failed_stage(tmp_path):
    config = read_hw2_settings(ROOT / "configs/hw2.yaml")
    store = LocalTracking(tmp_path / "mlflow", "tracking-test")
    first = store.create_runs("first", config["variants"])
    second = store.create_runs("second", config["variants"])
    assert set(first.values()).isdisjoint(second.values())
    artifact = tmp_path / "outputs"
    artifact.mkdir()
    (artifact / "predictions.csv").write_text("id,probability\n1,0.5\n")
    with store.stage(first, "training"):
        for run_id in first.values():
            store.log(run_id, params={"threshold": 0.5}, metrics={"validation.score": 0.7}, directory=artifact)
    with store.stage(first, "evaluation"):
        for run_id in first.values():
            store.log(run_id, metrics={"test.score": 0.6})
    with pytest.raises(RuntimeError):
        with store.stage(second, "training"):
            raise RuntimeError("test failure")
    for run_id in first.values():
        run = store.client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["stage"] == "evaluation_complete"
        assert run.data.metrics == {"validation.score": 0.7, "test.score": 0.6}
        assert store.client.list_artifacts(run_id, "series")[0].path == "series/predictions.csv"
    assert len(store.client.search_runs([store.experiment_id])) == 8
    assert all(store.client.get_run(r).info.status == "FAILED" for r in second.values())
