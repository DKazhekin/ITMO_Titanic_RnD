"""Exercise stage separation and artifact consistency on synthetic passengers."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src import evaluate, plots, train
from src.experiment import feature_frame, read_json, rows_by_id
from tests.test_experiment import config, passenger_frame


def test_training_evaluation_and_plotting_have_separate_responsibilities(
        tmp_path, monkeypatch, config, passenger_frame):
    root = Path(__file__).resolve().parents[1]
    data_dir, artifacts, reports = (tmp_path / name for name in ("data", "artifacts", "reports"))
    data_dir.mkdir()
    raw = passenger_frame.to_csv(index=False).encode()
    (data_dir / "train.csv").write_bytes(raw)
    data_config = yaml.safe_load((root / "configs/data.yaml").read_text())
    data_config["files"]["train.csv"].update(rows=len(passenger_frame), sha256=hashlib.sha256(raw).hexdigest())
    source_path, config_path = tmp_path / "data.yaml", tmp_path / "baseline.yaml"
    source_path.write_text(yaml.safe_dump(data_config))
    config["search"]["C"] = [0.01, 1.0]
    config["bootstrap"]["n_resamples"] = 100
    config_path.write_text(yaml.safe_dump(config))
    fitted_inputs, scored_inputs = [], []
    original_fit, original_score = train.fit_converged, train.score

    def fitting(estimator, features, target):
        fitted_inputs.append(features.copy())
        return original_fit(estimator, features, target)

    def scoring(estimator, features, target):
        scored_inputs.append(features.copy())
        return original_score(estimator, features, target)

    monkeypatch.setattr(train, "fit_converged", fitting)
    monkeypatch.setattr(train, "score", scoring)
    run = train.train(config_path, source_path, data_dir, artifacts, reports)
    splits = read_json(artifacts / "splits.json")
    assert len(fitted_inputs) == 21
    for index, features in enumerate(fitted_inputs[:-1]):
        expected = feature_frame(rows_by_id(passenger_frame, splits["inner_folds"][index % 5]["train_ids"]))
        pd.testing.assert_frame_equal(features, expected, check_dtype=False)
    expected_final = feature_frame(rows_by_id(passenger_frame, splits["development_ids"]))
    pd.testing.assert_frame_equal(fitted_inputs[-1], expected_final, check_dtype=False)
    assert len(scored_inputs) == 25
    for index, features in enumerate(scored_inputs):
        expected = feature_frame(rows_by_id(passenger_frame, splits["inner_folds"][index % 5]["validation_ids"]))
        pd.testing.assert_frame_equal(features, expected, check_dtype=False)
    assert not (artifacts / "predictions.csv").exists()
    assert not (data_dir / "test.csv").exists()

    def forbidden(*args, **kwargs):
        pytest.fail("This stage must not fit or predict")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(DummyClassifier, "fit", forbidden)
    report = evaluate.evaluate(config_path, source_path, data_dir, artifacts, reports)
    assert report["run_id"] == run["run_id"]
    predictions = pd.read_csv(artifacts / "predictions.csv")
    assert predictions.PassengerId.tolist() == splits["test_ids"]
    truth, prediction = predictions.Survived, predictions.prediction_model
    expected_ba = np.mean([(prediction[truth == c] == c).mean() for c in (0, 1)])
    assert report["metrics"]["model"]["balanced_accuracy"] == pytest.approx(expected_ba)
    assert report["bootstrap"]["invalid_resamples"] == 0
    slices = pd.read_csv(reports / "slice_metrics.csv")
    for feature in ("Sex", "Pclass"):
        assert slices.loc[(slices.feature == feature) & (slices.model == "model"), "n"].sum() == len(predictions)

    monkeypatch.setattr(Pipeline, "predict", forbidden)
    monkeypatch.setattr(Pipeline, "predict_proba", forbidden)
    monkeypatch.setattr(DummyClassifier, "predict", forbidden)
    monkeypatch.setattr(DummyClassifier, "predict_proba", forbidden)
    monkeypatch.setattr(evaluate, "load_labeled", forbidden)
    paths = plots.make_plots(artifacts, reports)
    assert len(paths) == 3
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)
    (artifacts / "predictions.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        plots.make_plots(artifacts, reports)
    config_path.write_text(config_path.read_text() + "\n# different config snapshot\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate.evaluate(config_path, source_path, data_dir, artifacts, reports)
