"""Evaluate saved models once on the fixed final test and export diagnostics."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, verify_checksum
from src.experiment import (
    digest, feature_frame, load_labeled, load_run, read_json, rows_by_id,
    runtime_versions, stage_parser, validate_splits, write_csv, write_json,
)
from src.metrics import classification_metrics, paired_group_bootstrap


def slice_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ("Sex", "Pclass"):
        for value, subset in predictions.groupby(feature, sort=True, observed=True):
            for model in ("model", "control"):
                metrics = classification_metrics(
                    subset["Survived"], subset[f"prediction_{model}"], subset[f"probability_{model}"],
                )
                rows.append({"feature": feature, "value": value, "model": model, **{
                    name: metrics[name] for name in (
                        "n", "n_0", "n_1", "balanced_accuracy", "recall_0", "recall_1",
                        "precision_0", "precision_1", "errors",
                    )
                }})
    return pd.DataFrame(rows)


def error_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    errors = predictions.loc[predictions["Survived"] != predictions["prediction_model"]].copy()
    errors["confidence"] = np.where(
        errors["prediction_model"] == 1, errors["probability_model"], 1 - errors["probability_model"],
    )
    errors = errors.sort_values(["confidence", "PassengerId"], ascending=[False, True])
    selected, labels = [], []
    for label, condition in (
        ("missed_survivor", errors["Survived"] == 1),
        ("false_survival", errors["Survived"] == 0),
        ("missing_age", errors["Age"].isna()),
        ("other_error", pd.Series(True, index=errors.index)),
    ):
        candidates = errors.loc[condition & ~errors.index.isin(selected)]
        if len(candidates) and len(selected) < 3:
            selected.append(candidates.index[0])
            labels.append(label)
    examples = errors.loc[selected].drop(columns=[
        "Ticket", "confidence", "prediction_control", "probability_control",
    ])
    examples.insert(0, "case", labels)
    return examples


def evaluate(config_path: Path, data_config: Path, data_dir: Path,
             artifacts_dir: Path, reports_dir: Path) -> dict:
    run = load_run(artifacts_dir)
    verify_checksum(config_path.read_bytes(), run["config_sha256"], "baseline configuration")
    verify_checksum(data_config.read_bytes(), run["data_config_sha256"], "data configuration")
    verify_checksum((PROJECT_ROOT / "uv.lock").read_bytes(), run["lock_sha256"], "uv.lock")
    if runtime_versions() != run["versions"]:
        raise ValueError("Evaluation must use the training environment versions")
    frame, data_sha = load_labeled(data_dir, data_config)
    if data_sha != run["data_sha256"]:
        raise ValueError("Evaluation data differ from the training snapshot")
    splits = read_json(artifacts_dir / "splits.json")
    validate_splits(frame, splits)
    test = rows_by_id(frame, splits["test_ids"])
    features = feature_frame(test)
    predictions = test[["PassengerId", "Ticket", *features.columns, "Survived"]].copy()
    metrics = {}
    for name in ("model", "control"):
        estimator = joblib.load(artifacts_dir / f"{name}.joblib")
        predictions[f"prediction_{name}"] = estimator.predict(features)
        positive = list(estimator.classes_).index(1)
        predictions[f"probability_{name}"] = estimator.predict_proba(features)[:, positive]
        metrics[name] = classification_metrics(
            test["Survived"].to_numpy(dtype=int), predictions[f"prediction_{name}"],
            predictions[f"probability_{name}"],
        )
    config = run["config"]
    interval, draws = paired_group_bootstrap(
        predictions["Survived"].to_numpy(dtype=int), predictions["prediction_model"],
        predictions["prediction_control"], predictions["Ticket"].to_numpy(dtype=str),
        seed=config["seed"], **config["bootstrap"],
    )
    write_csv(artifacts_dir / "predictions.csv", predictions)
    write_csv(artifacts_dir / "bootstrap.csv", draws)
    report = {
        "run_id": run["run_id"], "partition": "fixed_final_test", "metrics": metrics,
        "bootstrap": interval,
        "predictions_sha256": digest((artifacts_dir / "predictions.csv").read_bytes()),
        "bootstrap_sha256": digest((artifacts_dir / "bootstrap.csv").read_bytes()),
        "evaluation_source_sha256": {
            name: digest((PROJECT_ROOT / "src" / name).read_bytes())
            for name in ("evaluate.py", "metrics.py")
        },
    }
    write_json(artifacts_dir / "evaluation.json", report)
    write_json(reports_dir / "metrics.json", report)
    write_csv(reports_dir / "slice_metrics.csv", slice_metrics(predictions))
    write_csv(reports_dir / "error_examples.csv", error_examples(predictions))
    print(f"Final test: model BA={metrics['model']['balanced_accuracy']:.6f}, "
          f"control BA={metrics['control']['balanced_accuracy']:.6f}; "
          f"paired {interval['confidence_level']:.0%} interval for delta="
          f"[{interval['lower']:.6f}, {interval['upper']:.6f}].")
    return report


def main() -> None:
    parser = stage_parser(__doc__)
    args = parser.parse_args()
    evaluate(args.config, args.data_config, args.data_dir, args.artifacts_dir, args.reports_dir)


if __name__ == "__main__":
    main()
