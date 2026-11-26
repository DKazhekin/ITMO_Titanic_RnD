"""Select logistic-regression settings using development folds and save final fits."""

import io
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

from src.data import PROJECT_ROOT, write_atomic
from src.experiment import (
    build_pipeline, digest, feature_frame, json_bytes, load_labeled, make_splits,
    read_settings, rows_by_id, runtime_versions, split_summary, stage_parser,
    write_csv, write_json,
)
from src.metrics import classification_metrics


def fit_converged(pipeline, features: pd.DataFrame, target: np.ndarray):
    if features.isna().all().any():
        raise ValueError("An entirely missing training feature has no imputation statistic")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features, target)
    return pipeline


def score(estimator, features: pd.DataFrame, target: np.ndarray) -> dict:
    positive = list(estimator.classes_).index(1)
    return classification_metrics(
        target, estimator.predict(features), estimator.predict_proba(features)[:, positive],
    )


def summarize_candidate(rows: list[dict], model: str, c: float | None, weight: str | None) -> dict:
    result = {"model": model, "C": c, "class_weight": weight, "folds": len(rows)}
    for name in ("balanced_accuracy", "recall_0", "recall_1"):
        values = np.array([row[name] for row in rows])
        result[f"mean_{name}"] = float(values.mean())
        result[f"std_{name}"] = float(values.std(ddof=0))
    return result


def select_candidate(summaries: list[dict]) -> dict:
    """Use the unweighted fold mean, then smaller C and no class weights."""
    candidates = [row for row in summaries if row["model"] == "logistic_regression"]
    if not candidates or any(not np.isfinite(row["mean_balanced_accuracy"]) for row in candidates):
        raise ValueError("Selection requires finite candidate scores")
    return min(candidates, key=lambda row: (
        -row["mean_balanced_accuracy"], row["C"], row["class_weight"] is not None,
    ))


def model_bytes(estimator) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    return buffer.getvalue()


def train(config_path: Path, data_config: Path, data_dir: Path,
          artifacts_dir: Path, reports_dir: Path) -> dict:
    config = read_settings(config_path)
    frame, data_sha = load_labeled(data_dir, data_config)
    splits = make_splits(frame, config)
    folds = []
    for fold in splits["inner_folds"]:
        left, right = (rows_by_id(frame, fold[key]) for key in ("train_ids", "validation_ids"))
        folds.append((feature_frame(left), left["Survived"].to_numpy(dtype=int),
                      feature_frame(right), right["Survived"].to_numpy(dtype=int)))

    results, summaries = [], []
    control_rows = []
    for number, (x_train, y_train, x_val, y_val) in enumerate(folds):
        control = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
        row = {"model": "majority", "C": None, "class_weight": None, "fold": number,
               "n_train": len(y_train), "n_validation": len(y_val), "n_iter": None,
               **score(control, x_val, y_val)}
        control_rows.append(row)
    results.extend(control_rows)
    summaries.append(summarize_candidate(control_rows, "majority", None, None))

    for c in config["search"]["C"]:
        for weight in config["search"]["class_weight"]:
            candidate_rows = []
            for number, (x_train, y_train, x_val, y_val) in enumerate(folds):
                pipeline = fit_converged(build_pipeline(config, c, weight), x_train, y_train)
                candidate_rows.append({
                    "model": "logistic_regression", "C": c, "class_weight": weight,
                    "fold": number, "n_train": len(y_train), "n_validation": len(y_val),
                    "n_iter": int(pipeline.named_steps["model"].n_iter_.max()),
                    **score(pipeline, x_val, y_val),
                })
            results.extend(candidate_rows)
            summary = summarize_candidate(candidate_rows, "logistic_regression", c, weight)
            summaries.append(summary)
            print(f"C={c:g}, class_weight={weight}: "
                  f"balanced accuracy={summary['mean_balanced_accuracy']:.6f} "
                  f"+/- {summary['std_balanced_accuracy']:.6f}", flush=True)

    best = select_candidate(summaries)
    development = rows_by_id(frame, splits["development_ids"])
    x_development = feature_frame(development)
    y_development = development["Survived"].to_numpy(dtype=int)
    model = fit_converged(build_pipeline(config, best["C"], best["class_weight"]),
                          x_development, y_development)
    control = DummyClassifier(strategy="most_frequent").fit(x_development, y_development)
    training = {
        "selected": best, "cv_summary": summaries, "cv_folds": results,
        "std_ddof": 0, "split_summary": split_summary(frame, splits),
        "final_fit": {
            "n_train": len(development),
            "n_iter": int(model.named_steps["model"].n_iter_.max()),
            "majority_class": int(control.predict(x_development.iloc[:1])[0]),
        },
    }
    contents = {
        "model.joblib": model_bytes(model), "control.joblib": model_bytes(control),
        "splits.json": json_bytes(splits), "training.json": json_bytes(training),
    }
    run = {
        "config": config, "config_sha256": digest(config_path.read_bytes()),
        "data_sha256": data_sha, "data_config_sha256": digest(data_config.read_bytes()),
        "lock_sha256": digest((PROJECT_ROOT / "uv.lock").read_bytes()),
        "versions": runtime_versions(),
        "training_source_sha256": {
            name: digest((PROJECT_ROOT / "src" / name).read_bytes())
            for name in ("data.py", "experiment.py", "metrics.py", "train.py")
        },
        "files": {name: digest(content) for name, content in contents.items()},
    }
    run["run_id"] = digest(json_bytes(run))
    for name, content in contents.items():
        write_atomic(artifacts_dir / name, content)
    write_json(artifacts_dir / "run.json", run)
    write_json(reports_dir / "run_summary.json", run)
    write_json(reports_dir / "split_summary.json", training["split_summary"])
    write_csv(reports_dir / "cv_summary.csv", pd.DataFrame(summaries))
    write_csv(reports_dir / "cv_folds.csv", pd.DataFrame(results).drop(columns="confusion_matrix"))
    print(f"Saved final fits on {len(development)} development rows; "
          f"C={best['C']:g}, class_weight={best['class_weight']}. Final test has not been scored.")
    return run


def main() -> None:
    parser = stage_parser(__doc__)
    args = parser.parse_args()
    train(args.config, args.data_config, args.data_dir, args.artifacts_dir, args.reports_dir)


if __name__ == "__main__":
    main()
