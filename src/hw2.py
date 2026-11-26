"""Run four controlled feature/threshold variants on the fixed Titanic split."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, prepare_data, verify_checksum, write_atomic
from src.experiment import (
    build_pipeline, digest, feature_frame, json_bytes, load_labeled, make_splits,
    read_json, read_settings, rows_by_id, runtime_versions, validate_splits, write_csv, write_json,
)
from src.metrics import classification_metrics, paired_group_bootstrap
from src.thresholds import predict_at_threshold, select_threshold, threshold_table
from src.tracking import LocalTracking
from src.train import fit_converged, model_bytes


DEFAULT_CONFIG = PROJECT_ROOT / "configs/hw2.yaml"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs/data.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "artifacts/hw2/runs"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports/hw2"
VARIANTS = [("baseline", False, False), ("interaction", True, False),
            ("threshold", False, True), ("full", True, True)]
SCORE_NAMES = ("balanced_accuracy", "accuracy", "recall_0", "recall_1", "roc_auc")


def read_hw2_settings(path: Path) -> dict:
    config = read_settings(path)
    baseline = read_json(PROJECT_ROOT / "reports/run_summary.json")["config"]
    for key in ("seed", "features", "validation", "model", "bootstrap"):
        if config[key] != baseline[key]:
            raise ValueError(f"The controlled experiment must retain baseline {key}")
    if config["search"] != {"C": [1.0], "class_weight": [None]}:
        raise ValueError("Only the previously selected C=1 and no class weights are supported")
    expected = {
        "candidates": "unique_validation_probabilities_plus_0_0.5_1",
        "objective": "unweighted_mean_fold_balanced_accuracy",
        "comparison": "strict_greater", "ties": "closest_to_0.5_then_lower",
    }
    if config.get("threshold_search") != expected:
        raise ValueError("Unsupported threshold selection protocol")
    actual = [(v["id"], v["interaction"], v["tune_threshold"]) for v in config["variants"]]
    if actual != VARIANTS or any(not v.get("hypothesis") for v in config["variants"]):
        raise ValueError("Exactly four declared variants and their hypotheses are required")
    if not config["tracking"]["experiment"] or not config["tracking"]["directory"]:
        raise ValueError("Local tracking needs an experiment and directory")
    return config


def source_hashes() -> dict:
    return {p.name: digest(p.read_bytes()) for p in sorted((PROJECT_ROOT / "src").glob("*.py"))}


def versions() -> dict:
    return {**runtime_versions(), "mlflow": importlib.metadata.version("mlflow")}


def save_manifest(directory: Path, manifest: dict) -> None:
    manifest["files"] = {
        str(p.relative_to(directory)): digest(p.read_bytes())
        for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"
    }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = digest(json_bytes(manifest))
    write_json(directory / "manifest.json", manifest)


def load_series(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    identity = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(json_bytes(identity)) != manifest["manifest_sha256"]:
        raise ValueError("Series manifest checksum mismatch")
    for name, expected in manifest["files"].items():
        verify_checksum((directory / name).read_bytes(), expected, name)
    verify_checksum((PROJECT_ROOT / "uv.lock").read_bytes(), manifest["lock_sha256"], "uv.lock")
    if versions() != manifest["versions"] or source_hashes() != manifest["source_sha256"]:
        raise ValueError("Series stages must use the same source and dependency versions")
    return manifest


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Ticket": str})


def validation_metrics(predictions: pd.DataFrame, threshold: float) -> dict:
    folds = []
    for number, part in predictions.groupby("fold", sort=True):
        folds.append({"fold": int(number), **classification_metrics(
            part.Survived, predict_at_threshold(part.probability, threshold), part.probability,
        )})
    summary = {}
    for name in SCORE_NAMES:
        values = [row[name] for row in folds]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values, ddof=0))
    return {"folds": folds, "summary": summary}


def tracking_for(manifest: dict) -> LocalTracking:
    return LocalTracking(Path(manifest["tracking_directory"]), manifest["config"]["tracking"]["experiment"])


def record_failure(directory: Path, manifest: dict, stage: str, error: Exception) -> None:
    manifest["stage"] = f"{stage}_failed"
    write_json(directory / "failure.json", {"stage": stage, "error": str(error)})
    save_manifest(directory, manifest)


def train(config_path: Path = DEFAULT_CONFIG, data_config: Path = DEFAULT_DATA_CONFIG,
          data_dir: Path = DEFAULT_DATA_DIR, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    config = read_hw2_settings(config_path)
    reference = read_json(PROJECT_ROOT / "reports/run_summary.json")
    if runtime_versions() != reference["versions"]:
        raise ValueError("Core runtime versions must match the original baseline")
    frame, data_sha = load_labeled(data_dir, data_config)
    if data_sha != reference["data_sha256"]:
        raise ValueError("The controlled comparison requires the original baseline data")
    splits = make_splits(frame, config)
    if digest(json_bytes(splits)) != reference["files"]["splits.json"]:
        raise ValueError("The partitions differ from the original baseline")
    series_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = (runs_dir / series_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    tracking = LocalTracking(PROJECT_ROOT / config["tracking"]["directory"], config["tracking"]["experiment"])
    manifest = {
        "series_id": series_id, "stage": "training", "config": config,
        "data_sha256": data_sha, "data_directory": str(data_dir.resolve()),
        "source_sha256": source_hashes(), "versions": versions(),
        "lock_sha256": digest((PROJECT_ROOT / "uv.lock").read_bytes()),
        "tracking_directory": str(tracking.directory),
        "mlflow_runs": tracking.create_runs(series_id, config["variants"]),
    }
    write_atomic(directory / "config.yaml", config_path.read_bytes())
    write_atomic(directory / "data_config.yaml", data_config.read_bytes())
    write_json(directory / "splits.json", splits)
    write_json(directory / "baseline_reference.json", reference)
    write_json(directory / "baseline_metrics.json", read_json(PROJECT_ROOT / "reports/metrics.json"))
    save_manifest(directory, manifest)
    try:
        with tracking.stage(manifest["mlflow_runs"], "training"):
            oof, thresholds, fits = {}, {}, {}
            for interaction, name in ((False, "original"), (True, "interaction")):
                records = []
                for fold in splits["inner_folds"]:
                    left = rows_by_id(frame, fold["train_ids"])
                    right = rows_by_id(frame, fold["validation_ids"])
                    estimator = fit_converged(
                        build_pipeline(config, 1.0, None, interaction=interaction),
                        feature_frame(left), left.Survived.to_numpy(dtype=int),
                    )
                    part = right[["PassengerId", "Ticket", "Survived"]].copy()
                    part["fold"] = fold["fold"]
                    part["probability"] = estimator.predict_proba(feature_frame(right))[:, 1]
                    records.append(part)
                predictions = pd.concat(records, ignore_index=True).sort_values("PassengerId")
                if sorted(predictions.PassengerId) != sorted(splits["development_ids"]):
                    raise ValueError("Validation predictions must cover development passengers exactly once")
                table = threshold_table(predictions.Survived, predictions.probability, predictions.fold)
                threshold = select_threshold(table)
                oof[name], thresholds[name] = predictions, threshold
                write_csv(directory / name / "validation_predictions.csv", predictions)
                write_csv(directory / name / "threshold_candidates.csv", table)
                development = rows_by_id(frame, splits["development_ids"])
                estimator = fit_converged(
                    build_pipeline(config, 1.0, None, interaction=interaction),
                    feature_frame(development), development.Survived.to_numpy(dtype=int),
                )
                write_atomic(directory / name / "model.joblib", model_bytes(estimator))
                fits[name] = {"n_train": len(development),
                              "n_iter": int(estimator.named_steps["model"].n_iter_.max())}
                print(f"{name}: five validation fits and final fit complete; threshold={threshold:.17g}", flush=True)
            validation = {}
            for variant in config["variants"]:
                key = variant["id"]
                name = "interaction" if variant["interaction"] else "original"
                threshold = thresholds[name] if variant["tune_threshold"] else 0.5
                validation[key] = {"threshold": threshold,
                                   **validation_metrics(oof[name], threshold)}
                params = {
                    "interaction": variant["interaction"], "tune_threshold": variant["tune_threshold"],
                    "threshold": threshold, "C": 1.0, "class_weight": "None", "seed": config["seed"],
                    "hypothesis": variant["hypothesis"], "data_sha256": data_sha,
                    "splits_sha256": digest(json_bytes(splits)), "lock_sha256": manifest["lock_sha256"],
                    "source_sha256": digest(json_bytes(manifest["source_sha256"])),
                    **{f"model.{k}": v for k, v in config["model"].items()},
                    **{f"threshold_search.{k}": v for k, v in config["threshold_search"].items()},
                    **{f"validation.{k}": v for k, v in config["validation"].items()},
                    **{f"bootstrap.{k}": v for k, v in config["bootstrap"].items()},
                    **{f"version.{k}": v for k, v in manifest["versions"].items()},
                }
                metrics = {f"validation.{k}": v for k, v in validation[key]["summary"].items()}
                for fold in validation[key]["folds"]:
                    metrics.update({f"validation.fold_{fold['fold']}.{k}": fold[k] for k in SCORE_NAMES})
                tracking.log(manifest["mlflow_runs"][key], params=params, metrics=metrics)
            write_json(directory / "training.json", {
                "thresholds": thresholds, "validation": validation, "final_fits": fits,
                "n_fits": 12, "std_ddof": 0,
                "baseline_model_bytes_match": digest((directory / "original/model.joblib").read_bytes())
                == reference["files"]["model.joblib"],
            })
            manifest["stage"] = "trained"
            save_manifest(directory, manifest)
            for run_id in manifest["mlflow_runs"].values():
                tracking.log(run_id, directory=directory)
    except Exception as error:
        record_failure(directory, manifest, "training", error)
        raise
    print(f"Series directory: {directory}\nFinal test has not been scored.", flush=True)
    return directory


def evaluate(directory: Path) -> dict:
    directory = directory.resolve()
    manifest = load_series(directory)
    if manifest["stage"] not in ("trained", "evaluated", "complete"):
        raise ValueError("Evaluation requires a successfully trained series")
    config = manifest["config"]
    frame, data_sha = load_labeled(Path(manifest["data_directory"]), directory / "data_config.yaml")
    if data_sha != manifest["data_sha256"]:
        raise ValueError("Evaluation data differ from the training snapshot")
    splits = read_json(directory / "splits.json")
    validate_splits(frame, splits)
    test = rows_by_id(frame, splits["test_ids"])
    features = feature_frame(test)
    training = read_json(directory / "training.json")
    tracking = tracking_for(manifest)
    try:
        with tracking.stage(manifest["mlflow_runs"], "evaluation"):
            probabilities = {
                name: joblib.load(directory / name / "model.joblib").predict_proba(features)[:, 1]
                for name in ("original", "interaction")
            }
            predictions = test[["PassengerId", "Ticket", "Survived"]].copy()
            metrics, experiment_rows = {}, []
            for variant in config["variants"]:
                key = variant["id"]
                probability = probabilities["interaction" if variant["interaction"] else "original"]
                threshold = training["validation"][key]["threshold"]
                prediction = predict_at_threshold(probability, threshold)
                predictions[f"probability_{key}"] = probability
                predictions[f"prediction_{key}"] = prediction
                metrics[key] = classification_metrics(test.Survived, prediction, probability)
                row = {**variant, "threshold": threshold, "C": 1.0, "class_weight": "None", "seed": config["seed"],
                       **{f"validation_{k}": v for k, v in training["validation"][key]["summary"].items()},
                       **{f"test_{k}": v for k, v in metrics[key].items() if k != "confusion_matrix"}}
                experiment_rows.append(row)
                tracking.log(manifest["mlflow_runs"][key], metrics={
                    f"test.{k}": v for k, v in metrics[key].items() if k != "confusion_matrix"
                })
            # Reconstruct the original prediction file for an exact identity check, including labels.
            baseline = test[["PassengerId", "Ticket", *features.columns, "Survived"]].copy()
            baseline["prediction_model"] = predictions.prediction_baseline
            baseline["probability_model"] = probabilities["original"]
            baseline["prediction_control"] = np.zeros(len(test), dtype=int)
            baseline["probability_control"] = np.zeros(len(test), dtype=float)
            baseline_sha = digest(baseline.to_csv(index=False, float_format="%.17g").encode())
            original_metrics = read_json(directory / "baseline_metrics.json")
            if baseline_sha != original_metrics["predictions_sha256"]:
                raise ValueError("Baseline predictions do not reproduce the original experiment")
            comparisons = []
            for other, purpose in (("baseline", "joint_effect"), ("interaction", "threshold_ablation"),
                                   ("threshold", "interaction_ablation")):
                interval, draws = paired_group_bootstrap(
                    predictions.Survived, predictions.prediction_full, predictions[f"prediction_{other}"],
                    predictions.Ticket, seed=config["seed"], **config["bootstrap"],
                )
                comparisons.append({"comparison": f"full_minus_{other}", "purpose": purpose, **interval})
                write_csv(directory / "bootstrap" / f"full_minus_{other}.csv", draws)
                tracking.log(manifest["mlflow_runs"]["full"], metrics={
                    f"comparison.{other}.{k}": interval[k]
                    for k in ("delta_balanced_accuracy", "lower", "upper")
                })
            result = {"metrics": metrics, "comparisons": comparisons,
                      "baseline_predictions_sha256": baseline_sha, "baseline_reproduced": True}
            write_csv(directory / "predictions.csv", predictions)
            write_csv(directory / "experiments.csv", pd.DataFrame(experiment_rows))
            write_csv(directory / "comparisons.csv", pd.DataFrame(comparisons))
            write_json(directory / "evaluation.json", result)
            manifest["stage"] = "evaluated"
            save_manifest(directory, manifest)
            for run_id in manifest["mlflow_runs"].values():
                tracking.log(run_id, directory=directory)
    except Exception as error:
        record_failure(directory, manifest, "evaluation", error)
        raise
    for key, values in metrics.items():
        print(f"{key}: test balanced accuracy={values['balanced_accuracy']:.9f}", flush=True)
    return result


def make_plots(directory: Path, reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[Path]:
    from src.hw2_report import render_report

    directory = directory.resolve()
    manifest = load_series(directory)
    if manifest["stage"] not in ("evaluated", "complete"):
        raise ValueError("Plots require an evaluated series")
    if reports_dir.resolve() == directory or reports_dir.resolve() in directory.parents:
        raise ValueError("The report directory must not replace the series or its parents")
    tracking = tracking_for(manifest)
    try:
        with tracking.stage(manifest["mlflow_runs"], "report"):
            paths = render_report(directory, manifest)
            manifest["stage"] = "complete"
            reports_dir.mkdir(parents=True, exist_ok=True)
            for name in ("report.md", "experiments.csv", "comparisons.csv", "predictions.csv", "evaluation.json"):
                shutil.copyfile(directory / name, reports_dir / name)
            shutil.copytree(directory / "figures", reports_dir / "figures", dirs_exist_ok=True)
            summary = {
                "series_id": manifest["series_id"],
                "data_sha256": manifest["data_sha256"], "source_sha256": manifest["source_sha256"],
                "versions": manifest["versions"], "mlflow_runs": manifest["mlflow_runs"],
                "lock_sha256": manifest["lock_sha256"], "config": manifest["config"],
            }
            write_json(directory / "series.json", summary)
            write_json(reports_dir / "series.json", summary)
            pending_verification = {
                "status": "not_run", "series_id": manifest["series_id"], "repeat_checked": False,
            }
            write_json(directory / "verification.json", pending_verification)
            write_json(reports_dir / "verification.json", pending_verification)
            save_manifest(directory, manifest)
            for run_id in manifest["mlflow_runs"].values():
                tracking.log(run_id, directory=directory)
    except Exception as error:
        record_failure(directory, manifest, "report", error)
        raise
    print(f"Report: {reports_dir / 'report.md'}", flush=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "train"):
        stage = sub.add_parser(command)
        stage.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        stage.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
        stage.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        stage.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
        if command == "run":
            stage.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    for command in ("evaluate", "plots"):
        stage = sub.add_parser(command)
        stage.add_argument("--run-dir", type=Path, required=True)
        if command == "plots":
            stage.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    ui = sub.add_parser("ui")
    ui.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ui.add_argument("--port", type=int, default=8080)
    check = sub.add_parser("verify")
    check.add_argument("--run-dir", type=Path, required=True)
    check.add_argument("--repeat-dir", type=Path)
    check.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()
    if args.command in ("run", "train"):
        if args.command == "run":
            prepare_data(args.data_config, args.data_dir, args.runs_dir / "data_summary.json")
        directory = train(args.config, args.data_config, args.data_dir, args.runs_dir)
        if args.command == "run":
            evaluate(directory)
            make_plots(directory, args.reports_dir)
            from src.hw2_verify import verify
            verify(directory, reports_dir=args.reports_dir)
    elif args.command == "evaluate":
        evaluate(args.run_dir)
    elif args.command == "plots":
        make_plots(args.run_dir, args.reports_dir)
    elif args.command == "verify":
        from src.hw2_verify import verify
        verify(args.run_dir, args.repeat_dir, args.reports_dir)
    else:
        config = read_hw2_settings(args.config)
        tracking = LocalTracking(PROJECT_ROOT / config["tracking"]["directory"], config["tracking"]["experiment"])
        subprocess.run([
            sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(args.port),
            "--backend-store-uri", tracking.uri, "--default-artifact-root",
            (tracking.directory / "files").as_uri(), "--no-serve-artifacts",
        ], check=True)


if __name__ == "__main__":
    main()
