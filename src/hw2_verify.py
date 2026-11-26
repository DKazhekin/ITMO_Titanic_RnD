"""Independently verify saved scores, expanded-group resamples and repeated runs."""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from src.experiment import read_json, write_json


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(directory: Path, repeat_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    from src.hw2 import load_series, read_csv, save_manifest, tracking_for

    directory = directory.resolve()
    manifest = load_series(directory)
    require(manifest["stage"] == "complete", "Verification requires a completed series")
    training = read_json(directory / "training.json")
    evaluation = read_json(directory / "evaluation.json")
    splits = read_json(directory / "splits.json")
    predictions = read_csv(directory / "predictions.csv")
    require(predictions.PassengerId.tolist() == splits["test_ids"], "Test passengers or order differ")
    require(training["n_fits"] == 12, "Unexpected number of fits")
    require(len(splits["development_ids"]) == 712 and len(predictions) == 179, "Unexpected sample sizes")
    candidates_checked = 0
    for name in ("original", "interaction"):
        oof = read_csv(directory / name / "validation_predictions.csv")
        table = read_csv(directory / name / "threshold_candidates.csv")
        require(sorted(oof.PassengerId) == sorted(splits["development_ids"]), "Invalid validation coverage")
        expected_candidates = np.unique(np.r_[oof.probability, 0, 0.5, 1])
        np.testing.assert_array_equal(table.threshold, expected_candidates)
        for fold in splits["inner_folds"]:
            require(sorted(oof.loc[oof.fold == fold["fold"], "PassengerId"]) == sorted(fold["validation_ids"]),
                    "Validation membership differs from the saved split")
        means = []
        for row in table.to_dict("records"):
            values = []
            for fold, part in oof.groupby("fold", sort=True):
                value = balanced_accuracy_score(part.Survived, part.probability > row["threshold"])
                np.testing.assert_allclose(value, row[f"fold_{fold}_balanced_accuracy"], rtol=0, atol=1e-15)
                values.append(value)
            mean = float(np.mean(values))
            means.append(mean)
            np.testing.assert_allclose(mean, row["mean_balanced_accuracy"], rtol=0, atol=1e-15)
        selected = min(zip(table.threshold, means), key=lambda x: (-x[1], abs(x[0] - 0.5), x[0]))[0]
        require(selected == training["thresholds"][name], "Selected threshold violates the selection rule")
        candidates_checked += len(table)
    for variant in manifest["config"]["variants"]:
        key = variant["id"]
        probability = predictions[f"probability_{key}"]
        prediction = predictions[f"prediction_{key}"]
        np.testing.assert_array_equal(prediction, probability > training["validation"][key]["threshold"])
        expected = evaluation["metrics"][key]
        recalls = recall_score(predictions.Survived, prediction, labels=[0, 1], average=None)
        observed = {"balanced_accuracy": balanced_accuracy_score(predictions.Survived, prediction),
                    "accuracy": accuracy_score(predictions.Survived, prediction),
                    "recall_0": recalls[0], "recall_1": recalls[1],
                    "roc_auc": roc_auc_score(predictions.Survived, probability)}
        for metric, value in observed.items():
            np.testing.assert_allclose(value, expected[metric], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(confusion_matrix(predictions.Survived, prediction), expected["confusion_matrix"])
    for left, right in (("baseline", "threshold"), ("interaction", "full")):
        np.testing.assert_array_equal(predictions[f"probability_{left}"], predictions[f"probability_{right}"])

    # Expand sampled groups into actual row indices, independently of the optimized count method.
    groups = [np.flatnonzero(predictions.Ticket.to_numpy() == ticket) for ticket in np.unique(predictions.Ticket)]
    require(len(groups) == 136, "Unexpected number of test ticket groups")
    n_resamples = manifest["config"]["bootstrap"]["n_resamples"]
    rng = np.random.default_rng(manifest["config"]["seed"])
    observed_draws = {key: [] for key in ("baseline", "interaction", "threshold")}
    truth = predictions.Survived.to_numpy()
    full = predictions.prediction_full.to_numpy()
    others = {k: predictions[f"prediction_{k}"].to_numpy() for k in observed_draws}
    for _ in range(n_resamples):
        selected = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        require(set(truth[selected]) == {0, 1}, "An expanded resample lacks a true class")
        full_score = balanced_accuracy_score(truth[selected], full[selected])
        for key, other in others.items():
            observed_draws[key].append(full_score - balanced_accuracy_score(truth[selected], other[selected]))
    for key, values in observed_draws.items():
        name = f"full_minus_{key}"
        saved = read_csv(directory / "bootstrap" / f"{name}.csv")
        np.testing.assert_allclose(values, saved.delta_balanced_accuracy, rtol=0, atol=1e-15)
        comparison = next(row for row in evaluation["comparisons"] if row["comparison"] == name)
        np.testing.assert_allclose(np.quantile(values, [0.025, 0.975], method="linear"),
                                   [comparison["lower"], comparison["upper"]], rtol=0, atol=1e-15)
        point = evaluation["metrics"]["full"]["balanced_accuracy"] - evaluation["metrics"][key]["balanced_accuracy"]
        np.testing.assert_allclose(point, comparison["delta_balanced_accuracy"], rtol=0, atol=1e-15)

    tracking = tracking_for(manifest)
    series_runs = tracking.client.search_runs([tracking.experiment_id],
                                              filter_string=f"tags.series_id = '{manifest['series_id']}'")
    require({r.info.run_id for r in series_runs} == set(manifest["mlflow_runs"].values()), "Unexpected MLflow run set")
    for key, run_id in manifest["mlflow_runs"].items():
        run = tracking.client.get_run(run_id)
        require(run.info.status == "FINISHED" and run.data.tags["stage"] == "report_complete", "Incomplete MLflow stage")
        require(run.data.tags["hypothesis"] != "", "Missing hypothesis")
        require(float(run.data.params["threshold"]) == training["validation"][key]["threshold"], "MLflow threshold mismatch")
        require(run.data.metrics["test.balanced_accuracy"] == evaluation["metrics"][key]["balanced_accuracy"], "MLflow metric mismatch")
        require(any(f.path == "series/predictions.csv" for f in tracking.client.list_artifacts(run_id, "series")),
                "MLflow predictions are missing")

    repeated_files = []
    repeat = None
    if repeat_dir is not None:
        repeat = load_series(repeat_dir)
        require(repeat["stage"] == "complete" and repeat["series_id"] != manifest["series_id"], "Repeat must be a different complete series")
        require(set(repeat["mlflow_runs"].values()).isdisjoint(manifest["mlflow_runs"].values()), "Repeated series reused run identifiers")
        require(repeat["source_sha256"] == manifest["source_sha256"] and repeat["config"] == manifest["config"],
                "Repeat source or settings differ")
        for name in manifest["files"]:
            if name in ("report.md", "series.json", "verification.json"):
                continue
            require(name in repeat["files"] and manifest["files"][name] == repeat["files"][name],
                    f"Repeat differs: {name}")
            repeated_files.append(name)
        for run_id in repeat["mlflow_runs"].values():
            require(tracking.client.get_run(run_id).info.status == "FINISHED", "Previous run history is missing or incomplete")
    result = {
        "status": "passed", "series_id": manifest["series_id"],
        "repeat_series_id": repeat["series_id"] if repeat else None,
        "repeat_checked": repeat is not None, "byte_identical_files": repeated_files,
        "threshold_candidates_checked": candidates_checked, "test_passengers": len(predictions),
        "bootstrap_draws_per_comparison": n_resamples, "comparisons_checked": 3,
        "mlflow_runs_checked": 4 + (4 if repeat else 0),
        "baseline_predictions_sha256": evaluation["baseline_predictions_sha256"],
        "source_sha256": manifest["source_sha256"], "versions": manifest["versions"],
    }
    write_json(directory / "verification.json", result)
    save_manifest(directory, manifest)
    for run_id in manifest["mlflow_runs"].values():
        tracking.log(run_id, directory=directory)
    if reports_dir is not None:
        exported = read_json(reports_dir / "series.json")
        require(exported["series_id"] == manifest["series_id"], "Report export belongs to a different series")
        write_json(reports_dir / "verification.json", result)
    print(f"Verified {candidates_checked} thresholds, three paired comparisons and {len(repeated_files)} repeated files.", flush=True)
    return result
