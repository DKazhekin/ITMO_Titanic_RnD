"""Explicit local MLflow records shared across training, evaluation and plotting."""

from contextlib import contextmanager
from pathlib import Path
import os

# Tracking stays local; experiment records do not require usage telemetry.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")


class LocalTracking:
    def __init__(self, directory: Path, experiment: str):
        from mlflow import MlflowClient

        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        artifacts = self.directory / "files"
        artifacts.mkdir(exist_ok=True)
        self.uri = f"sqlite:///{self.directory / 'metadata.db'}"
        self.client = MlflowClient(tracking_uri=self.uri)
        existing = self.client.get_experiment_by_name(experiment)
        if existing is not None and existing.artifact_location != artifacts.as_uri():
            raise ValueError("The experiment already points to a different artifact directory")
        self.experiment_id = existing.experiment_id if existing else self.client.create_experiment(
            experiment, artifact_location=artifacts.as_uri(),
        )

    def create_runs(self, series_id: str, variants: list[dict]) -> dict[str, str]:
        runs = {}
        try:
            for variant in variants:
                run = self.client.create_run(self.experiment_id, tags={
                    "mlflow.runName": f"{series_id}/{variant['id']}",
                    "series_id": series_id, "variant": variant["id"],
                    "hypothesis": variant["hypothesis"], "stage": "created",
                })
                runs[variant["id"]] = run.info.run_id
        except Exception:
            for run_id in runs.values():
                self.client.set_terminated(run_id, status="FAILED")
            raise
        return runs

    @contextmanager
    def stage(self, runs: dict[str, str], stage: str):
        try:
            for run_id in runs.values():
                self.client.set_tag(run_id, "stage", f"{stage}_running")
            yield
            for run_id in runs.values():
                self.client.set_tag(run_id, "stage", f"{stage}_complete")
                self.client.set_terminated(run_id, status="FINISHED")
        except Exception:
            for run_id in runs.values():
                self.client.set_tag(run_id, "stage", f"{stage}_failed")
                self.client.set_terminated(run_id, status="FAILED")
            raise

    def log(self, run_id: str, *, params=None, metrics=None, directory: Path | None = None):
        for key, value in (params or {}).items():
            self.client.log_param(run_id, key, value)
        for key, value in (metrics or {}).items():
            if value is not None:
                self.client.log_metric(run_id, key, float(value))
        if directory is not None:
            self.client.log_artifacts(run_id, str(directory), artifact_path="series")
