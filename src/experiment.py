"""Shared inputs, preprocessing, group partitions and experiment artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from src.data import PROJECT_ROOT, load_config, read_table, verify_checksum, write_atomic
from src.features import AgeMaleInteraction


NUMERIC = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL = ["Sex", "Pclass", "Embarked"]


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: object) -> None:
    write_atomic(path, json_bytes(value))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    write_atomic(path, frame.to_csv(index=False, float_format="%.17g").encode())


def read_settings(path: Path) -> dict:
    """Validate the supported baseline configuration before any fitting."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["features"] != {"numeric": NUMERIC, "categorical": CATEGORICAL}:
        raise ValueError("The baseline must use exactly the seven declared features")
    if config["prediction"]["threshold"] != 0.5:
        raise ValueError("This experiment uses predict with a fixed threshold of 0.5")
    model = config["model"]
    if model["l1_ratio"] != 0 or model["solver"] != "lbfgs" or model["fit_intercept"] is not True:
        raise ValueError("The baseline requires L2, lbfgs and an unpenalized intercept")
    if model["max_iter"] < 1 or not 0 < model["tol"] < 1:
        raise ValueError("Invalid solver settings")
    validation = config["validation"]
    if min(validation["outer_splits"], validation["inner_splits"]) < 2:
        raise ValueError("Cross-validation needs at least two folds")
    if not 0 <= validation["test_fold"] < validation["outer_splits"]:
        raise ValueError("Invalid final test fold")
    search = config["search"]
    if not search["C"] or any(not np.isfinite(c) or c <= 0 for c in search["C"]):
        raise ValueError("C must contain finite positive values")
    if len(set(search["C"])) != len(search["C"]):
        raise ValueError("Duplicate C values")
    if not search["class_weight"] or any(w not in (None, "balanced") for w in search["class_weight"]):
        raise ValueError("Only None and balanced class weights are supported")
    if len(set(search["class_weight"])) != len(search["class_weight"]):
        raise ValueError("Duplicate class weights")
    if config["bootstrap"]["n_resamples"] < 2 or not 0 < config["bootstrap"]["confidence_level"] < 1:
        raise ValueError("Invalid bootstrap settings")
    return config


def load_labeled(data_dir: Path, data_config: Path) -> tuple[pd.DataFrame, str]:
    """Read only train.csv; the unlabeled competition test is never evaluated."""
    spec = load_config(data_config)["files"]["train.csv"]
    content = (data_dir / "train.csv").read_bytes()
    verify_checksum(content, spec["sha256"], "train.csv")
    return read_table(content, "train.csv", spec["rows"]), digest(content)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select inputs and normalize missing-value types without learning statistics."""
    features = frame[NUMERIC + CATEGORICAL].copy()
    features[NUMERIC] = features[NUMERIC].astype(float)
    for name in CATEGORICAL:
        features[name] = features[name].astype(object).where(features[name].notna(), np.nan)
    return features


def build_pipeline(config: dict, c: float, class_weight: str | None, *,
                   interaction: bool = False) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ]
    if interaction:
        transformers.append(("age_male", Pipeline([
            ("interact", AgeMaleInteraction()), ("scale", StandardScaler()),
        ]), ["Age", "Sex"]))
    return Pipeline([
        ("prepare", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(C=c, class_weight=class_weight, **config["model"])),
    ])


def rows_by_id(frame: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    return frame.set_index("PassengerId", drop=False).loc[ids].reset_index(drop=True)


def check_partition(frame: pd.DataFrame, left: list[int], right: list[int]) -> None:
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("Duplicate passengers in a partition")
    if set(left) & set(right) or set(left) | set(right) != set(frame["PassengerId"]):
        raise ValueError("Partitions must cover each passenger exactly once")
    train, validation = rows_by_id(frame, left), rows_by_id(frame, right)
    if set(train["Ticket"]) & set(validation["Ticket"]):
        raise ValueError("Ticket groups cross the partition boundary")
    if any(set(part["Survived"]) != {0, 1} for part in (train, validation)):
        raise ValueError("Every training and validation partition must contain both classes")


def validate_splits(frame: pd.DataFrame, splits: dict) -> None:
    check_partition(frame, splits["development_ids"], splits["test_ids"])
    development = rows_by_id(frame, splits["development_ids"])
    validation_ids = []
    for fold in splits["inner_folds"]:
        check_partition(development, fold["train_ids"], fold["validation_ids"])
        validation_ids.extend(fold["validation_ids"])
    if sorted(validation_ids) != sorted(splits["development_ids"]):
        raise ValueError("Each development passenger needs exactly one inner validation prediction")


def make_splits(frame: pd.DataFrame, config: dict) -> dict:
    """Fix one outer test fold, then split the remaining rows afresh."""
    validation = config["validation"]
    outer = StratifiedGroupKFold(
        n_splits=validation["outer_splits"], shuffle=True, random_state=config["seed"],
    )
    outer_folds = list(outer.split(frame, frame["Survived"].astype(int), frame["Ticket"]))
    train_index, test_index = outer_folds[validation["test_fold"]]
    development = frame.iloc[train_index]
    inner = StratifiedGroupKFold(
        n_splits=validation["inner_splits"], shuffle=True, random_state=config["seed"],
    )
    splits = {
        "development_ids": development["PassengerId"].astype(int).tolist(),
        "test_ids": frame.iloc[test_index]["PassengerId"].astype(int).tolist(),
        "inner_folds": [
            {"fold": number,
             "train_ids": development.iloc[train]["PassengerId"].astype(int).tolist(),
             "validation_ids": development.iloc[val]["PassengerId"].astype(int).tolist()}
            for number, (train, val) in enumerate(inner.split(
                development, development["Survived"].astype(int), development["Ticket"],
            ))
        ],
    }
    validate_splits(frame, splits)
    return splits


def sample_summary(frame: pd.DataFrame) -> dict:
    return {
        "passengers": len(frame), "ticket_groups": int(frame["Ticket"].nunique()),
        "class_0": int((frame["Survived"] == 0).sum()),
        "class_1": int((frame["Survived"] == 1).sum()),
        "survival_fraction": float(frame["Survived"].mean()),
    }


def split_summary(frame: pd.DataFrame, splits: dict) -> dict:
    return {
        "development": sample_summary(rows_by_id(frame, splits["development_ids"])),
        "test": sample_summary(rows_by_id(frame, splits["test_ids"])),
        "inner_folds": [
            {"fold": fold["fold"],
             "train": sample_summary(rows_by_id(frame, fold["train_ids"])),
             "validation": sample_summary(rows_by_id(frame, fold["validation_ids"]))}
            for fold in splits["inner_folds"]
        ],
    }


def runtime_versions() -> dict:
    packages = ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "PyYAML")
    return {"python": platform.python_version(), **{
        package: importlib.metadata.version(package) for package in packages
    }}


def load_run(artifacts_dir: Path) -> dict:
    """Reject incomplete or mismatched artifacts before loading fitted objects."""
    run = read_json(artifacts_dir / "run.json")
    identity = {key: value for key, value in run.items() if key != "run_id"}
    if digest(json_bytes(identity)) != run["run_id"]:
        raise ValueError("Run manifest checksum mismatch")
    for name, expected in run["files"].items():
        verify_checksum((artifacts_dir / name).read_bytes(), expected, name)
    return run


def stage_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    parser.add_argument("--data-config", type=Path, default=PROJECT_ROOT / "configs/data.yaml")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts/baseline")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser
