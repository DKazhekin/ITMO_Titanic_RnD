"""Download, verify and describe the pinned Titanic dataset."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_NAMES = ("train.csv", "test.csv")
INPUT_COLUMNS = (
    "PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
    "Ticket", "Fare", "Cabin", "Embarked",
)
DTYPES = {
    "PassengerId": "Int64",
    "Survived": "Int64",
    "Pclass": "Int64",
    "Name": "string",
    "Sex": "string",
    "Age": "Float64",
    "SibSp": "Int64",
    "Parch": "Int64",
    "Ticket": "string",
    "Fare": "Float64",
    "Cabin": "string",
    "Embarked": "string",
}


def verify_checksum(content: bytes, expected: str, name: str) -> None:
    """Reject changed bytes instead of silently using a different dataset."""
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")


def load_config(path: Path) -> dict:
    """Read the source manifest and check the fields needed for acquisition."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config.get("files", {})) != set(FILE_NAMES):
        raise ValueError("The source manifest must describe train.csv and test.csv")
    if not config["archive"]["url"].startswith("https://"):
        raise ValueError("The archive URL must use HTTPS")
    for name in FILE_NAMES:
        spec = config["files"][name]
        if not isinstance(spec["rows"], int) or spec["rows"] <= 0:
            raise ValueError(f"Expected row count must be positive for {name}")
    for spec in (config["archive"], *config["files"].values()):
        digest = spec["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Each source checksum must be a 64-character SHA-256 value")
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Source checksums must use lowercase hexadecimal characters")
    return config


def read_table(content: bytes, name: str, expected_rows: int) -> pd.DataFrame:
    """Validate the source schema without imputing or changing observations."""
    try:
        frame = pd.read_csv(
            io.BytesIO(content), dtype=DTYPES, keep_default_na=False, na_values=[""],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid column types or CSV content in {name}: {error}") from error
    expected_columns = list(INPUT_COLUMNS)
    if name == "train.csv":
        expected_columns.insert(1, "Survived")
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Unexpected columns in {name}: {list(frame.columns)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Unexpected row count in {name}: expected {expected_rows}, got {len(frame)}")

    required = ["PassengerId", "Pclass", "Name", "Sex", "SibSp", "Parch", "Ticket"]
    if name == "train.csv":
        required.append("Survived")
    missing_required = frame[required].columns[frame[required].isna().any()].tolist()
    if missing_required:
        raise ValueError(f"Missing required values in {name}: {missing_required}")
    if frame["PassengerId"].duplicated().any() or (frame["PassengerId"] <= 0).any():
        raise ValueError(f"PassengerId must be unique and positive in {name}")

    categories = {"Pclass": {1, 2, 3}, "Sex": {"female", "male"}, "Embarked": {"C", "Q", "S"}}
    if name == "train.csv":
        categories["Survived"] = {0, 1}
    for column, allowed in categories.items():
        if not frame[column].dropna().isin(allowed).all():
            raise ValueError(f"Unexpected values in {name}:{column}")
    for column in ("Age", "Fare", "SibSp", "Parch"):
        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Values must be finite and nonnegative in {name}:{column}")
    return frame


def validate_pair(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check that the two source files refer to different passenger records."""
    overlap = set(train["PassengerId"]) & set(test["PassengerId"])
    if overlap:
        raise ValueError(f"PassengerId overlaps between source files: {len(overlap)}")


def summarize_tables(tables: dict[str, pd.DataFrame], config: dict) -> dict:
    """Describe raw data; do not fit preprocessing or choose a validation split."""
    summary = {"source": config["source"], "archive": config["archive"], "files": {}}
    for name, frame in tables.items():
        summary["files"][name] = {
            "sha256": config["files"][name]["sha256"],
            "rows": len(frame),
            "unique_passenger_ids": int(frame["PassengerId"].nunique()),
            "duplicate_rows_without_id": int(frame.drop(columns="PassengerId").duplicated().sum()),
            "columns": {
                column: {
                    "dtype": str(frame[column].dtype),
                    "missing": int(frame[column].isna().sum()),
                    "unique_nonmissing": int(frame[column].nunique()),
                }
                for column in frame.columns
            },
        }
    train, test = tables["train.csv"], tables["test.csv"]
    summary["files"]["train.csv"]["target_counts"] = {
        str(label): int((train["Survived"] == label).sum()) for label in (0, 1)
    }
    ticket_counts = train["Ticket"].value_counts()
    shared_tickets = set(train["Ticket"]) & set(test["Ticket"])
    summary["record_checks"] = {
        "overlapping_passenger_ids": len(set(train["PassengerId"]) & set(test["PassengerId"])),
        "train_repeated_ticket_groups": int((ticket_counts > 1).sum()),
        "train_passengers_with_repeated_ticket": int(ticket_counts[ticket_counts > 1].sum()),
        "ticket_values_shared_between_source_files": len(shared_tickets),
        "train_passengers_with_ticket_in_test": int(train["Ticket"].isin(shared_tickets).sum()),
        "test_passengers_with_ticket_in_train": int(test["Ticket"].isin(shared_tickets).sum()),
    }
    return summary


def write_atomic(path: Path, content: bytes) -> None:
    """Replace a complete output file and leave unchanged outputs untouched."""
    if path.exists() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_data(config_path: Path, data_dir: Path, summary_path: Path) -> dict:
    """Acquire and validate all source data before publishing local outputs."""
    config = load_config(config_path)
    archive_path = data_dir / "titanic.tgz"
    inputs = {config_path.resolve(), archive_path.resolve()}
    inputs.update((data_dir / name).resolve() for name in FILE_NAMES)
    if summary_path.resolve() in inputs:
        raise ValueError("The summary output must not overwrite source data or configuration")

    for name in FILE_NAMES:
        path = data_dir / name
        if path.exists():
            verify_checksum(path.read_bytes(), config["files"][name]["sha256"], str(path))

    if archive_path.exists():
        archive = archive_path.read_bytes()
    else:
        request = Request(config["archive"]["url"], headers={"User-Agent": "titanic-rnd"})
        with urlopen(request, timeout=30) as response:
            archive = response.read()
    verify_checksum(archive, config["archive"]["sha256"], "titanic.tgz")

    contents = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for name in FILE_NAMES:
            spec = config["files"][name]
            members = [member for member in bundle.getmembers() if member.name == spec["member"]]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(f"Expected one regular archive member for {name}")
            with bundle.extractfile(members[0]) as source:
                contents[name] = source.read()
            verify_checksum(contents[name], spec["sha256"], name)

    tables = {
        name: read_table(contents[name], name, config["files"][name]["rows"])
        for name in FILE_NAMES
    }
    validate_pair(tables["train.csv"], tables["test.csv"])
    summary = summarize_tables(tables, config)

    write_atomic(archive_path, archive)
    for name in FILE_NAMES:
        write_atomic(data_dir / name, contents[name])
    write_atomic(summary_path, (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return summary


def main() -> None:
    """Run source acquisition and descriptive checks from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data.yaml")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "reports/data_summary.json")
    args = parser.parse_args()
    try:
        summary = prepare_data(args.config, args.data_dir, args.summary)
    except (OSError, ValueError, KeyError, tarfile.TarError, URLError, yaml.YAMLError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(
        f"Verified {summary['files']['train.csv']['rows']} labeled and "
        f"{summary['files']['test.csv']['rows']} unlabeled passengers. Summary: {args.summary}"
    )


if __name__ == "__main__":
    main()
