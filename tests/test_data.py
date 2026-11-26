"""Check source integrity, schema validation and repeatable local acquisition."""

import hashlib
import io
import json
import tarfile

import pandas as pd
import pytest
import yaml

from src import data


TRAIN = (
    "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n"
    "1,0,3,Alpha,male,22,0,0,0007,7.25,,S\n"
    "2,1,1,Beta,female,,1,0,NA,71.5,C85,C\n"
    "3,0,3,Gamma,male,30,1,0,0007,7.25,,S\n"
    "4,1,2,Delta,female,19,0,0,T4,13,,Q\n"
).encode()
TEST = (
    "PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n"
    "5,3,Epsilon,female,27,0,0,0007,7.25,,S\n"
    "6,2,Zeta,male,,0,1,T6,,,\n"
).encode()


def make_archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture
def source(tmp_path, monkeypatch):
    archive = make_archive({"titanic/train.csv": TRAIN, "titanic/test.csv": TEST})
    config = {
        "source": {"original_url": "https://example.test/original", "revision": "fixture"},
        "archive": {"url": "https://example.test/titanic.tgz", "sha256": hashlib.sha256(archive).hexdigest()},
        "files": {
            name: {"member": f"titanic/{name}", "sha256": hashlib.sha256(content).hexdigest(), "rows": rows}
            for name, content, rows in (("train.csv", TRAIN, 4), ("test.csv", TEST, 2))
        },
    }
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(archive))
    return config_path, tmp_path / "raw", tmp_path / "summary.json", config, archive


def test_acquisition_reports_raw_data_and_repeats_without_network(source, monkeypatch):
    config_path, data_dir, report_path, _, archive = source
    summary = data.prepare_data(config_path, data_dir, report_path)
    assert (data_dir / "titanic.tgz").read_bytes() == archive
    assert (data_dir / "train.csv").read_bytes() == TRAIN
    assert (data_dir / "test.csv").read_bytes() == TEST
    assert json.loads(report_path.read_text()) == summary
    assert summary["files"]["train.csv"]["target_counts"] == {"0": 2, "1": 2}
    assert summary["files"]["train.csv"]["columns"]["Age"]["missing"] == 1
    assert summary["files"]["test.csv"]["columns"]["Fare"]["missing"] == 1
    assert summary["record_checks"]["train_passengers_with_repeated_ticket"] == 2
    assert summary["record_checks"]["ticket_values_shared_between_source_files"] == 1

    paths = [data_dir / "titanic.tgz", data_dir / "train.csv", data_dir / "test.csv", report_path]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    def unexpected_network(*args, **kwargs):
        pytest.fail("A cached dataset must not require the network")

    monkeypatch.setattr(data, "urlopen", unexpected_network)
    assert data.prepare_data(config_path, data_dir, report_path) == summary
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before


def test_text_identifiers_and_missing_values_are_preserved():
    frame = data.read_table(TRAIN, "train.csv", 4)
    assert frame.loc[0, "Ticket"] == "0007"
    assert frame.loc[1, "Ticket"] == "NA"
    assert pd.isna(frame.loc[1, "Age"])
    assert pd.isna(frame.loc[0, "Cabin"])


def test_changed_download_is_rejected_before_writing(source, monkeypatch):
    config_path, data_dir, report_path, _, archive = source
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(archive + b"changed"))
    with pytest.raises(ValueError, match="SHA-256 mismatch for titanic.tgz"):
        data.prepare_data(config_path, data_dir, report_path)
    assert not data_dir.exists()
    assert not report_path.exists()


def test_member_checksum_is_verified_independently(source):
    config_path, data_dir, report_path, config, _ = source
    config["files"]["test.csv"]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="SHA-256 mismatch for test.csv"):
        data.prepare_data(config_path, data_dir, report_path)
    assert not data_dir.exists()
    assert not report_path.exists()


@pytest.mark.parametrize("filename", ["titanic.tgz", "train.csv", "test.csv"])
def test_corrupt_local_input_is_rejected_and_not_overwritten(source, filename):
    config_path, data_dir, report_path, _, _ = source
    data.prepare_data(config_path, data_dir, report_path)
    previous_report = report_path.read_bytes()
    damaged_path = data_dir / filename
    damaged_path.write_bytes(b"local corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.prepare_data(config_path, data_dir, report_path)
    assert damaged_path.read_bytes() == b"local corruption"
    assert report_path.read_bytes() == previous_report


def test_missing_archive_member_is_rejected(source, monkeypatch):
    config_path, data_dir, report_path, config, _ = source
    archive = make_archive({"titanic/train.csv": TRAIN})
    config["archive"]["sha256"] = hashlib.sha256(archive).hexdigest()
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(archive))
    with pytest.raises(ValueError, match="Expected one regular archive member for test.csv"):
        data.prepare_data(config_path, data_dir, report_path)
    assert not data_dir.exists()


def test_unrequested_archive_paths_are_never_extracted(source, monkeypatch):
    config_path, data_dir, report_path, config, _ = source
    archive = make_archive({
        "titanic/train.csv": TRAIN, "titanic/test.csv": TEST, "../escaped.txt": b"unexpected",
    })
    config["archive"]["sha256"] = hashlib.sha256(archive).hexdigest()
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(archive))
    data.prepare_data(config_path, data_dir, report_path)
    assert not (data_dir.parent / "escaped.txt").exists()
    assert {path.name for path in data_dir.iterdir()} == {"titanic.tgz", "train.csv", "test.csv"}


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (b"PassengerId,Survived", b"PassengerId,Other", "Unexpected columns"),
        (b"1,0,3,Alpha", b"1,2,3,Alpha", "Unexpected values in train.csv:Survived"),
        (b"1,0,3,Alpha", b",0,3,Alpha", "Missing required values"),
        (b"2,1,1,Beta", b"1,1,1,Beta", "PassengerId must be unique"),
        (b"1,0,3,Alpha", b"1.5,0,3,Alpha", "Invalid column types"),
        (b"Alpha,male,22", b"Alpha,male,-22", "finite and nonnegative"),
        (b"0007,7.25", b"0007,inf", "finite and nonnegative"),
    ],
)
def test_invalid_observations_are_rejected(original, replacement, message):
    with pytest.raises(ValueError, match=message):
        data.read_table(TRAIN.replace(original, replacement, 1), "train.csv", 4)


def test_row_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Unexpected row count"):
        data.read_table(TRAIN, "train.csv", 5)


def test_test_file_cannot_contain_target():
    with pytest.raises(ValueError, match="Unexpected columns"):
        data.read_table(TRAIN, "test.csv", 4)


def test_passenger_overlap_is_rejected():
    train = data.read_table(TRAIN, "train.csv", 4)
    test = data.read_table(TEST.replace(b"5,3,Epsilon", b"1,3,Epsilon"), "test.csv", 2)
    with pytest.raises(ValueError, match="PassengerId overlaps"):
        data.validate_pair(train, test)


@pytest.mark.parametrize("destination", ["config", "train.csv", "test.csv", "titanic.tgz"])
def test_summary_cannot_replace_an_input(source, destination):
    config_path, data_dir, _, _, _ = source
    report_path = config_path if destination == "config" else data_dir / destination
    with pytest.raises(ValueError, match="must not overwrite"):
        data.prepare_data(config_path, data_dir, report_path)
