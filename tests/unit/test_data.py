"""Synthetic fixtures and local official-copy checks; no network in tests."""

from copy import deepcopy
import io
import json
from pathlib import Path
import stat
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import ParameterGrid
from pydantic import ValidationError

from bank_marketing.config import BASELINE, load_config, validate_config
from bank_marketing.data.audit import adapt_demo_records, audit_dataset, normalize_records
from bank_marketing.data.ingest import _archive_files, ingest_dataset, read_csv
from bank_marketing.data.splits import create_splits, inner_folds
from bank_marketing.schemas import CORE_FEATURES, DATASET_SHA256, DICTIONARY_SHA256, DomainError, SPLIT_RANGES, TRAINING_COLUMNS, ScoreResult, ScoreRow, RankingResult
from bank_marketing.services.ranking import rank_records

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def core():
    return read_csv(FIXTURES / "demo_valid.csv")


@pytest.fixture
def training(core):
    values = {"contact": "cellular", "month": "may", "day_of_week": "mon", "duration": 100, "campaign": 1,
              "emp.var.rate": 1.1, "cons.price.idx": 93.2, "cons.conf.idx": -40.0, "euribor3m": 4.5, "nr.employed": 5100.0}
    data = core.drop(columns="record_id").copy()
    for name, value in values.items():
        data[name] = value
    data["y"] = ["no", "yes", "no", "yes"]
    return data.loc[:, list(TRAINING_COLUMNS)]


def test_domain_error_contract():
    error = DomainError("ROW_INVALID", "Invalid row", {"column": "age"})
    assert error.code == "ROW_INVALID" and error.message == str(error)
    assert error.details == {"column": "age"} and len(error.error_id) == 32


def test_valid_invalid_demo_files(core):
    assert audit_dataset(core, "scoring")["valid"]
    report = audit_dataset(read_csv(FIXTURES / "demo_invalid.csv"), "scoring")
    assert not report["valid"]
    assert {error["column"] for error in report["errors"]} >= {"record_id", "age", "job", "pdays", "default"}
    assert {error["input_row_id"] for error in report["errors"]} == {0, 1, 2}
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("separator", [";", ","])
def test_csv_utf8_preserves_ids_and_order(core, separator):
    core["record_id"] = ["001", "002", "003", "004"]
    data = read_csv(("\ufeff" + core.to_csv(index=False, sep=separator)).encode())
    assert data["record_id"].tolist() == ["001", "002", "003", "004"]
    assert list(data.columns) == list(core.columns)
    assert data.attrs["separator"] == separator
    assert audit_dataset(data, "scoring")["valid"]


@pytest.mark.parametrize("content", [b"age;age\n35;35\n", b"age;job\n35\n", b"age;job\n35;admin.;extra\n",
    b"age;job\n35;\"admin.\n", b"age\tjob\n35\tadmin.\n", b"age;job\n\n", b"age;job\n", b"\xff;job\n35;x\n", b"age ;job\n35;x\n", b"age;job\n35;\x00\n"])
def test_bad_csv_rejected_before_pandas(content):
    with pytest.raises(DomainError) as error:
        read_csv(content)
    assert error.value.code == "SCHEMA_INVALID"


def test_csv_limits(core):
    content = core.to_csv(index=False, sep=";").encode()
    with pytest.raises(DomainError):
        read_csv(content, max_bytes=len(content) - 1)
    with pytest.raises(DomainError):
        read_csv(content, max_rows=3)
    assert len(read_csv(content, max_rows=4, max_bytes=len(content))) == 4


@pytest.mark.parametrize("column,value", [("age", 0), ("age", 121), ("age", 1.1), ("age", True), ("age", float("nan")),
    ("age", float("inf")), ("pdays", -1), ("pdays", 1000), ("previous", -1), ("previous", "garbage"),
    ("job", "Admin."), ("job", ""), ("job", None), ("loan", "YES"), ("record_id", 123)])
def test_invalid_rows(core, column, value):
    data = core.astype(object)
    data.index = [12, 22, 32, 42]
    data.iloc[1, data.columns.get_loc(column)] = value
    report = audit_dataset(data, "scoring")
    assert not report["valid"]
    assert any(error["column"] == column and error["input_row_id"] == 1 for error in report["errors"])
    json.dumps(report, allow_nan=False)


def test_whitespace_unknown_and_semantic_warnings(core):
    core.loc[0, "job"] = " unknown "
    core.loc[0, "age"] = 120
    core.loc[0, "previous"] = 3
    original = core.copy(deep=True)
    report = audit_dataset(core, "scoring")
    assert report["valid"]
    assert {item["code"] for item in report["warnings"]} >= {"WHITESPACE_NORMALIZED", "SEMANTIC_INCONSISTENCY", "OUTSIDE_HISTORICAL_RANGE"}
    pd.testing.assert_frame_equal(core, original)
    assert normalize_records(core).loc[0, "job"] == "unknown"
    core.loc[1, "record_id"] = " " + core.loc[0, "record_id"] + " "
    assert not audit_dataset(core, "scoring")["valid"]


def test_strict_modes_and_demo_adapter(core, training):
    assert audit_dataset(training)["valid"]
    assert not audit_dataset(core)["valid"]
    assert not audit_dataset(training, "scoring")["valid"]
    assert not audit_dataset(training.assign(row_id=np.arange(4)))["valid"]
    assert not audit_dataset(core.assign(phone="123"), "scoring")["valid"]
    demo = adapt_demo_records(training)
    assert list(demo["records"].columns) == list(CORE_FEATURES)
    assert demo["labels"].tolist() == [0, 1, 0, 1]
    assert set(demo["dropped_columns"]) == set(TRAINING_COLUMNS) - set(CORE_FEATURES)
    training.loc[0, "y"] = "1"
    assert not audit_dataset(training)["valid"]
    with pytest.raises(DomainError):
        adapt_demo_records(training)
    duplicate = pd.concat([core, core[["age"]]], axis=1)
    assert not audit_dataset(duplicate, "scoring")["valid"]


def test_exact_config_and_candidate_counts():
    config = load_config(ROOT / "configs" / "experiment.yaml")
    assert [len(ParameterGrid(config["grids"][model])) for model in config["models"]] == [6, 3, 16, 8, 8]
    assert config["model_parameters"]["random_forest"]["n_jobs"] == 2
    assert validate_config(config) == config


@pytest.mark.parametrize("section,field,value", [(None, "seed", True), (None, "seed", 1), (None, "models", ["logistic_regression"]),
    ("capacity", "default_fraction", 1.1), ("split", "selection", [28830, 32950]), ("cv", "gap", -1),
    ("upload", "invalid_policy", "partial"), ("features", "set", "invalid"), ("dataset", "expected_rows", 41188.0)])
def test_invalid_config(section, field, value):
    config = deepcopy(BASELINE)
    (config if section is None else config[section])[field] = value
    with pytest.raises(DomainError):
        validate_config(config)


def test_config_missing_and_changed_grid(tmp_path):
    config = validate_config(BASELINE)
    config["grids"]["gaussian_nb"]["estimator__var_smoothing"] = [1e-9]
    with pytest.raises(DomainError):
        validate_config(config)
    config = deepcopy(BASELINE)
    del config["upload"]["max_bytes"]
    with pytest.raises(DomainError):
        validate_config(config)
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 42\nseed: 1\n")
    with pytest.raises(DomainError):
        load_config(path)


def test_fixed_splits_and_folds(training):
    data = training.iloc[np.arange(41188) % len(training)].reset_index(drop=True)
    data["row_id"] = np.arange(len(data))
    manifest = create_splits(data, BASELINE)
    assert manifest["ranges"] == SPLIT_RANGES
    assert [manifest["counts"][name]["rows"] for name in SPLIT_RANGES] == [24712, 4119, 4119, 8238]
    folds = inner_folds(data.iloc[:24712])
    assert [(int(fit[0]), int(fit[-1]) + 1, int(valid[0]), int(valid[-1]) + 1) for fit, valid in folds] == [(0,6178,6178,12356), (0,12356,12356,18534), (0,18534,18534,24712)]
    for fit, valid in inner_folds(data.iloc[:24712], gap=2):
        assert fit[-1] + 3 == valid[0]
    shuffled_ids = data.copy()
    shuffled_ids.loc[0, "row_id"] = 1
    with pytest.raises(DomainError):
        create_splits(shuffled_ids, BASELINE)
    with pytest.raises(DomainError):
        inner_folds(data.iloc[:24712].assign(y="no"))
    with pytest.raises(DomainError):
        inner_folds(data)


def _zip(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return stream.getvalue()


def test_safe_nested_archive_selects_full():
    inner = _zip([("bank-additional/bank-additional-full.csv", b"full"), ("bank-additional/bank-additional-names.txt", b"dictionary"), ("bank-additional/bank-additional.csv", b"sample")])
    outer = _zip([("bank-additional.zip", inner), ("bank.zip", b"not read")])
    assert _archive_files(outer) == {"bank-additional-full.csv": b"full", "bank-additional-names.txt": b"dictionary"}


@pytest.mark.parametrize("name", ["../evil", "/evil", "C:/evil", "folder/../../evil", "folder\\..\\evil", "file:stream"])
def test_zip_traversal_rejected(name):
    with pytest.raises(DomainError):
        _archive_files(_zip([(name, b"bad")]))


def test_nested_zip_and_symlink_rejected(tmp_path):
    inner = _zip([("../evil", b"bad")])
    with pytest.raises(DomainError):
        _archive_files(_zip([("bank-additional.zip", inner)]))
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(DomainError):
        _archive_files(_zip([(link, b"../outside")]))
    archive = tmp_path / "bank.zip"
    archive.write_bytes(_zip([("bank-full.csv", b"wrong")]))
    with pytest.raises(DomainError) as error:
        ingest_dataset(archive, tmp_path / "data" / "raw")
    assert error.value.code == "DATASET_VARIANT_MISMATCH"
    archive.write_bytes(_zip([("bank-additional-full.csv", b"wrong"), ("bank-additional-names.txt", b"wrong")]))
    with pytest.raises(DomainError) as error:
        ingest_dataset(archive, tmp_path / "data" / "raw")
    assert error.value.code == "DATASET_HASH_MISMATCH"
    assert not (tmp_path / "data" / "raw").exists()


def test_official_local_ingest_integrity_and_immutable_manifest(tmp_path):
    source = ROOT / "data" / "raw"
    if not (source / "bank-additional-full.csv").is_file():
        pytest.skip("Official local dataset absent; run ingest before official-copy checks.")
    destination = tmp_path / "data" / "raw"
    manifest = ingest_dataset(source, destination)
    assert manifest["dataset_sha256"] == DATASET_SHA256
    assert manifest["dictionary_sha256"] == DICTIONARY_SHA256
    assert manifest["audit_counts"]["target"] == {"no": 36548, "yes": 4640}
    assert manifest["audit_counts"]["duplicate_rows"] == 12
    path = destination.parent / "manifests" / "dataset_manifest.json"
    before = path.read_bytes()
    assert ingest_dataset(source, destination) == manifest
    assert path.read_bytes() == before
    dictionary = destination / "bank-additional-names.txt"
    dictionary.write_bytes(dictionary.read_bytes() + b"tampered")
    with pytest.raises(DomainError) as error:
        ingest_dataset(source, destination)
    assert error.value.code == "DATASET_HASH_MISMATCH"
    assert path.read_bytes() == before


@pytest.mark.parametrize("column,value", [("age", [1, 2]), ("record_id", {"id": "x"}), ("job", ["admin."])])
def test_non_scalar_row_errors_are_serializable(core, column, value):
    data = core.astype(object)
    data.at[0, column] = value
    report = audit_dataset(data, "scoring")
    assert not report["valid"]
    assert any(error["column"] == column and error["input_row_id"] == 0 for error in report["errors"])
    json.dumps(report, allow_nan=False)


def test_finite_extreme_numeric_summary(training):
    training["euribor3m"] = 1e308
    report = audit_dataset(training)
    assert report["valid"]
    assert report["descriptive"]["euribor3m"]["mean"] == 1e308
    assert report["descriptive"]["euribor3m"]["median"] == 1e308
    json.dumps(report, allow_nan=False)

def test_typed_service_contracts_preserve_dictionary_results():
    from pydantic import ValidationError
    from bank_marketing.schemas import ScoreResult, ScoreRow, RankingResult
    from bank_marketing.services.ranking import rank_records
    score = {"batch_id": "synthetic", "run_id": "run", "model_id": "logistic_regression", "feature_set": "core_v1", "threshold": 0.5,
             "rows": [{"input_row_id": 0, "record_id": "r0", "p_yes": 0.4, "predicted_label": "no", "warnings": []},
                      {"input_row_id": 1, "record_id": "r1", "p_yes": 0.8, "predicted_label": "yes", "warnings": []}], "warnings": []}
    typed = ScoreResult.model_validate(score)
    assert typed.model_dump(exclude_unset=True) == score
    ranked = rank_records(score, {"k": 1})
    assert isinstance(ranked, dict)
    assert RankingResult.model_validate(ranked).model_dump(exclude_unset=True) == ranked
    assert score["rows"][0]["input_row_id"] == 0
    for probability in (float("nan"), float("inf"), -0.1, 1.1, True):
        with pytest.raises(ValidationError):
            ScoreRow.model_validate({**score["rows"][0], "p_yes": probability})
    with pytest.raises(ValidationError):
        ScoreRow.model_validate({**score["rows"][0], "input_row_id": True})
    with pytest.raises(ValidationError):
        ScoreResult.model_validate({**score, "rows": [score["rows"][0], score["rows"][0]]})
    with pytest.raises(ValidationError):
        RankingResult.model_validate({**ranked, "capacity_k": 3})
    with pytest.raises(ValidationError):
        RankingResult.model_validate({**ranked, "call_fraction": 1.0})
    ranked["rows"][0]["selected_for_contact"] = False
    with pytest.raises(ValidationError):
        RankingResult.model_validate(ranked)
@pytest.mark.parametrize('content', [b'[]', b'null', b'42'])
def test_local_ingest_malformed_manifest_fails_safely(tmp_path, content):
    source = ROOT / 'data/raw'
    if not (source / 'bank-additional-full.csv').exists():
        pytest.skip('Official local dataset absent')
    destination = tmp_path / 'data/raw'
    ingest_dataset(source, destination)
    manifest = destination.parent / 'manifests/dataset_manifest.json'
    manifest.write_bytes(content)
    with pytest.raises(DomainError) as failure:
        ingest_dataset(source, destination)
    assert failure.value.code == 'DATASET_HASH_MISMATCH'
def test_csv_numeric_whitespace_preserved_for_audit():
    content = (ROOT / 'tests/fixtures/demo_valid.csv').read_bytes()
    text = content.decode('utf-8-sig').splitlines()
    fields = text[1].split(';')
    age = text[0].split(';').index('age')
    fields[age] = ' ' + fields[age].strip() + ' '
    text[1] = ';'.join(fields)
    records = read_csv('\n'.join(text).encode())
    report = audit_dataset(records, 'scoring')
    assert report['valid']
    assert any(w['column'] == 'age' and w['code'] == 'WHITESPACE_NORMALIZED' for w in report['warnings'])
